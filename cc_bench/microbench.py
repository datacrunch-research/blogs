import torch
import time
import argparse
import sys
import math
import os
import warnings
import subprocess
import ctypes

# --- Suppress known "False Positive" Warnings for Blackwell ---
warnings.filterwarnings("ignore", message=".*NVIDIA GB300.*")
warnings.filterwarnings("ignore", message=".*sm_103.*")

# --- Robust Initialization ---
def init_torch_cuda():
    """
    Trigger PyTorch lazy initialization immediately to catch environment issues early.
    """
    try:
        if not torch.cuda.is_available():
            return
        # Standard way to force initialization
        _ = torch.zeros(1).cuda()
    except Exception as e:
        # Print this immediately so we see it before any other logic
        print(f"⚠️  PyTorch CUDA Init Failed: {e}")

# Initialize PyTorch CUDA immediately
init_torch_cuda()

# --- System Diagnostics ---
def check_cublas_version():
    print("\n--- 🔍 System Diagnostics ---")
    try:
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   PyTorch CUDA:    {torch.version.cuda}")
    except:
        print("   PyTorch:         Unknown")
    
    try:
        libs = [line for line in os.popen('ldconfig -p | grep cublas')]
        print(f"   System cuBLAS:   {len(libs)} libs found via ldconfig")
    except:
        pass

def get_gpu_status(device_id):
    try:
        cmd = f"nvidia-smi --query-gpu=utilization.gpu,clocks.current.sm,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits -i {device_id}"
        output = subprocess.check_output(cmd.split()).decode('utf-8').strip()
        util, clock, pwr, pwr_lim, temp = output.split(',')
        return {
            "util": f"{util.strip()}%",
            "clock": f"{clock.strip()} MHz",
            "power": f"{pwr.strip()}/{pwr_lim.strip()} W",
            "temp": f"{temp.strip()} C"
        }
    except:
        return {"util": "N/A", "clock": "N/A", "power": "N/A", "temp": "N/A"}

# --- Low Level cuBLAS Control ---
class CuBLASController:
    CUBLAS_DEFAULT_MATH = 0
    CUBLAS_TENSOR_OP_MATH = 1 

    def __init__(self):
        self.lib = None
        self.set_mode_func = None
        # Lazy load in set_math_mode to prevent init conflicts

    def _load_lib(self):
        if self.lib: return
        try:
            candidates = ['libcublas.so', 'libcublas.so.12', 'libcublas.so.11']
            for c in candidates:
                try:
                    self.lib = ctypes.cdll.LoadLibrary(c)
                    break
                except OSError:
                    pass
            
            if self.lib:
                self.set_mode_func = self.lib.cublasSetMathMode
                self.set_mode_func.argtypes = [ctypes.c_void_p, ctypes.c_int]
                self.set_mode_func.restype = ctypes.c_int
        except Exception as e:
            print(f"   ⚠️  CuBLAS Controller Init Failed: {e}")

    def set_math_mode(self, mode):
        self._load_lib()
        if not self.set_mode_func: return False
        try:
            torch.matmul(torch.ones(1,1, device='cuda'), torch.ones(1,1, device='cuda'))
            handle_ptr = torch.cuda.current_blas_handle()
            if handle_ptr == 0: return False
            status = self.set_mode_func(ctypes.c_void_p(handle_ptr), mode)
            return status == 0
        except Exception:
            return False

def get_device():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires an NVIDIA GPU.")
        sys.exit(1)
    return torch.device("cuda")

class GPUBenchmark:
    def __init__(self, device_id=0):
        self.device_id = device_id
        
        # Defensive Check
        try:
            self.device = torch.device(f"cuda:{device_id}")
            # Ensure context is active before doing anything else
            torch.zeros(1).to(self.device)
        except RuntimeError as e:
            print(f"\n❌ FATAL: Failed to initialize GPU {device_id}.")
            print(f"   Error: {e}")
            print("\n   Troubleshooting:")
            print("   1. Driver: Check 'nvidia-smi'. If it hangs or fails, the driver/fabric-manager is down.")
            print("   2. Permissions: Ensure your user has access to /dev/nvidia*.")
            print("   3. Env: Check 'CUDA_VISIBLE_DEVICES'.")
            sys.exit(1)
            
        self.props = torch.cuda.get_device_properties(self.device)
        self.total_mem = self.props.total_memory
        self.cublas = CuBLASController()
        
        print(f"\n🔥 Target GPU: {self.props.name} (Compute Cap: {self.props.major}.{self.props.minor})")
        print(f"   Total VRAM: {self.total_mem / 1024**3:.2f} GB")
        
        status = get_gpu_status(self.device_id)
        print(f"   Status: Clock={status['clock']} | Power={status['power']} | Temp={status['temp']}")

    def _measure_time(self, func, *args, iterations=10, warmup=3):
        try:
            for _ in range(warmup):
                func(*args)
            torch.cuda.synchronize(self.device)
        except Exception as e:
            raise e

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iterations):
            func(*args)
        end_event.record()
        torch.cuda.synchronize(self.device)
        
        return start_event.elapsed_time(end_event) / 1000.0

    def benchmark_compute_precision_sweep(self, size=None, iterations=20):
        base_size = size if size is not None else 4096
        boost_size = 16384 if (size is None and self.props.major >= 9) else base_size
        tensor_size_fp64 = (base_size // 128) * 128 if base_size >= 128 else 128

        print(f"\n--- 🧮 Compute Performance ---")
        print(f"   {'Precision':<18} | {'Backend/Mode':<15} | {'Size':<6} | {'Avg Time (ms)':<13} | {'TFLOPS':<10}")
        print("   " + "-"*80)

        test_configs = [
            ("FP64 (Double)", torch.float64, base_size, None, 0, "Vector (Leg)"),
            ("FP64 (Double)", torch.float64, tensor_size_fp64, None, 1, "Tensor (TC)"),
            ("FP32 (Float)",  torch.float32, base_size, False, 0, "Vector (F32)"),
            ("TF32 (Tensor)", torch.float32, base_size, True,  0, "Tensor (TF32)"),
            ("FP16 (Half)",   torch.float16, boost_size, None, 0, "Tensor (TC)"),
            ("BF16 (BFloat)", torch.bfloat16, boost_size, None, 0, "Tensor (TC)")
        ]

        for name, dtype, run_size, tf32_setting, cublas_mode, label in test_configs:
            try:
                if tf32_setting is not None:
                    torch.backends.cuda.matmul.allow_tf32 = tf32_setting
                self.cublas.set_math_mode(cublas_mode)
                
                ops = 2 * (run_size ** 3)
                a = torch.randn(run_size, run_size, device=self.device, dtype=dtype)
                b = torch.randn(run_size, run_size, device=self.device, dtype=dtype)
                
                def workload(): torch.matmul(a, b)
                
                duration = self._measure_time(workload, iterations=iterations)
                avg_time = duration / iterations
                tflops = (ops / avg_time) / 1e12
                
                print(f"   {name:<18} | {label:<15} | {run_size:<6} | {avg_time*1000:<13.2f} | {tflops:<10.2f}")
                del a, b
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"   {name:<18} | {label:<15} | {run_size:<6} | {'Not Supported':<13} | -")

        torch.backends.cuda.matmul.allow_tf32 = True
        self.cublas.set_math_mode(0)

        if hasattr(torch, 'float8_e4m3fn') and hasattr(torch, '_scaled_mm'):
            try:
                run_size = boost_size
                ops = 2 * (run_size ** 3)
                scale_a = torch.tensor([1.0], device=self.device)
                scale_b = torch.tensor([1.0], device=self.device)
                a_fp8 = torch.randn(run_size, run_size, device=self.device, dtype=torch.float32).to(torch.float8_e4m3fn).contiguous()
                b_fp8 = torch.randn(run_size, run_size, device=self.device, dtype=torch.float32).to(torch.float8_e4m3fn).contiguous()
                b_fp8_col = b_fp8.t()

                def workload_fp8():
                    torch._scaled_mm(a_fp8, b_fp8_col, scale_a=scale_a, scale_b=scale_b, 
                                     out_dtype=torch.bfloat16, use_fast_accum=True)

                duration = self._measure_time(workload_fp8, iterations=iterations)
                avg_time = duration / iterations
                tflops = (ops / avg_time) / 1e12
                print(f"   {'Torch FP8 (E4M3)':<18} | {'Tensor':<15} | {run_size:<6} | {avg_time*1000:<13.2f} | {tflops:<10.2f}")
                del a_fp8, b_fp8
            except Exception as e:
                 print(f"   {'Torch FP8 (E4M3)':<18} | {'Tensor':<15} | {boost_size:<6} | {'Failed':<13} | {str(e)[:20]}...")
        
        # Lazy load TE here to prevent context corruption on import
        self._benchmark_te(boost_size, iterations)

    def _benchmark_te(self, size, iterations):
        # Lazy import inside the method to protect global state
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import Format, DelayedScaling, MXFP8BlockScaling
            TE_INSTALLED = True
        except ImportError:
            print(f"   {'TE FP8/FP4':<18} | {'N/A':<15} | {'-':<6} | {'Not Installed':<13} | -")
            return

        ALIGNMENT = 512
        if size % ALIGNMENT != 0:
            aligned_size = math.ceil(size / ALIGNMENT) * ALIGNMENT
            size = aligned_size
        
        ops = 2 * (size ** 3)
        inp_a = torch.randn(size, size, device=self.device, dtype=torch.bfloat16).contiguous()

        def run_te_gemm(fmt_name, fmt_enum, use_mxfp8=False):
            try:
                if use_mxfp8:
                    recipe = MXFP8BlockScaling(fp8_format=fmt_enum)
                else:
                    recipe = DelayedScaling(fp8_format=fmt_enum)
                
                layer = te.Linear(size, size, bias=False).to(device=self.device, dtype=torch.bfloat16)
                
                def workload():
                    with torch.no_grad():
                        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                            layer(inp_a)

                duration = self._measure_time(workload, iterations=iterations)
                avg_time = duration / iterations
                tflops = (ops / avg_time) / 1e12
                label = f"TE {'MX' if use_mxfp8 else ''}FP8 ({fmt_name})"
                print(f"   {label:<18} | {'Tensor':<15} | {size:<6} | {avg_time*1000:<13.2f} | {tflops:<10.2f}")
            except Exception as e:
                 err_str = str(e)
                 label = f"TE {'MX' if use_mxfp8 else ''}FP8 ({fmt_name})"
                 if "cublas_version" in err_str:
                     print(f"   {label:<18} | {'Tensor':<15} | {size:<6} | {'Env Error':<13} | -")
                 elif "CUBLAS_STATUS_NOT_SUPPORTED" in err_str:
                     print(f"   {label:<18} | {'Tensor':<15} | {size:<6} | {'Not Supp.':<13} | -")
                 else:
                     print(f"   {label:<18} | {'Tensor':<15} | {size:<6} | {'Failed':<13} | {err_str[:20]}...")

        run_te_gemm("E4M3", Format.E4M3, use_mxfp8=False)
        run_te_gemm("E4M3", Format.E4M3, use_mxfp8=True)

        fp4_fmt = getattr(Format, 'E2M1', None)
        if fp4_fmt:
            try:
                recipe = MXFP8BlockScaling(fp8_format=fp4_fmt)
                
                try:
                    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                        with te.fp8_model_init(enabled=True):
                            layer_fp4 = te.Linear(size, size, bias=False).to(device=self.device, dtype=torch.bfloat16)
                except Exception:
                    layer_fp4 = te.Linear(size, size, bias=False).to(device=self.device, dtype=torch.bfloat16)

                def workload_fp4():
                    with torch.no_grad():
                        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                            layer_fp4(inp_a)

                duration = self._measure_time(workload_fp4, iterations=iterations)
                avg_time = duration / iterations
                tflops = (ops / avg_time) / 1e12
                print(f"   {'TE FP4 (E2M1)':<18} | {'Tensor':<15} | {size:<6} | {avg_time*1000:<13.2f} | {tflops:<10.2f}")
            except Exception as e:
                 err_str = str(e)
                 if "cublas_version" in err_str:
                     print(f"   {'TE FP4 (E2M1)':<18} | {'Tensor':<15} | {size:<6} | {'Env Error':<13} | -")
                 elif "CUBLAS_STATUS_NOT_SUPPORTED" in err_str:
                     print(f"   {'TE FP4 (E2M1)':<18} | {'Tensor':<15} | {size:<6} | {'Not Supp.':<13} | -")
                 else:
                     print(f"   {'TE FP4 (E2M1)':<18} | {'Tensor':<15} | {size:<6} | {'Failed':<13} | {str(e)[:20]}...")
        else:
            print(f"   {'TE FP4 (E2M1)':<18} | {'Tensor':<15} | {size:<6} | {'Not in TE':<13} | -")

    def benchmark_bandwidth_sweep(self, mode="d2d", iterations=50):
        target_device = self.device
        is_p2p = False
        if mode == "p2p":
            if torch.cuda.device_count() < 2:
                print("\n--- ⏭️  Skipping P2P Sweep (Only 1 GPU) ---")
                return
            is_p2p = True
            target_device = torch.device(f"cuda:{1 if self.device_id == 0 else 0}")
            can_access = torch.cuda.can_device_access_peer(self.device_id, target_device.index)
            status = "✅ NVLink/P2P" if can_access else "⚠️  PCIe"
        else:
            status = "✅ Intra-Device"

        title_map = {
            "d2d": f"Intra-Device Bandwidth (D2D) - {self.props.name}",
            "h2d": "Host-to-Device Bandwidth (PCIe)",
            "d2h": "Device-to-Host Bandwidth (PCIe)",
            "p2p": f"Inter-Device P2P (GPU{self.device_id} -> GPU{target_device.index}) [{status}]"
        }
        
        print(f"\n--- 🚌 {title_map.get(mode, mode)} ---")
        print(f"   {'Size':>10} | {'Elements':>12} | {'Time (us)':>12} | {'BusBW (GB/s)':>15}")
        print("   " + "-"*55)

        limit_bytes = int(self.total_mem * 0.5)
        if is_p2p:
            mem2 = torch.cuda.get_device_properties(target_device).total_memory
            limit_bytes = min(limit_bytes, int(mem2 * 0.5))
        if mode in ["h2d", "d2h"]:
            limit_bytes = min(limit_bytes, 16 * 1024**3)

        size_bytes = 4096 
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        while size_bytes <= limit_bytes:
            try:
                if mode == "h2d":
                    src = torch.randint(0, 255, (size_bytes,), dtype=torch.uint8, pin_memory=True)
                    dst = torch.empty(size_bytes, dtype=torch.uint8, device=self.device)
                    def copy_fn(): dst.copy_(src, non_blocking=True)
                elif mode == "d2h":
                    src = torch.randint(0, 255, (size_bytes,), dtype=torch.uint8, device=self.device)
                    dst = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
                    def copy_fn(): dst.copy_(src, non_blocking=True)
                elif mode == "d2d":
                    src = torch.randint(0, 255, (size_bytes,), dtype=torch.uint8, device=self.device)
                    dst = torch.empty(size_bytes, dtype=torch.uint8, device=self.device)
                    def copy_fn(): dst.copy_(src)
                elif mode == "p2p":
                    src = torch.randint(0, 255, (size_bytes,), dtype=torch.uint8, device=self.device)
                    dst = torch.empty(size_bytes, dtype=torch.uint8, device=target_device)
                    def copy_fn(): dst.copy_(src)

                for _ in range(5): copy_fn()
                torch.cuda.synchronize()

                start_event.record()
                for _ in range(iterations):
                    copy_fn()
                end_event.record()
                torch.cuda.synchronize()

                avg_time_ms = start_event.elapsed_time(end_event) / iterations
                avg_time_us = avg_time_ms * 1000
                gb_per_sec = (size_bytes / 1e9) / (avg_time_ms / 1000)

                if size_bytes < 1024**2: size_str = f"{size_bytes/1024:.0f} KB"
                elif size_bytes < 1024**3: size_str = f"{size_bytes/1024**2:.0f} MB"
                else: size_str = f"{size_bytes/1024**3:.1f} GB"

                print(f"   {size_str:>10} | {size_bytes:>12} | {avg_time_us:>12.1f} | {gb_per_sec:>15.2f}")

            except RuntimeError:
                print(f"   {size_bytes:>10} | {'OOM / Skipped':>30}")
                break
            except Exception as e:
                print(f"   {size_bytes:>10} | Error: {str(e)}")
                break

            del src, dst
            torch.cuda.empty_cache()
            size_bytes *= 4

    def benchmark_streams(self, num_streams=4, iterations=20):
        print(f"\n--- 🌊 Stream Concurrency ({num_streams} Streams) ---")
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        small_size = 2048 
        a_s = torch.randn(small_size, small_size, device=self.device)
        b_s = torch.randn(small_size, small_size, device=self.device)

        start_serial = torch.cuda.Event(enable_timing=True)
        end_serial = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        start_serial.record()
        for _ in range(iterations):
            for _ in range(num_streams):
                torch.matmul(a_s, b_s)
        end_serial.record()
        torch.cuda.synchronize(self.device)
        time_serial = start_serial.elapsed_time(end_serial)

        start_par = torch.cuda.Event(enable_timing=True)
        end_par = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        start_par.record()
        for _ in range(iterations):
            for s in streams:
                with torch.cuda.stream(s):
                    torch.matmul(a_s, b_s)
        end_par.record()
        torch.cuda.synchronize(self.device)
        time_par = start_par.elapsed_time(end_par)

        print(f"   Serial Time:   {time_serial:.2f} ms")
        print(f"   Parallel Time: {time_par:.2f} ms")
        speedup = time_serial / time_par
        print(f"   Overlap Speedup: {speedup:.2f}x")
        if speedup < 1.1:
            print("   (Note: Low speedup implies GPU was fully saturated by a single stream or kernels were too small)")

    def benchmark_p2p_matrix(self, device_ids):
        print("\n--- 🌉 P2P Bandwidth Matrix (GB/s) ---")
        
        header_label = "SRC \\ DST"
        print(f"{header_label:>10} |", end="")
        for dst in device_ids:
            print(f"{f'GPU {dst}':>10} |", end="")
        print("\n" + "-" * (12 + 13 * len(device_ids)))

        size_bytes = 256 * 1024 * 1024 
        num_elements = size_bytes // 1 
        iterations = 20

        for src_id in device_ids:
            print(f"{f'GPU {src_id}':>10} |", end="")
            src_dev = torch.device(f"cuda:{src_id}")
            for dst_id in device_ids:
                if src_id == dst_id:
                    print(f"{'-':>10} |", end="")
                    continue
                dst_dev = torch.device(f"cuda:{dst_id}")
                try:
                    t_src = torch.zeros(num_elements, dtype=torch.uint8, device=src_dev)
                    t_dst = torch.zeros(num_elements, dtype=torch.uint8, device=dst_dev)
                    can_p2p = torch.cuda.can_device_access_peer(src_id, dst_id)
                    for _ in range(5): t_dst.copy_(t_src)
                    torch.cuda.synchronize(src_dev)
                    torch.cuda.synchronize(dst_dev)
                    start = time.time()
                    for _ in range(iterations):
                        t_dst.copy_(t_src)
                    torch.cuda.synchronize(src_dev)
                    torch.cuda.synchronize(dst_dev)
                    duration = time.time() - start
                    gb = (size_bytes * iterations) / 1e9
                    bw = gb / duration
                    marker = "⚡" if can_p2p else " " 
                    print(f"{marker}{bw:8.1f} |", end="")
                    del t_src, t_dst
                except Exception:
                    print(f"{'Err':>10} |", end="")
                sys.stdout.flush()
            print("") 
        print("-" * (12 + 13 * len(device_ids)))
        print("⚡ = P2P/NVLink Enabled")

    def _print_monitor_stats(self, gpu_ids):
        """
        Prints status for all listed GPUs using nvidia-smi in one go.
        """
        try:
            id_str = ",".join(map(str, gpu_ids))
            # Query multiple GPUs at once
            cmd = f"nvidia-smi --query-gpu=index,utilization.gpu,clocks.current.sm,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits -i {id_str}"
            output = subprocess.check_output(cmd.split()).decode('utf-8').strip().split('\n')
            
            status_line = []
            for line in output:
                parts = line.split(',')
                if len(parts) < 6: continue
                idx, util, clock, pwr, pwr_lim, temp = [x.strip() for x in parts]
                status_line.append(f"GPU {idx}: {util}% {clock}MHz {pwr}/{pwr_lim}W {temp}C")
            
            # Print timestamped status
            timestamp = time.strftime("%H:%M:%S")
            print(f"   [{timestamp}] {' | '.join(status_line)}")
        except Exception:
            pass # Suppress monitor errors to keep stress loop clean

    def stress_test_multi_gpu(self, duration_sec=60, gpu_ids=[0], mode="gemm", fill="std", monitor_interval=None):
        print(f"\n--- ☢️  MULTI-GPU STRESS TEST ({mode.upper()}) ---")
        print(f"   Targets: {gpu_ids}")
        print(f"   Duration: {duration_sec}s")
        print(f"   Fill: {fill} (Affects switching noise)")
        if monitor_interval:
            print(f"   Monitor: Every {monitor_interval}s")
        print("   Press Ctrl+C to stop early.")
        
        STRESS_SIZE = 16384 
        
        devices_data = []
        
        print("   Allocating resources...")
        for gid in gpu_ids:
            try:
                d = torch.device(f"cuda:{gid}")
                
                # Copy buffers for P2P stress if needed
                p2p_peer = None
                if mode == "p2p" and len(gpu_ids) > 1:
                    # Ring topology target: next gpu in list
                    curr_idx = gpu_ids.index(gid)
                    next_idx = (curr_idx + 1) % len(gpu_ids)
                    p2p_peer = torch.device(f"cuda:{gpu_ids[next_idx]}")

                def make_tensor(shape, dtype, dev=d):
                    if fill == "ones":
                        return torch.ones(shape, device=dev, dtype=dtype)
                    else:
                        return torch.randn(shape, device=dev, dtype=dtype)

                if mode == "gemm" or mode == "hybrid":
                    a = make_tensor((STRESS_SIZE, STRESS_SIZE), torch.bfloat16)
                    b = make_tensor((STRESS_SIZE, STRESS_SIZE), torch.bfloat16)
                    devices_data.append({'type': 'gemm', 'dev': d, 'a': a, 'b': b, 'stream': torch.cuda.Stream(device=d)})
                
                if mode == "alu_toggle" or mode == "hybrid":
                    num_ints = (STRESS_SIZE * STRESS_SIZE * 2) // 4 
                    if fill == "ones":
                        a = torch.full((num_ints,), -1, device=d, dtype=torch.int32) 
                    else:
                        a = torch.randint(0, 2**31, (num_ints,), device=d, dtype=torch.int32)
                    devices_data.append({'type': 'alu', 'dev': d, 'a': a, 'stream': torch.cuda.Stream(device=d)})
                
                if mode == "p2p":
                    # For P2P we need a source buffer on current GPU and we will copy to Peer
                    # Alloc 1GB buffer
                    p2p_size = 256 * 1024 * 1024 # 256MB elements (1GB if float32, but we use ByteTensor)
                    # Use ByteTensor for raw throughput
                    src_t = torch.zeros(p2p_size, dtype=torch.uint8, device=d)
                    # We need a dst buffer on the PEER device
                    # Note: We create the DST tensor here, but it lives on p2p_peer
                    dst_t = torch.zeros(p2p_size, dtype=torch.uint8, device=p2p_peer)
                    
                    devices_data.append({'type': 'p2p', 'dev': d, 'src': src_t, 'dst': dst_t, 'stream': torch.cuda.Stream(device=d)})

            except RuntimeError as e:
                print(f"   ❌ Failed to alloc on GPU {gid}: {e}")
                return

        print("   Running stress loops...")
        start_time = time.time()
        last_monitor_time = start_time
        try:
            iters = 0
            while (time.time() - start_time) < duration_sec:
                
                # Check monitor interval
                if monitor_interval and (time.time() - last_monitor_time) >= monitor_interval:
                    self._print_monitor_stats(gpu_ids)
                    last_monitor_time = time.time()

                for item in devices_data:
                    with torch.cuda.device(item['dev']):
                        with torch.cuda.stream(item['stream']):
                            if item['type'] == 'gemm':
                                torch.matmul(item['a'], item['b'])
                            elif item['type'] == 'alu':
                                item['a'].mul_(1664525).add_(1013904223)
                            elif item['type'] == 'p2p':
                                # Copy from local src to remote dst (Push)
                                item['dst'].copy_(item['src'], non_blocking=True)
                
                if iters % 100 == 0:
                    for item in devices_data:
                        torch.cuda.synchronize(item['dev'])

                iters += 1
                if iters % 10 == 0 and not monitor_interval:
                    # Only show simplified progress if monitoring is off, to avoid clutter
                    elapsed = time.time() - start_time
                    sys.stdout.write(f"\r   Running... Time remaining: {int(duration_sec - elapsed)}s")
                    sys.stdout.flush()
            print("\n--- ✅ Multi-GPU Stress Test Complete ---")
            
        except KeyboardInterrupt:
            print("\n--- 🛑 Stress Test Stopped by User ---")
        except RuntimeError as e:
            print(f"\n❌ OOM or Error during stress test: {e}")

if __name__ == "__main__":
    init_torch_cuda()
    check_cublas_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bench", "stress", "all"], default="all")
    parser.add_argument("--stress-type", choices=["gemm", "alu_toggle", "hybrid", "p2p"], default="gemm", help="Type of stress load")
    parser.add_argument("--fill", choices=["std", "ones"], default="std", help="Data pattern (std=random, ones=all 1s)")
    parser.add_argument("--size", type=int, default=None, help="Matmul size for compute test (default: auto-detect)")
    parser.add_argument("--duration", type=int, default=60, help="Stress test duration in seconds")
    parser.add_argument("--monitor-interval", type=float, default=None, help="Interval (sec) to print GPU stats during stress")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs or 'all'")
    args = parser.parse_args()
    
    # Check device availability BEFORE doing anything else
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("\n❌ FATAL: PyTorch cannot find any CUDA devices.")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA (Torch):    {torch.version.cuda}")
        
        print("\n   Troubleshooting:")
        print("   1. Hardware: Check 'nvidia-smi'. If it fails, your driver/fabric-manager is down.")
        print("   2. Environment: Verify 'CUDA_VISIBLE_DEVICES' is not empty.")
        print("   3. Installation: Ensure you didn't install a CPU-only PyTorch.")
        print("      (Run: python -c 'import torch; print(torch.version.cuda)')")
        print("   4. Containers: Ensure '--gpus all' or '--device nvidia.com/gpu=all' is used.")
        sys.exit(1)

    # Parse GPU list
    if args.gpus.lower() == "all":
        gpu_list = list(range(torch.cuda.device_count()))
    else:
        gpu_list = [int(x) for x in args.gpus.split(",")]

    if args.mode in ["bench", "all"]:
        print(f"Selected GPUs for Bench: {gpu_list}")
        # Safe execution wrapper
        try:
            primary_gpu = gpu_list[0]
            bm = GPUBenchmark(device_id=primary_gpu)
            bm.benchmark_compute_precision_sweep(size=args.size)
            bm.benchmark_bandwidth_sweep(mode="d2d")
            bm.benchmark_bandwidth_sweep(mode="h2d")
            bm.benchmark_bandwidth_sweep(mode="d2h")
            if len(gpu_list) > 1:
                bm.benchmark_p2p_matrix(gpu_list)
            else:
                print("\n(Skipping P2P Matrix - need >1 GPU)")
            bm.benchmark_streams()
        except RuntimeError as e:
            print(f"\n❌ BENCHMARK ERROR: {e}")

    if args.mode in ["stress", "all"]:
        try:
            bm = GPUBenchmark(device_id=gpu_list[0]) 
            bm.stress_test_multi_gpu(duration_sec=args.duration, gpu_ids=gpu_list, mode=args.stress_type, fill=args.fill, monitor_interval=args.monitor_interval)
        except RuntimeError as e:
            print(f"\n❌ STRESS TEST ERROR: {e}")
