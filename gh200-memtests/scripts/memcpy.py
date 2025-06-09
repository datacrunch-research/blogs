import time
import ctypes
from typing import Union, Tuple, Any, Callable, List
from cuda.bindings import driver as cuda

TEST_SIZES = [1024**2, 10*1024**2, 500*1024**2, 10*1024**3, 25*1024**3, 50*1024**3]

def format_size(size_bytes):
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.0f}GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.0f}MB"
    else:
        return f"{size_bytes / 1024:.0f}KB"

def checkCudaErrors(result: Union[int, Tuple[int, ...]]) -> Any:
    if isinstance(result, tuple):
        err, *values = result
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Error: {err}")
        return values[0] if len(values) == 1 else values
    else:
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Error: {result}")
        return result

def findCudaDeviceDRV() -> int:
    device_count = checkCudaErrors(cuda.cuDeviceGetCount())
    if device_count == 0:
        raise RuntimeError("No CUDA devices found")
    return checkCudaErrors(cuda.cuDeviceGet(0))

def timeit(fxn: Callable[[], Any]) -> float:
    times = []
    for _ in range(10):
        start = time.perf_counter()
        fxn()
        times.append(time.perf_counter() - start)
    return min(times)

# Initialize CUDA once
checkCudaErrors(cuda.cuInit(0))
cuDevice = findCudaDeviceDRV()
cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cuDevice))

NUM_DEVICES = checkCudaErrors(cuda.cuDeviceGetCount())
print(f"Found {NUM_DEVICES} GPU devices")

# Iterate over different buffer sizes
for sz_bytes in TEST_SIZES:
    print(f"\n=== Testing with buffer size: {format_size(sz_bytes)} ===")
    
    # Allocate host memory for current size
    cuResult_inp, inp_ptr_raw = cuda.cuMemHostAlloc(
        sz_bytes, cuda.CU_MEMHOSTALLOC_DEVICEMAP
    )
    cuResult_out, out_ptr_raw = cuda.cuMemHostAlloc(
        sz_bytes, cuda.CU_MEMHOSTALLOC_DEVICEMAP
    )
    inp = ctypes.c_void_p(inp_ptr_raw)
    out = ctypes.c_void_p(out_ptr_raw)

    # ***** CPU timing *****
    def cpu_memcpy() -> None:
        """Host-side memory copy used as a CPU bandwidth baseline."""
        ctypes.memmove(inp, out, sz_bytes)

    print(f"CPU copy {(tm:=timeit(cpu_memcpy))*1000:.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")

    # ***** GPU timing *****
    STREAMS = 16
    sz_bytes_chunk = sz_bytes // STREAMS
    buf = [cuda.cuMemAlloc(sz_bytes_chunk)[1] for _ in range(STREAMS)]
    streams = [cuda.cuStreamCreate(0)[1] for _ in range(STREAMS)]

    def gpu_roundtrip() -> None:
        """Round-trip host⇄device memory transfer on a *single* GPU."""
        for i in range(STREAMS):
            offset_inp: int = inp.value + sz_bytes_chunk * i
            offset_out: int = out.value + sz_bytes_chunk * i
            cuda.cuMemcpyHtoDAsync(buf[i], offset_inp, sz_bytes_chunk, streams[i])
            cuda.cuMemcpyDtoHAsync(offset_out, buf[i], sz_bytes_chunk, streams[i])

        for stream in streams:
            checkCudaErrors(cuda.cuStreamSynchronize(stream))

    print(f"GPU copy {(tm:=timeit(gpu_roundtrip))*1000:.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")

    # ***** One-way CPU to GPU timing *****
    def gpu_oneway_htod() -> None:
        """One-way host-to-device memory transfer."""
        for i in range(STREAMS):
            offset_inp: int = inp.value + sz_bytes_chunk * i
            cuda.cuMemcpyHtoDAsync(buf[i], offset_inp, sz_bytes_chunk, streams[i])

        for stream in streams:
            checkCudaErrors(cuda.cuStreamSynchronize(stream))

    print(f"GPU H->D {(tm:=timeit(gpu_oneway_htod))*1000:.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")

    # Cleanup single GPU resources
    for buf_mem in buf:
        checkCudaErrors(cuda.cuMemFree(buf_mem))
    for stream in streams:
        checkCudaErrors(cuda.cuStreamDestroy(stream))

    # ***** multiGPU timing *****
    if NUM_DEVICES > 1:
        STREAMS = 4
        sz_bytes_chunk: int = sz_bytes // (STREAMS * NUM_DEVICES)
        
        buf: List[List[int]] = []
        streams: List[List[int]] = []
        device_contexts: List[int] = []
        
        # Initialize devices and allocate memory
        for device_id in range(NUM_DEVICES):
            device: int = checkCudaErrors(cuda.cuDeviceGet(device_id))
            ctx: int = checkCudaErrors(cuda.cuCtxCreate(0, device))
            device_contexts.append(ctx)
            
            # Allocate buffers for this device
            device_buf: List[int] = [checkCudaErrors(cuda.cuMemAlloc(sz_bytes_chunk)) for _ in range(STREAMS)]
            buf.append(device_buf)
            
            # Create streams for this device
            device_streams: List[int] = [checkCudaErrors(cuda.cuStreamCreate(0)) for _ in range(STREAMS)]
            streams.append(device_streams)
        
        def multigpu_roundtrip() -> None:
            """Round-trip memory transfer distributed over *all* available GPUs."""
            for i in range(STREAMS):
                for device_id in range(NUM_DEVICES):
                    checkCudaErrors(cuda.cuCtxSetCurrent(device_contexts[device_id]))
                    
                    offset: int = sz_bytes_chunk * (device_id * STREAMS + i)
                    cuda.cuMemcpyHtoDAsync(
                        buf[device_id][i],
                        inp.value + offset,
                        sz_bytes_chunk,
                        streams[device_id][i],
                    )
                    cuda.cuMemcpyDtoHAsync(
                        out.value + offset,
                        buf[device_id][i],
                        sz_bytes_chunk,
                        streams[device_id][i],
                    )
            
            for i in range(STREAMS):
                for device_id in range(NUM_DEVICES):
                    checkCudaErrors(cuda.cuCtxSetCurrent(device_contexts[device_id]))
                    checkCudaErrors(cuda.cuStreamSynchronize(streams[device_id][i]))
        
        print(f"GPU  {NUM_DEVICES}x  {(tm:=timeit(multigpu_roundtrip))*1000:.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")
        
        # ***** Multi-GPU one-way CPU to GPU timing *****
        def multigpu_oneway_htod() -> None:
            """One-way host-to-device memory transfer distributed over all GPUs."""
            for i in range(STREAMS):
                for device_id in range(NUM_DEVICES):
                    checkCudaErrors(cuda.cuCtxSetCurrent(device_contexts[device_id]))
                    
                    offset: int = sz_bytes_chunk * (device_id * STREAMS + i)
                    cuda.cuMemcpyHtoDAsync(
                        buf[device_id][i],
                        inp.value + offset,
                        sz_bytes_chunk,
                        streams[device_id][i],
                    )
            
            for i in range(STREAMS):
                for device_id in range(NUM_DEVICES):
                    checkCudaErrors(cuda.cuCtxSetCurrent(device_contexts[device_id]))
                    checkCudaErrors(cuda.cuStreamSynchronize(streams[device_id][i]))
        
        print(f"GPU  {NUM_DEVICES}x H->D  {(tm:=timeit(multigpu_oneway_htod))*1000:.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")
        
        # Cleanup multi-GPU resources
        for device_id in range(NUM_DEVICES):
            checkCudaErrors(cuda.cuCtxSetCurrent(device_contexts[device_id]))
            for buf_mem in buf[device_id]:
                checkCudaErrors(cuda.cuMemFree(buf_mem))
            for stream in streams[device_id]:
                checkCudaErrors(cuda.cuStreamDestroy(stream))
            checkCudaErrors(cuda.cuCtxDestroy(device_contexts[device_id]))
    else:
        print("Only 1 GPU found, skipping multi-GPU test")

    # Free host memory for current size
    checkCudaErrors(cuda.cuMemFreeHost(inp_ptr_raw))
    checkCudaErrors(cuda.cuMemFreeHost(out_ptr_raw))

# Final cleanup
checkCudaErrors(cuda.cuCtxDestroy(cuContext))