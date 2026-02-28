import torch
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cuda.bindings.driver import CUstream
from torch.cuda import current_stream


@cute.kernel
def shared_mem_gemm_kernel(A, B, C, M, N, K):
    """Shared memory GEMM kernel with swizzled layouts: C = A @ B

    Uses swizzled shared memory to reduce bank conflicts.
    Multiple threads cooperatively load and compute.
    """
    # Get block and thread indices
    block_m, block_n, _ = cute.arch.block_idx()
    thread_x, thread_y, _ = cute.arch.thread_idx()

    # Fixed tile sizes
    block_tile_m = 64
    block_tile_n = 64
    block_tile_k = 16

    # Calculate this block's starting position
    block_start_m = block_m * block_tile_m
    block_start_n = block_n * block_tile_n

    # Create swizzle pattern: Swizzle<2,4,3> (SW64 for Float32)
    swizzle = cute.make_swizzle(2, 4, 3)

    # Create base layouts for shared memory tiles
    a_base_layout = cute.make_layout((64, 16), stride=(16, 1))
    b_base_layout = cute.make_layout((16, 64), stride=(64, 1))

    # Apply swizzling to create composed layouts
    a_smem_layout = cute.make_composed_layout(swizzle, 0, a_base_layout)
    b_smem_layout = cute.make_composed_layout(swizzle, 0, b_base_layout)

    # Allocate shared memory (64*16*4 = 4096 bytes each)
    smem = utils.SmemAllocator()
    ptr_a_bytes = smem.allocate(4096, byte_alignment=128)
    ptr_b_bytes = smem.allocate(4096, byte_alignment=128)

    # Cast to Float32 pointers
    ptr_a = cute.recast_ptr(ptr_a_bytes, dtype=cute.Float32)
    ptr_b = cute.recast_ptr(ptr_b_bytes, dtype=cute.Float32)

    # Create shared memory tensors with swizzled layouts
    sA = cute.make_tensor(ptr_a, a_smem_layout)
    sB = cute.make_tensor(ptr_b, b_smem_layout)

    # Each thread computes a 4x4 tile
    thread_tile_m = 4
    thread_tile_n = 4

    thread_start_m = thread_y * thread_tile_m
    thread_start_n = thread_x * thread_tile_n

    # Each thread computes its assigned elements
    for i in range(4):
        for j in range(4):
            m = block_start_m + thread_start_m + i
            n = block_start_n + thread_start_n + j

            if m < M and n < N:
                acc = 0.0

                # Process K dimension in tiles of 16
                for k_tile_idx in range((K + 15) // 16):
                    k_start = k_tile_idx * 16
                    k_end = min(k_start + 16, K)

                    # Cooperatively load tile of A (64x16)
                    thread_id = thread_y * 16 + thread_x
                    
                    # Each thread loads 4 elements of A
                    for elem_idx in range(4):
                        idx = thread_id + elem_idx * 256
                        if idx < 1024:  # 64*16
                            row = idx // 16
                            col = idx % 16
                            gm = block_start_m + row
                            gk = k_start + col
                            if gm < M and gk < K:
                                sA[row, col] = A[gm, gk]

                    # Each thread loads 4 elements of B
                    for elem_idx in range(4):
                        idx = thread_id + elem_idx * 256
                        if idx < 1024:  # 16*64
                            row = idx // 64
                            col = idx % 64
                            gk = k_start + row
                            gn = block_start_n + col
                            if gk < K and gn < N:
                                sB[row, col] = B[gk, gn]

                    # Synchronize threads
                    cute.arch.sync_threads()

                    # Compute using swizzled shared memory
                    for k_offset in range(k_end - k_start):
                        acc += sA[thread_start_m + i, k_offset] * sB[k_offset, thread_start_n + j]

                    # Synchronize before next tile
                    cute.arch.sync_threads()

                C[m, n] = acc


@cute.jit
def launch_shared_mem_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
                           M: int, N: int, K: int, stream: CUstream):
    """Launch shared memory GEMM with swizzled layouts"""
    grid_m = (M + 63) // 64
    grid_n = (N + 63) // 64

    shared_mem_gemm_kernel(A, B, C, M, N, K).launch(
        grid=[grid_m, grid_n, 1],
        block=[16, 16, 1],
        stream=stream
    )


def run():
    M = N = K = 1024


    A = torch.full((M, K), 1.0, dtype=torch.float32, device="cuda")
    B = torch.full((K, N), 2.0, dtype=torch.float32, device="cuda")
    C = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    # Convert to CuTe tensors
    A_cute, B_cute, C_cute = map(from_dlpack, (A, B, C))

    # Compile kernel
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(launch_shared_mem_gemm, A_cute, B_cute, C_cute, M, N, K, stream)

    # Warmup
    compiled(A_cute, B_cute, C_cute, M, N, K, stream)
    torch.cuda.synchronize()

    # Time kernel (average of 5 runs)
    num_runs = 5
    times = []

    for _ in range(num_runs):
        C.zero_()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        compiled(A_cute, B_cute, C_cute, M, N, K, stream)
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    time_ms = sum(times) / len(times)
    flops = 2 * M * N * K
    tflops = (flops / (time_ms * 1e-3)) / 1e12

    #print(f"\n{'='*60}")
    #print(f"Block tile: 64x64x16")
    #print(f"Threads per block: 16x16 = 256 threads")
    #print(f"Each thread computes: 4x4 = 16 elements")
    #print(f"✨ REAL swizzled shared memory ✨")
    #print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"Shared Memory GEMM: Matrix size: {M}x{N}x{K} with Swizzle<2,4,3>")
    print(f"Performance Results:")
    print(f"Average time: {time_ms:.2f} ms")
    print(f"Performance: {tflops:.4f} TFLOP/s")
    print(f"{'='*60}")

    # Verify
    expected = K * 1.0 * 2.0
    is_correct = torch.allclose(C, torch.full_like(C, expected))
    print(f"\n✓ Result is correct: {is_correct}")

    if not is_correct:
        diff = (C - expected).abs()
        print(f"❌ Max error: {diff.max().item()}")


if __name__ == "__main__":
    run()
