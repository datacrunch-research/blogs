import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings.driver import CUstream
from torch.cuda import current_stream


@cute.kernel
def tiled_gemm_kernel(A, B, C, M, N, K, tile_size):
    """Tiled GEMM kernel: C = A @ B (with K-tiling)"""
    # Get block indices
    block_m, block_n, _ = cute.arch.block_idx()

    # Calculate tile starting positions
    m_start = block_m * tile_size
    n_start = block_n * tile_size

    # Each thread block computes one tile_size x tile_size output tile
    for m_local in range(tile_size):
        for n_local in range(tile_size):
            m = m_start + m_local
            n = n_start + n_local

            if m < M and n < N:
                acc = 0.0

                # Tile along K: load tile_size chunks at a time
                for k_tile in range(0, K, tile_size):
                    for k_local in range(tile_size):
                        k = k_tile + k_local
                        if k < K:
                            acc += A[m, k] * B[k, n]

                C[m, n] = acc


@cute.jit
def launch_tiled_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
                      M: int, N: int, K: int, tile_size: int, stream: CUstream):
    """Launch tiled GEMM kernel with multiple blocks"""
    # Calculate grid dimensions (number of tiles in each dimension)
    grid_m = (M + tile_size - 1) // tile_size
    grid_n = (N + tile_size - 1) // tile_size

    tiled_gemm_kernel(A, B, C, M, N, K, tile_size).launch(
        grid=[grid_m, grid_n, 1],
        block=[1, 1, 1],  # Single thread per block for now
        stream=stream
    )

def run():
    M = N = K = 1024
    tile_size = 32  # Each block computes a 32x32 tile

    A = torch.full((M, K), 1.0, dtype=torch.float32, device="cuda")
    B = torch.full((K, N), 2.0, dtype=torch.float32, device="cuda")
    C = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    # Convert to CuTe tensors
    A_cute, B_cute, C_cute = map(from_dlpack, (A, B, C))

    # Compile kernel
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(launch_tiled_gemm, A_cute, B_cute, C_cute, M, N, K, tile_size, stream)

    # Time kernel execution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    compiled(A_cute, B_cute, C_cute, M, N, K, tile_size, stream)
    end_event.record()

    # Wait for kernel to complete
    torch.cuda.synchronize()

    # Calculate performance
    time_ms = start_event.elapsed_time(end_event)
    flops = 2 * M * N * K  # Each output element: K multiplies + K adds
    tflops = (flops / (time_ms * 1e-3)) / 1e12
    grid_m = (M + tile_size - 1) // tile_size
    grid_n = (N + tile_size - 1) // tile_size

    print(f"\n{'='*50}")
    print(f"Tiled GEMM Performance: {M}x{N}x{K}")
    print(f"Tile size: {tile_size}x{tile_size}, Grid: {grid_m}x{grid_n}")
    print(f"Time: {time_ms:.2f} ms ({time_ms/1000:.3f} s)")
    print(f"Performance: {tflops:.4f} TFLOP/s")
    print(f"{'='*50}")

    # Verify result
    expected = K * 1.0 * 2.0
    is_correct = torch.allclose(C, torch.full_like(C, expected))
    print(f"\n✓ Result is correct: {is_correct}")

if __name__ == "__main__":
    run()
