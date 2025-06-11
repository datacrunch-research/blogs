#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <numa.h>

#define GIB (1024LL*1024*1024)
#define SIZE_BYTES (1LL * GIB)
#define N_ELEMENTS (SIZE_BYTES / sizeof(float))

__global__ void processing_kernel(float* data, long long num_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void first_touch_kernel(float* data, long long num_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = (float)idx * 0.1f;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int get_numa_node_of_pointer(void* ptr) {
    int numa_node = -1;
    if (numa_available() >= 0) {
        void* pages[1] = {ptr};
        int status[1] = {-1};
        int ret = numa_move_pages(0, 1, pages, NULL, status, 0);
        if (ret == 0) {
            numa_node = status[0];
        }
    }
    return numa_node;
}

const char* get_numa_location_name(int numa_node) {
    switch(numa_node) {
        case 0: return "CPU DDR (Node 0)";
        case 1: return "GPU HBM (Node 1+)";
        case -1: return "Unknown";
        default: return (numa_node > 1) ? "Other NUMA Node" : "Error";
    }
}

void run_gpu_intensive_processing(float* data) {
    int blockSize = 256;
    int gridSize = (N_ELEMENTS + blockSize - 1) / blockSize;

    processing_kernel<<<gridSize, blockSize>>>(data, N_ELEMENTS);
    cudaDeviceSynchronize();

    double start_time = get_time();
    processing_kernel<<<gridSize, blockSize>>>(data, N_ELEMENTS);
    cudaError_t err = cudaDeviceSynchronize();
    double elapsed_time = get_time() - start_time;

    if (err != cudaSuccess) {
        printf("  Kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    double bandwidth_gib_s = (2.0 * SIZE_BYTES / GIB) / elapsed_time;
    printf("  GPU processing: %.2f ms, %.2f GiB/s\n", elapsed_time * 1000, bandwidth_gib_s);
}

void check_numa_location(float* data, const char* description) {
    int numa_node = get_numa_node_of_pointer(data);
    printf("  %s: NUMA node %d (%s)\n", description, numa_node, get_numa_location_name(numa_node));
}

void cpu_first_touch(float* data, long long n) {
    for (long long i = 0; i < n; i++) {
        data[i] = (float)i * 0.1f;
    }
}

void gpu_first_touch(float* data) {
    int blockSize = 256;
    int gridSize = (N_ELEMENTS + blockSize - 1) / blockSize;
    
    double start_time = get_time();
    first_touch_kernel<<<gridSize, blockSize>>>(data, N_ELEMENTS);
    cudaError_t err = cudaDeviceSynchronize();
    double elapsed_time = get_time() - start_time;
    
    if (err != cudaSuccess) {
        printf("  GPU first touch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    double bandwidth_gib_s = (SIZE_BYTES / GIB) / elapsed_time;
    printf("  GPU init: %.2f ms, %.2f GiB/s\n", elapsed_time * 1000, bandwidth_gib_s);
}


void run_test_iterations(float* ptr, const char* test_name, int iterations) {
    for (int i = 1; i <= iterations; i++) {
        printf("\n  Iteration %d:\n", i);
        run_gpu_intensive_processing(ptr);
        check_numa_location(ptr, "After GPU processing");
        if (i < iterations) {
            printf("  Waiting 1 second...\n");
            sleep(1);
        }
    }
}

int main() {
    printf("Memory Migration & Bandwidth Test - %.1f GiB buffer\n", (double)SIZE_BYTES / GIB);
    printf("=================================================\n");

    if (numa_available() >= 0) {
        int num_nodes = numa_max_node() + 1;
        printf("NUMA nodes: %d\n", num_nodes);
        for (int i = 0; i < num_nodes; i++) {
            long long node_size = numa_node_size64(i, NULL);
            if (node_size >= 0) {
                printf("  Node %d: %.1f GiB\n", i, (double)node_size / GIB);
            }
        }
    }

    // Test 1: malloc
    printf("\n1. malloc (CPU host memory):\n");
    float* malloc_ptr = (float*)malloc(SIZE_BYTES);
    if (!malloc_ptr) return 1;
    cpu_first_touch(malloc_ptr, N_ELEMENTS);
    check_numa_location(malloc_ptr, "After malloc & CPU touch");
    run_test_iterations(malloc_ptr, "malloc", 3);
    free(malloc_ptr);

    // Test 2: malloc with GPU first touch
    printf("\n\n2. malloc with GPU first touch:\n");
    float* malloc_gpu_ptr = (float*)malloc(SIZE_BYTES);
    if (!malloc_gpu_ptr) return 1;
    gpu_first_touch(malloc_gpu_ptr);
    check_numa_location(malloc_gpu_ptr, "After malloc & GPU touch");
    run_test_iterations(malloc_gpu_ptr, "malloc_gpu", 3);
    free(malloc_gpu_ptr);

    // Test 3: cudaMallocManaged with CPU first touch
    printf("\n\n2. cudaMallocManaged with CPU first touch:\n");
    float* managed_cpu;
    if (cudaMallocManaged(&managed_cpu, SIZE_BYTES) != cudaSuccess) return 1;
    double start = get_time();
    cpu_first_touch(managed_cpu, N_ELEMENTS);
    double elapsed = get_time() - start;
    printf("  CPU init: %.2f ms, %.2f GiB/s\n", elapsed * 1000, (SIZE_BYTES / GIB) / elapsed);
    check_numa_location(managed_cpu, "After CPU first touch");
    run_test_iterations(managed_cpu, "managed_cpu", 3);
    cudaFree(managed_cpu);

    // Test 4: cudaMallocManaged with GPU first touch
    printf("\n\n4. cudaMallocManaged with GPU first touch:\n");
    float* managed_gpu;
    if (cudaMallocManaged(&managed_gpu, SIZE_BYTES) != cudaSuccess) return 1;
    printf("  GPU first touch:\n");
    run_gpu_intensive_processing(managed_gpu);
    check_numa_location(managed_gpu, "After GPU first touch");
    run_test_iterations(managed_gpu, "managed_gpu", 2);
    cudaFree(managed_gpu);

    // Test 5: cudaMalloc baseline
    printf("\n\n5. cudaMalloc (GPU device memory):\n");
    float* device_ptr;
    if (cudaMalloc(&device_ptr, SIZE_BYTES) != cudaSuccess) return 1;
    printf("  Location: GPU HBM\n");
    run_gpu_intensive_processing(device_ptr);
    cudaFree(device_ptr);

    return 0;
}