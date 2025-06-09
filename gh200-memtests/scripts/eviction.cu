#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <numa.h>

#define KIB (1024LL)
#define MIB (1024LL * KIB)
#define GIB (1024LL * MIB)

const size_t TARGET_SIZE_BYTES = 100 * MIB;
const size_t EVICTION_SIZE_BYTES = 500 * MIB;

__global__ void vector_add_kernel(const float* a, const float* b, float* c, long long n_chunk) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_chunk) {
        c[idx] = a[idx] + b[idx];
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
        if (numa_move_pages(0, 1, pages, NULL, status, 0) == 0) {
            numa_node = status[0];
        }
    }
    return numa_node;
}

const char* get_numa_location_name(int numa_node) {
    switch(numa_node) {
        case 0: return "CPU DDR (Node 0)";
        case 1: return "GPU HBM (Node 1)";
        case -1: return "Unknown";
        default: return (numa_node > 1) ? "Other NUMA Node" : "Error";
    }
}

void init_arrays(float* a, float* b, float* c, long long n_elements, const char* prefix) {
    printf("  Initializing %s arrays...\n", prefix);
    double start_time = get_time();
    for (long long i = 0; i < n_elements; i++) {
        a[i] = (float)i * 0.001f + (prefix[0] * 0.1f);
        b[i] = (float)i * 0.002f + (prefix[0] * 0.1f);
        if (c) c[i] = 0.0f;
    }
    double elapsed_time = get_time() - start_time;
    size_t bytes = n_elements * sizeof(float) * (c ? 3 : 2);
    double bandwidth = elapsed_time > 0 ? ((double)bytes / GIB) / elapsed_time : 0.0;
    printf("  CPU init: %.2f ms, %.2f GiB/s\n", elapsed_time * 1000, bandwidth);
}

double run_kernel(const float* a, const float* b, float* c, long long n_elements) {
    if (n_elements == 0) return 0.0;
    
    int block_size = 256;
    long long grid_size = (n_elements + block_size - 1) / block_size;
    
    double start_time = get_time();
    vector_add_kernel<<<grid_size, block_size>>>(a, b, c, n_elements);
    cudaError_t err = cudaDeviceSynchronize();
    double elapsed_time = get_time() - start_time;
    
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return elapsed_time;
}

void print_numa_info(void* a, void* b, void* c, const char* prefix) {
    int node_a = get_numa_node_of_pointer(a);
    int node_b = get_numa_node_of_pointer(b);
    int node_c = get_numa_node_of_pointer(c);
    printf("  %s NUMA: A=%d, B=%d, C=%d\n", prefix, node_a, node_b, node_c);
}

void test_malloc_eviction() {
    printf("\n=== MALLOC TEST ===\n");
    
    long long target_n_elements = TARGET_SIZE_BYTES / sizeof(float);
    long long eviction_n_elements = EVICTION_SIZE_BYTES / sizeof(float);
    float* target_a_h = NULL;
    float* target_b_h = NULL;
    float* target_c_h = NULL;
    float* evict_a_h = NULL;
    float* evict_b_h = NULL;
    float* evict_c_h = NULL;
    double time_cold = 0.0;
    double time_warm1 = 0.0;
    double time_warm2 = 0.0;
    double eviction_time = 0.0;
    double time_after_eviction = 0.0;
    double bandwidth_gib_s = 0.0;
    size_t total_bytes = 3 * TARGET_SIZE_BYTES;
    size_t eviction_bytes = 3 * EVICTION_SIZE_BYTES;

    target_a_h = (float*)malloc(TARGET_SIZE_BYTES);
    target_b_h = (float*)malloc(TARGET_SIZE_BYTES);
    target_c_h = (float*)malloc(TARGET_SIZE_BYTES);

    if (!target_a_h || !target_b_h || !target_c_h) {
        fprintf(stderr, "Failed to allocate host arrays\n");
        goto cleanup_malloc;
    }
    
    init_arrays(target_a_h, target_b_h, target_c_h, target_n_elements, "Target");
    print_numa_info(target_a_h, target_b_h, target_c_h, "Initial");

    time_cold = run_kernel(target_a_h, target_b_h, target_c_h, target_n_elements);
    time_warm1 = run_kernel(target_a_h, target_b_h, target_c_h, target_n_elements);
    time_warm2 = run_kernel(target_a_h, target_b_h, target_c_h, target_n_elements);
    
    bandwidth_gib_s = 0.0;
    if(time_cold > 0) bandwidth_gib_s = ((double)total_bytes / GIB) / time_cold;
    printf("  Cold: %.3f ms, %.2f GiB/s\n", time_cold * 1000, bandwidth_gib_s);
    bandwidth_gib_s = 0.0;
    if(time_warm1 > 0) bandwidth_gib_s = ((double)total_bytes / GIB) / time_warm1;
    printf("  Warm: %.3f ms, %.2f GiB/s\n", time_warm1 * 1000, bandwidth_gib_s);
    print_numa_info(target_a_h, target_b_h, target_c_h, "After warmup");

    evict_a_h = (float*)malloc(EVICTION_SIZE_BYTES);
    evict_b_h = (float*)malloc(EVICTION_SIZE_BYTES);
    evict_c_h = (float*)malloc(EVICTION_SIZE_BYTES);

    if (!evict_a_h || !evict_b_h || !evict_c_h) {
        fprintf(stderr, "Failed to allocate eviction arrays\n");
        goto cleanup_malloc;
    }
    
    init_arrays(evict_a_h, evict_b_h, evict_c_h, eviction_n_elements, "Eviction");

    printf("  Eviction workload (%.1f MiB x3)...\n", EVICTION_SIZE_BYTES / (float)MIB);
    eviction_time = run_kernel(evict_a_h, evict_b_h, evict_c_h, eviction_n_elements);
    bandwidth_gib_s = 0.0;
    if(eviction_time > 0) bandwidth_gib_s = ((double)eviction_bytes / GIB) / eviction_time;
    printf("  Eviction: %.2f ms, %.2f GiB/s\n", eviction_time * 1000, bandwidth_gib_s);

    time_after_eviction = run_kernel(target_a_h, target_b_h, target_c_h, target_n_elements);
    bandwidth_gib_s = 0.0;
    if(time_after_eviction > 0) bandwidth_gib_s = ((double)total_bytes / GIB) / time_after_eviction;
    printf("  After eviction: %.3f ms, %.2f GiB/s\n", time_after_eviction * 1000, bandwidth_gib_s);
    print_numa_info(target_a_h, target_b_h, target_c_h, "After eviction");

    printf("\nSummary (malloc):\n");
    printf("  Cold: %.3f ms\n", time_cold * 1000);
    printf("  Warm: %.3f ms\n", (time_warm1 + time_warm2) / 2.0 * 1000);
    printf("  After eviction: %.3f ms\n", time_after_eviction * 1000);

cleanup_malloc:
    if (evict_a_h) free(evict_a_h);
    if (evict_b_h) free(evict_b_h);
    if (evict_c_h) free(evict_c_h);
    if (target_a_h) free(target_a_h);
    if (target_b_h) free(target_b_h);
    if (target_c_h) free(target_c_h);
}

void test_managed_eviction() {
    printf("\n=== CUDAMALLOCMANAGED TEST ===\n");

    long long target_n_elements = TARGET_SIZE_BYTES / sizeof(float);
    long long eviction_n_elements = EVICTION_SIZE_BYTES / sizeof(float);
    float* target_a_m = NULL;
    float* target_b_m = NULL;
    float* target_c_m = NULL;
    float* evict_a_m = NULL;
    float* evict_b_m = NULL;
    float* evict_c_m = NULL;
    double time_cold = 0.0;
    double time_warm1 = 0.0;
    double time_warm2 = 0.0;
    double eviction_time = 0.0;
    double time_after_eviction = 0.0;
    double bandwidth_gib_s = 0.0;
    size_t total_bytes = 3 * TARGET_SIZE_BYTES;
    size_t eviction_bytes = 3 * EVICTION_SIZE_BYTES;
    cudaError_t err;
    
    err = cudaMallocManaged(&target_a_m, TARGET_SIZE_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for target_a_m\n");
        goto cleanup_managed;
    }
    
    err = cudaMallocManaged(&target_b_m, TARGET_SIZE_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for target_b_m\n");
        goto cleanup_managed;
    }
    
    err = cudaMallocManaged(&target_c_m, TARGET_SIZE_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for target_c_m\n");
        goto cleanup_managed;
    }

    init_arrays(target_a_m, target_b_m, target_c_m, target_n_elements, "Target");
    print_numa_info(target_a_m, target_b_m, target_c_m, "Initial");

    time_cold = run_kernel(target_a_m, target_b_m, target_c_m, target_n_elements);
    time_warm1 = run_kernel(target_a_m, target_b_m, target_c_m, target_n_elements);
    time_warm2 = run_kernel(target_a_m, target_b_m, target_c_m, target_n_elements);
    
    bandwidth_gib_s = 0.0;
    if(time_cold > 0) bandwidth_gib_s = ((double)total_bytes / GIB) / time_cold;
    printf("  Cold (migration): %.3f ms, %.2f GiB/s\n", time_cold * 1000, bandwidth_gib_s);
    bandwidth_gib_s = 0.0;
    if(time_warm1 > 0) bandwidth_gib_s = ((double)total_bytes / GIB) / time_warm1;
    printf("  Warm (HBM): %.3f ms, %.2f GiB/s\n", time_warm1 * 1000, bandwidth_gib_s);
    print_numa_info(target_a_m, target_b_m, target_c_m, "After warmup");
    
    err = cudaMallocManaged(&evict_a_m, EVICTION_SIZE_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for evict_a_m\n");
        goto cleanup_managed;
    }
    
    err = cudaMallocManaged(&evict_b_m, EVICTION_SIZE_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for evict_b_m\n");
        goto cleanup_managed;
    }
    
    err = cudaMallocManaged(&evict_c_m, EVICTION_SIZE_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for evict_c_m\n");
        goto cleanup_managed;
    }

    init_arrays(evict_a_m, evict_b_m, evict_c_m, eviction_n_elements, "Eviction");

    printf("  Eviction workload (%.1f MiB x3)...\n", EVICTION_SIZE_BYTES / (float)MIB);
    eviction_time = run_kernel(evict_a_m, evict_b_m, evict_c_m, eviction_n_elements);
    bandwidth_gib_s = 0.0;
    if(eviction_time > 0) bandwidth_gib_s = ((double)eviction_bytes / GIB) / eviction_time;
    printf("  Eviction: %.2f ms, %.2f GiB/s\n", eviction_time * 1000, bandwidth_gib_s);

    time_after_eviction = run_kernel(target_a_m, target_b_m, target_c_m, target_n_elements);
    bandwidth_gib_s = 0.0;
    if(time_after_eviction > 0) bandwidth_gib_s = ((double)total_bytes / GIB) / time_after_eviction;
    printf("  After eviction: %.3f ms, %.2f GiB/s\n", time_after_eviction * 1000, bandwidth_gib_s);
    print_numa_info(target_a_m, target_b_m, target_c_m, "After eviction");

    printf("\nSummary (managed):\n");
    printf("  Cold: %.3f ms\n", time_cold * 1000);
    printf("  Warm: %.3f ms\n", (time_warm1 + time_warm2) / 2.0 * 1000);
    printf("  After eviction: %.3f ms\n", time_after_eviction * 1000);

cleanup_managed:
    if (evict_a_m) cudaFree(evict_a_m);
    if (evict_b_m) cudaFree(evict_b_m);
    if (evict_c_m) cudaFree(evict_c_m);
    if (target_a_m) cudaFree(target_a_m);
    if (target_b_m) cudaFree(target_b_m);
    if (target_c_m) cudaFree(target_c_m);
}

int main() {
    printf("GPU HBM Cache Eviction Test\n");
    printf("Target: %.1f MiB, Eviction: %.1f MiB\n",
           TARGET_SIZE_BYTES / (float)MIB, EVICTION_SIZE_BYTES / (float)MIB);
    printf("================================\n");

    cudaSetDevice(0);

    if (numa_available() >= 0) {
        printf("\nNUMA nodes:\n");
        int max_node = numa_max_node();
        for (int i = 0; i <= max_node; i++) {
            long long node_size = numa_node_size64(i, NULL);
            if (node_size > 0) {
                printf("  Node %d: %lld MiB\n", i, node_size / MIB);
            }
        }
    }

    test_malloc_eviction();
    test_managed_eviction();

    printf("\nInterpretation:\n");
    printf("- malloc: GPU accesses host memory over NVLink\n");
    printf("- managed: Data migrates between CPU DDR and GPU HBM\n");
    printf("- Look for timing changes after eviction workload\n");

    return 0;
}