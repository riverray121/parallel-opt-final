#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "bfs_utils.cuh"

#define WARP_SIZE 32
#define CTA_SIZE 256 // Must be a multiple of WARP_SIZE
#define SCRATCH_SIZE 128 // Warp-based scratch size for duplicate detection

// Expansion Kernel
// Reads vertices from input vertex queue and writes neighbors into edge queue
__global__ void bfs_expand(const int *R, const int *C, const int *frontier_in, int frontier_size, int *edge_queue, int *edge_queue_size) {
    // Shared memory allocation
    extern __shared__ int shmem[];
    int *degrees = shmem;
    int *prefix_coarse = &shmem[CTA_SIZE];
    int *prefix_fine = &shmem[CTA_SIZE * 2];

    int thread_id = threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;

    // Each thread reads a vertex from input frontier
    int idx = blockIdx.x * blockDim.x + thread_id;
    int vertex = (idx < frontier_size) ? frontier_in[idx] : -1;

    // Each thread computes degree of its vertex
    int degree = 0;
    if (vertex != -1) {
        degree = R[vertex + 1] - R[vertex];
    }

    // Classify gathering strategy based on degree
    int coarse_val = (degree > WARP_SIZE) ? degree : 0;
    int fine_val = (degree > 0 && degree <= WARP_SIZE) ? degree : 0;

    // Store values for prefix sums
    degrees[thread_id] = degree;
    prefix_coarse[thread_id] = coarse_val;
    prefix_fine[thread_id] = fine_val;
    __syncthreads();

    int total_coarse = 0;
    int total_fine = 0;

    // Perform prefix sums to compute scatter offsets
    prefix_sum(prefix_coarse, CTA_SIZE, &total_coarse);
    prefix_sum(prefix_fine, CTA_SIZE, &total_fine);

    // Thread 0 obtains base enqueue offset using atomicAdd
    int total_neighbors = total_coarse + total_fine;
    int base_offset = 0;
    if (thread_id == 0 && total_neighbors > 0) {
        base_offset = atomicAdd(edge_queue_size, total_neighbors);
    }
    base_offset = __shfl_sync(0xffffffff, base_offset, 0);
    __syncthreads();

    // Coarse-grained gathering
    if (coarse_val > 0) {
        int enqueue_offset = base_offset + prefix_coarse[thread_id];
        int row_start = R[vertex];
        int row_end = R[vertex + 1];
        int neighbor_count = row_end - row_start;

        // Enlist other threads in CTA to gather neighbors
        for (int i = thread_id; i < neighbor_count; i += CTA_SIZE) {
            int neighbor = C[row_start + i];
            edge_queue[enqueue_offset + i] = neighbor;
        }
    }

    // Fine-grained scan-based gathering
    if (fine_val > 0) {
        int enqueue_offset = base_offset + total_coarse + prefix_fine[thread_id];
        int row_start = R[vertex];
        int row_end = R[vertex + 1];

        // Each thread processes its neighbors
        for (int i = 0; i < degree; ++i) {
            int neighbor = C[row_start + i];
            edge_queue[enqueue_offset + i] = neighbor;
        }
    }
}

// Contraction Kernel
// Reads neighbors from edge queue, filters duplicates, and writes valid vertices into output frontier
__global__ void bfs_contract(const int *edge_queue, int edge_queue_size, int *status, int *frontier_out, int *frontier_out_size, int current_level, int *labels) {
    // Shared memory allocation
    extern __shared__ int shmem[];
    int *validity = shmem;
    int *scratchpad = &shmem[CTA_SIZE];

    int thread_id = threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;
    int warps_per_cta = CTA_SIZE / WARP_SIZE;

    volatile int *warp_scratch = &scratchpad[warp_id * SCRATCH_SIZE];

    // Initialize warp scratchpad for duplicate detection
    for (int i = lane_id; i < SCRATCH_SIZE; i += WARP_SIZE) {
        warp_scratch[i] = -1;
    }
    __syncwarp();

    // Each thread reads a neighbor from edge queue
    int idx = blockIdx.x * blockDim.x + thread_id;
    int neighbor = (idx < edge_queue_size) ? edge_queue[idx] : -1;

    // Test neighbor for validity
    bool is_valid = false;
    if (neighbor != -1) {
        // Status-lookup
        int old_status = atomicExch(&status[neighbor], 1);  // Mark as visited
        if (old_status == 0) {
            is_valid = true;
            labels[neighbor] = current_level;
        }

        // Warp-based duplicate culling
        bool is_duplicate = warp_cull(neighbor, warp_scratch, thread_id);
        if (is_duplicate) {
            is_valid = false;
        }
    }

    // Store validity flag
    validity[thread_id] = is_valid ? 1 : 0;
    __syncthreads();

    // Perform prefix sum to compute enqueue offsets
    int total_valid = 0;
    prefix_sum(validity, CTA_SIZE, &total_valid);

    // Thread 0 obtains base enqueue offset using atomicAdd
    int base_offset = 0;
    if (thread_id == 0 && total_valid > 0) {
        base_offset = atomicAdd(frontier_out_size, total_valid);
    }
    base_offset = __shfl_sync(0xffffffff, base_offset, 0);
    __syncthreads();

    // Enqueue valid neighbors into output frontier
    if (is_valid) {
        int enqueue_offset = base_offset + validity[thread_id];
        frontier_out[enqueue_offset] = neighbor;
    }
}
