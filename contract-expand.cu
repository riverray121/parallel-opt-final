#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "bfs_utils.cuh"

#define WARP_SIZE 32
#define CTA_SIZE 256 // Must be a multiple of WARP_SIZE
#define SCRATCH_SIZE 128 // Warp-based scratch size for duplicate detection

// Contract-expand BFS kernel
__global__ void bfs_contract_expand(const int *R, const int *C, int *status, int *frontier_in, int *frontier_out, int *labels, int queue_size, int *next_queue_size, int current_level) {
    // Shared memory allocation
    extern __shared__ int shmem[];
    int *scratchpad = shmem;
    int *validity = &shmem[CTA_SIZE];
    int *prefix_coarse = &shmem[CTA_SIZE * 2];
    int *prefix_fine = &shmem[CTA_SIZE * 3];

    int thread_id = threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;
    int warps_per_cta = CTA_SIZE / WARP_SIZE;

    volatile int *warp_scratch = &scratchpad[warp_id * SCRATCH_SIZE];

    // Initialize warp scratch
    for (int i = lane_id; i < SCRATCH_SIZE; i += WARP_SIZE) {
        warp_scratch[i] = -1;
    }
    __syncwarp();

    // Process tile of input from incoming edge-frontier queue
    for (int idx = blockIdx.x * blockDim.x + thread_id; idx < queue_size; idx += gridDim.x * blockDim.x) {
        int ni = frontier_in[idx];  // Neighbor vertex identifier

        // Test ni for validity
        bool is_valid = false;

        // Status-lookup
        int old_status = atomicExch(&status[ni], 1);  // Mark as visited
        if (old_status == 0) {
            is_valid = true;
        }

        // Warp-based duplicate culling
        bool is_duplicate = warp_cull(ni, warp_scratch, thread_id);
        if (is_duplicate) {
            is_valid = false;
        }

        // Store validity flag
        validity[thread_id] = is_valid ? 1 : 0;

        // Update labels and obtain row ranges
        int row_start = 0;
        int row_end = 0;
        int neighbor_degree = 0;
        if (is_valid) {
            labels[ni] = current_level;
            row_start = R[ni];
            row_end = R[ni + 1];
            neighbor_degree = row_end - row_start;
        }

        // Perform two concurrent CTA-wide prefix sums
        int coarse_val = (neighbor_degree > WARP_SIZE) ? neighbor_degree : 0;
        int fine_val = (neighbor_degree > 0 && neighbor_degree <= WARP_SIZE) ? neighbor_degree : 0;

        prefix_coarse[thread_id] = coarse_val;
        prefix_fine[thread_id] = fine_val;
        __syncthreads();

        int total_coarse = 0;
        int total_fine = 0;

        // Perform prefix sums
        prefix_sum(prefix_coarse, CTA_SIZE, &total_coarse);
        prefix_sum(prefix_fine, CTA_SIZE, &total_fine);

        // Thread0 obtains base enqueue offset
        int total_neighbors = total_coarse + total_fine;
        int base_offset = 0;
        if (thread_id == 0 && total_neighbors > 0) {
            base_offset = atomicAdd(next_queue_size, total_neighbors);
        }
        base_offset = __shfl_sync(0xffffffff, base_offset, 0);
        __syncthreads();

        // Coarse-grained gathering
        // Each thread processes own neighbor if neighbor_degree > WARP_SIZE
        if (coarse_val > 0) {
            // Enqueue index for this thread's neighbors
            int enqueue_offset = base_offset + prefix_coarse[thread_id];
            // Coarse-grained gathering
            int gather_start = row_start;
            int gather_end = row_end;
            int gather_count = gather_end - gather_start;

            // Enlist other threads in CTA or warp
            if (gather_count <= CTA_SIZE) {
                for (int i = thread_id; i < gather_count; i += CTA_SIZE) {
                    int neighbor = C[gather_start + i];
                    frontier_out[enqueue_offset + i] = neighbor;
                }
            } else {
                // For large degrees use multiple passes
                int num_iterations = (gather_count + CTA_SIZE - 1) / CTA_SIZE;
                for (int iter = 0; iter < num_iterations; ++iter) {
                    int i = thread_id + iter * CTA_SIZE;
                    if (i < gather_count) {
                        int neighbor = C[gather_start + i];
                        frontier_out[enqueue_offset + i] = neighbor;
                    }
                }
            }
        }
        __syncthreads();

        // Fine-grained scan-based gathering
        if (fine_val > 0) {
            int fine_offset = base_offset + total_coarse;  // Starting offset for fine-grained enqueue
            int rsv_rank = prefix_fine[thread_id];

            // Process fine-grained batches
            int cta_progress = 0;
            int remain = neighbor_degree;
            while (remain > 0) {
                int batch_size = min(remain, CTA_SIZE);
                // Each thread in CTA can process one neighbor
                if (thread_id < batch_size) {
                    int neighbor = C[row_start + cta_progress + thread_id];
                    frontier_out[fine_offset + cta_progress + thread_id] = neighbor;
                }
                cta_progress += batch_size;
                remain -= batch_size;
                __syncthreads();
            }
        }
    }
}
