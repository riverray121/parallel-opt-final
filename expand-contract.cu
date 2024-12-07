#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "bfs_utils.cuh"

#define WARP_SIZE 32
#define CTA_SIZE 256 // Must be a multiple of WARP_SIZE
#define SCRATCH_SIZE 128 // Warp-based scratch size for duplicate detection

__global__ void bfs_expand_contract(const int *R, const int *C, int *status, int *labels, const int *frontier_in, int frontier_size, int *frontier_out, int *next_queue_size, int current_level) {
    // Shared memory allocation
    extern __shared__ int shmem[];
    int warps_per_cta = CTA_SIZE / WARP_SIZE;

    int *scratchpad = shmem;
    int *validity = &shmem[warps_per_cta * SCRATCH_SIZE];
    int *prefix_array = &validity[CTA_SIZE];

    int thread_id = threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;

    volatile int *warp_scratch = &scratchpad[warp_id * SCRATCH_SIZE];

    // Initialize warp scratch
    for (int i = lane_id; i < SCRATCH_SIZE; i += WARP_SIZE) {
        warp_scratch[i] = -1;
    }
    __syncwarp();

    // Process tiles of vertices from input frontier
    for (int idx = blockIdx.x * blockDim.x + thread_id; idx < frontier_size; idx += gridDim.x * blockDim.x) {
        int v = frontier_in[idx];

        // Test v for validity
        bool is_valid = false;

        // Status-lookup
        int old_status = atomicExch(&status[v], 1);  // Mark as visited
        if (old_status == 0) {
            is_valid = true;
        }

        // Warp-based duplicate culling
        bool is_duplicate = warp_cull(v, warp_scratch, thread_id);
        if (is_duplicate) {
            is_valid = false;
        }

        // Store validity flag
        validity[thread_id] = (is_valid ? 1 : 0);
        __syncthreads();

        if (!is_valid) {
            // Move to next vertex if this one is invalid
            __syncthreads();
            continue;
        }

        // Update labels and obtain row ranges
        int row_start = R[v];
        int row_end = R[v + 1];
        int neighbor_degree = row_end - row_start;
        labels[v] = current_level;

        __syncthreads();

        // Coarse-grained if neighbor_degree > WARP_SIZE
        // Fine-grained (scan-based) if neighbor_degree <= WARP_SIZE and > 0
        if (neighbor_degree > WARP_SIZE) {
            // Coarse-grained CTA-based neighbor-gathering
            int total_processed = 0;
            while (total_processed < neighbor_degree) {
                int batch_size = min(neighbor_degree - total_processed, CTA_SIZE);
                __syncthreads();

                int neighbor = -1;
                if (thread_id < batch_size) {
                    neighbor = C[row_start + total_processed + thread_id];
                }

                // Status-lookup for neighbors
                bool n_valid = false;
                if (thread_id < batch_size) {
                    int old_st = atomicExch(&status[neighbor], 1);
                    if (old_st == 0) {
                        n_valid = true;
                        labels[neighbor] = current_level + 1;
                    }
                }

                validity[thread_id] = (n_valid && thread_id < batch_size) ? 1 : 0;
                __syncthreads();

                for (int i = 0; i < CTA_SIZE; i++) prefix_array[i] = validity[i];
                __syncthreads();

                int total_valid = 0;
                
                prefix_sum(prefix_array, CTA_SIZE, &total_valid);

                if (total_valid > 0) {
                    int base_offset = 0;
                    if (thread_id == 0) {
                        base_offset = atomicAdd(next_queue_size, total_valid);
                    }
                    base_offset = __shfl_sync(0xffffffff, base_offset, 0);
                    __syncthreads();

                    // Enqueue valid neighbors
                    if (thread_id < batch_size && n_valid) {
                        int rank = prefix_array[thread_id]; 
                        frontier_out[base_offset + rank] = neighbor;
                    }
                }

                total_processed += batch_size;
            }
        } else if (neighbor_degree > 0) {
            // Fine-grained scan-based gathering
            int remaining = neighbor_degree;
            int cta_progress = 0;

            while (remaining > 0) {
                int batch_size = min(remaining, CTA_SIZE);
                __syncthreads();

                int neighbor = -1;
                if (thread_id < batch_size) {
                    neighbor = C[row_start + cta_progress + thread_id];
                }

                // Status-lookup
                bool n_valid = false;
                if (thread_id < batch_size) {
                    int old_st = atomicExch(&status[neighbor], 1);
                    if (old_st == 0) {
                        n_valid = true;
                        labels[neighbor] = current_level + 1;
                    }
                }

                validity[thread_id] = (n_valid && thread_id < batch_size) ? 1 : 0;
                __syncthreads();

                for (int i = 0; i < CTA_SIZE; i++) prefix_array[i] = validity[i];
                __syncthreads();

                int total_valid = 0;
                
                prefix_sum(prefix_array, CTA_SIZE, &total_valid);

                if (total_valid > 0) {
                    int base_offset = 0;
                    if (thread_id == 0) {
                        base_offset = atomicAdd(next_queue_size, total_valid);
                    }
                    base_offset = __shfl_sync(0xffffffff, base_offset, 0);
                    __syncthreads();

                    if (thread_id < batch_size && n_valid) {
                        int rank = prefix_array[thread_id];
                        frontier_out[base_offset + rank] = neighbor;
                    }
                }

                cta_progress += batch_size;
                remaining -= batch_size;
            }
        }

        __syncthreads();
    }
}
