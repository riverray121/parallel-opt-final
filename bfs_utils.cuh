#ifndef BFS_UTILS_CUH
#define BFS_UTILS_CUH

#define WARP_SIZE 32
#define SCRATCH_SIZE 128

// Warp-based duplicate detection
__device__ bool warp_cull(int neighbor, volatile int *scratch, int thread_id) {
    int hash = neighbor & (SCRATCH_SIZE - 1);
    scratch[hash] = neighbor;
    int retrieved = scratch[hash];

    if (retrieved == neighbor) {
        // attempt uniqueness
        scratch[hash] = thread_id;
        if (scratch[hash] != thread_id) {
            // already claimed
            return true;
        }
    }
    return false;
}

// CTA-wide prefix sum
__device__ void prefix_sum(int *data, int n, int *total_sum) {
    int tid = threadIdx.x;
    int offset = 1;

    // Up-sweep
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            data[bi] += data[ai];
        }
        offset <<= 1;
    }

    // Save total sum before clearing last element
    if (tid == 0) {
        *total_sum = data[n - 1];
        data[n - 1] = 0;
    }

    // Down-sweep
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }
    }
    __syncthreads();
}

#endif