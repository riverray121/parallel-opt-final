#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for prefix sum
__global__ void prefix_sum_kernel(int* d_input, int* d_output, int n) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    
    // Load input into shared memory
    if (block_offset + tid < n) {
        temp[tid] = d_input[block_offset + tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Write results back
    if (block_offset + tid < n) {
        d_output[block_offset + tid] = temp[tid];
    }
}

// CUDA kernel for parallel neighbor processing
__global__ void process_level_kernel(
    int* d_adjacency_list,
    int* d_adjacency_offsets,
    int* d_distances,
    int* d_frontier,
    int* d_new_frontier,
    int* d_frontier_size,
    int* d_new_frontier_size,
    int* d_local_sizes,
    int* d_local_offsets,
    int current_depth,
    int max_neighbors
) {
    extern __shared__ int shared_mem[];
    int* local_indices = shared_mem;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int local_count = 0;
    
    // Process nodes and store neighbors locally
    for (int idx = tid; idx < *d_frontier_size; idx += stride) {
        int current = d_frontier[idx];
        int start = d_adjacency_offsets[current];
        int end = d_adjacency_offsets[current + 1];
        
        for (int i = start; i < end && local_count < max_neighbors; i++) {
            int neighbor = d_adjacency_list[i];
            // Simple distance check and update, no atomic needed due to benign race condition 
            // as described in the NVIDIA paper
            if (d_distances[neighbor] == INT_MAX) {
                d_distances[neighbor] = current_depth + 1;
                local_indices[local_count++] = neighbor;
            }
        }
    }
    
    // Store local count
    d_local_sizes[tid] = local_count;
    __syncthreads();
    
    // Wait for prefix sum to be computed externally
    
    // Copy local results to final positions
    int write_offset = (tid == 0) ? 0 : d_local_offsets[tid - 1];
    for (int i = 0; i < local_count; i++) {
        d_new_frontier[write_offset + i] = local_indices[i];
    }
    
    // Update total frontier size (only done by last thread)
    if (tid == gridDim.x * blockDim.x - 1) {
        *d_new_frontier_size = d_local_offsets[tid] + d_local_sizes[tid];
    }
}

void BFS_GPU(const std::vector<std::vector<int>>& graph, int source, int branching_factor) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int n = graph.size();
    
    // Convert graph to CSR format
    std::vector<int> adjacency_list;
    std::vector<int> adjacency_offsets(n + 1, 0);
    
    for (int i = 0; i < n; i++) {
        adjacency_offsets[i + 1] = adjacency_offsets[i] + graph[i].size();
        for (int neighbor : graph[i]) {
            adjacency_list.push_back(neighbor);
        }
    }

    // Allocate device memory
    int *d_adjacency_list, *d_adjacency_offsets, *d_distances;
    int *d_frontier, *d_new_frontier;
    int *d_frontier_size, *d_new_frontier_size;
    
    cudaMalloc(&d_adjacency_list, adjacency_list.size() * sizeof(int));
    cudaMalloc(&d_adjacency_offsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_distances, n * sizeof(int));
    cudaMalloc(&d_frontier, n * sizeof(int));
    cudaMalloc(&d_new_frontier, n * sizeof(int));
    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_new_frontier_size, sizeof(int));

    // Initialize host arrays
    std::vector<int> distances(n, INT_MAX);
    distances[source] = 0;
    std::vector<int> frontier = {source};
    int frontier_size = 1;
    int new_frontier_size = 0;
    
    // Copy data to device
    cudaMemcpy(d_adjacency_list, adjacency_list.data(), adjacency_list.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjacency_offsets, adjacency_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, distances.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    int current_depth = 0;
    int max_depth = 0;
    int nodes_visited = 1;

    // Additional allocations
    int max_threads = 1024;  // Adjust based on your GPU
    int max_neighbors = 256; // Adjust based on your needs
    int *d_local_sizes, *d_local_offsets;
    cudaMalloc(&d_local_sizes, max_threads * sizeof(int));
    cudaMalloc(&d_local_offsets, max_threads * sizeof(int));
    
    // Keep these arrays on GPU throughout the entire BFS
    int *d_frontier_size, *d_new_frontier_size;
    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_new_frontier_size, sizeof(int));
    
    // Initialize directly on device instead of copying
    cudaMemset(d_frontier_size, 0, sizeof(int));
    cudaMemset(d_new_frontier_size, 0, sizeof(int));
    
    // Initialize first frontier directly on device
    cudaMemset(d_frontier, 0, n * sizeof(int));
    int init_frontier_size = 1;
    cudaMemcpy(d_frontier_size, &init_frontier_size, sizeof(int), cudaMemcpyHostToDevice);
    
    // BFS iterations
    bool continue_bfs = true;
    while (continue_bfs) {
        cudaMemset(d_new_frontier_size, 0, sizeof(int));
        
        // Launch kernel with shared memory
        int block_size = 256;
        int num_blocks = (frontier_size + block_size - 1) / block_size;
        int shared_mem_size = block_size * max_neighbors * sizeof(int);
        
        process_level_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            d_adjacency_list,
            d_adjacency_offsets,
            d_distances,
            d_frontier,
            d_new_frontier,
            d_frontier_size,
            d_new_frontier_size,
            d_local_sizes,
            d_local_offsets,
            current_depth,
            max_neighbors
        );
        
        prefix_sum_kernel<<<1, max_threads, max_threads * sizeof(int)>>>(
            d_local_sizes,
            d_local_offsets,
            max_threads
        );

        // Swap frontier pointers instead of copying
        int* temp = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = temp;
        
        // Copy only the size to check termination
        int current_frontier_size;
        cudaMemcpy(&current_frontier_size, d_new_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
        continue_bfs = (current_frontier_size > 0);
        
        if (continue_bfs) {
            nodes_visited += current_frontier_size;
            max_depth = ++current_depth;
        }
    }

    // Only copy distances back at the very end if needed
    // cudaMemcpy(distances.data(), d_distances, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_adjacency_list);
    cudaFree(d_adjacency_offsets);
    cudaFree(d_distances);
    cudaFree(d_frontier);
    cudaFree(d_new_frontier);
    cudaFree(d_frontier_size);
    cudaFree(d_new_frontier_size);
    cudaFree(d_local_sizes);
    cudaFree(d_local_offsets);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    printf("%lu,%d,GPU,%d,%.3f,%d,%d\n", 
           graph.size(),          // graph_size
           branching_factor,      // branching_factor
           source,               // source_node
           duration.count() / 1000.0,  // time_ms
           max_depth,            // max_depth
           nodes_visited);       // nodes_visited
}

std::vector<std::vector<int>> read_graph(std::ifstream& file) {
    std::string line;
    std::getline(file, line);
    int n = std::stoi(line);
    
    std::vector<std::vector<int>> graph(n);
    
    for (int i = 0; i < n; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        std::string vertex;
        iss >> vertex;
        
        int neighbor;
        while (iss >> neighbor) {
            graph[i].push_back(neighbor);
        }
    }
    
    std::getline(file, line);
    return graph;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        // std::cerr << "Usage: " << argv[0] << " <branching_factor>\n";
        return 1;
    }
    
    const int branching_factor = std::stoi(argv[1]);
    
    auto total_start_time = std::chrono::high_resolution_clock::now();
    
    std::ifstream file("random_graphs.txt");
    if (!file.is_open()) {
        // std::cerr << "Error: Could not open random_graphs.txt\n";
        return 1;
    }

    int graph_number = 1;
    int total_searches = 0;
    
    while (!file.eof()) {
        std::string peek;
        if (!std::getline(file, peek)) break;
        file.seekg(-peek.length()-1, std::ios::cur);
        
        std::vector<std::vector<int>> graph = read_graph(file);
        if (graph.empty()) break;
        
        // std::cout << "\nGraph " << graph_number << " (Size: " << graph.size() << "):\n";
        
        BFS_GPU(graph, 0, branching_factor);
        BFS_GPU(graph, graph.size() / 2, branching_factor);
        
        graph_number++;
        total_searches += 2;
    }

    file.close();
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
    
    // std::cout << "\nTotal Statistics:\n";
    // std::cout << "Total time: " << total_duration.count() / 1000.0 << " milliseconds\n";
    // std::cout << "Graphs processed: " << graph_number - 1 << "\n";
    // std::cout << "Total searches performed: " << total_searches << "\n";
    // std::cout << "Average time per search: " << (total_duration.count() / total_searches) / 1000.0 << " milliseconds\n";

    return 0;
}