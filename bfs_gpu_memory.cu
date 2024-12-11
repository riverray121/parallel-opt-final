#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// SHARED MEMORY on device memory for entire search

// Add this structure to track BFS state on GPU
struct BFSState {
    int frontier_size;
    int max_depth;
    int nodes_visited;
    bool finished;
};

// Modify kernel to update BFS state
__global__ void process_level_kernel(
    int* d_adjacency_list,
    int* d_adjacency_offsets,
    int* d_distances,
    int* d_current_frontier,
    int* d_next_frontier,
    BFSState* d_state,
    int current_depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Reset new frontier size at the start
    if (tid == 0) {
        d_state->frontier_size = 0;
    }
    __syncthreads();
    
    // Process nodes
    for (int idx = tid; idx < d_state->frontier_size; idx += stride) {
        int current = d_current_frontier[idx];
        int start = d_adjacency_offsets[current];
        int end = d_adjacency_offsets[current + 1];

        for (int i = start; i < end; i++) {
            int neighbor = d_adjacency_list[i];
            if (d_distances[neighbor] == INT_MAX) {
                d_distances[neighbor] = current_depth + 1;
                int frontier_idx = atomicAdd(&d_state->frontier_size, 1);
                d_next_frontier[frontier_idx] = neighbor;
            }
        }
    }
    
    // Update BFS state
    if (tid == 0) {
        if (d_state->frontier_size > 0) {
            d_state->max_depth = current_depth + 1;
            atomicAdd(&d_state->nodes_visited, d_state->frontier_size);
        } else {
            d_state->finished = true;
        }
    }
}

void BFS_GPU(const std::vector<std::vector<int>>& graph, int source, int branching_factor) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int n = graph.size();
    
    // Create CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Convert graph to CSR format
    std::vector<int> adjacency_list;
    std::vector<int> adjacency_offsets(n + 1, 0);
    
    for (int i = 0; i < n; i++) {
        adjacency_offsets[i + 1] = adjacency_offsets[i] + graph[i].size();
        for (int neighbor : graph[i]) {
            adjacency_list.push_back(neighbor);
        }
    }

    // Allocate and initialize BFS state
    BFSState* d_state;
    cudaMalloc(&d_state, sizeof(BFSState));
    BFSState initial_state = {1, 0, 1, false};  // Start with source node
    cudaMemcpyAsync(d_state, &initial_state, sizeof(BFSState), cudaMemcpyHostToDevice, stream);
    
    // Allocate and initialize other GPU memory
    int *d_adjacency_list, *d_adjacency_offsets, *d_distances;
    int *d_current_frontier, *d_next_frontier;
    
    cudaMalloc(&d_adjacency_list, adjacency_list.size() * sizeof(int));
    cudaMalloc(&d_adjacency_offsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_distances, n * sizeof(int));
    cudaMalloc(&d_current_frontier, n * sizeof(int));
    cudaMalloc(&d_next_frontier, n * sizeof(int));

    // Initialize arrays asynchronously
    cudaMemsetAsync(d_distances, INT_MAX, n * sizeof(int), stream);
    cudaMemcpyAsync(d_adjacency_list, adjacency_list.data(), adjacency_list.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_adjacency_offsets, adjacency_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Initialize source node
    int initial_frontier[] = {source};
    cudaMemcpyAsync(d_current_frontier, initial_frontier, sizeof(int), cudaMemcpyHostToDevice, stream);

    // BFS iterations
    int current_depth = 0;
    BFSState host_state;
    do {
        int block_size = 1024;
        int num_blocks = 256;  // Adjust based on GPU capabilities
        
        process_level_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_adjacency_list,
            d_adjacency_offsets,
            d_distances,
            d_current_frontier,
            d_next_frontier,
            d_state,
            current_depth
        );

        // Swap frontier buffers
        std::swap(d_current_frontier, d_next_frontier);
        current_depth++;

        // Only check state periodically or for termination
        if (current_depth % 100 == 0) {
            cudaMemcpyAsync(&host_state, d_state, sizeof(BFSState), 
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    } while (!host_state.finished);

    // Get final state
    cudaMemcpyAsync(&host_state, d_state, sizeof(BFSState), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Cleanup with stream
    cudaFree(d_adjacency_list);
    cudaFree(d_adjacency_offsets);
    cudaFree(d_distances);
    cudaFree(d_current_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_state);
    cudaStreamDestroy(stream);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    printf("%lu,%d,GPU,%d,%.3f,%d,%d\n", 
           graph.size(),
           branching_factor,
           source,
           duration.count() / 1000.0,
           host_state.max_depth,
           host_state.nodes_visited);
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