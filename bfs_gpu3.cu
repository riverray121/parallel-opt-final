#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// Add this kernel for processing neighbors
__global__ void processNeighborsKernel(int* d_adjacency_list, int* d_distances, 
                                      int* d_visited, int level, 
                                      int start_idx, int end_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int idx = start_idx + tid; idx < end_idx; idx += stride) {
        int neighbor = d_adjacency_list[idx];
        if (!d_visited[neighbor]) {
            d_distances[neighbor] = level + 1;
            d_visited[neighbor] = 1;
        }
    }
}

// Modified main BFS kernel with dynamic parallelism
__global__ void bfs_kernel_dynamic(int* d_adjacency_list, int* d_vertices, 
                                 int* d_distances, int* d_visited,
                                 int num_vertices, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices && d_distances[tid] == level) {
        int start_idx = d_vertices[tid];
        int end_idx = d_vertices[tid + 1];
        
        // Calculate number of neighbors
        int num_neighbors = end_idx - start_idx;
        
        if (num_neighbors > 0) {
            // Launch a new kernel to process neighbors
            int block_size = 256;
            int num_blocks = (num_neighbors + block_size - 1) / block_size;
            
            // Limit the number of blocks to prevent excessive parallelism
            num_blocks = min(num_blocks, 32);
            
            processNeighborsKernel<<<num_blocks, block_size>>>(
                d_adjacency_list, d_distances, d_visited,
                level, start_idx, end_idx);
        }
    }
}

// Modified BFS function
__host__ void bfs_gpu(int* adjacency_list, int* vertices, int* distances,
                     int num_vertices, int source) {
    // ... (existing device memory allocation code) ...

    // Initialize distances and visited arrays
    int block_size = 256;
    int num_blocks = (num_vertices + block_size - 1) / block_size;

    // Set initial distance for source vertex
    cudaMemset(d_distances, -1, num_vertices * sizeof(int));
    cudaMemset(d_visited, 0, num_vertices * sizeof(int));
    
    int level = 0;
    d_distances[source] = level;
    d_visited[source] = 1;

    bool continue_bfs = true;
    while (continue_bfs) {
        continue_bfs = false;
        
        // Launch the dynamic kernel
        bfs_kernel_dynamic<<<num_blocks, block_size>>>(
            d_adjacency_list, d_vertices, d_distances,
            d_visited, num_vertices, level);
            
        // Synchronize to ensure all dynamic kernels complete
        cudaDeviceSynchronize();
        
        // Check if we need to continue
        int* h_distances = new int[num_vertices];
        cudaMemcpy(h_distances, d_distances, num_vertices * sizeof(int), 
                  cudaMemcpyDeviceToHost);
                  
        for (int i = 0; i < num_vertices; i++) {
            if (h_distances[i] == level) {
                continue_bfs = true;
                break;
            }
        }
        
        delete[] h_distances;
        level++;
    }

    // ... (existing cleanup code) ...
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
