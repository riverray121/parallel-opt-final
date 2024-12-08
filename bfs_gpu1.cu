#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for parallel neighbor processing
__global__ void process_level_kernel(
    int* d_adjacency_list,
    int* d_adjacency_offsets,
    int* d_distances,
    int* d_frontier,
    int* d_new_frontier,
    int* d_frontier_size,
    int* d_new_frontier_size,
    int current_depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *d_frontier_size) return;

    int current = d_frontier[tid];
    int start = d_adjacency_offsets[current];
    int end = d_adjacency_offsets[current + 1];

    for (int i = start; i < end; i++) {
        int neighbor = d_adjacency_list[i];
        if (d_distances[neighbor] == INT_MAX) {
            d_distances[neighbor] = current_depth + 1;
            int idx = atomicAdd(d_new_frontier_size, 1);
            d_new_frontier[idx] = neighbor;
        }
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

    // BFS iterations
    while (frontier_size > 0) {
        cudaMemcpy(d_frontier, frontier.data(), frontier_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier_size, &frontier_size, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_frontier_size, &new_frontier_size, sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int num_blocks = (frontier_size + block_size - 1) / block_size;
        process_level_kernel<<<num_blocks, block_size>>>(
            d_adjacency_list,
            d_adjacency_offsets,
            d_distances,
            d_frontier,
            d_new_frontier,
            d_frontier_size,
            d_new_frontier_size,
            current_depth
        );

        // Get new frontier size
        cudaMemcpy(&new_frontier_size, d_new_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Get new frontier
        frontier.resize(new_frontier_size);
        cudaMemcpy(frontier.data(), d_new_frontier, new_frontier_size * sizeof(int), cudaMemcpyDeviceToHost);
        
        frontier_size = new_frontier_size;
        new_frontier_size = 0;
        
        current_depth++;
        if (frontier_size > 0) {
            max_depth = current_depth;
            nodes_visited += frontier_size;
        }
    }

    // Clean up
    cudaFree(d_adjacency_list);
    cudaFree(d_adjacency_offsets);
    cudaFree(d_distances);
    cudaFree(d_frontier);
    cudaFree(d_new_frontier);
    cudaFree(d_frontier_size);
    cudaFree(d_new_frontier_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    printf("%d,%d,GPU,%d,%.3f,%d,%d\n", 
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
        std::cerr << "Usage: " << argv[0] << " <branching_factor>\n";
        return 1;
    }
    
    const int branching_factor = std::stoi(argv[1]);
    
    auto total_start_time = std::chrono::high_resolution_clock::now();
    
    std::ifstream file("random_graphs.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open random_graphs.txt\n";
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
        
        std::cout << "\nGraph " << graph_number << " (Size: " << graph.size() << "):\n";
        
        BFS_GPU(graph, 0, branching_factor);
        BFS_GPU(graph, graph.size() / 2, branching_factor);
        
        graph_number++;
        total_searches += 2;
    }

    file.close();
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);
    
    std::cout << "\nTotal Statistics:\n";
    std::cout << "Total time: " << total_duration.count() / 1000.0 << " milliseconds\n";
    std::cout << "Graphs processed: " << graph_number - 1 << "\n";
    std::cout << "Total searches performed: " << total_searches << "\n";
    std::cout << "Average time per search: " << (total_duration.count() / total_searches) / 1000.0 << " milliseconds\n";

    return 0;
}
