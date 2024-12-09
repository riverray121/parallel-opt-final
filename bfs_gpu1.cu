#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

using std::vector;
using std::queue;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::istringstream;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

__global__ void bfs_kernel(
    const int* adjacency_list,
    const int* adjacency_offset,
    int* distances,
    bool* frontier,
    bool* next_frontier,
    int n) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if (frontier[tid]) {
        frontier[tid] = false;
        int start = adjacency_offset[tid];
        int end = adjacency_offset[tid + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = adjacency_list[i];
            if (distances[neighbor] == INT_MAX) {
                distances[neighbor] = distances[tid] + 1;
                next_frontier[neighbor] = true;
            }
        }
    }
}

void BFS_GPU(const vector<vector<int>>& graph, int source, int branching_factor) {
    auto start_time = high_resolution_clock::now();
    
    int n = graph.size();
    
    // Convert graph to GPU-friendly format
    vector<int> adjacency_list;
    vector<int> adjacency_offset(n + 1, 0);
    
    for (int i = 0; i < n; i++) {
        adjacency_offset[i + 1] = adjacency_offset[i] + graph[i].size();
        adjacency_list.insert(adjacency_list.end(), graph[i].begin(), graph[i].end());
    }
    
    // Allocate device memory
    int *d_adjacency_list, *d_adjacency_offset, *d_distances;
    bool *d_frontier, *d_next_frontier;
    
    cudaMalloc(&d_adjacency_list, adjacency_list.size() * sizeof(int));
    cudaMalloc(&d_adjacency_offset, (n + 1) * sizeof(int));
    cudaMalloc(&d_distances, n * sizeof(int));
    cudaMalloc(&d_frontier, n * sizeof(bool));
    cudaMalloc(&d_next_frontier, n * sizeof(bool));
    
    // Initialize host arrays
    vector<int> distances(n, INT_MAX);
    vector<bool> frontier(n, false);
    distances[source] = 0;
    frontier[source] = true;
    
    // Copy data to device
    cudaMemcpy(d_adjacency_list, adjacency_list.data(), 
        adjacency_list.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjacency_offset, adjacency_offset.data(), 
        (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, distances.data(), 
        n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, frontier.data(), 
        n * sizeof(bool), cudaMemcpyHostToDevice);
    
    // BFS iterations
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    bool has_frontier = true;
    int max_depth = 0;
    int nodes_visited = 1;

    while (has_frontier) {
        cudaMemset(d_next_frontier, false, n * sizeof(bool));
        
        bfs_kernel<<<num_blocks, block_size>>>(
            d_adjacency_list, d_adjacency_offset, d_distances,
            d_frontier, d_next_frontier, n);
        
        // Swap frontiers
        std::swap(d_frontier, d_next_frontier);
        
        // Check if we should continue
        vector<bool> current_frontier(n);
        cudaMemcpy(current_frontier.data(), d_frontier, 
            n * sizeof(bool), cudaMemcpyDeviceToHost);
        
        has_frontier = false;
        for (bool b : current_frontier) {
            if (b) {
                has_frontier = true;
                nodes_visited++;
            }
        }
    }

    // Copy final distances back to host
    cudaMemcpy(distances.data(), d_distances, 
        n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Calculate max_depth
    for (int d : distances) {
        if (d != INT_MAX) {
            max_depth = std::max(max_depth, d);
        }
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    
    printf("%lu,%d,GPU,%d,%.3f,%d,%d\n", 
           graph.size(),
           branching_factor,
           source,
           duration.count() / 1000.0,
           max_depth,
           nodes_visited);

    // Cleanup
    cudaFree(d_adjacency_list);
    cudaFree(d_adjacency_offset);
    cudaFree(d_distances);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
}

vector<vector<int>> read_graph(ifstream& file) {
    string line;
    getline(file, line);
    int n = std::stoi(line);
    
    vector<vector<int>> graph(n);
    
    for (int i = 0; i < n; i++) {
        getline(file, line);
        istringstream iss(line);
        string vertex;
        iss >> vertex;
        
        int neighbor;
        while (iss >> neighbor) {
            graph[i].push_back(neighbor);
        }
    }
    
    getline(file, line);
    return graph;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }
    
    const int branching_factor = std::stoi(argv[1]);
    
    auto total_start_time = high_resolution_clock::now();
    
    ifstream file("random_graphs.txt");
    if (!file.is_open()) {
        // cerr << "Error: Could not open random_graphs.txt\n";
        return 1;
    }

    int graph_number = 1;
    int total_searches = 0;
    
    while (!file.eof()) {
        string peek;
        if (!getline(file, peek)) break;
        file.seekg(-peek.length()-1, std::ios::cur);
        
        vector<vector<int>> graph = read_graph(file);
        if (graph.empty()) break;
        
        BFS_GPU(graph, 0, branching_factor);
        BFS_GPU(graph, graph.size() / 2, branching_factor);
        
        graph_number++;
        total_searches += 2;
    }

    file.close();
    
    auto total_end_time = high_resolution_clock::now();
    auto total_duration = duration_cast<microseconds>(total_end_time - total_start_time);
    
    // cout << "\nTotal Statistics:\n";
    // cout << "Total time: " << total_duration.count() / 1000.0 << " milliseconds\n";
    // cout << "Graphs processed: " << graph_number - 1 << "\n";
    // cout << "Total searches performed: " << total_searches << "\n";
    // cout << "Average time per search: " << (total_duration.count() / total_searches) / 1000.0 << " milliseconds\n";

    return 0;
}