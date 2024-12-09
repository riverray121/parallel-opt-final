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

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void bfs_kernel(
    const int* adjacency_list,
    const int* adjacency_offsets,
    int* distances,
    bool* frontier,
    bool* next_frontier,
    int n,
    int level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if (frontier[tid]) {
        frontier[tid] = false;
        int start = adjacency_offsets[tid];
        int end = adjacency_offsets[tid + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = adjacency_list[i];
            if (distances[neighbor] == INT_MAX) {
                distances[neighbor] = level;
                next_frontier[neighbor] = true;
            }
        }
    }
}

void BFS_GPU(const vector<vector<int>>& graph, int source, int branching_factor) {
    auto start_time = high_resolution_clock::now();
    
    int n = graph.size();
    
    vector<int> adjacency_list;
    vector<int> adjacency_offsets(n + 1, 0);
    
    for (int i = 0; i < n; i++) {
        adjacency_offsets[i + 1] = adjacency_offsets[i] + graph[i].size();
        adjacency_list.insert(adjacency_list.end(), graph[i].begin(), graph[i].end());
    }
    
    int *d_adjacency_list, *d_adjacency_offsets, *d_distances;
    bool *d_frontier, *d_next_frontier;
    
    CUDA_CHECK(cudaMalloc(&d_adjacency_list, adjacency_list.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adjacency_offsets, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(bool)));
    
    vector<int> h_distances(n, INT_MAX);
    vector<bool> h_frontier(n, false);
    h_distances[source] = 0;
    h_frontier[source] = true;
    
    CUDA_CHECK(cudaMemcpy(d_adjacency_list, adjacency_list.data(), adjacency_list.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacency_offsets, adjacency_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_distances, h_distances.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, (void*)h_frontier.data(), n * sizeof(bool), cudaMemcpyHostToDevice));
    
    int level = 1;
    int max_depth = 0;
    int nodes_visited = 1;
    
    const int BLOCK_SIZE = 256;
    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    bool continue_bfs;
    vector<bool> h_next_frontier(n);
    
    do {
        CUDA_CHECK(cudaMemset(d_next_frontier, 0, n * sizeof(bool)));
        
        bfs_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_adjacency_list,
            d_adjacency_offsets,
            d_distances,
            d_frontier,
            d_next_frontier,
            n,
            level
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::swap(d_frontier, d_next_frontier);
        
        CUDA_CHECK(cudaMemcpy((void*)h_next_frontier.data(), d_frontier, n * sizeof(bool), cudaMemcpyDeviceToHost));
        
        continue_bfs = false;
        for (bool b : h_next_frontier) {
            if (b) {
                continue_bfs = true;
                nodes_visited++;
            }
        }
        
        if (continue_bfs) max_depth = level;
        level++;
        
    } while (continue_bfs);

    CUDA_CHECK(cudaFree(d_adjacency_list));
    CUDA_CHECK(cudaFree(d_adjacency_offsets));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    
    printf("%lu,%d,GPU,%d,%.3f,%d,%d\n", 
           graph.size(),
           branching_factor,
           source,
           duration.count() / 1000.0,
           max_depth,
           nodes_visited);
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