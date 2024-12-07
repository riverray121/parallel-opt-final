#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) do {                                                                                       \
    cudaError_t err = call;                                                                                         \
    if (err != cudaSuccess) {                                                                                       \
        cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                               \
} while (0)

#define CTA_SIZE 256

using namespace std;

extern __global__ void bfs_expand_contract(const int *R, const int *C, int *status, int *labels, const int *frontier_in, int frontier_size, int *frontier_out, int *next_queue_size, int current_level);

extern __global__ void bfs_contract_expand(const int *R, const int *C, int *status, int *frontier_in, int *frontier_out, int *labels, int queue_size, int *next_queue_size, int current_level);

// Two-phase kernels
__global__ void bfs_expand(const int *R, const int *C, const int *frontier_in, int frontier_size, int *edge_queue, int *edge_queue_size);
__global__ void bfs_contract(const int *edge_queue, int edge_queue_size, int *status, int *frontier_out, int *frontier_out_size, int current_level, int *labels);

// Algorithms operate on CSR format graphs
// Can convert edge-list graphs to CSR with included Python script
static void load_csr(const string &row_offsets_file, const string &column_indices_file, vector<int> &row_offsets, vector<int> &column_indices) {
    ifstream row_file(row_offsets_file);
    if (!row_file.is_open()) {
        cerr << "Error opening: " << row_offsets_file << endl;
        exit(EXIT_FAILURE);
    }
    int val;
    while (row_file >> val) {
        row_offsets.push_back(val);
    }
    row_file.close();

    ifstream col_file(column_indices_file);
    if (!col_file.is_open()) {
        cerr << "Error opening: " << column_indices_file << endl;
        exit(EXIT_FAILURE);
    }
    int val;
    while (col_file >> val) {
        column_indices.push_back(val);
    }
    col_file.close();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cerr << "Usage: ./bfs_host <row_offsets.txt> <column_indices.txt> <kernel_name>" << endl;
        cerr << "kernel_name in {contract-expand, expand-contract, two-phase}" << endl;
        return 1;
    }

    string row_offsets_file = argv[1], column_indices_file = argv[2], kernel_name = argv[3];
    vector<int> row_offsets, column_indices;
    load_csr(row_offsets_file, column_indices_file, row_offsets, column_indices);

    int num_nodes = static_cast<int>(row_offsets.size() - 1), num_edges = static_cast<int>(column_indices.size());

    // Allocate memory
    int *d_R, *d_C, *d_status, *d_labels;
    int *d_frontier_in, *d_frontier_out, *d_next_queue_size;

    CUDA_CHECK(cudaMalloc(&d_R, row_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C, column_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_status, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_in, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_out, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_queue_size, sizeof(int)));

    // Copy graph to device
    CUDA_CHECK(cudaMemcpy(d_R, row_offsets.data(), row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, column_indices.data(), column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize BFS arrays
    CUDA_CHECK(cudaMemset(d_status, 0, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_labels, -1, num_nodes * sizeof(int)));

    int source_node = 0;
    int h_queue_size = 1;
    CUDA_CHECK(cudaMemcpy(&d_frontier_in[0], &source_node, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_queue_size, &h_queue_size, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&d_labels[source_node], &h_queue_size, sizeof(int), cudaMemcpyHostToDevice));

    int current_level = 1;

    // Allocate edge queue for two-phase
    int *d_edge_queue = nullptr, *d_edge_queue_size = nullptr;
    if (kernel_name == "two-phase") {
        CUDA_CHECK(cudaMalloc(&d_edge_queue, num_edges * sizeof(int))); // worst case
        CUDA_CHECK(cudaMalloc(&d_edge_queue_size, sizeof(int)));
    }

    // BFS loop
    while (true) {
        CUDA_CHECK(cudaMemcpy(&h_queue_size, d_next_queue_size, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_queue_size <= 0)
            break;
        CUDA_CHECK(cudaMemset(d_next_queue_size, 0, sizeof(int)));
        
        int num_blocks = (h_queue_size + CTA_SIZE - 1) / CTA_SIZE;
        size_t shared_mem_size = 0;

        if (kernel_name == "expand-contract") {
            shared_mem_size = CTA_SIZE * sizeof(int) * 3;
            
            bfs_expand_contract<<<num_blocks, CTA_SIZE, shared_mem_size>>>(d_R, d_C, d_status, d_labels, d_frontier_in, h_queue_size,d_frontier_out, d_next_queue_size, current_level);
            CUDA_CHECK(cudaDeviceSynchronize());
            swap(d_frontier_in, d_frontier_out);
        
        } else if (kernel_name == "contract-expand") {
            shared_mem_size = CTA_SIZE * sizeof(int) * 3;
            
            bfs_contract_expand<<<num_blocks, CTA_SIZE, shared_mem_size>>>(d_R, d_C, d_status, d_frontier_in, d_frontier_out, d_labels, h_queue_size, d_next_queue_size, current_level);
            CUDA_CHECK(cudaDeviceSynchronize());
            swap(d_frontier_in, d_frontier_out);
        
        } else if (kernel_name == "two-phase") {
            // Expansion
            CUDA_CHECK(cudaMemset(d_edge_queue_size, 0, sizeof(int)));
            shared_mem_size = CTA_SIZE * sizeof(int) * 3;
            
            bfs_expand<<<num_blocks, CTA_SIZE, shared_mem_size>>>(d_R, d_C, d_frontier_in, h_queue_size, d_edge_queue, d_edge_queue_size);
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_edge_queue_size = 0;
            CUDA_CHECK(cudaMemcpy(&h_edge_queue_size, d_edge_queue_size, sizeof(int), cudaMemcpyDeviceToHost));

            // Contraction
            if (h_edge_queue_size > 0) {
                int cta_ctr = (h_edge_queue_size + CTA_SIZE - 1) / CTA_SIZE;
                shared_mem_size = CTA_SIZE * sizeof(int);

                bfs_contract<<<cta_ctr, CTA_SIZE, shared_mem_size>>>(d_edge_queue, h_edge_queue_size, d_status, d_frontier_out, d_next_queue_size, current_level + 1, d_labels);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        
            std::swap(d_frontier_in, d_frontier_out);
        
        } else {
            cerr << "Invalid kernel name: " << kernel_name << endl;
            exit(1);
        }

        current_level++;
    }

    // Copy labels to host
    vector<int> h_labels(num_nodes);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    cout << "BFS Levels:" << endl;
    for (int i = 0; i < num_nodes; ++i)
        cout << "Node " << i << ": " << h_labels[i] << std::endl;

    // Free device memory
    if (kernel_name == "two-phase") {
        CUDA_CHECK(cudaFree(d_edge_queue));
        CUDA_CHECK(cudaFree(d_edge_queue_size));
    }

    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_status));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_frontier_in));
    CUDA_CHECK(cudaFree(d_frontier_out));
    CUDA_CHECK(cudaFree(d_next_queue_size));

    return 0;
}