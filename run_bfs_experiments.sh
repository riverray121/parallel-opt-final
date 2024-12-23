#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Create CSV file with headers
echo "graph_size,branching_factor,algorithm,source_node,time_ms,max_depth,nodes_visited" > results/bfs_results.csv

# Compile all implementations
echo "Compiling programs..."
nvcc -O3 generate_graphs_bf.cu -o generate_graphs
nvcc -O3 bfs_gpu.cu -o bfs_gpu
nvcc -O3 bfs_cpu.cu -o bfs_cpu

# Loop through branching factors (10, 30, 90, 270, 810)
# for bf in 2 5 10 15 20; do
for bf in 10 30 90 270 810; do
    echo "Testing with branching factor: $bf"
    
    # Generate graphs with current branching factor
    ./generate_graphs $bf
    
    # Run CPU version
    ./bfs_cpu $bf >> results/bfs_results.csv
    
    # Run GPU version
    ./bfs_gpu1 $bf >> results/bfs_results.csv
done

echo "Experiments completed! Results saved in results/bfs_results.csv" 