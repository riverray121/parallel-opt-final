#!/bin/bash

# Compile the graph generator
echo "Compiling graph generator..."
nvcc -O3 generate_graphs_bf100.cu -o generate_graphs

# Compile the GPU BFS implementation
echo "Compiling GPU BFS implementation..."
nvcc -O3 bfs_gpu1.cu -o bfs_gpu1

# Generate the graphs
echo "Generating graphs..."
./generate_graphs

# Run the BFS
echo "Running BFS experiments..."
./bfs_gpu1

echo "Experiments completed!" 