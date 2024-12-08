#!/bin/bash

# Compile the graph generator
echo "Compiling graph generator..."
nvcc -O3 generate_graphs_bf100.cu -o generate_graphs

# Compile both BFS implementations
echo "Compiling BFS implementations..."
nvcc -O3 bfs_gpu1.cu -o bfs_gpu1
g++ -O3 bfs_cpu.cpp -o bfs_cpu

# Generate the graphs
echo "Generating graphs..."
./generate_graphs

# Run both BFS implementations
echo "Running BFS experiments..."
echo -e "\nCPU BFS Results:"
./bfs_cpu
echo -e "\nGPU BFS Results:"
./bfs_gpu1

echo "Experiments completed!" 