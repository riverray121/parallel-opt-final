#!/bin/bash

echo "Compiling graph generator..."
nvcc graph_generator.cu -o graph_generator

echo "Compiling BFS..."
nvcc bfs.cu -o bfs

echo "Running graph generator..."
./graph_generator

echo "Running BFS..."
./bfs

# Compile and run GPU BFS version 1
echo "Compiling GPU BFS version 1..."
nvcc -O3 bfs_gpu1.cu -o bfs_gpu1
echo "Running GPU BFS version 1..."
./bfs_gpu1

echo "Done!" 