#!/bin/bash

echo "Compiling graph generator..."
nvcc graph_generator.cu -o graph_generator

echo "Compiling BFS..."
nvcc bfs.cu -o bfs

echo "Running graph generator..."
./graph_generator

echo "Running BFS..."
./bfs

echo "Done!" 