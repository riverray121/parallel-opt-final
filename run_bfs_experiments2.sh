#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Create CSV file with headers
echo "graph_size,branching_factor,algorithm,source_node,time_ms,max_depth,nodes_visited" > results/bfs_results_dynamic.csv

# Compile the dynamic parallelism version
echo "Compiling programs..."
nvcc -O3 generate_graphs_bf1000.cu -o generate_graphs
nvcc -arch=sm_35 -rdc=true bfs_gpu3.cu -o bfs_gpu3

# Loop through branching factors (10, 30, 90, 270, 810)
for bf in 10 30 90 270 810; do
    echo "Testing with branching factor: $bf"
    
    # Generate graphs with current branching factor
    ./generate_graphs $bf
    
    # Run GPU version with dynamic parallelism
    ./bfs_gpu3 $bf >> results/bfs_results_dynamic.csv
done

echo "Experiments completed! Results saved in results/bfs_results_dynamic.csv"

# Optional: Run the Python plotting script if it exists
if [ -f "plot_results.py" ]; then
    python3 plot_results.py results/bfs_results_dynamic.csv
fi 