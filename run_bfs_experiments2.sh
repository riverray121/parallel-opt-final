#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Create CSV file with headers
echo "graph_size,branching_factor,algorithm,source_node,time_ms,max_depth,nodes_visited" > results/bfs_results.csv

# Compile all implementations with explicit runtime and compute capability
echo "Compiling programs..."
nvcc -O3 generate_graphs_bf1000.cu -o generate_graphs --cudart shared
nvcc -O3 bfs_gpu3.cu -o bfs_gpu3 --cudart shared -arch=compute_35 -code=sm_35 -rdc=true
nvcc -O3 bfs.cu -o bfs_cpu --cudart shared

# Loop through branching factors (10, 30, 90, 270)
for bf in 10 30 90 270; do
    echo "Testing with branching factor: $bf"
    
    # Generate graphs with current branching factor
    ./generate_graphs $bf
    
    # Run CPU version
    ./bfs_cpu $bf >> results/bfs_results.csv
    
    # Run GPU version
    ./bfs_gpu3 $bf >> results/bfs_results.csv
done

echo "Experiments completed! Results saved in results/bfs_results.csv"

# Optional: Run the Python plotting script if it exists
if [ -f "plot_results.py" ]; then
    python3 plot_results.py results/bfs_results.csv
fi 