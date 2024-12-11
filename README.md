# CUDA Graph Search and Analysis

This project involves running CUDA-based graph search experiments and visualizing the results using Python. Follow the steps below to set up and execute the project.

---

## Prerequisites

### System Requirements
1. **CUDA Toolkit**  
   Ensure the CUDA Toolkit and NVIDIA CUDA Compiler Driver (`nvcc`) are installed on your system.

2. **Python 3 and Required Libraries**  
   Install the following Python libraries:
   - `matplotlib`
   - `pandas`
   - `seaborn`

   You can install these libraries using pip:
   ```bash
   pip install matplotlib pandas seaborn
   ```

---

## Configuration

1. **Edit Graph Sizes**  
   Open the `generate_graphs_bf.cu` file and modify the following line to specify the graph sizes to search:
   ```cpp
   std::vector<int> sizes = {1000000, 10000000};
   ```
   > Note: Large graphs may result in running out of memory and crashing.

2. **Edit Branching Factors**  
   Open the `run_bfs_experiments.sh` file and update the following line to set the branching factors to loop through:
   ```bash
   for bf in 10 30 90; do
   ```

3. **Make Script Executable**  
   Ensure the `run_bfs_experiments.sh` script is executable:
   ```bash
   chmod +x run_bfs_experiments.sh
   ```

---

## Running the Experiments

1. Execute the experiment script:
   ```bash
   ./run_bfs_experiments.sh
   ```

2. Once the graph searches conclude, generate the result graphs using the Python script:
   ```bash
   python plot_results.py
   ```

---

## Notes

- Be cautious of memory usage when configuring large graph sizes.

---