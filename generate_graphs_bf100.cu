#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <set>

void generate_graph(std::ofstream& file, int size, int branching_factor) {
    // Write the number of vertices
    file << size << "\n";
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size - 1);
    
    // Generate edges for each vertex
    for (int i = 0; i < size; i++) {
        file << i;  // Write vertex number
        std::set<int> neighbors;  // Use set to avoid duplicates
        
        // Generate branching_factor unique neighbors
        while (neighbors.size() < std::min(branching_factor, size - 1)) {
            int neighbor = dis(gen);
            if (neighbor != i) {  // Avoid self-loops
                neighbors.insert(neighbor);
            }
        }
        
        // Write neighbors
        for (int neighbor : neighbors) {
            file << " " << neighbor;
        }
        file << "\n";
    }
    
    file << "-1\n";  // Graph separator
}

int main() {
    std::ofstream file("random_graphs.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing\n";
        return 1;
    }
    
    // Generate graphs of different sizes
    std::vector<int> sizes = {1000, 5000, 10000, 50000, 100000};
    const int branching_factor = 1000;
    
    for (int size : sizes) {
        generate_graph(file, size, branching_factor);
    }
    
    file.close();
    std::cout << "Generated " << sizes.size() << " graphs with branching factor " << branching_factor << "\n";
    return 0;
} 