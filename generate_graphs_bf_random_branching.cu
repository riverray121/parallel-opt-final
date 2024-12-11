#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>

void generate_graph(std::ofstream& file, int size, int branching_factor) {
    file << size << "\n";
    
    for (int i = 0; i < size; i++) {
        file << i;
        for (int j = 0; j < branching_factor && i + j + 1 < size; j++) {
            file << " " << (i + j + 1);
        }
        file << "\n";
    }
    file << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <branching_factor>\n";
        return 1;
    }
    
    const int branching_factor = std::stoi(argv[1]);
    
    std::ofstream file("random_graphs.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing\n";
        return 1;
    }
    
    // Generate graphs of different sizes
    std::vector<int> sizes = {1000, 2500, 5000, 10000, 20000};
    // std::vector<int> sizes = {1000000, 10000000};
    
    for (int size : sizes) {
        generate_graph(file, size, branching_factor);
    }
    
    file.close();
    return 0;
} 