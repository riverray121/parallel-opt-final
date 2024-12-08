#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <fstream>

using namespace std;

// Generate a random graph with n vertices and approximately edge_density percentage of possible edges
vector<vector<int>> generate_random_graph(int n, double edge_density) {
    vector<vector<int>> graph(n);
    
    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> prob(0.0, 1.0);
    uniform_int_distribution<> vertex_dist(0, n-1);
    
    // For each pair of vertices
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // Add edge with probability edge_density
            if (prob(gen) < edge_density) {
                graph[i].push_back(j);
                graph[j].push_back(i); // Since it's an undirected graph
            }
        }
    }
    
    return graph;
}

// Save graph to file
void save_graph_to_file(const vector<vector<int>>& graph, const string& filename) {
    ofstream out(filename, ios::app);
    out << graph.size() << "\n"; // First line contains number of vertices
    
    // Write each adjacency list
    for (size_t i = 0; i < graph.size(); i++) {
        out << i << ": ";
        for (int neighbor : graph[i]) {
            out << neighbor << " ";
        }
        out << "\n";
    }
    out << "\n"; // Empty line between graphs
    out.close();
}

int main() {
    // Seed for reproducibility
    srand(time(0));
    
    // Clear the output file
    ofstream out("random_graphs.txt", ios::trunc);
    out.close();
    
    // Generate 20 random graphs
    for (int i = 0; i < 20; i++) {
        // Random size between 5 and 50 vertices
        int size = rand() % 46 + 5;
        
        // Random edge density between 0.1 and 0.5
        double density = (rand() % 40 + 10) / 100.0;
        
        // Generate and save the graph
        vector<vector<int>> graph = generate_random_graph(size, density);
        
        cout << "Generated graph " << i + 1 << " with " << size << " vertices"
             << " and density " << density << "\n";
             
        save_graph_to_file(graph, "random_graphs.txt");
    }
    
    cout << "\nAll graphs have been saved to 'random_graphs.txt'\n";
    return 0;
} 