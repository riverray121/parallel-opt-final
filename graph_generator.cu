#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <fstream>

using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ofstream;
using std::ios;

vector<vector<int>> generate_deep_graph(int n) {
    vector<vector<int>> graph(n);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 2);  // Reduced to 1-2 connections
    std::uniform_real_distribution<> prob(0.0, 1.0);
    
    // Create several separate long paths
    int num_paths = n / 100;  // More separate paths for larger graphs
    vector<int> path_starts;
    
    for (int p = 0; p < num_paths; p++) {
        int start = (p * n) / num_paths;
        path_starts.push_back(start);
        
        // Create a winding path from this start point
        int current = start;
        int remaining_length = (n / num_paths) - 1;
        
        while (remaining_length > 0 && current < n-1) {
            int next = current + 1;
            graph[current].push_back(next);
            graph[next].push_back(current);
            current = next;
            remaining_length--;
        }
    }
    
    // Add sparse connections between paths
    for (int i = 0; i < n; i++) {
        if (prob(gen) < 0.05) {  // Only 5% chance to add cross-path connection
            int jump = dist(gen) * 100;  // Make jumps between paths larger
            if (i + jump < n) {
                graph[i].push_back(i + jump);
                graph[i + jump].push_back(i);
            }
        }
    }
    
    // Add some dead ends and branches
    for (int i = 0; i < n; i++) {
        if (prob(gen) < 0.1) {  // 10% chance to add a branch
            int branch_length = dist(gen);
            int current = i;
            
            // Create a branch that leads to nowhere
            for (int j = 0; j < branch_length && current < n-1; j++) {
                int next = current + 1;
                graph[current].push_back(next);
                graph[next].push_back(current);
                current = next;
            }
        }
    }
    
    return graph;
}

void save_graph_to_file(const vector<vector<int>>& graph, const string& filename) {
    ofstream out(filename, ios::app);
    out << graph.size() << "\n";
    
    for (size_t i = 0; i < graph.size(); i++) {
        out << i << ": ";
        for (int neighbor : graph[i]) {
            out << neighbor << " ";
        }
        out << "\n";
    }
    out << "\n";
    out.close();
}

int main() {
    srand(time(0));
    
    ofstream out("random_graphs.txt", ios::trunc);
    out.close();
    
    // Generate 10 graphs
    for (int i = 0; i < 10; i++) {
        // Random size between 500 and 2000 vertices
        int size = rand() % 1501 + 500;
        
        vector<vector<int>> graph = generate_deep_graph(size);
        
        cout << "Generated graph " << i + 1 << " with " << size << " vertices\n";
        
        save_graph_to_file(graph, "random_graphs.txt");
    }
    
    cout << "\nAll graphs have been saved to 'random_graphs.txt'\n";
    return 0;
} 