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
    std::uniform_int_distribution<> dist(1, 3); // Each node connects to 1-3 nodes ahead
    std::uniform_real_distribution<> prob(0.0, 1.0);
    
    // First ensure a main deep path exists
    for (int i = 0; i < n-1; i++) {
        int next = i + 1;
        graph[i].push_back(next);
        graph[next].push_back(i);
    }
    
    // Then add some branching paths
    for (int i = 0; i < n-3; i++) {
        int num_branches = dist(gen);
        for (int b = 0; b < num_branches; b++) {
            if (prob(gen) < 0.3) { // 30% chance to add a branch
                int jump = dist(gen) + 1; // Connect to a node 2-4 steps ahead
                if (i + jump < n) {
                    graph[i].push_back(i + jump);
                    graph[i + jump].push_back(i);
                }
            }
        }
    }
    
    // Add a few random long-range connections
    int num_long_range = n / 50; // Add more long range connections for larger graphs
    std::uniform_int_distribution<> long_range(0, n-1);
    for (int i = 0; i < num_long_range; i++) {
        int from = long_range(gen);
        int to = long_range(gen);
        if (from != to && prob(gen) < 0.1) { // 10% chance for long range connection
            graph[from].push_back(to);
            graph[to].push_back(from);
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