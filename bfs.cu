#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>  // Add this for timing

// Use specific using declarations for frequently used components
using std::vector;
using std::queue;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::istringstream;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

void BFS(const vector<vector<int>>& graph, int source) {
    auto start_time = high_resolution_clock::now();  // Start timing
    
    int n = graph.size();
    vector<int> dist(n, INT_MAX);
    queue<int> Q;
    
    int max_depth = 0;
    int nodes_visited = 0;
    vector<int> nodes_at_depth(n, 0);

    dist[source] = 0;
    Q.push(source);
    nodes_visited++;
    nodes_at_depth[0] = 1;

    while (!Q.empty()) {
        int current = Q.front();
        Q.pop();

        for (int neighbor : graph[current]) {
            if (dist[neighbor] == INT_MAX) {
                dist[neighbor] = dist[current] + 1;
                max_depth = std::max(max_depth, dist[neighbor]);
                nodes_visited++;
                nodes_at_depth[dist[neighbor]]++;
                Q.push(neighbor);
            }
        }
    }

    auto end_time = high_resolution_clock::now();  // End timing
    auto duration = duration_cast<microseconds>(end_time - start_time);
    
    // Print detailed statistics
    cout << "BFS Statistics:\n";
    cout << "Time taken: " << duration.count() / 1000.0 << " milliseconds\n";
    cout << "Maximum depth reached: " << max_depth << "\n";
    cout << "Total nodes visited: " << nodes_visited << " out of " << n << "\n";
    cout << "Nodes at each depth:\n";
    for (int d = 0; d <= max_depth; d++) {
        if (nodes_at_depth[d] > 0) {
            cout << "Depth " << d << ": " << nodes_at_depth[d] << " nodes\n";
        }
    }
}

vector<vector<int>> read_graph(ifstream& file) {
    string line;
    
    getline(file, line);
    int n = std::stoi(line);
    
    vector<vector<int>> graph(n);
    
    for (int i = 0; i < n; i++) {
        getline(file, line);
        istringstream iss(line);
        
        string vertex;
        iss >> vertex;
        
        int neighbor;
        while (iss >> neighbor) {
            graph[i].push_back(neighbor);
        }
    }
    
    getline(file, line);
    
    return graph;
}

int main() {
    ifstream file("random_graphs.txt");
    if (!file.is_open()) {
        cerr << "Error: Could not open random_graphs.txt\n";
        return 1;
    }

    int graph_number = 1;
    while (!file.eof()) {
        string peek;
        if (!getline(file, peek)) {
            break;
        }
        file.seekg(-peek.length()-1, std::ios::cur);
        
        vector<vector<int>> graph = read_graph(file);
        if (graph.empty()) break;
        
        cout << "\n=== Graph " << graph_number << " (Size: " << graph.size() << ") ===\n";
        
        cout << "\nStarting from vertex 0:\n";
        BFS(graph, 0);
        
        int mid_vertex = graph.size() / 2;
        cout << "\nStarting from vertex " << mid_vertex << ":\n";
        BFS(graph, mid_vertex);
        
        graph_number++;
    }

    file.close();
    return 0;
}
