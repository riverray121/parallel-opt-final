#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

void BFS(const vector<vector<int>>& graph, int source) {
    int n = graph.size();
    vector<int> dist(n, INT_MAX);
    queue<int> Q;

    dist[source] = 0;
    Q.push(source);

    while (!Q.empty()) {
        int current = Q.front();
        Q.pop();

        for (int neighbor : graph[current]) {
            if (dist[neighbor] == INT_MAX) {
                dist[neighbor] = dist[current] + 1;
                Q.push(neighbor);
            }
        }
    }

    // Print distances from the source
    cout << "Distances from node " << source << ":\n";
    for (int i = 0; i < n; ++i) {
        if (dist[i] == INT_MAX) {
            cout << "Node " << i << ": Unreachable\n";
        } else {
            cout << "Node " << i << ": " << dist[i] << "\n";
        }
    }
}

// Function to read a graph from the input stream
vector<vector<int>> read_graph(ifstream& file) {
    string line;
    
    // Read number of vertices
    getline(file, line);
    int n = stoi(line);
    
    vector<vector<int>> graph(n);
    
    // Read each adjacency list
    for (int i = 0; i < n; i++) {
        getline(file, line);
        istringstream iss(line);
        
        string vertex;
        iss >> vertex; // Skip the vertex number and colon
        
        int neighbor;
        while (iss >> neighbor) {
            graph[i].push_back(neighbor);
        }
    }
    
    // Skip the empty line between graphs
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
        // Try to read the next graph
        string peek;
        if (!getline(file, peek)) {
            break;  // Exit if we can't read anymore
        }
        file.seekg(-peek.length()-1, ios::cur); // Go back to start of graph
        
        vector<vector<int>> graph = read_graph(file);
        if (graph.empty()) break;  // Exit if we got an empty graph
        
        cout << "\n=== Graph " << graph_number << " (Size: " << graph.size() << ") ===\n";
        
        // Run BFS from vertex 0 and from a middle vertex
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
