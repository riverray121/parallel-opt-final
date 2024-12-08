#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>

// Use specific using declarations for frequently used components
using std::vector;
using std::queue;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::istringstream;

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

    cout << "Distances from node " << source << ":\n";
    for (int i = 0; i < n; ++i) {
        if (dist[i] == INT_MAX) {
            cout << "Node " << i << ": Unreachable\n";
        } else {
            cout << "Node " << i << ": " << dist[i] << "\n";
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
