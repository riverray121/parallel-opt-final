#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

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

void BFS(const vector<vector<int>>& graph, int source, int branching_factor) {
    auto start_time = high_resolution_clock::now();
    
    int n = graph.size();
    vector<int> dist(n, INT_MAX);
    queue<int> Q;
    
    int max_depth = 0;
    int nodes_visited = 0;

    dist[source] = 0;
    Q.push(source);
    nodes_visited++;

    while (!Q.empty()) {
        int current = Q.front();
        Q.pop();

        for (int neighbor : graph[current]) {
            if (dist[neighbor] == INT_MAX) {
                dist[neighbor] = dist[current] + 1;
                max_depth = std::max(max_depth, dist[neighbor]);
                nodes_visited++;
                Q.push(neighbor);
            }
        }
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    
    printf("%lu,%d,CPU,%d,%.3f,%d,%d\n", graph.size(), branching_factor, source, duration.count() / 1000.0, max_depth, nodes_visited);
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }
    
    const int branching_factor = std::stoi(argv[1]);
    
    auto total_start_time = high_resolution_clock::now();
    
    ifstream file("random_graphs.txt");
    if (!file.is_open()) {
        // cerr << "Error: Could not open random_graphs.txt\n";
        return 1;
    }

    int graph_number = 1;
    int total_searches = 0;
    
    while (!file.eof()) {
        string peek;
        if (!getline(file, peek)) break;
        file.seekg(-peek.length()-1, std::ios::cur);
        
        vector<vector<int>> graph = read_graph(file);
        if (graph.empty()) break;
        
        BFS(graph, 0, branching_factor);
        BFS(graph, graph.size() / 2, branching_factor);
        
        graph_number++;
        total_searches += 2;
    }

    file.close();
    
    auto total_end_time = high_resolution_clock::now();
    auto total_duration = duration_cast<microseconds>(total_end_time - total_start_time);
    
    // cout << "\nTotal Statistics:\n";
    // cout << "Total time: " << total_duration.count() / 1000.0 << " milliseconds\n";
    // cout << "Graphs processed: " << graph_number - 1 << "\n";
    // cout << "Total searches performed: " << total_searches << "\n";
    // cout << "Average time per search: " << (total_duration.count() / total_searches) / 1000.0 << " milliseconds\n";

    return 0;
}
