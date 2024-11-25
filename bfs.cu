#include <iostream>
#include <vector>
#include <queue>
#include <climits>

void BFS(const std::vector<std::vector<int>>& graph, int source) {
    int n = graph.size();
    std::vector<int> dist(n, INT_MAX);
    std::queue<int> Q;

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
    std::cout << "Distances from node " << source << ":\n";
    for (int i = 0; i < n; ++i) {
        if (dist[i] == INT_MAX) {
            std::cout << "Node " << i << ": Unreachable\n";
        } else {
            std::cout << "Node " << i << ": " << dist[i] << "\n";
        }
    }
}

int main() {
    // Example graphs
    std::vector<std::vector<int>> graph1 = {
        {1, 2},      // Neighbors of node 0
        {0, 3, 4},   // Neighbors of node 1
        {0, 4},      // Neighbors of node 2
        {1, 5},      // Neighbors of node 3
        {1, 2},      // Neighbors of node 4
        {3}          // Neighbors of node 5
    };

    std::vector<std::vector<int>> graph2 = {
        {1},         // Neighbors of node 0
        {0, 2, 3},   // Neighbors of node 1
        {1, 4},      // Neighbors of node 2
        {1, 5},      // Neighbors of node 3
        {2},         // Neighbors of node 4
        {3}          // Neighbors of node 5
    };

    std::cout << "Graph 1:\n";
    BFS(graph1, 0);

    std::cout << "\nGraph 2:\n";
    BFS(graph2, 1);

    return 0;
}
