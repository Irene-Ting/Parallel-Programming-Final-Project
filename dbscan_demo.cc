#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "dbscan.h"

#define V 10
#define D 2
// #define DEBUG

float cal_distance(float* a, float* b) {
    float sum = 0;
    for (int i = 0; i < D; i++) {
        sum += (a[i] - b[i])*(a[i] - b[i]);
    }
    return sqrt(sum);
}

int main(int argc, char** argv) {
    int** raw_vertices = new int *[V];
    for (int i = 0; i < V; i++) {
        raw_vertices[i] = new int[D];
    }
    int tmp[V][D] = {{2, 2}, {2, 3}, {1, 3}, {1, 4}, {1, 5}, {8, 7}, {8, 8}, {9, 8}, {9, 7}, {15, 30}};
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < D; j++) {
            raw_vertices[i][j] = tmp[i][j];
        }
    }

    graph neighbor = constuct_neighbor_pts(V, D, raw_vertices, 3);

    #ifdef DEBUG
    std::cout << neighbor.num_of_vertices << std::endl;
    std::cout << neighbor.num_of_edges << std::endl;
    for (int i = 0; i < neighbor.num_of_vertices; i++) {
        std::cout << neighbor.degree[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < neighbor.num_of_vertices; i++) {
        std::cout << neighbor.edge_pos[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < neighbor.num_of_edges; i++) {
        std::cout << neighbor.edge[i] << " ";
    }
    std::cout << std::endl;
    #endif

    DBSCAN dbscan(3, 2);
    int* cluster_label = dbscan.cluster(neighbor);

    dbscan.print_degree();
    dbscan.print_edge_pos();
    dbscan.print_edge();
    dbscan.print_cluster();
    /*
    3 4 4 4 3 3 3 3 3 0 
    0 3 7 11 15 18 21 24 27 30 
    1 2 3 0 2 3 4 0 1 3 4 0 1 2 4 1 2 3 6 7 8 5 7 8 5 6 8 5 6 7 
    0 0 0 0 0 1 1 1 1 -1 
    */
}
