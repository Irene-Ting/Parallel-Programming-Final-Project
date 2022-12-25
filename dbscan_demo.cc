#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "dbscan.h"

#define V 10
#define D 2

float cal_distance(float* a, float* b) {
    float sum = 0;
    for (int i = 0; i < D; i++) {
        sum += (a[i] - b[i])*(a[i] - b[i]);
    }
    return sqrt(sum);
}

int main(int argc, char** argv) {
    int** graph = new int *[V];
    for (int i = 0; i < V; i++) {
        graph[i] = new int[D];
    }
    int tmp[V][D] = {{2, 2}, {2, 3}, {1, 3}, {1, 4}, {1, 5}, {8, 7}, {8, 8}, {9, 8}, {9, 7}, {15, 30}};
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < D; j++) {
            graph[i][j] = tmp[i][j];
        }
    }

    DBSCAN dbscan(3, 2);
    int* cluster_label = new int[V];
    cluster_label = dbscan.cluster(V, D, graph);
    dbscan.print_cluster();
    // ans: 0 0 0 0 0 1 1 1 1 -1 
}
