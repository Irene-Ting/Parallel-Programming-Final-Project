#include <assert.h>
#include <math.h>
#include <iostream>
#include <queue>
#include "dbscan.h"
// #define DEBUG

bool
DBSCAN::is_neighbor(int a, int b) {
    for (auto n : vertices[a].neighbors) {
        if (n == b) {
            return true;
        }
    }
    return false;
}

void 
DBSCAN::BFS(int id) {
    std::queue<int> vtx_queue;
    vtx_queue.push(id);
    int num_of_set_true = 0;
    while (!vtx_queue.empty()) {
        int target_idx = vtx_queue.front();
        if (vertices[target_idx].visited) {
            vtx_queue.pop();
            continue;
        }
        vertices[target_idx].visited = true;
        vertex target = vertices[target_idx];
        if (target.type == Core) {
            for (int i = 0; i < num_of_vertices; i++) {
                if (is_neighbor(target_idx, i) && !vertices[i].visited) {
                    vtx_queue.push(i);
                    vertices[i].cluster = target.cluster;
                    if (vertices[i].type != Core) {
                        vertices[i].type = Border;
                    }
                }
            }
        }
        vtx_queue.pop();
    }
}

void 
DBSCAN::print_adjacency_lists() {
    for (int i = 0; i < num_of_vertices; i++) {
        std::cout << "vertex " << i << ": ";
        for (auto n : vertices[i].neighbors) {
            std::cout << n << " ";
        }
        std::cout << "\n";
    }
}

void 
DBSCAN::print_cluster() {
    for (int i = 0; i < num_of_vertices; i++) {
        std::cout << cluster_label[i] << " ";
    }
    std::cout << "\n";
}

void
DBSCAN::set_cluster_label() {
    cluster_label = new int[num_of_vertices];
    for (int i = 0; i < num_of_vertices; i++) {
        cluster_label[i] = vertices[i].cluster;
        num_of_cluster = (vertices[i].cluster > num_of_cluster) ? vertices[i].cluster : num_of_cluster;
    }
}

void
DBSCAN::set_cluster_color() {
    colors = new int [num_of_cluster];
    for (int i = 0; i < num_of_cluster; i++) {
        colors[i] = -1;
    }
    for (int i = 0; i < num_of_vertices; i++) {
        if (colors[vertices[i].cluster] == -1) {
            colors[vertices[i].cluster] = i;
        }
    }
}

bool  
DBSCAN::is_close(int* a, int* b) {
    float sum = 0;
    int eps_square = eps * eps;
    for (int i = 0; i < dimension; i++) {
        int diff = a[i] - b[i];
        if (diff >= eps) {
            return false;
        }
        sum += diff * diff;
        if (diff >= eps_square) {
            return false;
        }
    }
    return sqrt(sum) < eps;
}

void
DBSCAN::constuct_neighbor(int** raw_vertices) {
    for (int i = 0; i < num_of_vertices; i++) {
        #ifdef DEBUG
        std::cout << i << " / " << num_of_vertices << "\n";
        #endif
        for (int j = 0; j < num_of_vertices; j++) {
            if (i == j) continue;
            if (is_close(raw_vertices[i], raw_vertices[j])) {
                vertices[i].neighbors.push_back(j);
            }
        }
    }
}

int*
DBSCAN::cluster(int v, int d, int** raw_vertices) {
    num_of_vertices = v;
    dimension = d;
    vertices = new vertex[num_of_vertices];
    constuct_neighbor(raw_vertices);

    for (int i = 0; i < num_of_vertices; i++) {
        int pts = vertices[i].neighbors.size();
        if (pts > minPts) {
            vertices[i].type = Core;
        } else if (pts == 0) {
            vertices[i].type = Noise;
        }
    }
    int cluster_id = 0;
    for (int i = 0; i < num_of_vertices; i++) {
        #ifdef DEBUG
        std::cout << i << " / " << num_of_vertices << "\n";
        #endif
        if (!vertices[i].visited && vertices[i].type == Core) {
            vertices[i].cluster = cluster_id;
            BFS(i);
            cluster_id += 1;
        }
    }

    set_cluster_label();
    set_cluster_color();
    return cluster_label;
}
