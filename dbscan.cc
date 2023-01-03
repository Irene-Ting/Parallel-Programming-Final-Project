#include <assert.h>
#include <math.h>
#include <iostream>
#include <queue>
#include <chrono>
#include "dbscan.h"

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

float 
DBSCAN::cal_distance(int* a, int* b) {
    float sum = 0;
    for (int i = 0; i < dimension; i++) {
        sum += (a[i] - b[i])*(a[i] - b[i]);
    }
    float ans = sqrt(sum);
    // std::cout << ans << " ";
    return ans;
}

int*
DBSCAN::cluster(int v, int d, int** raw_vertices) {
    num_of_vertices = v;
    dimension = d;
    vertices = new vertex[num_of_vertices];

    std::chrono::steady_clock::time_point timeBegin;
    std::chrono::steady_clock::time_point timeEnd;
    cluster_construct = 0;
    neighbor_construct = 0;
    std::cout << "neighbor_construct: " << neighbor_construct << " cluster_construct: " << cluster_construct << "\n";
    timeBegin = std::chrono::steady_clock::now();
    for (int i = 0; i < num_of_vertices; i++) {
        int cnt = 0;
        for (int j = 0; j < num_of_vertices; j++) {
            if (i == j) continue;
            if (cal_distance(raw_vertices[i], raw_vertices[j]) < eps) {
                vertices[i].neighbors.push_back(j);
                cnt++;
            }
        }
        if (cnt > minPts) {
            vertices[i].type = Core;
        } else if (cnt == 0) {
            vertices[i].type = Noise;
        }
    }
    timeEnd = std::chrono::steady_clock::now();
    neighbor_construct += std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeBegin).count();
    std::cout << "neighbor_construct: " << neighbor_construct << "\n";
    timeBegin = std::chrono::steady_clock::now();
    int cluster_id = 0;
    for (int i = 0; i < num_of_vertices; i++) {
        if (!vertices[i].visited && vertices[i].type == Core) {
            timeBegin = std::chrono::steady_clock::now();
            vertices[i].cluster = cluster_id;
            BFS(i);
            cluster_id += 1;
            timeEnd = std::chrono::steady_clock::now();
            cluster_construct += std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeBegin).count();
            std::cout << "i: " << i << " cluster_construct: " << cluster_construct << "\n";
        }
    }
    timeEnd = std::chrono::steady_clock::now();
    cluster_construct += std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeBegin).count();
    std::cout << "cluster_construct: " << cluster_construct << "\n";
    std::cout << "Total: " << cluster_construct + neighbor_construct << "\n";
    set_cluster_label();
    set_cluster_color();
    return cluster_label;
}
