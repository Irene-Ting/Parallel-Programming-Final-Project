#include <assert.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <string.h>
#include "dbscan.h"
#define BS 1024
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

__global__ void bfs(int *edge, int *edge_pos, int *degree, bool *is_core, bool *frontier, int *cluster_label, int cluster_id, bool *done, int num_of_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_of_vertices) return;
    if (frontier[tid] && cluster_label[tid] == -1 && is_core[tid]) {
        frontier[tid] = false;
        cluster_label[tid] = cluster_id;
        __syncthreads();
        for (int i = edge_pos[tid]; i < edge_pos[tid] + degree[tid]; i++) {
            if (!cluster_label[edge[i]] != -1) {
                frontier[edge[i]] = true;
                *done = false;
            }
        }
    }
}

void 
DBSCAN::BFS(int id, int cluster_id) {
    *done = false;
    memset(frontier, false, sizeof(bool) * num_of_vertices);
    frontier[id] = true;
    cudaMemcpy(d_frontier, frontier, sizeof(bool) * num_of_vertices, cudaMemcpyHostToDevice); 
    while (!(*done)) {
        *done = true;
        cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice); 
        bfs<<<(num_of_vertices+BS-1)/BS, BS>>>(d_edge, d_edge_pos, d_degree, d_is_core, d_frontier, d_cluster_label, cluster_id, d_done, num_of_vertices);
        cudaMemcpy(done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
    }
}

void 
DBSCAN::print_adjacency_lists() {
    for (int i = 0; i < num_of_vertices; i++) {
        std::cout << "vertex " << i << ": ";
        for (auto n : vertices[i].neighbors) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
    }
}

void 
DBSCAN::print_edge() {
    for (int i = 0; i < num_of_edges; i++) {
        std::cout << edge[i] << " ";
    }
    std::cout << std::endl;
}

void 
DBSCAN::print_edge_pos() {
    for (int i = 0; i < num_of_vertices; i++) {
        std::cout << edge_pos[i] << " ";
    }
    std::cout << std::endl;
}

void 
DBSCAN::print_degree() {
    for (int i = 0; i < num_of_vertices; i++) {
        std::cout << degree[i] << " ";
    }
    std::cout << std::endl;
}

void 
DBSCAN::print_cluster() {
    for (int i = 0; i < num_of_vertices; i++) {
        std::cout << cluster_label[i] << " ";
    }
    std::cout << std::endl;
}

void 
DBSCAN::print_type() {
    for (int i = 0; i < num_of_vertices; i++) {
        std::cout << is_core[i] << " ";
    }
    std::cout << std::endl;
}

void
DBSCAN::set_cluster_color() {
    num_of_cluster = 0;
    for (int i = 0; i < num_of_vertices; i++) {
        num_of_cluster = (cluster_label[i] > num_of_cluster) ? cluster_label[i] : num_of_cluster;
    }
    colors = new int [num_of_cluster+1];
    memset(colors, -1, sizeof(int) * (num_of_cluster+1));
    for (int i = 0; i < num_of_cluster; i++) {
        colors[i] = -1;
    }
    for (int i = 0; i < num_of_vertices; i++) {
        if (colors[cluster_label[i]] == -1) {
            colors[cluster_label[i]] = i;
        }
    }
}

bool  
DBSCAN::is_close(int* a, int* b) {
    float sum = 0;
    int eps_square = eps * eps;
    for (int i = 0; i < dimension; i++) {
        int diff = a[i] - b[i];
        if (diff > eps) {
            return false;
        }
        sum += diff * diff;
        if (diff > eps_square) {
            return false;
        }
    }
    return sqrt(sum) <= eps;
}

void
DBSCAN::constuct_neighbor(int** raw_vertices) {
    for (int i = 0; i < num_of_vertices; i++) {
        #ifdef DEBUG
        std::cout << "constuct_neighbor: " << i << " / " << num_of_vertices << std::endl;
        #endif
        for (int j = 0; j < num_of_vertices; j++) {
            if (i == j) continue;
            if (is_close(raw_vertices[i], raw_vertices[j])) {
                vertices[i].neighbors.push_back(j);
                degree[i]++;
            }
        }
    }
}

int*
DBSCAN::cluster(int v, int d, int** raw_vertices) {
    num_of_vertices = v;
    dimension = d;
    vertices = new vertex[num_of_vertices];
    edge_pos = new int[num_of_vertices];
    degree = new int[num_of_vertices];
    memset(degree, 0, num_of_vertices);
    is_core = new bool[num_of_vertices];
    frontier = new bool[num_of_vertices];
    cluster_label = new int[num_of_vertices];
    memset(cluster_label, -1, sizeof(int) * num_of_vertices);
    done = new bool;

    constuct_neighbor(raw_vertices);
    
    num_of_edges = 0;
    for (int i = 0; i < num_of_vertices; i++) {
        num_of_edges += degree[i];
        if (i > 0) {
            edge_pos[i] = edge_pos[i-1] + degree[i-1];
        } else {
            edge_pos[i] = 0;
        }
    }

    edge = new int[num_of_edges];
    for (int i = 0; i < num_of_vertices; i++) {
        for (int j = 0; j < vertices[i].neighbors.size(); j++) {
            edge[edge_pos[i] + j] = vertices[i].neighbors[j];
        }
    }

    for (int i = 0; i < num_of_vertices; i++) {
        int pts = degree[i];
        if (pts > minPts) {
            is_core[i] = true;
        } else {
            is_core[i] = false;
        }
    }
    cudaMalloc((void **)&d_edge, sizeof(int) * num_of_edges);
    cudaMalloc((void **)&d_edge_pos, sizeof(int) * num_of_vertices);
    cudaMalloc((void **)&d_degree, sizeof(int) * num_of_vertices);
    cudaMalloc((void **)&d_is_core, sizeof(bool) * num_of_vertices);
    cudaMalloc((void **)&d_frontier, sizeof(bool) * num_of_vertices);
    cudaMalloc((void **)&d_cluster_label, sizeof(int) * num_of_vertices);
    cudaMalloc((void **)&d_done, sizeof(bool));

    cudaMemcpy(d_edge, edge, sizeof(int) * num_of_edges, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_edge_pos, edge_pos, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_degree, degree, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_is_core, is_core, sizeof(bool) * num_of_vertices, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_cluster_label, cluster_label, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice); 
    
    int cluster_id = 0;
    for (int i = 0; i < num_of_vertices; i++) {
        #ifdef DEBUG
        std::cout << "cluster: " << i << " / " << num_of_vertices << std::endl;
        #endif
        if (cluster_label[i] == -1 && is_core[i]) {
            #ifdef DEBUG
            std::cout << "BFS(" << i << ", " << cluster_id << ")\n";
            #endif
            BFS(i, cluster_id);
            cluster_id += 1;
        }
        cudaMemcpy(cluster_label, d_cluster_label, sizeof(int) * num_of_vertices, cudaMemcpyDeviceToHost);
    }
    set_cluster_color();
    return cluster_label;
}
