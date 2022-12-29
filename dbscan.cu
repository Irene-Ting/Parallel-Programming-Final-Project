#include <assert.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <string.h>
#include "dbscan.h"
#define BS 1024
// #define DEBUG

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
        cudaMemcpy(d_done, done, sizeof(bool), cudaMemcpyHostToDevice); 
        bfs<<<(num_of_vertices+BS-1)/BS, BS>>>(d_edge, d_edge_pos, d_degree, d_is_core, d_frontier, d_cluster_label, cluster_id, d_done, num_of_vertices);
        cudaMemcpy(done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
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

// bool  
// DBSCAN::is_close(int* a, int* b) {
//     float sum = 0;
//     int eps_square = eps * eps;
//     for (int i = 0; i < dimension; i++) {
//         int diff = a[i] - b[i];
//         if (diff > eps) {
//             return false;
//         }
//         sum += diff * diff;
//         if (diff > eps_square) {
//             return false;
//         }
//     }
//     return sqrt(sum) <= eps;
// }

// void
// DBSCAN::constuct_neighbor(int** raw_vertices) {
//     for (int i = 0; i < num_of_vertices; i++) {
//         #ifdef DEBUG
//         std::cout << "constuct_neighbor: " << i << " / " << num_of_vertices << std::endl;
//         #endif
//         for (int j = 0; j < num_of_vertices; j++) {
//             if (i == j) continue;
//             if (is_close(raw_vertices[i], raw_vertices[j])) {
//                 vertices[i].neighbors.push_back(j);
//                 degree[i]++;
//             }
//         }
//     }
// }

int*
DBSCAN::cluster(graph neighbors) {
    num_of_vertices = neighbors.num_of_vertices;
    num_of_edges = neighbors.num_of_edges;
    edge_pos = neighbors.edge_pos;
    degree = neighbors.degree;
    edge = neighbors.edge;

    is_core = new bool[num_of_vertices];
    frontier = new bool[num_of_vertices];
    cluster_label = new int[num_of_vertices];
    memset(cluster_label, -1, sizeof(int) * num_of_vertices);
    done = new bool;
    
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
            std::cout << "BFS(" << i << ", " << cluster_id << ")" << std::endl;
            #endif
            BFS(i, cluster_id);
            cluster_id += 1;
        }
        cudaMemcpy(cluster_label, d_cluster_label, sizeof(int) * num_of_vertices, cudaMemcpyDeviceToHost);
    }
    set_cluster_color();
    return cluster_label;
}

bool is_close(int* a, int* b, int dimension, int eps) {
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

graph constuct_neighbor_pts(int num_of_vertices, int dimension, int** raw_vertices, int eps) {
    graph neighbor;
    neighbor.num_of_vertices = num_of_vertices;
    neighbor.edge_pos = new int[num_of_vertices];
    neighbor.degree = new int[num_of_vertices];
    memset(neighbor.degree, 0, sizeof(int) * num_of_vertices);
    std::vector<std::vector<int>> vertices(num_of_vertices);
    
    for (int i = 0; i < num_of_vertices; i++) {
        #ifdef DEBUG
        std::cout << "constuct_neighbor: " << i << " / " << num_of_vertices << std::endl;
        #endif
        for (int j = 0; j < num_of_vertices; j++) {
            if (i == j) continue;
            if (is_close(raw_vertices[i], raw_vertices[j], dimension, eps)) {
                vertices[i].push_back(j);
                neighbor.degree[i]++;
            }
        }
    }
        
    neighbor.num_of_edges = 0;
    for (int i = 0; i < num_of_vertices; i++) {
        neighbor.num_of_edges += neighbor.degree[i];
        if (i > 0) {
            neighbor.edge_pos[i] = neighbor.edge_pos[i-1] + neighbor.degree[i-1];
        } else {
            neighbor.edge_pos[i] = 0;
        }
    }

    neighbor.edge = new int[neighbor.num_of_edges];
    for (int i = 0; i < num_of_vertices; i++) {
        for (int j = 0; j < vertices[i].size(); j++) {
            neighbor.edge[neighbor.edge_pos[i] + j] = vertices[i][j];
        }
    }

    return neighbor;
}

graph constuct_neighbor_img(unsigned char* img, int channels, int width, int height, int eps) {
    int num_of_pixels = height * width;
    int dimension = 5;

    int** raw_vertices = new int *[num_of_pixels];
    graph neighbor;
    neighbor.num_of_vertices = num_of_pixels;
    neighbor.edge_pos = new int[num_of_pixels];
    neighbor.degree = new int[num_of_pixels];
    memset(neighbor.degree, 0, sizeof(int) * num_of_pixels);
    std::vector<std::vector<int>> vertices(num_of_pixels);

    for (int i = 0; i < num_of_pixels; i++) {
        raw_vertices[i] = new int[dimension];
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            raw_vertices[i * width + j][0] = img[channels * (width * i + j) + 0];
            raw_vertices[i * width + j][1] = img[channels * (width * i + j) + 1];
            raw_vertices[i * width + j][2] = img[channels * (width * i + j) + 2];
            raw_vertices[i * width + j][3] = i;
            raw_vertices[i * width + j][4] = j;
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int m = i - eps; m <= i + eps; m++) {
                for (int n = j - eps; n < j + eps; n++) {
                    if (m >= 0 && m < height && n >= 0 && n < width) {
                        if (is_close(raw_vertices[i * width + j], raw_vertices[m * width + n], dimension, eps)) {
                            vertices[i * width + j].push_back(j);
                            neighbor.degree[i * width + j]++;
                        }
                    }
                }
            }
        }
    }

    neighbor.num_of_edges = 0;
    for (int i = 0; i < num_of_pixels; i++) {
        neighbor.num_of_edges += neighbor.degree[i];
        if (i > 0) {
            neighbor.edge_pos[i] = neighbor.edge_pos[i-1] + neighbor.degree[i-1];
        } else {
            neighbor.edge_pos[i] = 0;
        }
    }

    neighbor.edge = new int[neighbor.num_of_edges];
    for (int i = 0; i < num_of_pixels; i++) {
        for (int j = 0; j < vertices[i].size(); j++) {
            neighbor.edge[neighbor.edge_pos[i] + j] = vertices[i][j];
        }
    }

    return neighbor;
}

