#include <assert.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <thrust/scan.h>
#include <string.h>
#include "dbscan.h"
#define BS 64
#define abs(a, b) ((a > b) ? a - b : b - a)
// #define DEBUG

__global__ void bfs(int *edge, int *edge_pos, int *degree, bool *is_core, bool *frontier, int *cluster_label, int cluster_id, bool *done, int num_of_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    *done = true;
    if (tid >= num_of_vertices) return;
    if (frontier[tid] && cluster_label[tid] == -1 && is_core[tid]) {
        frontier[tid] = false;
        cluster_label[tid] = cluster_id;
        __syncthreads();
        for (int i = edge_pos[tid]; i < edge_pos[tid] + degree[tid]; i++) {
            if (cluster_label[edge[i]] == -1) {
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

int*
DBSCAN::cluster(graph neighbor) {
    num_of_vertices = neighbor.num_of_vertices;
    num_of_edges = neighbor.num_of_edges;
    edge_pos = neighbor.edge_pos;
    degree = neighbor.degree;
    edge = neighbor.edge;

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
    if (!neighbor.on_device) {
        cudaMalloc((void **)&d_edge, sizeof(int) * num_of_edges);
        cudaMalloc((void **)&d_edge_pos, sizeof(int) * num_of_vertices);
        cudaMalloc((void **)&d_degree, sizeof(int) * num_of_vertices);

        cudaMemcpy(d_edge, edge, sizeof(int) * num_of_edges, cudaMemcpyHostToDevice); 
        cudaMemcpy(d_edge_pos, edge_pos, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice); 
        cudaMemcpy(d_degree, degree, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice); 
    } else {
        d_edge = neighbor.d_edge;
        d_edge_pos = neighbor.d_edge_pos;
        d_degree = neighbor.d_degree;
    }
    cudaMalloc((void **)&d_is_core, sizeof(bool) * num_of_vertices);
    cudaMalloc((void **)&d_frontier, sizeof(bool) * num_of_vertices);
    cudaMalloc((void **)&d_cluster_label, sizeof(int) * num_of_vertices);
    cudaMalloc((void **)&d_done, sizeof(bool));

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
        if (i + 1 < num_of_vertices) {
            cudaMemcpy(cluster_label+i+1, d_cluster_label+i+1, sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    set_cluster_color();
    cudaFree(d_edge); 
    cudaFree(d_edge_pos); 
    cudaFree(d_degree); 
    cudaFree(d_is_core); 
    cudaFree(d_cluster_label); 
    return cluster_label;
}

bool is_close(int* a, int* b, int dimension, int eps) {
    float sum = 0;
    int eps_square = eps * eps;
    for (int i = 0; i < dimension; i++) {
        int diff = abs(a[i], b[i]);
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

// for dbscan_demo
graph construct_neighbor_pts(int num_of_vertices, int dimension, int** raw_vertices, int eps) {
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

inline __device__ bool d_is_close(int a[], int b[], int dimension, int eps) {
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

inline __device__ void translate(int* shm_r, int* shm_c, int r, int c, int blockidx_x, int blockidx_y, int eps) {
    *shm_r = r - blockidx_y + eps;
    *shm_c = c - blockIdx.x * BS + eps;
}

extern __shared__ unsigned char shared_img[];

__global__ void get_degree(int* degree, unsigned char* img, int channels, int height, int width, int eps, int dimension) {    
    int r = blockIdx.y;
    int c = blockIdx.x * BS + threadIdx.x;
    
    for (int i = r-eps; i <= r + eps; i++) {
        if (i >= 0 && i < height) {
            for (int j = c - eps; j <= int((blockIdx.x + 1) * BS + eps - 1); j += BS) {
                if (j >= 0 && j < width) {
                    int shm_r, shm_c;
                    translate(&shm_r, &shm_c, i, j, blockIdx.x, blockIdx.y, eps);
                    shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 2] = img[channels * (width * i + j) + 2];
                    shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 1] = img[channels * (width * i + j) + 1];
                    shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 0] = img[channels * (width * i + j) + 0];
                }
            }
        }
    }
    __syncthreads();

    if (c >= width) return;
    for (int m = r - eps; m <= r + eps; m++) {
        for (int n = c - eps; n <= c + eps; n++) {
            if (m >= 0 && m < height && n >= 0 && n < width && m != r && n != c) {
                int src[5], dst[5];
                int shm_r, shm_c;
                translate(&shm_r, &shm_c, r, c, blockIdx.x, blockIdx.y, eps);
                src[0] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 0];
                src[1] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 1];
                src[2] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 2];
                src[3] = r;
                src[4] = c;
                
                translate(&shm_r, &shm_c, m, n, blockIdx.x, blockIdx.y, eps);
                dst[0] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 0];
                dst[1] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 1];
                dst[2] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 2];
                dst[3] = m;
                dst[4] = n;
                if (d_is_close(src, dst, dimension, eps)) {
                    degree[r * width + c]++;
                }
            }
        }
    }
}

__global__ void get_neighbor(int* edge, int* edge_pos, unsigned char* img, int channels, int height, int width, int eps, int dimension) {
    int r = blockIdx.y;
    int c = blockIdx.x * BS + threadIdx.x;

    for (int i = r-eps; i <= r + eps; i++) {
        if (i >= 0 && i < height) {
            for (int j = c - eps; j <= int((blockIdx.x + 1) * BS + eps - 1); j += BS) {
                if (j >= 0 && j < width) {
                    int shm_r, shm_c;
                    translate(&shm_r, &shm_c, i, j, blockIdx.x, blockIdx.y, eps);
                    shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 2] = img[channels * (width * i + j) + 2];
                    shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 1] = img[channels * (width * i + j) + 1];
                    shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 0] = img[channels * (width * i + j) + 0];
                }
            }
        }
    }
    __syncthreads();

    if (c >= width) return;
    int pos = edge_pos[r * width + c];
    for (int m = r - eps; m <= r + eps; m++) {
        for (int n = c - eps; n <= c + eps; n++) {
            if (m >= 0 && m < height && n >= 0 && n < width && m != r && n != c) {
                int src[5], dst[5];
                int shm_r, shm_c;
                translate(&shm_r, &shm_c, r, c, blockIdx.x, blockIdx.y, eps);
                src[0] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 0];
                src[1] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 1];
                src[2] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 2];
                src[3] = r;
                src[4] = c;

                translate(&shm_r, &shm_c, m, n, blockIdx.x, blockIdx.y, eps);
                dst[0] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 0];
                dst[1] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 1];
                dst[2] = shared_img[channels * ((BS + 2 * eps) * shm_r + shm_c) + 2];
                dst[3] = m;
                dst[4] = n;
                if (d_is_close(src, dst, dimension, eps)) {
                    edge[pos++] = width * m + n;
                }
            }
        }
    }
}

// for image_seg
graph construct_neighbor_img(unsigned char* img, int channels, int width, int height, int eps) {
    int num_of_vertices = height * width;
    graph neighbor;
    neighbor.num_of_vertices = num_of_vertices;
    neighbor.edge_pos = new int[num_of_vertices];
    neighbor.degree = new int[num_of_vertices];
    memset(neighbor.degree, 0, sizeof(int) * num_of_vertices);

    dim3 num_threads(BS, 1);
    dim3 num_blocks(width / BS + 1, height);

    neighbor.on_device = true;
    unsigned char *d_img;
    cudaMalloc((void **)&neighbor.d_degree, sizeof(int) * num_of_vertices);
    cudaMalloc((void **)&d_img, sizeof(int) * height * width * channels);
    cudaMemcpy(neighbor.d_degree, neighbor.degree, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_img, img, sizeof(unsigned char) * height * width * channels, cudaMemcpyHostToDevice); 
    get_degree<<<num_blocks, num_threads, sizeof(unsigned char)*(2*eps+1)*(BS+2*eps)*channels>>>(neighbor.d_degree, d_img, channels, height, width, eps, 5);
    cudaMemcpy(neighbor.degree, neighbor.d_degree, sizeof(int) * num_of_vertices, cudaMemcpyDeviceToHost);

    thrust::exclusive_scan(neighbor.degree, neighbor.degree + num_of_vertices, neighbor.edge_pos);

    int num_of_edges = neighbor.edge_pos[num_of_vertices-1] + neighbor.degree[num_of_vertices-1];
    neighbor.num_of_edges = num_of_edges;
    neighbor.edge = new int[neighbor.num_of_edges];
    cudaMalloc((void **)&neighbor.d_edge, sizeof(int) * num_of_edges);
    cudaMalloc((void **)&neighbor.d_edge_pos, sizeof(int) * num_of_vertices);
    cudaMemcpy(neighbor.d_edge, neighbor.edge, sizeof(int) * num_of_edges, cudaMemcpyHostToDevice); 
    cudaMemcpy(neighbor.d_edge_pos, neighbor.edge_pos, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice); 
    get_neighbor<<<num_blocks, num_threads, sizeof(unsigned char)*(2*eps+1)*(BS+2*eps)*channels>>>(neighbor.d_edge, neighbor.d_edge_pos, d_img, channels, height, width, eps, 5);
    cudaMemcpy(neighbor.edge, neighbor.d_edge, sizeof(int) * num_of_edges, cudaMemcpyDeviceToHost);
    return neighbor;
}

