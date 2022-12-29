#include <vector>
#include <string.h>

typedef struct graph {
    int num_of_vertices = 0;
    int num_of_edges = 0;
    int *edge, *edge_pos, *degree;
} graph;

class DBSCAN{
    private:
        float eps;
        int minPts;

        int num_of_vertices;
        int num_of_edges;
        int dimension;
        int* colors; // cluster id -> vertex id
        int num_of_cluster;

        int *edge, *edge_pos, *degree, *cluster_label; // cluster id of each vertex
        bool *is_core, *frontier, *done;

        int *d_edge, *d_edge_pos, *d_degree, *d_cluster_label;
        bool *d_is_core, *d_frontier, *d_done;
        
        bool is_neighbor(int, int);
        // bool is_close(int*, int*);
        void BFS(int, int);
        void set_cluster_color();

    public:
        DBSCAN(float e, int mp) { eps = e, minPts = mp; };
        ~DBSCAN() { }; 				
        int* cluster(graph);
        int* get_colors() { return colors; };
        // debug
        void print_cluster(); 
        void print_degree();
        void print_edge_pos();
        void print_edge();
        void print_type();
};

bool is_close(int* a, int* b, int dimension, int eps);
graph constuct_neighbor_pts(int num_of_vertices, int dimension, int** raw_vertices, int eps);
graph constuct_neighbor_img(unsigned char* img, int channels, int width, int height, int eps);