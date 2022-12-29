#include <vector>

typedef struct vertex {
    std::vector<int> neighbors;
} vertex;
class DBSCAN{
    private:
        float eps;
        int minPts;

        int num_of_vertices;
        int num_of_edges;
        int dimension;
        vertex* vertices; // vertices representation
        int* colors; // cluster id -> vertex id
        int num_of_cluster;

        int *edge, *edge_pos, *degree, *cluster_label; // cluster id of each vertex
        bool *is_core, *frontier, *done;

        int *d_edge, *d_edge_pos, *d_degree, *d_cluster_label;
        bool *d_is_core, *d_frontier, *d_done;
        
        bool is_neighbor(int, int);
        bool is_close(int*, int*);
        void constuct_neighbor(int**);
        void BFS(int, int);
        void set_cluster_color();

    public:
        DBSCAN(float e, int mp) { eps = e, minPts = mp; };
        ~DBSCAN() { }; 				
        int* cluster(int num_of_vertices, int dimension, int** raw_vertices);
        int* get_colors() { return colors; };
        void print_cluster(); // debug
        void print_adjacency_lists(); // debug
        void print_degree();
        void print_edge_pos();
        void print_edge();
        void print_type();
};