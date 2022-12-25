#include <vector>

enum Type { Undefined, Core, Noise, Border };

typedef struct vertex {
    Type type = Undefined;
    int cluster = -1;
    bool visited = false;
    std::vector<int> neighbors;
} vertex;

class DBSCAN{
    private:
        float eps;
        int minPts;
        int num_of_vertices;
        int dimension;
        vertex* vertices; // vertices representation
        std::vector<std::vector<int>> adjacency_lists;
        int* colors; // cluster id -> vertex id
        int num_of_cluster;
        int* cluster_label; // cluster id of each vertex
        
        bool is_neighbor(int, int);
        float cal_distance(int*, int*);
        void BFS(int id);
        void set_cluster_label();
        void set_cluster_color();

    public:
        DBSCAN(float e, int mp) { eps = e, minPts = mp; };
        ~DBSCAN() { }; 				
        int* cluster(int num_of_vertices, int dimension, int** raw_vertices);
        int* get_colors() { return colors; };
        void print_cluster(); // debug
        void print_adjacency_lists(); // debug
};