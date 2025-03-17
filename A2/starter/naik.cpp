#include "template.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <mpi.h>

using namespace std;

void init_mpi(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
}

void end_mpi() {
    MPI_Finalize();
}

vector<vector<int>> degree_cen(vector<pair<int,int>>& partial_edge_list, 
    map<int,int>& partial_vector_color, int k) {
        
        int rank,size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (rank!=0){
            int edge_list_size = partial_edge_list.size();
            MPI_Send(&edge_list_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            for (auto edge : partial_edge_list) {
                int edge_data[2] = {edge.first, edge.second};
                MPI_Send(edge_data, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }

            int vector_color_size = partial_vector_color.size();
            MPI_Send(&vector_color_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            for (auto vertex_color : partial_vector_color) {
                int color_data[2] = {vertex_color.first, vertex_color.second};
                MPI_Send(color_data, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
            return {};
        }
        else{
            vector<pair<int, int>> global_edge_list = partial_edge_list;
            map<int, int> global_vector_color = partial_vector_color;

            for (int i = 1; i < size; i++) {
                int total_edges;
                MPI_Recv(&total_edges, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < total_edges; j++) {
                    int edge_data[2];
                    MPI_Recv(edge_data, 2, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    global_edge_list.emplace_back(edge_data[0], edge_data[1]);
                }

                int total_colors;
                MPI_Recv(&total_colors, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < total_colors; j++) {
                    int color_data[2];
                    MPI_Recv(color_data, 2, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    global_vector_color[color_data[0]] = color_data[1];
                }
            }
        int n = global_vector_color.size();
        set<int> colors;
        for (auto it = global_vector_color.begin(); it != global_vector_color.end(); it++) {
            colors.insert(it->second);
        }
        int vertex_count = global_vector_color.size();
        vector<vector<int>> result;
        map<pair<int,int>,int> vcolor_count;
        for (const auto &vertex_color : global_vector_color) {
            int vertex = vertex_color.first;
            for (int c : colors) {
                vcolor_count[{vertex, c}] = 0;
            }
        }
        for (auto edge : global_edge_list) {
            int u = edge.first;
            int v = edge.second;
            int color_u = global_vector_color[u];
            int color_v = global_vector_color[v];
            vcolor_count[{u,color_v}]++;
            vcolor_count[{v,color_u}]++;
        }

        for (auto &c : colors){
            vector<pair<int,int>> color_vertices;
            for (auto kp: vcolor_count){
                if (kp.first.second == c){
                    color_vertices.push_back({kp.first.first,kp.second});}
            }
            sort(color_vertices.begin(), color_vertices.end(), [](pair<int,int> a, pair<int,int> b){
                return (a.second > b.second)||(a.second == b.second && a.first < b.first);
            });
            vector<int> topk;
            for (int i = 0; i < k; i++){
                topk.push_back(color_vertices[i].first);
            }
            result.push_back(topk);
            }
        
        return result;
    }
}