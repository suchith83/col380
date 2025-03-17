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
    //Code Here
    MPI_Init(&argc, &argv);
}

void end_mpi() {
    //Code Here
    MPI_Finalize();
}

vector<vector<int>> degree_cen(vector<pair<int, int>>& partial_edge_list, map<int, int>& partial_vertex_color, int k) {
    //Code Here
    int numprocs, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // broadcast colors in current process to all other processes and receive colors from all other processes
    vector<int> local_colors_serialized;
    for (auto &entry : partial_vertex_color) {
        local_colors_serialized.push_back(entry.first);
        local_colors_serialized.push_back(entry.second);
    }

    int local_colors_size = local_colors_serialized.size();
    vector<int> recv_clr_sizes(numprocs);
    MPI_Allgather(&local_colors_size, 1, MPI_INT, &recv_clr_sizes[0], 1, MPI_INT, MPI_COMM_WORLD);

    // compute displacements for receiving data
    vector<int> displs_recv(numprocs);
    int total_recv_size = 0;
    for (int i = 0; i < numprocs; i++) {
        displs_recv[i] = total_recv_size;
        total_recv_size += recv_clr_sizes[i];
    }

    vector<int> global_colors_serialized(total_recv_size);
    MPI_Allgatherv(local_colors_serialized.data(), local_colors_size, MPI_INT, global_colors_serialized.data(),
                   recv_clr_sizes.data(), displs_recv.data(), MPI_INT, MPI_COMM_WORLD);

    map<int, int> global_vertex_color;
    for (int i = 0; i < total_recv_size; i += 2) {
        global_vertex_color[global_colors_serialized[i]] = global_colors_serialized[i + 1];
    }

    map<int, map<int, int>> local_centrality;
    for (auto edge : partial_edge_list) {
        int u = edge.first;
        int v = edge.second;
        if (global_vertex_color.count(v)) {
            int color_v = global_vertex_color[v];
            local_centrality[u][color_v]++;
        }
        if (global_vertex_color.count(u)) {
            int color_u = global_vertex_color[u];
            local_centrality[v][color_u]++;
        }
    }

    vector<int> send_data;
    for (auto &node_entry : local_centrality) {
        int node = node_entry.first;
        for (auto &color_entry : node_entry.second) {
            send_data.push_back(node);
            send_data.push_back(color_entry.first);
            send_data.push_back(color_entry.second);
        }
    }

    int local_size = send_data.size();
    vector<int> recv_sizes(numprocs);
    MPI_Gather(&local_size, 1, MPI_INT, &recv_sizes[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> recv_data;
    vector<int> displs(numprocs);
    if (my_rank == 0) {
        int total_size = 0;
        for (int i = 0; i < numprocs; i++) {
            displs[i] = total_size;
            total_size += recv_sizes[i];
        }
        recv_data.resize(total_size);
    }

    // MPI_Gatherv(&send_data[0], local_size, MPI_INT, &recv_data[0], &recv_sizes[0], &displs[0], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(send_data.data(),local_size,MPI_INT,
                recv_data.data(),recv_sizes.data(),displs.data(),MPI_INT,
                0,MPI_COMM_WORLD);

    vector<vector<int>> result;

    if (my_rank == 0) {
        map<int, map<int, int>> global_centrality;
        set<int> colors_present;

        for (int i = 0; i < recv_data.size(); i += 3) {
            int node = recv_data[i];
            int color = recv_data[i + 1];
            int count = recv_data[i + 2];
            global_centrality[node][color] += count;
            colors_present.insert(color);
        }

        vector<int> sorted_colors(colors_present.begin(), colors_present.end());
        sort(sorted_colors.begin(), sorted_colors.end());

        for (auto color : sorted_colors) {
            vector<pair<int, int>> nodes_scores;
            for (auto &node_entry : global_centrality) {
                int node = node_entry.first;
                int cen = node_entry.second[color];
                nodes_scores.push_back({cen, node});
            }
            // sort(nodes_scores.begin(), nodes_scores.end());
            sort(nodes_scores.begin(),nodes_scores.end(),[](auto &a,auto &b){
                return a.first!=b.first?a.first>b.first:a.second<b.second;});

            // print node_scores
            // cout << "\nnode_scores\n";
            // cout << "color: " << color << endl;
            // for (auto &node_score : nodes_scores) {
            //     cout << node_score.first << " " << node_score.second << endl;
            // }

            vector<int> top_k_nodes;
            for (int i = 0; i < min(k, (int)nodes_scores.size()); i++) {
                top_k_nodes.push_back(nodes_scores[i].second);
            }
            result.push_back(top_k_nodes);
        }
    }
    return result;
}
