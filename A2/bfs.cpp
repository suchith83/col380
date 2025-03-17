#include "mpi.h"
#include <iostream>
#include<stdio.h>
#include<stdlib.h>

using namespace std;

#define MAX_QUEUE_SIZE 5

int areAllVisited(int visited[], int n) {
    for(int i = 0; i < n; i++) {
        if(visited[i] == 0) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int numprocs, my_rank;
    int adjacency_matrix[100];
	int adjacency_queue[MAX_QUEUE_SIZE];
	int source_vertex;
	int no_of_vertices;
	int adjacency_row[10];
	int bfs_traversal[100];
	int visited[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        cout<<"Enter the number of vertices\n";
		cin>>no_of_vertices;

		//Entering the Adjacency Matrix
		cout<<"Enter the Adjacency Matrix\n";
		for(int i = 0; i < no_of_vertices * no_of_vertices; i++)
		{
			cin>>adjacency_matrix[i];
		}
		cout<<endl;

		//Entering the Source Vertex
		cout<<"Enter the Source Vertex\n";
		cin>>source_vertex;
		cout<<endl;
    }

    MPI_Bcast(&no_of_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source_vertex, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(adjacency_matrix, no_of_vertices, MPI_INT, adjacency_row, no_of_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < MAX_QUEUE_SIZE; i++) {
        adjacency_queue[i] = -1;
    }

    int index = 0;
    if(my_rank >= source_vertex) {
        for(int i = 0; i < no_of_vertices; i++) {
            if(adjacency_row[i] == 1) {
                adjacency_queue[index] = i;
                index++;
            }
        }
    }

    cout << "Process " << my_rank << ": ";
    for(int i = 0; i < index; i++) {
        cout << adjacency_queue[i] << " ";
    }
    cout << endl;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(adjacency_queue, MAX_QUEUE_SIZE, MPI_INT, bfs_traversal, MAX_QUEUE_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < no_of_vertices; i++) {
        visited[i] = 0;
    }

    if (my_rank == 0) {
        cout << "\nBFS Traversal: " << endl;
        cout << source_vertex << " ";
        for (int i = 0; i < MAX_QUEUE_SIZE; i++) {
            if (areAllVisited(visited, no_of_vertices)) {
                break;
            }
            if (bfs_traversal[i] != -1) {
                if (visited[bfs_traversal[i]] == 0) {
                    cout << " -> " << bfs_traversal[i];
                    visited[bfs_traversal[i]] = 1;
                }
            }
            else {
                continue;
            }
        }
    }

    MPI_Finalize();
    return 0;   
}