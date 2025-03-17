using namespace std;
#include <mpi.h>
#include <iostream>
int main (int nargs, char* args[])
{
int numprocs, my_rank, i;
MPI_Init (&nargs, &args);
MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
for (i = 0; i < numprocs; i++) {
MPI_Barrier (MPI_COMM_WORLD);
if (i == my_rank) {
cout << "Hello world, I have rank " << my_rank <<
" out of " << numprocs << endl;}
}
MPI_Finalize ();
}