module load compiler/gcc/9.1/openmpi/4.1.2 
module load compiler/gcc/7.1.0/compilervars 
module load compiler/cuda/10.0/compilervars

make clean

make

# mpirun -n 4 ./a4 small_test


mpirun -n 4 ./a4 medium_test

# mpirun -n 4 ./a4 large_test


# mpirun -n 4 ./a4 new_test

# ./a4 small_test

