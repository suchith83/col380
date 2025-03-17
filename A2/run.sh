module load compiler/gcc/9.1.0
module load compiler/gcc/9.1/mpich/3.3.1

module load compiler/gcc/9.1/openmpi/4.1.2




mpic++ -o check check.cpp template.cpp
mpirun --hostfile my_hostfile -np 2 ./check TestCase1 first_out_2.txt 10
echo "first test case ran"
# mpirun --hostfile $PBS_NODEFILE -np 4 ./check TestCase1 first_out_4.txt 10
mpirun --hostfile my_hostfile -np 4 ./check TestCase1 first_out_4.txt 10

# mpirun --hostfile $PBS_NODEFILE --display-allocation --display-map -np 4 ./check TestCase1 first_out_4.txt 10

echo "2nd case ran"
