module load compiler/gcc/9.1.0
module load compiler/gcc/9.1/mpich/3.3.1



# mpic++ -o check check.cpp template.cpp
# mpirun -np 2 ./check TestCase1 outputs/bigOut_2_10.txt 10
# echo "big test case ran for k = 10"
# mpirun -np 2 ./check TestCase1 outputs/bigOut_2_50.txt 50
# echo "big test case ran for k = 50"
# mpirun -np 2 ./check SmallTestCase outputs/smallOut_2_10.txt 10
# echo "small test case ran for k = 10"
# mpirun -np 2 ./check SmallTestCase outputs/smallOut_2_20.txt 20
# echo "small test case ran for k = 20"



# mpic++ -o check check.cpp template.cpp
# mpirun -np 4 ./check TestCase1 outputs/bigOut_4_10.txt 10
# echo "big test case ran for k = 10"
# mpirun -np 4 ./check TestCase1 outputs/bigOut_4_50.txt 50
# echo "big test case ran for k = 50"
# mpirun -np 4 ./check SmallTestCase outputs/smallOut_4_10.txt 10
# echo "small test case ran for k = 10"
# mpirun -np 4 ./check SmallTestCase outputs/smallOut_4_20.txt 20
# echo "small test case ran for k = 20"


# mpic++ -o check check.cpp template.cpp
# mpirun -np 6 ./check TestCase1 outputs/bigOut_6_10.txt 10
# echo "big test case ran for k = 10"
# mpirun -np 6 ./check TestCase1 outputs/bigOut_6_50.txt 50
# echo "big test case ran for k = 50"
# mpirun -np 6 ./check SmallTestCase outputs/smallOut_6_10.txt 10
# echo "small test case ran for k = 10"
# mpirun -np 6 ./check SmallTestCase outputs/smallOut_6_20.txt 20
# echo "small test case ran for k = 20"


mpic++ -o check check.cpp template.cpp
mpirun -np 8 ./check TestCase1 outputs/bigOut_8_10.txt 10
echo "big test case ran for k = 10"
mpirun -np 8 ./check TestCase1 outputs/bigOut_8_50.txt 50
echo "big test case ran for k = 50"
mpirun -np 8 ./check SmallTestCase outputs/smallOut_8_10.txt 10
echo "small test case ran for k = 10"
mpirun -np 8 ./check SmallTestCase outputs/smallOut_8_20.txt 20
echo "small test case ran for k = 20"



# mpirun -np 5 ./check TestCase1 first_out_5.txt 10
