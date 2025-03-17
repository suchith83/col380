#!/bin/bash
#PBS -P col380.cs1210572.course
#PBS -q standard
#PBS -lselect=4:ncpus=40:ngpus=0:centos=skylake
#PBS -lwalltime=00:05:00
#PBS -N run_tests

cd $PBS_O_WORKDIR

module purge
module load compiler/gcc/9.1.0
module load compiler/gcc/9.1/mpich/3.3.1

module load compiler/gcc/9.1/openmpi/4.1.2

CSV_FILE="data.csv"
echo "Optimizations,V,E,C,k,1 node (2 procs),2 nodes (4 procs),3 nodes (6 procs),4 nodes (8 procs)" > "$CSV_FILE"

TEST_CASE="TestCase1"
OUTPUT_FILE="output_new.txt"
V=1000
E=20000
C=5
K_VALUES=(10 50)
NODE_COUNTS=(1 2 3 4)

echo "Compiling O1 ..."
mpic++ -o check_O1 check.cpp template.cpp

# echo "Compiling O2 (optimized -O3 -march=native)..."
# mpic++ -march=native -o check_O2 check.cpp template.cpp

if [ ! -x "./check_O1" ]; then
    echo "Error: Executable ./check_O1 not found. Exiting."
    exit 1
fi
# if [ ! -x "./check_O2" ]; then
#     echo "Error: Executable ./check_O2 not found. Exiting."
#     exit 1
# fi

OPT_LABELS=("O1")

for opt in "${OPT_LABELS[@]}"; do
    if [ "$opt" = "O1" ]; then
        EXEC="./check_O1"
    fi

    for k in "${K_VALUES[@]}"; do
        runtimes=()
        for nodes in "${NODE_COUNTS[@]}"; do
            np=$(( nodes * 2 ))
            echo "Running $opt with k=$k on $nodes node(s) ($np processes)..."
            start=$(date +%s.%N)
            # Using -hostfile ensures processes spread across allocated nodes properly
            mpirun -hostfile $PBS_NODEFILE -np "$np" "$EXEC" "$TEST_CASE" "$OUTPUT_FILE" "$k" > /dev/null
            end=$(date +%s.%N)
            runtime=$(echo "scale=2; ($end - $start)*1000" | bc -l)
            runtimes+=("$runtime")
        done
        echo "$opt,$V,$E,$C,$k,${runtimes[0]},${runtimes[1]},${runtimes[2]},${runtimes[3]}" >> "$CSV_FILE"
        echo "Row for $opt with k=$k appended to $CSV_FILE."
    done
done

echo "Testing complete. Results saved to $CSV_FILE."
ls -l "$CSV_FILE"