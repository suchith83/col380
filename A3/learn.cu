#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cassert>

using namespace std;

// Computes the sum of two arrays
__global__ void vectorAdd(int *a, int *b, int *c, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        c[tid] = a[tid] + b[tid];
}

void init_array(int *a, int N){
    for(int i = 0; i < N; i++){
        a[i] = rand() % 100;
    }
}

// Verify the vector addition computation on the CPU
void verify_result(int *a, int *b, int *c, int N){
    for(int i = 0; i < N; i++){
        assert(c[i] == a[i] + b[i]);
    }
}

int main(){
    int N = 1<<20; // 1M elements
    size_t bytes = N * sizeof(int); // FIXED SIZE CALCULATION

    // Allocate memory
    int *a, *b, *c;
    if (cudaMallocManaged(&a, bytes) != cudaSuccess ||
        cudaMallocManaged(&b, bytes) != cudaSuccess ||
        cudaMallocManaged(&c, bytes) != cudaSuccess) {
        cerr << "CUDA malloc failed!" << endl;
        return -1;
    }

    // Initialize arrays
    init_array(a, N);
    init_array(b, N);

    int THREADS = 256;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Launch the kernel
    vectorAdd<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    // Verify the result
    verify_result(a, b, c, N);

    cout << "PROGRAM COMPLETED CORRECTLY" << endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}

// https://www.youtube.com/watch?v=uUEHuF5i_qI&list=PLxNPSjHT5qvvwoy6KXzUbLaF5A8NdJvuo