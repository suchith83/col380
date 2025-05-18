#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "modify.cuh"

using namespace std;

#define BLOCK_SIZE 1024

__global__ void compute_frequency(int *matrix, int *freq, int total_size, int max_range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int value = matrix[idx];
        if (value >= 0 && value <= max_range) {
            atomicAdd(&freq[value], 1);
        }
    }
}

// scan part of code 


#define BLOCK_SIZE 1024  // Maximum threads per block

__global__ void block_prefix_sum(int *d_input, int *d_output, int *d_block_sums, int n) {
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (index < n) {
        temp[tid] = d_input[index];
    } else {
        temp[tid] = 0;  // Handle out-of-bounds case
    }
    __syncthreads();

    // **Work-efficient Parallel Scan (Blelloch's Algorithm)**
    // Up-Sweep (Reduction)
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < BLOCK_SIZE) {
            temp[idx] += temp[idx - stride];
        }
        __syncthreads();
    }

    // Store block sum before zeroing the last element
    if (tid == BLOCK_SIZE - 1) {
        d_block_sums[blockIdx.x] = temp[tid];
        temp[tid] = 0;
    }
    __syncthreads();

    // Down-Sweep (Exclusive Scan)
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < BLOCK_SIZE) {
            int tempVal = temp[idx - stride];
            temp[idx - stride] = temp[idx];
            temp[idx] += tempVal;
        }
        __syncthreads();
    }

    // Write results to global memory
    if (index < n) {
        d_output[index] = temp[tid];
    }
}

__global__ void add_block_sums(int *d_output, int *d_block_sums, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && index < n) {
        d_output[index] += d_block_sums[blockIdx.x - 1];
    }
}

void parallel_prefix_sum(int *d_input, int *d_pref_output, int n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / (BLOCK_SIZE);
    int *d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));

    block_prefix_sum<<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_input, d_pref_output, d_block_sums, n);
    cudaStreamSynchronize(stream);

    int *h_block_sums = new int[num_blocks];
    cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < num_blocks; i++) {
        h_block_sums[i] += h_block_sums[i-1];
    }

    cudaMemcpy(d_block_sums, h_block_sums, num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    add_block_sums<<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_pref_output, d_block_sums, n);
    cudaStreamSynchronize(stream);

    cudaFree(d_block_sums);
    delete[] h_block_sums;
    
}


__global__ void modify_matrix(int *matrix, int *prefix_sum, int *output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int val = matrix[idx];
        int new_pos = atomicSub(&prefix_sum[val+1], 1) - 1;
        output[new_pos] = val;
    }
}


// Updated `modify` function with CPU-GPU comparison
vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
    int N = matrices.size();
    vector<int*> d_matrices(N), d_freqs(N), d_prefix_copy(N), d_output(N);
    cudaStream_t streams[N];
    vector<vector<vector<int>>> modified_matrices(N);

    for (int i = 0; i < N; i++) {
        int rows = matrices[i].size();
        int cols = matrices[i][0].size();
        int size = rows * cols * sizeof(int);

        vector<int> flat_matrix(rows * cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                flat_matrix[r * cols + c] = matrices[i][r][c];

        cudaMalloc(&d_matrices[i], size);
        cudaMalloc(&d_freqs[i], (ranges[i] + 2) * sizeof(int));
        cudaMalloc(&d_prefix_copy[i], (ranges[i] + 2) * sizeof(int));
        cudaMalloc(&d_output[i], size);

        cudaStreamCreate(&streams[i]);

        cudaMemcpy(d_matrices[i], flat_matrix.data(), size, cudaMemcpyHostToDevice);
        cudaMemset(d_freqs[i], 0, (ranges[i] + 2) * sizeof(int));

        int tot_threads = rows * cols;
        int block_size = 256;
        int noBlocks = (tot_threads + block_size - 1) / block_size;

        compute_frequency<<<noBlocks, block_size, 0, streams[i]>>>(d_matrices[i], d_freqs[i], rows * cols, ranges[i]);
        cudaStreamSynchronize(streams[i]);


        
        parallel_prefix_sum(d_freqs[i], d_prefix_copy[i], ranges[i] + 2, streams[i]);

        cudaFree(d_freqs[i]);
        
        cout << "modifying matrix: " << endl;
        modify_matrix<<<(rows * cols + 255) / 256, 256, 0, streams[i]>>>(d_matrices[i], d_prefix_copy[i], d_output[i], rows * cols);
        cudaStreamSynchronize(streams[i]);

        cudaFree(d_prefix_copy[i]);
        cudaFree(d_matrices[i]);
        
        vector<int> flat_matrix2(rows * cols);

        cudaMemcpy(flat_matrix2.data(), d_output[i], rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        modified_matrices[i].resize(rows, vector<int>(cols));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                modified_matrices[i][r][c] = flat_matrix2[r * cols + c];

                
        cudaFree(d_output[i]);
        cudaStreamDestroy(streams[i]);
    }

    return modified_matrices;
}


