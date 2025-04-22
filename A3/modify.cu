#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "modify.cuh"

using namespace std;


__global__ void compute_frequency(int *matrix, int *freq, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        atomicAdd(&freq[matrix[idx]], 1);
    }
}


__global__ void prefix_sum(int *freq, int range) {
    extern __shared__ int temp[]; // Dynamically allocated shared memory

    int tid = threadIdx.x;

    if (tid <= range) temp[tid] = freq[tid];
    __syncthreads();

    for (int stride = 1; stride <= range; stride *= 2) {
        int val = 0;
        if (tid >= stride) val = temp[tid - stride];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (tid <= range) freq[tid] = temp[tid];
}


__global__ void distribute_elements(int *matrix, int *prefix_sum, int *output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int val = matrix[idx];
        int new_pos = atomicSub(&prefix_sum[val], 1) - 1;
        output[new_pos] = val;
    }
}


void modify_matrix(vector<vector<int>> &matrix, int range, cudaStream_t stream) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int total_size = rows * cols;

    
    vector<int> flat_matrix;
    for (auto &row : matrix) {
        flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }

    
    int *my_mat, *my_mat_freq, *d_prefix_sum, *d_output;
    cudaMalloc(&my_mat, total_size * sizeof(int));
    cudaMalloc(&my_mat_freq, (range + 1) * sizeof(int));
    cudaMalloc(&d_prefix_sum, (range + 1) * sizeof(int));
    cudaMalloc(&d_output, total_size * sizeof(int));

    
    cudaMemcpyAsync(my_mat, flat_matrix.data(), total_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(my_mat_freq, 0, (range + 1) * sizeof(int), stream);


    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;
    compute_frequency<<<grid_size, block_size, 0, stream>>>(my_mat, my_mat_freq, total_size);


    prefix_sum<<<1, range + 1, (range + 1) * sizeof(int), stream>>>(my_mat_freq, range);


    cudaMemcpyAsync(d_prefix_sum, my_mat_freq, (range + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream);


    distribute_elements<<<grid_size, block_size, 0, stream>>>(my_mat, d_prefix_sum, d_output, total_size);


    cudaMemcpyAsync(flat_matrix.data(), d_output, total_size * sizeof(int), cudaMemcpyDeviceToHost, stream);


    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = flat_matrix[i * cols + j];
        }
    }

    cudaFree(my_mat);
    cudaFree(my_mat_freq);
    cudaFree(d_prefix_sum);
    cudaFree(d_output);
}


vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
    int num_matrices = matrices.size();

    vector<vector<vector<int>>> modified_matrices = matrices;


    vector<cudaStream_t> streams(num_matrices);

    for (int i = 0; i < num_matrices; ++i) {
        cudaStreamCreate(&streams[i]);
        

        modify_matrix(modified_matrices[i], ranges[i], streams[i]);

        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return modified_matrices;
}
