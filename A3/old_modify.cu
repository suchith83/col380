#include <cuda_runtime.h>
#include <vector>
#include <iostream>

using namespace std;

// CUDA Kernel to compute the frequency array
__global__ void compute_frequency(int *matrix, int *freq, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        atomicAdd(&freq[matrix[idx]], 1);
    }
}

// Inclusive scan (prefix sum) kernel
__global__ void prefix_sum(int *freq, int range) {
    __shared__ int temp[1024]; // Adjust based on the max range
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

// CUDA Kernel to distribute elements into the modified matrix
__global__ void distribute_elements(int *matrix, int *prefix_sum, int *output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int val = matrix[idx];
        int new_pos = atomicSub(&prefix_sum[val], 1) - 1;
        output[new_pos] = val;
    }
}

// Host function to modify a matrix
void modify_matrix(vector<vector<int>> &matrix, int range) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int total_size = rows * cols;

    // Flatten the matrix
    vector<int> flat_matrix;
    for (auto &row : matrix) {
        flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }

    int *d_matrix, *d_freq, *d_prefix_sum, *d_output;
    cudaMalloc(&d_matrix, total_size * sizeof(int));
    cudaMalloc(&d_freq, (range + 1) * sizeof(int));
    cudaMalloc(&d_prefix_sum, (range + 1) * sizeof(int));
    cudaMalloc(&d_output, total_size * sizeof(int));

    cudaMemcpy(d_matrix, flat_matrix.data(), total_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_freq, 0, (range + 1) * sizeof(int));

    // Step 1: Compute frequency
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;
    compute_frequency<<<grid_size, block_size>>>(d_matrix, d_freq, total_size);
    cudaDeviceSynchronize();

    // Step 2: Compute prefix sum in parallel
    prefix_sum<<<1, 1024>>>(d_freq, range);
    cudaDeviceSynchronize();

    // Copy prefix sum back to device memory for atomic updates
    cudaMemcpy(d_prefix_sum, d_freq, (range + 1) * sizeof(int), cudaMemcpyDeviceToDevice);



    // Step 3: Distribute elements
    distribute_elements<<<grid_size, block_size>>>(d_matrix, d_prefix_sum, d_output, total_size);
    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(flat_matrix.data(), d_output, total_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Reconstruct matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = flat_matrix[i * cols + j];
        }
    }

    cudaFree(d_matrix);
    cudaFree(d_freq);
    cudaFree(d_prefix_sum);
    cudaFree(d_output);
}

// Main function to modify multiple matrices
vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
    int num_matrices = matrices.size();
    vector<vector<vector<int>>> modified_matrices = matrices;

    for (int i = 0; i < num_matrices; ++i) {
        modify_matrix(modified_matrices[i], ranges[i]);
    }

    return modified_matrices;
}


// #include <cuda_runtime.h>
// #include <vector>
// #include <iostream>

// using namespace std;

// // CUDA Kernel to compute the frequency array
// __global__ void compute_frequency(int *matrix, int *freq, int total_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < total_size) {
//         atomicAdd(&freq[matrix[idx]], 1);
//     }
// }

// // Inclusive scan (prefix sum) kernel
// __global__ void prefix_sum(int *freq, int range) {
//     __shared__ int temp[1024]; // Adjust based on the max range
//     int tid = threadIdx.x;
    
//     if (tid <= range) temp[tid] = freq[tid];
//     __syncthreads();

//     for (int stride = 1; stride <= range; stride *= 2) {
//         int val = 0;
//         if (tid >= stride) val = temp[tid - stride];
//         __syncthreads();
//         temp[tid] += val;
//         __syncthreads();
//     }

//     if (tid <= range) freq[tid] = temp[tid];
// }

// // CUDA Kernel to distribute elements into the modified matrix
// __global__ void distribute_elements(int *matrix, int *prefix_sum, int *output, int total_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < total_size) {
//         int val = matrix[idx];
//         int new_pos = atomicSub(&prefix_sum[val], 1) - 1;
//         output[new_pos] = val;
//     }
// }

// // Function to print arrays for debugging
// void print_array(const vector<int> &arr, const string &label) {
//     cout << label << ": ";
//     for (size_t i = 0; i < arr.size(); ++i) {
//         cout << arr[i] << " ";
//     }
//     cout << endl;
// }

// // Host function to modify a matrix
// void modify_matrix(vector<vector<int>> &matrix, int range) {
//     int rows = matrix.size();
//     int cols = matrix[0].size();
//     int total_size = rows * cols;

//     // Flatten the matrix
//     vector<int> flat_matrix;
//     for (auto &row : matrix) {
//         flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
//     }

//     int *d_matrix, *d_freq, *d_prefix_sum, *d_output;
//     cudaMalloc(&d_matrix, total_size * sizeof(int));
//     cudaMalloc(&d_freq, (range + 1) * sizeof(int));
//     cudaMalloc(&d_prefix_sum, (range + 1) * sizeof(int));
//     cudaMalloc(&d_output, total_size * sizeof(int));

//     cudaMemcpy(d_matrix, flat_matrix.data(), total_size * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemset(d_freq, 0, (range + 1) * sizeof(int));

//     // Step 1: Compute frequency
//     int block_size = 256;
//     int grid_size = (total_size + block_size - 1) / block_size;
//     compute_frequency<<<grid_size, block_size>>>(d_matrix, d_freq, total_size);
//     cudaDeviceSynchronize();

//     // Copy frequency array back to host and print
//     vector<int> h_freq(range + 1);
//     cudaMemcpy(h_freq.data(), d_freq, (range + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//     print_array(h_freq, "Frequency Array");

//     // Step 2: Compute prefix sum in parallel
//     prefix_sum<<<1, 1024>>>(d_freq, range);
//     cudaDeviceSynchronize();

//     // Copy prefix sum back to host and print
//     vector<int> h_prefix_sum(range + 1);
//     cudaMemcpy(h_prefix_sum.data(), d_freq, (range + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//     print_array(h_prefix_sum, "Prefix Sum Array");

//     // Copy prefix sum back to device memory for atomic updates
//     cudaMemcpy(d_prefix_sum, h_freq.data(), (range + 1) * sizeof(int), cudaMemcpyHostToDevice);

//     // Step 3: Distribute elements
//     distribute_elements<<<grid_size, block_size>>>(d_matrix, d_prefix_sum, d_output, total_size);
//     cudaDeviceSynchronize();

//     // Copy back results
//     cudaMemcpy(flat_matrix.data(), d_output, total_size * sizeof(int), cudaMemcpyDeviceToHost);

//     // Reconstruct matrix
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             matrix[i][j] = flat_matrix[i * cols + j];
//         }
//     }

//     cudaFree(d_matrix);
//     cudaFree(d_freq);
//     cudaFree(d_prefix_sum);
//     cudaFree(d_output);
// }

// // Main function to modify multiple matrices
// vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
//     int num_matrices = matrices.size();
//     vector<vector<vector<int>>> modified_matrices = matrices;

//     for (int i = 0; i < num_matrices; ++i) {
//         modify_matrix(modified_matrices[i], ranges[i]);
//     }

//     return modified_matrices;
// }



// #include <cuda_runtime.h>
// #include <vector>
// #include <iostream>

// using namespace std;

// // CUDA Kernel to compute the frequency array
// __global__ void compute_frequency(int *matrix, int *freq, int total_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < total_size) {
//         atomicAdd(&freq[matrix[idx]], 1);
//     }
// }

// // Inclusive scan (prefix sum) kernel
// __global__ void prefix_sum(int *freq, int range) {
//     __shared__ int temp[1024]; // Adjust based on the max range
//     int tid = threadIdx.x;

//     if (tid <= range) temp[tid] = freq[tid];
//     __syncthreads();

//     for (int stride = 1; stride <= range; stride *= 2) {
//         int val = 0;
//         if (tid >= stride) val = temp[tid - stride];
//         __syncthreads();
//         temp[tid] += val;
//         __syncthreads();
//     }

//     if (tid <= range) freq[tid] = temp[tid];
// }

// // CUDA Kernel to distribute elements into the modified matrix
// __global__ void distribute_elements(int *matrix, int *prefix_sum, int *output, int total_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < total_size) {
//         int val = matrix[idx];
//         int new_pos = atomicSub(&prefix_sum[val], 1) - 1;
//         output[new_pos] = val;
//     }
// }

// // Host function to modify a single matrix
// void modify_matrix(vector<vector<int>> &matrix, int range, cudaStream_t stream) {
//     int rows = matrix.size();
//     int cols = matrix[0].size();
//     int total_size = rows * cols;

//     // Flatten the matrix
//     vector<int> flat_matrix;
//     for (auto &row : matrix) {
//         flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
//     }

//     // Allocate device memory
//     int *d_matrix, *d_freq, *d_prefix_sum, *d_output;
//     cudaMalloc(&d_matrix, total_size * sizeof(int));
//     cudaMalloc(&d_freq, (range + 1) * sizeof(int));
//     cudaMalloc(&d_prefix_sum, (range + 1) * sizeof(int));
//     cudaMalloc(&d_output, total_size * sizeof(int));

//     // Copy data to device
//     cudaMemcpyAsync(d_matrix, flat_matrix.data(), total_size * sizeof(int), cudaMemcpyHostToDevice, stream);
//     cudaMemsetAsync(d_freq, 0, (range + 1) * sizeof(int), stream);

//     // Step 1: Compute frequency
//     int block_size = 256;
//     int grid_size = (total_size + block_size - 1) / block_size;
//     compute_frequency<<<grid_size, block_size, 0, stream>>>(d_matrix, d_freq, total_size);

//     // Step 2: Compute prefix sum in parallel
//     prefix_sum<<<1, 1024, 0, stream>>>(d_freq, range);

//     // Copy prefix sum back to device memory for atomic updates
//     cudaMemcpyAsync(d_prefix_sum, d_freq, (range + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream);

//     // Step 3: Distribute elements
//     distribute_elements<<<grid_size, block_size, 0, stream>>>(d_matrix, d_prefix_sum, d_output, total_size);

//     // Copy back results
//     cudaMemcpyAsync(flat_matrix.data(), d_output, total_size * sizeof(int), cudaMemcpyDeviceToHost, stream);

//     // Reconstruct matrix
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             matrix[i][j] = flat_matrix[i * cols + j];
//         }
//     }

//     // Free device memory
//     cudaFree(d_matrix);
//     cudaFree(d_freq);
//     cudaFree(d_prefix_sum);
//     cudaFree(d_output);
// }

// // Main function to modify multiple matrices in parallel using CUDA streams
// vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
//     int num_matrices = matrices.size();
    
//     vector<vector<vector<int>>> modified_matrices = matrices;

//     // Create CUDA streams for concurrent execution
//     vector<cudaStream_t> streams(num_matrices);
    
//     for (int i = 0; i < num_matrices; ++i) {
//         cudaStreamCreate(&streams[i]);
//         modify_matrix(modified_matrices[i], ranges[i], streams[i]);
//         cudaStreamSynchronize(streams[i]);
//         cudaStreamDestroy(streams[i]);
//     }

//     return modified_matrices;
// }
