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

// __global__ void blelloch_scan_block(int *d_array, int *d_block_sums, int n) {
//     __shared__ int temp[2 * BLOCK_SIZE];
//     int tid = threadIdx.x;
//     int offset = 1;
//     int index = 2 * blockIdx.x * BLOCK_SIZE + 2 * tid;
//     temp[2 * tid] = (index < n) ? d_array[index] : 0;
//     temp[2 * tid + 1] = (index + 1 < n) ? d_array[index + 1] : 0;
//     for (int d = 1; d < 2 * BLOCK_SIZE; d *= 2) {
//         __syncthreads();
//         int ai = offset * (2 * tid + 1) - 1;
//         int bi = offset * (2 * tid + 2) - 1;
//         if (bi < 2 * BLOCK_SIZE) temp[bi] += temp[ai];
//         offset *= 2;
//     }
//     if (tid == 0) d_block_sums[blockIdx.x] = temp[2 * BLOCK_SIZE - 1];
//     if (tid == 0) temp[2 * BLOCK_SIZE - 1] = 0;
//     for (int d = 2 * BLOCK_SIZE / 2; d > 0; d /= 2) {
//         offset /= 2;
//         __syncthreads();
//         int ai = offset * (2 * tid + 1) - 1;
//         int bi = offset * (2 * tid + 2) - 1;
//         if (bi < 2 * BLOCK_SIZE) {
//             int t = temp[ai];
//             temp[ai] = temp[bi];
//             temp[bi] += t;
//         }
//     }
//     __syncthreads();
//     if (index < n) d_array[index] = temp[2 * tid];
//     if (index + 1 < n) d_array[index + 1] = temp[2 * tid + 1];
// }

// __global__ void add_block_sums(int *d_array, int *d_block_sums, int n) {
//     int index = 2 * blockIdx.x * BLOCK_SIZE + threadIdx.x;
//     if (blockIdx.x > 0 && index < n) {
//         d_array[index] += d_block_sums[blockIdx.x - 1];
//     }
// }


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
    cudaDeviceSynchronize();

    int *h_block_sums = new int[num_blocks];
    cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < num_blocks; i++) {
        h_block_sums[i] += h_block_sums[i-1];
    }

    cudaMemcpy(d_block_sums, h_block_sums, num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    add_block_sums<<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_pref_output, d_block_sums, n);
    cudaDeviceSynchronize();

    cudaFree(d_block_sums);
    delete[] h_block_sums;
    
}

// __global__ void modify_matrix(int *matrix, int *prefix_sum, int rows, int cols) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < rows * cols) {
//         int value = matrix[idx];
//         matrix[idx] = atomicAdd(&prefix_sum[value], 1);
//     }
// }

// __global__ void modify_matrix(int *matrix, int *prefix_sum, int rows, int cols, int *prefix_copy) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < rows * cols) {
//         int value = matrix[idx];
//         matrix[idx] = prefix_copy[value];  // Use precomputed prefix sum
//     }
// }

__global__ void modify_matrix(int *matrix, int *prefix_sum, int *output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx == 0) cout << "Hello from GPU" << endl;
    // if (idx == 0) printf("Hi from gpu!\n");
    if (idx < total_size) {
        int val = matrix[idx];
        // if (val > 100000000) printf("GPU thread no: %d \n", idx);
        int new_pos = atomicSub(&prefix_sum[val+1], 1) - 1;
        // if (new_pos < 20 || new_pos == 87) printf("gpu thread no: %d, %d, %d\n", idx, val, new_pos);
        // cout << "val: at : " << idx << " " << val << endl;
        // printf("prefix sum of val %d: %d \n",val, prefix_sum[val]);
        output[new_pos] = val;
    }
}




// vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
//     int N = matrices.size();
//     vector<int*> d_matrices(N), d_freqs(N);
//     int *h_freqs[N];
//     cudaStream_t streams[N];
//     for (int i = 0; i < N; i++) {
//         int rows = matrices[i].size();
//         int cols = matrices[i][0].size();
//         int size = rows * cols * sizeof(int);
//         vector<int> flat_matrix(rows * cols);
//         for (int r = 0; r < rows; r++){
//             for (int c = 0; c < cols; c++) {
//                 flat_matrix[r * cols + c] = matrices[i][r][c];
//             }
//         }
//         cudaMalloc(&d_matrices[i], size);
//         cudaMalloc(&d_freqs[i], (ranges[i] + 2) * sizeof(int)); // Allocate extra space
//         cudaMemcpy(d_matrices[i], flat_matrix.data(), size, cudaMemcpyHostToDevice);
//         cudaMemset(d_freqs[i], 0, (ranges[i] + 2) * sizeof(int));
//         h_freqs[i] = new int[ranges[i] + 2](); // Allocate extra space
//         cudaStreamCreate(&streams[i]);
//         int tot_threads = rows * cols;
//         int block_size = 256;
//         int noBlocks = (tot_threads + block_size - 1) / block_size;
//         compute_frequency<<<noBlocks, block_size, 0, streams[i]>>>(d_matrices[i], d_freqs[i], rows * cols, ranges[i]);
//         cudaMemcpy(h_freqs[i], d_freqs[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//         cout << "Frequency Array for matrix " << i << ": ";
//         for (int j = 0; j <= ranges[i] + 1; j++) cout << h_freqs[i][j] << " ";
//         cout << endl;
//         parallel_prefix_sum(d_freqs[i], ranges[i] + 2);
//         cudaMemcpy(h_freqs[i], d_freqs[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//         cout << "Prefix Sum Array for matrix " << i << ": ";
//         for (int j = 0; j <= ranges[i] + 1; j++) cout << h_freqs[i][j] << " ";
//         cout << endl;
//         modify_matrix<<<(ranges[i] + 255) / 256, 256>>>(matrices[i], d_freqs[i], d_matrices[i], rows * cols);
//     }
//     return d_matrices;
// }


// vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
//     int N = matrices.size();
//     vector<int*> d_matrices(N), d_freqs(N);
//     int *h_freqs[N];
//     cudaStream_t streams[N];

//     for (int i = 0; i < N; i++) {
//         int rows = matrices[i].size();
//         int cols = matrices[i][0].size();
//         int size = rows * cols * sizeof(int);

//         // Flatten matrix
//         vector<int> flat_matrix(rows * cols);
//         for (int r = 0; r < rows; r++){
//             for (int c = 0; c < cols; c++) {
//                 flat_matrix[r * cols + c] = matrices[i][r][c];
//             }
//         }

//         // Allocate device memory
//         cudaMalloc(&d_matrices[i], size);
//         cudaMalloc(&d_freqs[i], (ranges[i] + 2) * sizeof(int));

//         // Copy input data to device
//         cudaMemcpy(d_matrices[i], flat_matrix.data(), size, cudaMemcpyHostToDevice);
//         cudaMemset(d_freqs[i], 0, (ranges[i] + 2) * sizeof(int));

//         // Host-side frequency array
//         h_freqs[i] = new int[ranges[i] + 2]();

//         // Create CUDA streams for concurrent execution
//         cudaStreamCreate(&streams[i]);

//         // Launch frequency computation
//         int tot_threads = rows * cols;
//         int block_size = 256;
//         int noBlocks = (tot_threads + block_size - 1) / block_size;
//         compute_frequency<<<noBlocks, block_size, 0, streams[i]>>>(d_matrices[i], d_freqs[i], rows * cols, ranges[i]);

//         // Copy frequency array back to host (for debugging)
//         cudaMemcpy(h_freqs[i], d_freqs[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//         cout << "Frequency Array for matrix " << i << ": ";
//         for (int j = 0; j <= ranges[i] + 1; j++) cout << h_freqs[i][j] << " ";
//         cout << endl;

//         // Perform prefix sum
//         parallel_prefix_sum(d_freqs[i], ranges[i] + 2);

//         // Copy prefix sum array back to host (for debugging)
//         cudaMemcpy(h_freqs[i], d_freqs[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//         cout << "Prefix Sum Array for matrix " << i << ": ";
//         for (int j = 0; j <= ranges[i] + 1; j++) cout << h_freqs[i][j] << " ";
//         cout << endl;

//         // Modify matrix using prefix sum
//         modify_matrix<<<(tot_threads + block_size - 1) / block_size, block_size>>>(d_matrices[i], d_freqs[i], rows, cols);
//     }

//     // Prepare output matrices
//     vector<vector<vector<int>>> modified_matrices(N);
//     for (int i = 0; i < N; i++) {
//         int rows = matrices[i].size();
//         int cols = matrices[i][0].size();
//         vector<int> modified_flat(rows * cols);

//         // Copy modified data from device to host
//         cudaMemcpy(modified_flat.data(), d_matrices[i], rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

//         // Convert flat array back to 2D
//         modified_matrices[i].resize(rows, vector<int>(cols));
//         for (int r = 0; r < rows; r++) {
//             for (int c = 0; c < cols; c++) {
//                 modified_matrices[i][r][c] = modified_flat[r * cols + c];
//             }
//         }

//         // Free device memory
//         cudaFree(d_matrices[i]);
//         cudaFree(d_freqs[i]);
//         delete[] h_freqs[i];
//         cudaStreamDestroy(streams[i]);
//     }

//     return modified_matrices;
// }


// without print

// vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
//     int N = matrices.size();
//     vector<int*> d_matrices(N), d_freqs(N), d_prefix_copy(N), d_output(N);
//     int *h_prefix[N];
//     cudaStream_t streams[N];

//     for (int i = 0; i < N; i++) {
//         int rows = matrices[i].size();
//         int cols = matrices[i][0].size();
//         int size = rows * cols * sizeof(int);

//         vector<int> flat_matrix(rows * cols);
//         for (int r = 0; r < rows; r++)
//             for (int c = 0; c < cols; c++)
//                 flat_matrix[r * cols + c] = matrices[i][r][c];

//         cudaMalloc(&d_matrices[i], size);
//         cudaMalloc(&d_freqs[i], (ranges[i] + 2) * sizeof(int));
//         cudaMalloc(&d_prefix_copy[i], (ranges[i] + 2) * sizeof(int));
//         cudaMalloc(&d_output[i], size);

//         cudaMemcpy(d_matrices[i], flat_matrix.data(), size, cudaMemcpyHostToDevice);
//         cudaMemset(d_freqs[i], 0, (ranges[i] + 2) * sizeof(int));

//         h_prefix[i] = new int[ranges[i] + 2]();
//         cudaStreamCreate(&streams[i]);

//         int tot_threads = rows * cols;
//         int block_size = 256;
//         int noBlocks = (tot_threads + block_size - 1) / block_size;

//         compute_frequency<<<noBlocks, block_size, 0, streams[i]>>>(d_matrices[i], d_freqs[i], rows * cols, ranges[i]);
//         parallel_prefix_sum(d_freqs[i], ranges[i] + 2);

//         cudaMemcpy(h_prefix[i], d_freqs[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//         cudaMemcpy(d_prefix_copy[i], h_prefix[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyHostToDevice);

//         modify_matrix<<<(rows * cols + 255) / 256, 256>>>(d_matrices[i], d_prefix_copy[i], d_output[i], rows * cols);

//         delete[] h_prefix[i];
//     }

//     vector<vector<vector<int>>> modified_matrices(N);
//     for (int i = 0; i < N; i++) {
//         int rows = matrices[i].size();
//         int cols = matrices[i][0].size();
//         vector<int> flat_matrix(rows * cols);

//         cudaMemcpy(flat_matrix.data(), d_output[i], rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

//         modified_matrices[i].resize(rows, vector<int>(cols));
//         for (int r = 0; r < rows; r++)
//             for (int c = 0; c < cols; c++)
//                 modified_matrices[i][r][c] = flat_matrix[r * cols + c];

//         cudaFree(d_matrices[i]);
//         cudaFree(d_freqs[i]);
//         cudaFree(d_prefix_copy[i]);
//         cudaFree(d_output[i]);
//         cudaStreamDestroy(streams[i]);
//     }

//     return modified_matrices;
// }

// with print
vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &ranges) {
    int N = matrices.size();
    vector<int*> d_matrices(N), d_freqs(N), d_prefix_copy(N), d_output(N);
    // int *h_prefix[N], *h_freq[N];
    cudaStream_t streams[N];

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

        cudaMemcpy(d_matrices[i], flat_matrix.data(), size, cudaMemcpyHostToDevice);
        cudaMemset(d_freqs[i], 0, (ranges[i] + 2) * sizeof(int));

        // h_prefix[i] = new int[ranges[i] + 2]();
        // h_freq[i] = new int[ranges[i] + 2]();
        cudaStreamCreate(&streams[i]);

        int tot_threads = rows * cols;
        int block_size = 256;
        int noBlocks = (tot_threads + block_size - 1) / block_size;

        compute_frequency<<<noBlocks, block_size, 0, streams[i]>>>(d_matrices[i], d_freqs[i], rows * cols, ranges[i]);
        cudaDeviceSynchronize();
        // cudaMemcpy(h_freq[i], d_freqs[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyDeviceToHost);

        // Print a small range of frequency values
        // printf("Frequency array (first 10 values) for matrix %d:\n", i);
        // for (int j = 54989849 - 5; j < 54989849 + 5; j++) {
        //     printf("%d ", h_freq[i][j]);
        // }
        // printf("\n");
        // cout << ranges[i] << endl;
        // for (int j = 155051280 - 5; j < 155051280 + 5; j++) {
        //     printf("%d ", h_freq[i][j]);
        // }
        // printf("\n");

        parallel_prefix_sum(d_freqs[i], d_prefix_copy[i], ranges[i] + 2, streams[i]);
        // cudaMemcpy(h_prefix[i], d_prefix_copy[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(d_prefix_copy[i], h_prefix[i], (ranges[i] + 2) * sizeof(int), cudaMemcpyHostToDevice);

        // Print a small range of prefix sum values
        // printf("Prefix sum array (first 10 values) for matrix %d:\n", i);
        // for (int j = 54989849 - 5; j < 54989849 + 5; j++) {
        //     printf("%d ", h_prefix[i][j]);
        // }
        // printf("\n");
        // for (int j = 160666904 - 5; j < 160666904 + 5; j++) {
        //     printf("%d ", h_prefix[i][j]);
        // }
        // printf("\n");

        // check if prefix sum array is correct for frequency array
        // bool flag = true;
        // for (int j = 0; j < ranges[i] + 2; j++) {
        //     if (h_prefix[i][j+1] - h_prefix[i][j] != h_freq[i][j]) {
        //         cout << "went wront at : " << j <<  " " << h_freq[i][j] << " " << h_prefix[i][j+1] << " " << h_prefix[i][j] << endl;
        //         flag = false;
        //         break;
        //     }
        // }
        // cout << "flag: " << flag << endl;
        // int cnt = 0;
        // for (int j = 0; j < ranges[i] + 1; j++) {
        //     if (h_prefix[i][j] > h_prefix[i][j+1]) {
        //         cnt++;
        //         cout << j << ", " << h_prefix[i][j] << ", " << h_prefix[i][j+1] << endl;
        //         if (cnt == 50) break;
        //     }
        // }

        modify_matrix<<<(rows * cols + 255) / 256, 256, 0, streams[i]>>>(d_matrices[i], d_prefix_copy[i], d_output[i], rows * cols);
        cudaDeviceSynchronize();
        // delete[] h_prefix[i];
        // delete[] h_freq[i];
    }

    vector<vector<vector<int>>> modified_matrices(N);
    for (int i = 0; i < N; i++) {
        int rows = matrices[i].size();
        int cols = matrices[i][0].size();
        vector<int> flat_matrix(rows * cols);

        cudaMemcpy(flat_matrix.data(), d_output[i], rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

        modified_matrices[i].resize(rows, vector<int>(cols));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                modified_matrices[i][r][c] = flat_matrix[r * cols + c];

        cudaFree(d_matrices[i]);
        cudaFree(d_freqs[i]);
        cudaFree(d_prefix_copy[i]);
        cudaFree(d_output[i]);
        cudaStreamDestroy(streams[i]);
    }

    return modified_matrices;
}