// this is gpu_matrix.cu
#include "gpu_matrix.h"
#include <cuda_runtime.h>
#include <iostream>

struct GPUSparseMatrix {
    int height, width, blockSize, numBlocks;
    int *rowIndices, *colIndices;
    long long *values;
};

__global__ void sparseMatrixMultiplyKernel(
    int* A_rows, int* A_cols, long long* A_vals, int A_nBlocks,
    int* B_rows, int* B_cols, long long* B_vals, int B_nBlocks,
    int* C_rows, int* C_cols, long long* C_vals, int* C_count,
    int k, long long MOD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= A_nBlocks * B_nBlocks) return;
    
    int a = idx / B_nBlocks;
    int b = idx % B_nBlocks;
    
    if (A_cols[a] != B_rows[b]) return;
    
    int c_idx = atomicAdd(C_count, 1);
    C_rows[c_idx] = A_rows[a];
    C_cols[c_idx] = B_cols[b];
    
    long long* C_block = C_vals + c_idx * k * k;
    long long* A_block = A_vals + a * k * k;
    long long* B_block = B_vals + b * k * k;
    
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            long long sum = 0;
            for (int p = 0; p < k; p++) {
                sum = (sum + (A_block[i * k + p] * B_block[p * k + j]) % MOD) % MOD;
            }
            C_block[i * k + j] = sum;
        }
    }
}

GPUSparseMatrix toGPU(const SparseMatrix& m) {
    GPUSparseMatrix gm;
    gm.height = m.height;
    gm.width = m.width;
    gm.blockSize = m.blockSize;
    gm.numBlocks = m.blocks.size();
    
    std::vector<int> rows(gm.numBlocks), cols(gm.numBlocks);
    std::vector<long long> vals(gm.numBlocks * gm.blockSize * gm.blockSize);
    int idx = 0;
    for (const auto& entry : m.blocks) {
        rows[idx] = entry.first.first;
        cols[idx] = entry.first.second;
        for (int i = 0; i < gm.blockSize; i++) {
            for (int j = 0; j < gm.blockSize; j++) {
                vals[idx * gm.blockSize * gm.blockSize + i * gm.blockSize + j] = entry.second[i][j];
            }
        }
        idx++;
    }
    
    cudaMalloc(&gm.rowIndices, gm.numBlocks * sizeof(int));
    cudaMalloc(&gm.colIndices, gm.numBlocks * sizeof(int));
    cudaMalloc(&gm.values, gm.numBlocks * gm.blockSize * gm.blockSize * sizeof(long long));
    
    cudaMemcpy(gm.rowIndices, rows.data(), gm.numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gm.colIndices, cols.data(), gm.numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gm.values, vals.data(), gm.numBlocks * gm.blockSize * gm.blockSize * sizeof(long long), cudaMemcpyHostToDevice);
    
    return gm;
}

SparseMatrix fromGPU(const GPUSparseMatrix& gm, int actualBlocks) {
    SparseMatrix m;
    m.height = gm.height;
    m.width = gm.width;
    m.blockSize = gm.blockSize;
    
    std::vector<int> rows(actualBlocks), cols(actualBlocks);
    std::vector<long long> vals(actualBlocks * gm.blockSize * gm.blockSize);
    
    cudaMemcpy(rows.data(), gm.rowIndices, actualBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cols.data(), gm.colIndices, actualBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vals.data(), gm.values, actualBlocks * gm.blockSize * gm.blockSize * sizeof(long long), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < actualBlocks; i++) {
        Block block(gm.blockSize, std::vector<long long>(gm.blockSize));
        bool nonZero = false;
        for (int r = 0; r < gm.blockSize; r++) {
            for (int c = 0; c < gm.blockSize; c++) {
                long long val = vals[i * gm.blockSize * gm.blockSize + r * gm.blockSize + c];
                block[r][c] = val;
                if (val != 0) nonZero = true;
            }
        }
        if (nonZero) m.blocks[{rows[i], cols[i]}] = block;
    }
    
    return m;
}

void freeGPU(GPUSparseMatrix& gm) {
    cudaFree(gm.rowIndices);
    cudaFree(gm.colIndices);
    cudaFree(gm.values);
}

SparseMatrix multiplyMatricesGPU(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.width != B.height) {
        std::cerr << "Matrix dimensions mismatch\n";
        exit(1);
    }
    
    GPUSparseMatrix d_A = toGPU(A);
    GPUSparseMatrix d_B = toGPU(B);
    
    int maxBlocks = d_A.numBlocks * d_B.numBlocks;
    int* d_C_rows, *d_C_cols, *d_C_count;
    long long* d_C_vals;
    
    cudaMalloc(&d_C_rows, maxBlocks * sizeof(int));
    cudaMalloc(&d_C_cols, maxBlocks * sizeof(int));
    cudaMalloc(&d_C_vals, maxBlocks * A.blockSize * A.blockSize * sizeof(long long));
    cudaMalloc(&d_C_count, sizeof(int));
    cudaMemset(d_C_count, 0, sizeof(int));
    
    int threadsPerBlock = 256;
    int blocks = (d_A.numBlocks * d_B.numBlocks + threadsPerBlock - 1) / threadsPerBlock;
    
    sparseMatrixMultiplyKernel<<<blocks, threadsPerBlock>>>(
        d_A.rowIndices, d_A.colIndices, d_A.values, d_A.numBlocks,
        d_B.rowIndices, d_B.colIndices, d_B.values, d_B.numBlocks,
        d_C_rows, d_C_cols, d_C_vals, d_C_count,
        A.blockSize, 9223372036854775807LL
    );
    
    int h_C_count;
    cudaMemcpy(&h_C_count, d_C_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    GPUSparseMatrix d_C = {A.height, B.width, A.blockSize, h_C_count, d_C_rows, d_C_cols, d_C_vals};
    SparseMatrix C = fromGPU(d_C, h_C_count);
    
    cudaFree(d_C_rows);
    cudaFree(d_C_cols);
    cudaFree(d_C_vals);
    cudaFree(d_C_count);
    freeGPU(d_A);
    freeGPU(d_B);
    
    return C;
}