// this is matrix.cpp
#include "matrix.h"
#include <omp.h>
#include <iostream>

bool shouldUseGPU(const SparseMatrix& A, const SparseMatrix& B) {
    size_t totalBlocks = A.blocks.size() + B.blocks.size();
    return totalBlocks > 1000; // Threshold can be tuned
}

Block multiplyBlocks(const Block& A, const Block& B, int k) {
    Block result(k, std::vector<long long>(k, 0));
    const long long MOD = 9223372036854775807LL; // LLONG_MAX
    
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            long long sum = 0;
            for (int p = 0; p < k; p++) {
                sum = (sum + (A[i][p] * B[p][j]) % MOD) % MOD;
            }
            result[i][j] = sum;
        }
    }
    return result;
}

SparseMatrix multiplyMatricesCPU(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.width != B.height) {
        std::cerr << "Matrix dimensions mismatch\n";
        exit(1);
    }
    
    SparseMatrix C;
    C.height = A.height;
    C.width = B.width;
    C.blockSize = A.blockSize;
    
    int heightBlocks = (A.height + A.blockSize - 1) / A.blockSize;
    int widthBlocks = (B.width + B.blockSize - 1) / B.blockSize;
    int midBlocks = (A.width + A.blockSize - 1) / A.blockSize;
    
    std::vector<std::map<std::pair<int, int>, Block>> threadResults(omp_get_max_threads());
    
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for (int i = 0; i < heightBlocks; i++) {
            for (int j = 0; j < widthBlocks; j++) {
                for (int p = 0; p < midBlocks; p++) {
                    auto itA = A.blocks.find({i, p});
                    auto itB = B.blocks.find({p, j});
                    if (itA != A.blocks.end() && itB != B.blocks.end()) {
                        Block temp = multiplyBlocks(itA->second, itB->second, A.blockSize);
                        auto& Cij = threadResults[tid][{i, j}];
                        if (Cij.empty()) {
                            Cij.resize(A.blockSize, std::vector<long long>(A.blockSize, 0));
                        }
                        for (int r = 0; r < A.blockSize; r++) {
                            for (int c = 0; c < A.blockSize; c++) {
                                Cij[r][c] = (Cij[r][c] + temp[r][c]) % 9223372036854775807LL;
                            }
                        }
                    }
                }
            }
        }
    }
    
    for (const auto& tr : threadResults) {
        for (const auto& entry : tr) {
            auto& Cij = C.blocks[entry.first];
            if (Cij.empty()) {
                Cij = entry.second;
            } else {
                for (int r = 0; r < A.blockSize; r++) {
                    for (int c = 0; c < A.blockSize; c++) {
                        Cij[r][c] = (Cij[r][c] + entry.second[r][c]) % 9223372036854775807LL;
                    }
                }
            }
        }
    }
    
    // Remove zero blocks
    for (auto it = C.blocks.begin(); it != C.blocks.end();) {
        bool allZero = true;
        for (const auto& row : it->second) {
            for (long long val : row) {
                if (val != 0) {
                    allZero = false;
                    break;
                }
            }
            if (!allZero) break;
        }
        if (allZero) it = C.blocks.erase(it);
        else ++it;
    }
    
    return C;
}