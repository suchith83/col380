// this is matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <map>
#include <utility>

using Block = std::vector<std::vector<long long>>;

struct SparseMatrix {
    int height;
    int width;
    int blockSize;
    std::map<std::pair<int, int>, Block> blocks;
};

bool shouldUseGPU(const SparseMatrix& A, const SparseMatrix& B);
SparseMatrix multiplyMatricesCPU(const SparseMatrix& A, const SparseMatrix& B);

#endif // MATRIX_H