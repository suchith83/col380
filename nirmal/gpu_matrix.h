// this is gpu_matrix.h
#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

#include "matrix.h"

SparseMatrix multiplyMatricesGPU(const SparseMatrix& A, const SparseMatrix& B);

#endif // GPU_MATRIX_H