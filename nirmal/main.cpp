// this is main.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <mpi.h>
#include <omp.h>
#include "matrix.h"
#include "gpu_matrix.h"
#include <cmath> 

// Function to read matrix from file
SparseMatrix readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int height, width, numBlocks;
    file >> height >> width >> numBlocks;
    
    SparseMatrix matrix;
    matrix.height = height;
    matrix.width = width;
    matrix.blockSize = 0;  // Will be set after reading k
    
    std::vector<std::pair<int, int>> blockPositions(numBlocks);
    for (int i = 0; i < numBlocks; i++) {
        int row, col;
        file >> row >> col;
        blockPositions[i] = {row, col};
    }
    
    for (int i = 0; i < numBlocks; i++) {
        int blockSize;
        file >> blockSize;
        int k = static_cast<int>(std::sqrt(blockSize));
        if (matrix.blockSize == 0) matrix.blockSize = k;
        else if (matrix.blockSize != k) {
            std::cerr << "Inconsistent block size in " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        Block block(k, std::vector<long long>(k, 0));
        for (int j = 0; j < blockSize; j++) {
            long long value;
            file >> value;
            block[j / k][j % k] = value;
        }
        matrix.blocks[blockPositions[i]] = block;
    }
    
    file.close();
    return matrix;
}

// Function to write matrix to file
void writeMatrix(const SparseMatrix& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    file << matrix.height << " " << matrix.width << "\n";
    file << matrix.blocks.size() << "\n";
    
    for (const auto& entry : matrix.blocks) {
        file << entry.first.first << " " << entry.first.second << "\n";
    }
    
    for (const auto& entry : matrix.blocks) {
        const Block& block = entry.second;
        int k = matrix.blockSize;
        file << k * k << "\n";
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                file << block[i][j] << (j < k - 1 ? " " : "");
            }
            file << "\n";
        }
    }
    
    file.close();
}

// Serialize SparseMatrix to a vector for MPI communication
std::vector<long long> serialize(const SparseMatrix& matrix) {
    std::vector<long long> buffer;
    buffer.push_back(matrix.height);
    buffer.push_back(matrix.width);
    buffer.push_back(matrix.blockSize);
    buffer.push_back(matrix.blocks.size());
    
    for (const auto& entry : matrix.blocks) {
        buffer.push_back(entry.first.first);
        buffer.push_back(entry.first.second);
        for (int i = 0; i < matrix.blockSize; i++) {
            for (int j = 0; j < matrix.blockSize; j++) {
                buffer.push_back(entry.second[i][j]);
            }
        }
    }
    return buffer;
}

// Deserialize vector to SparseMatrix
SparseMatrix deserialize(const std::vector<long long>& buffer) {
    SparseMatrix matrix;
    size_t idx = 0;
    matrix.height = buffer[idx++];
    matrix.width = buffer[idx++];
    matrix.blockSize = buffer[idx++];
    int numBlocks = buffer[idx++];
    
    for (int i = 0; i < numBlocks; i++) {
        int row = buffer[idx++];
        int col = buffer[idx++];
        Block block(matrix.blockSize, std::vector<long long>(matrix.blockSize));
        for (int r = 0; r < matrix.blockSize; r++) {
            for (int c = 0; c < matrix.blockSize; c++) {
                block[r][c] = buffer[idx++];
            }
        }
        matrix.blocks[{row, col}] = block;
    }
    return matrix;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " <folder_path>\n";
        MPI_Finalize();
        return 1;
    }
    
    std::string folderPath = argv[1];
    int N, k;
    if (rank == 0) {
        std::ifstream sizeFile(folderPath + "/size");
        if (!sizeFile.is_open()) {
            std::cerr << "Failed to open size file\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        sizeFile >> N >> k;
        sizeFile.close();
    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate local range
    int localN = N / size + (rank < N % size ? 1 : 0);
    int start = rank * (N / size) + (rank < N % size ? rank : N % size);
    int end = start + localN;
    
    std::vector<SparseMatrix> localMatrices;
    for (int i = start; i < end; i++) {
        std::string path = folderPath + "/matrix" + std::to_string(i + 1);
        localMatrices.push_back(readMatrix(path));
    }
    
    SparseMatrix localResult = localMatrices.empty() ? SparseMatrix{0, 0, k, {}} : localMatrices[0];
    for (size_t i = 1; i < localMatrices.size(); i++) {
        if (shouldUseGPU(localResult, localMatrices[i])) {
            localResult = multiplyMatricesGPU(localResult, localMatrices[i]);
        } else {
            localResult = multiplyMatricesCPU(localResult, localMatrices[i]);
        }
    }
    
    // Tree reduction
    SparseMatrix globalResult = localResult;
    for (int step = 0; (1 << step) < size; step++) {
        int stride = 1 << step;
        if (rank % (2 * stride) == 0) {
            int partner = rank + stride;
            if (partner < size) {
                std::vector<long long> recvBuffer;
                MPI_Status status;
                int count;
                MPI_Probe(partner, 0, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, MPI_LONG_LONG, &count);
                recvBuffer.resize(count);
                MPI_Recv(recvBuffer.data(), count, MPI_LONG_LONG, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                SparseMatrix partnerResult = deserialize(recvBuffer);
                if (shouldUseGPU(globalResult, partnerResult)) {
                    globalResult = multiplyMatricesGPU(globalResult, partnerResult);
                } else {
                    globalResult = multiplyMatricesCPU(globalResult, partnerResult);
                }
            }
        } else if (rank % (2 * stride) == stride) {
            std::vector<long long> sendBuffer = serialize(globalResult);
            MPI_Send(sendBuffer.data(), sendBuffer.size(), MPI_LONG_LONG, rank - stride, 0, MPI_COMM_WORLD);
            break; // Exit after sending
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) writeMatrix(globalResult, "matrix");
    
    MPI_Finalize();
    return 0;
}