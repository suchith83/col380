#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <climits>
#include <time.h>

using namespace std;

typedef vector<vector<uint64_t>> Block;

struct SparseMatrix {
    int height;
    int width;
    int block_size;
    map<pair<int, int>, Block> blocks;
};

SparseMatrix readMatrix(const string& path, int k) {
    ifstream inFile(path);
    if (!inFile) {
        throw runtime_error("Error: cannot open file: " + path);
    }

    int h, w, blocks;
    if (! (inFile >> h >> w >> blocks)) {
        throw runtime_error("Error: Invalid header in file" + path);
    }

    SparseMatrix mat;
    mat.height = h;
    mat.width = w;
    mat.block_size = k;
    for (int b = 0; b < blocks; b++) {
        int r, c;
        inFile >> r >> c;

        Block blk(k, vector<uint64_t>(k));
        for(int i = 0; i < k; i++)
            for(int j = 0; j < k; j++)
                inFile >> blk[i][j];
        mat.blocks[{r, c}] = blk;
    }
    inFile.close();
    return mat;
}


void writeMatrix(const SparseMatrix& mat, const string& filename) {
    ofstream fout(filename);
    if(!fout) {
        cerr << "Error: cannot open result file.\n";
        return;
    }
    fout << mat.height << " " << mat.width << "\n";
    fout << mat.blocks.size() << "\n";

    for (const auto& entry : mat.blocks) {
        int r = entry.first.first;
        int c = entry.first.second;
        fout << r << " " << c << "\n";

        const auto& block = entry.second;
        int k = mat.block_size;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                fout << block[i][j] << " ";
            }
            fout << "\n";
        }
    }
    fout.close();
}


__global__ void block_multiply_kernel(const uint64_t* A, const uint64_t* B, uint64_t* C, int k) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < k && col < k) {
        uint64_t sum = 0;
        for (int p = 0; p < k; ++p)
            sum = (sum + A[row * k + p] * B[p * k + col] % LLONG_MAX) % LLONG_MAX;
        C[row * k + col] = sum;
    }
}

Block multiply_blocks_cuda(const Block& A, const Block& B, int k) {
    size_t bytes = sizeof(uint64_t) * k * k;
    uint64_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    vector<uint64_t> A_flat, B_flat;
    for (auto& row : A)
        for (auto x : row) A_flat.push_back(x);
    for (auto& row : B)
        for (auto x : row) B_flat.push_back(x);

    cudaMemcpy(d_A, A_flat.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(k, k);
    block_multiply_kernel<<<1, threads>>>(d_A, d_B, d_C, k);

    vector<uint64_t> C_flat(k * k);
    cudaMemcpy(C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    Block C(k, vector<uint64_t>(k));
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            C[i][j] = C_flat[i * k + j];

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return C;
}


SparseMatrix matmultCPU(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.width != B.height) {
        cerr << "Matrix dimensions mismatch";
        exit(1);
    }

    SparseMatrix C;
    C.height = A.height;
    C.width = B.width;
    C.block_size = A.block_size;
    int k = C.block_size;

    map<pair<int, int>, Block> c_blocks;

    #pragma omp parallel
    {
        #pragma omp single 
        {
            for(auto itA = A.blocks.begin(); itA != A.blocks.end(); ++itA) {
                const auto& a_coord = itA->first;
                const auto& a_block = itA->second;
                for (auto itB = B.blocks.begin(); itB != B.blocks.end(); ++itB) {
                    const auto& b_coord = itB->first;
                    const auto& b_block = itB->second;
                    if (a_coord.second == b_coord.first) {
                        #pragma omp task shared(c_blocks)
                        {
                            Block prod = multiply_blocks_cuda(a_block, b_block, k);
                            #pragma omp critical
                            {
                                if(c_blocks.find({a_coord.first, b_coord.second}) != c_blocks.end()) {
                                    for(int i = 0; i < k; i++) {
                                        for (int j = 0; j < k; j++) {
                                            c_blocks[{a_coord.first, b_coord.second}][i][j] += prod[i][j] % LLONG_MAX;
                                        }
                                    }
                                } else {
                                    c_blocks[{a_coord.first, b_coord.second}] = prod;
                                }
                            }
                        }
                    }

                }
            }
        }
       
    }
    C.blocks = c_blocks;
    return C;
}


// Helpers for serialization
vector<uint64_t> serialize(const SparseMatrix& mat) {
    vector<uint64_t> buf;
    buf.push_back(mat.height);
    buf.push_back(mat.width);
    buf.push_back(mat.block_size);
    buf.push_back(mat.blocks.size());
    for (const auto& entry : mat.blocks) {
        buf.push_back(entry.first.first);
        buf.push_back(entry.first.second);
        for (int i = 0; i < mat.block_size; i++)
            for (int j = 0; j < mat.block_size; j++)
                buf.push_back(entry.second[i][j]);
    }
    return buf;
}

SparseMatrix deserialize(const vector<uint64_t>& buf) {
    SparseMatrix mat;
    size_t idx = 0;
    mat.height = buf[idx++];
    mat.width = buf[idx++];
    mat.block_size = buf[idx++];
    int num_blocks = buf[idx++];
    for (int b = 0; b < num_blocks; b++) {
        int r = buf[idx++];
        int c = buf[idx++];
        Block blk(mat.block_size, vector<uint64_t>(mat.block_size));
        for (int i = 0; i < mat.block_size; i++)
            for (int j = 0; j < mat.block_size; j++)
                blk[i][j] = buf[idx++];
        mat.blocks[{r, c}] = blk;
    }
    return mat;
}

// write matmultGPU
// and use it in tree reduction.

__global__ void sparseMultKernel(
    int* Ar, int* Ac, uint64_t* Av, int aN,
    int* Br, int* Bc, uint64_t* Bv, int bN,
    int* Cr, int* Cc, uint64_t* Cv, int* count,
    int k, uint64_t MOD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= aN * bN) return;

    int ai = idx / bN;
    int bi = idx % bN;

    if (Ac[ai] != Br[bi]) return;

    int pos = atomicAdd(count, 1);
    Cr[pos] = Ar[ai];
    Cc[pos] = Bc[bi];

    uint64_t* A = Av + ai * k * k;
    uint64_t* B = Bv + bi * k * k;
    uint64_t* C = Cv + pos * k * k;

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            uint64_t sum = 0;
            for (int x = 0; x < k; x++) {
                sum = (sum + (A[i * k + x] * B[x * k + j]) % MOD) % MOD;
            }
            C[i * k + j] = sum;
        }
}


SparseMatrix gpuSparseMultiply(const SparseMatrix& A, const SparseMatrix& B, int k) {
    const uint64_t MOD = LLONG_MAX;

    int numA = A.blocks.size();
    int numB = B.blocks.size();
    int maxBlocks = numA * numB;

    vector<int> A_rows(numA), A_cols(numA), B_rows(numB), B_cols(numB);
    vector<uint64_t> A_vals(numA * k * k), B_vals(numB * k * k);

    int i = 0;
    for (auto it = A.blocks.begin(); it != A.blocks.end(); ++it) {
        auto pos = it->first;
        const Block& blk = it->second;
        A_rows[i] = pos.first;
        A_cols[i] = pos.second;
        for (int r = 0; r < k; r++)
            for (int c = 0; c < k; c++)
                if (A_rows[i] + r >= A.height || A_cols[i] + c >= A.width) A_vals[i * k * k + r * k + c] = 0;
                else A_vals[i * k * k + r * k + c] = blk[r][c];
        i++;
    }

    i = 0;
    for (auto it = B.blocks.begin(); it != B.blocks.end(); ++it) {
        auto pos = it->first;
        const Block& blk = it->second;
        B_rows[i] = pos.first;
        B_cols[i] = pos.second;
        for (int r = 0; r < k; r++)
            for (int c = 0; c < k; c++)
                if (B_rows[i] + r >= B.height || B_cols[i] + c >= B.width) B_vals[i * k * k + r * k + c] = 0;
                else B_vals[i * k * k + r * k + c] = blk[r][c];
        i++;
    }

    int *d_A_r, *d_A_c, *d_B_r, *d_B_c, *d_C_r, *d_C_c, *d_count;
    uint64_t *d_A_v, *d_B_v, *d_C_v;

    cudaMalloc(&d_A_r, numA * sizeof(int));
    cudaMalloc(&d_A_c, numA * sizeof(int));
    cudaMalloc(&d_B_r, numB * sizeof(int));
    cudaMalloc(&d_B_c, numB * sizeof(int));
    cudaMalloc(&d_A_v, numA * k * k * sizeof(uint64_t));
    cudaMalloc(&d_B_v, numB * k * k * sizeof(uint64_t));
    cudaMalloc(&d_C_r, maxBlocks * sizeof(int));
    cudaMalloc(&d_C_c, maxBlocks * sizeof(int));
    cudaMalloc(&d_C_v, maxBlocks * k * k * sizeof(uint64_t));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    cudaMemcpy(d_A_r, A_rows.data(), numA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_c, A_cols.data(), numA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_r, B_rows.data(), numB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_c, B_cols.data(), numB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_v, A_vals.data(), numA * k * k * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_v, B_vals.data(), numB * k * k * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (numA * numB + threads - 1) / threads;

    sparseMultKernel<<<blocks, threads>>>(
        d_A_r, d_A_c, d_A_v, numA,
        d_B_r, d_B_c, d_B_v, numB,
        d_C_r, d_C_c, d_C_v, d_count,
        k, MOD
    );

    int blockCount;
    cudaMemcpy(&blockCount, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    vector<int> C_rows(blockCount), C_cols(blockCount);
    vector<uint64_t> C_vals(blockCount * k * k);
    cudaMemcpy(C_rows.data(), d_C_r, blockCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_cols.data(), d_C_c, blockCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_vals.data(), d_C_v, blockCount * k * k * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    SparseMatrix result;
    result.height = A.height;
    result.width = B.width;
    result.block_size = k;

    map<pair<int, int>, Block> merged_blocks;

    for (int i = 0; i < blockCount; i++) {
        auto key = make_pair(C_rows[i], C_cols[i]);
        Block blk(k, vector<uint64_t>(k));
        for (int r = 0; r < k; r++)
            for (int c = 0; c < k; c++)
                blk[r][c] = C_vals[i * k * k + r * k + c];

        if (merged_blocks.count(key)) {
            for (int r = 0; r < k; r++)
                for (int c = 0; c < k; c++)
                    merged_blocks[key][r][c] = (merged_blocks[key][r][c] + blk[r][c]) % LLONG_MAX;
        } else {
            merged_blocks[key] = blk;
        }
    }

    result.blocks = merged_blocks;


    cudaFree(d_A_r); cudaFree(d_A_c); cudaFree(d_A_v);
    cudaFree(d_B_r); cudaFree(d_B_c); cudaFree(d_B_v);
    cudaFree(d_C_r); cudaFree(d_C_c); cudaFree(d_C_v); cudaFree(d_count);

    return result;
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // struct timespec t0, t1;
    // clock_gettime(CLOCK_MONOTONIC, &t0);

    string folder = argv[1];
    int k, N;
    if (rank == 0) {
        ifstream sizeFile(folder + "/size");
        if (!sizeFile) {
            cerr << "Error: cannot open size file\n";
            exit(1);
        }
        sizeFile >> N >> k;
        sizeFile.close();
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_count = N / size + (rank < N % size ? 1 : 0);
    int start_idx = rank * (N / size) + min(rank, N % size);
    int end_idx = start_idx + local_count;

    // cout << "Reading matrices in : " << rank << endl;
    vector<SparseMatrix> localMatrices;
    for (int i = start_idx; i < end_idx; i++) {
        string fileName = folder + "/matrix" + to_string(i+1);
        localMatrices.push_back(readMatrix(fileName, k));
    }
    // cout << "read all matrices: " << endl;
    // SparseMatrix local_res = localMatrices.empty() ? SparseMatrix(0, 0, k, {}) : localMatrices[0];
    SparseMatrix local_res;
    if (localMatrices.empty()) {
        local_res.height = 0;
        local_res.width = 0;
        local_res.block_size = 0;
        local_res.blocks = {};
    } else {
        local_res = localMatrices[0];
    }
    if (localMatrices.size() > 0) {
        // cout << "starting for loop in: " << rank << endl;
        for (size_t i = 1; i < localMatrices.size(); i++) {
            local_res = gpuSparseMultiply(local_res, localMatrices[i], k);
            
        }
    }
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    // cout << "calculated local resutl in : " << rank << endl;
    // Tree-based reduction to gather final result at rank 0
    SparseMatrix final_result = local_res;
    bool activity = true;
    if (rank >= N) activity = false;
    for (int step = 1; step < size; step *= 2) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                vector<uint64_t> recvBffr;
                MPI_Status status;
                int count;
                MPI_Probe(rank + step, 0, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, MPI_LONG_LONG, &count);
                recvBffr.resize(count);
                MPI_Recv(recvBffr.data(), count, MPI_UINT64_T, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                SparseMatrix incoming = deserialize(recvBffr);
                // final_result = matmultCPU(final_result, incoming);
                final_result = gpuSparseMultiply(final_result, incoming, k);
                
            }
        } else {
            if ((rank - step) >= 0 && activity) {
                vector<uint64_t> out_buf = serialize(final_result);
                MPI_Send(out_buf.data(), out_buf.size(), MPI_UINT64_T, rank - step, 0, MPI_COMM_WORLD);
                activity = false;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) writeMatrix(final_result , "matrix");
    // clock_gettime(CLOCK_MONOTONIC, &t1);

    // long mms = (t1.tv_sec - t0.tv_sec) *1000L + (t1.tv_nsec - t0.tv_nsec)/1000000L;
    // cout << "time: " << mms << "ms\n";

    MPI_Finalize();
    return 0;
}