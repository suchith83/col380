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

using namespace std;

typedef vector<vector<uint64_t>> Block;
typedef map<pair<int, int>, Block> SparseMatrix;

void print_block(const pair<int,int>& pos, const Block& block) {
    cout << "Block at (" << pos.first << ", " << pos.second << "):\n";
    for (const auto& row : block) {
        for (uint64_t val : row) cout << val << " ";
        cout << '\n';
    }
    cout << '\n';
}

SparseMatrix readMatrix(const string& path, int k, int& N, int& f_h, int& l_w) {
    ifstream inFile(path);
    if (!inFile) throw runtime_error("Error: cannot open file: " + path);

    int h, w, blocks;
    inFile >> h >> w >> blocks;
    if (path == "matrix1") f_h = h;
    if (path == "matrix" + to_string(N)) l_w = w;

    SparseMatrix mat;
    for (int b = 0; b < blocks; b++) {
        int r, c;
        inFile >> r >> c;
        Block blk(k, vector<uint64_t>(k));
        for(int i = 0; i < k; i++)
            for(int j = 0; j < k; j++)
                inFile >> blk[i][j];
        mat[{r, c}] = move(blk);
    }
    return mat;
}

void writeMatrix(const SparseMatrix& mat, int h, int w, int k) {
    ofstream fout("result");
    fout << h << " " << w << "\n" << mat.size() << "\n";
    for (const auto& entry : mat) {
        fout << entry.first.first << " " << entry.first.second << "\n";
    }
    for (const auto& entry : mat) {
        const Block& block = entry.second;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                fout << block[i][j] << " ";
            }
            fout << "\n";
        }
    }
}

vector<SparseMatrix> load_all_matrices(string folder, int& k, int& N, int& f_w, int& l_h) {
    ifstream sizeFile(folder + "/size");
    sizeFile >> N >> k;
    sizeFile.close();
    vector<SparseMatrix> matrices(N);
    for (int idx = 0; idx < N; idx++) {
        string filename = folder + "/matrix" + to_string(idx+1);
        matrices[idx] = readMatrix(filename, k, N, f_w, l_h);
    }
    return matrices;
}

__global__ void block_multiply_kernel(const uint64_t* A, const uint64_t* B, uint64_t* C, int k) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    uint64_t sum = 0;
    for (int p = 0; p < k; ++p)
        sum = (sum + A[row * k + p] * B[p * k + col] % 9223372036854775807LL) % 9223372036854775807LL;
    C[row * k + col] = sum;
}

Block multiply_blocks_cuda(const Block& A, const Block& B, int k) {
    size_t bytes = sizeof(uint64_t) * k * k;
    uint64_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    vector<uint64_t> A_flat, B_flat;
    for (auto& row : A) for (auto x : row) A_flat.push_back(x);
    for (auto& row : B) for (auto x : row) B_flat.push_back(x);

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

SparseMatrix multiply_gpu(const SparseMatrix& A, const SparseMatrix& B, int k) {
    SparseMatrix C;
    for (const auto& [a_pos, A_blk] : A) {
        for (const auto& [b_pos, B_blk] : B) {
            if (a_pos.second != b_pos.first) continue;
            auto c_pos = make_pair(a_pos.first, b_pos.second);
            Block prod = multiply_blocks_cuda(A_blk, B_blk, k);
            Block& C_blk = C[c_pos];
            if (C_blk.empty()) C_blk = prod;
            else {
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j)
                        C_blk[i][j] = (C_blk[i][j] + prod[i][j]) % 9223372036854775807LL;
            }
        }
    }
    return C;
}

SparseMatrix multiply_all_gpu(vector<SparseMatrix>& matrices, int k) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = matrices.size();
    int local_count = N / size + (rank < N % size ? 1 : 0);
    int start_idx = rank * (N / size) + min(rank, N % size);
    int end_idx = start_idx + local_count;

    // Each process multiplies its assigned subset
    SparseMatrix local_result = matrices[start_idx];
    for (int i = start_idx + 1; i < end_idx; ++i) {
        local_result = multiply_gpu(local_result, matrices[i], k);
    }

    // Tree-based reduction to gather final result at rank 0
    for (int step = 1; step < size; step <<= 1) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                MPI_Status status;
                int count;
                MPI_Probe(rank + step, 0, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, MPI_LONG_LONG, &count);
                vector<uint64_t> buffer(count);
                MPI_Recv(buffer.data(), count, MPI_LONG_LONG, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                SparseMatrix incoming = deserialize(buffer, k);
                local_result = multiply_gpu(local_result, incoming, k);
            }
        } else {
            vector<uint64_t> out_buf = serialize(local_result, k);
            MPI_Send(out_buf.data(), out_buf.size(), MPI_LONG_LONG, rank - step, 0, MPI_COMM_WORLD);
            break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    return local_result;
}

// Helpers for serialization
vector<uint64_t> serialize(const SparseMatrix& mat, int k) {
    vector<uint64_t> buf;
    buf.push_back(mat.size());
    for (const auto& [coord, block] : mat) {
        buf.push_back(coord.first);
        buf.push_back(coord.second);
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                buf.push_back(block[i][j]);
    }
    return buf;
}

SparseMatrix deserialize(const vector<uint64_t>& buf, int k) {
    SparseMatrix mat;
    size_t idx = 0;
    int num_blocks = buf[idx++];
    for (int b = 0; b < num_blocks; b++) {
        int r = buf[idx++];
        int c = buf[idx++];
        Block blk(k, vector<uint64_t>(k));
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                blk[i][j] = buf[idx++];
        mat[{r, c}] = blk;
    }
    return mat;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 2) {
        if (rank == 0) cerr << "Usage: ./a4 folder\n";
        MPI_Finalize(); return 1;
    }

    string folder = argv[1];
    int k, N, l_h, f_w;
    auto matrices = load_all_matrices(folder, k, N, f_w, l_h);
    SparseMatrix final_result = multiply_all_gpu(matrices, k);
    if (rank == 0) writeMatrix(final_result, f_w, l_h, k);
    MPI_Finalize();
    return 0;
}
