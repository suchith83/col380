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

typedef vector<vector<int>> Block;
typedef map<pair<int, int>, Block> SparseMatrix;


// Helper to print a single k×k block
void print_block(const pair<int,int>& pos, const vector<vector<int>>& block) {
    cout << "Block at (" << pos.first << ", " << pos.second << "):\n";
    for (const auto& row : block) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << '\n';
    }
    cout << '\n';
}

SparseMatrix readMatrix(const string& path, int k, int& N, int& f_h, int& l_w) {
    ifstream inFile(path);
    if (!inFile) {
        throw runtime_error("Error: cannot open file: " + path);
    }

    int h, w, blocks;
    if (! (inFile >> h >> w >> blocks)) {
        throw runtime_error("Error: Invalid header in file" + path);
    }
    if (path == "matrix1") {
        f_h = h;
    }
    if (path == "matrix" + to_string(N)) {
        l_w = w;
    }

    SparseMatrix mat;
    int printed_blocks = 0;
    for (int b = 0; b < blocks; b++) {
        int r, c;
        inFile >> r >> c;

        Block blk(k, vector<int>(k));
        for(int i = 0; i < k; i++)
            for(int j = 0; j < k; j++)
                inFile >> blk[i][j];
        if (printed_blocks < 5) {
            print_block({r, c}, blk);
            printed_blocks++;
        }
        mat[{r, c}] = move(blk);
    }
    return mat;
}


void writeMatrix(const SparseMatrix& mat, int h, int w, int k) {
    int blocks = mat.size();
    ofstream fout("result");
    if(!fout) {
        cerr << "Error: cannot open result file.\n";
        return;
    }
    fout << h << " " << w << "\n";
    fout << blocks << "\n";

    for (const auto& entry : mat) {
        int r = entry.first.first;
        int c = entry.first.second;
        fout << r << " " << c << "\n";

        const auto& block = entry.second;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                fout << block[i][j] << " ";
            }
            fout << "\n";
        }
    }
    cout << "wrote matrix1 to the file result \n";
}

vector<SparseMatrix> load_all_matrices(string folder, int& k, int& N, int& f_w, int& l_h) {
    ifstream sizeFile(folder + "/size");
    if (!sizeFile) {
        cerr << "Error: cannot open size file\n";
        exit(1);
    }
    sizeFile >> N >> k;
    sizeFile.close();
    vector<SparseMatrix> matrices(N);
    for (int idx = 0; idx < N; idx++) {
        string filename = folder + "/matrix" + to_string(idx+1);
        try {
            matrices[idx] = readMatrix(filename, k, N, f_w, l_h);

        }
        catch (const exception& e) {
            cerr << e.what() << "\n";
            exit(1);
        }
    }
    return matrices;
}



int main(int argc, char** argv) {
    string folder = argv[1];
    int k, N, l_h, f_w;
    auto matrices = load_all_matrices(folder, k, N, f_w, l_h);
    writeMatrix(matrices[0],f_w, l_h, k);
    
}
