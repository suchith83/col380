#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>

using namespace std;

// Function to read a matrix from a binary file
double* readMatrix(const string &path, int rows, int cols) {
    ifstream inFile(path, ios::binary);
    if (!inFile) {
        cerr << "Error opening file: " << path << endl;
        exit(EXIT_FAILURE);
    }
    double* mat = new double[rows * cols];
    inFile.read(reinterpret_cast<char *>(mat), rows * cols * sizeof(double));
    return mat;
}

// Function to write a matrix to a binary file
void writeMatrix(const string &path, double* mat, int rows, int cols) {
    ofstream outFile(path, ios::binary);
    if (!outFile) {
        cerr << "Error writing to file: " << path << endl;
        exit(EXIT_FAILURE);
    }
    outFile.write(reinterpret_cast<const char *>(mat), rows * cols * sizeof(double));
}

// Matrix multiplication functions for each loop permutation
void matrixMultiplyIJK(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyIKJ(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int k = 0; k < colsA; ++k) {
            for (int j = 0; j < colsB; ++j) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyJIK(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    for (int j = 0; j < colsB; ++j) {
        for (int i = 0; i < rowsA; ++i) {
            for (int k = 0; k < colsA; ++k) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyJKI(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    for (int j = 0; j < colsB; ++j) {
        for (int k = 0; k < colsA; ++k) {
            for (int i = 0; i < rowsA; ++i) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyKIJ(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    for (int k = 0; k < colsA; ++k) {
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyKJI(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    for (int k = 0; k < colsA; ++k) {
        for (int j = 0; j < colsB; ++j) {
            for (int i = 0; i < rowsA; ++i) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        cerr << "Usage: ./main <type> <rowsA> <colsA> <colsB> <input_path> <output_path>" << endl;
        return EXIT_FAILURE;
    }

    int type = stoi(argv[1]);
    int rowsA = stoi(argv[2]);
    int colsA = stoi(argv[3]);
    int colsB = stoi(argv[4]);
    string inputPath = argv[5];
    string outputPath = argv[6];

    // Read matrices A and B
    double* A = readMatrix(inputPath + "/mtx_A.bin", rowsA, colsA);
    double* B = readMatrix(inputPath + "/mtx_B.bin", colsA, colsB);
    double* C = new double[rowsA * colsB]();

    auto start = chrono::high_resolution_clock::now();

    // Call appropriate matrix multiplication function
    switch (type) {
        case 0: matrixMultiplyIJK(A, B, C, rowsA, colsA, colsB); break;
        case 1: matrixMultiplyIKJ(A, B, C, rowsA, colsA, colsB); break;
        case 2: matrixMultiplyJIK(A, B, C, rowsA, colsA, colsB); break;
        case 3: matrixMultiplyJKI(A, B, C, rowsA, colsA, colsB); break;
        case 4: matrixMultiplyKIJ(A, B, C, rowsA, colsA, colsB); break;
        case 5: matrixMultiplyKJI(A, B, C, rowsA, colsA, colsB); break;
        default:
            cerr << "Invalid type. Must be between 0 and 5." << endl;
            delete[] A;
            delete[] B;
            delete[] C;
            return EXIT_FAILURE;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    // Write result matrix C to file
    writeMatrix(outputPath + "/mtx_C.bin", C, rowsA, colsB);

    delete[] A;
    delete[] B;
    delete[] C;

    return EXIT_SUCCESS;
}
