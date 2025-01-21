#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

using namespace std;


void readMatrix(const char* filename, double* matrix, int rows, int cols){

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        cerr << "Error: Cannot open file " << endl;
        exit(1);
    }
    size_t elements =fread(matrix, sizeof(double), rows*cols, fp);
    fclose(fp);
}


void writeMatrix(const char* filename,double* matrix, int rows, int cols){
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        cerr << "Error: Cannot open file " << endl;
        exit(1);
    }
    fwrite(matrix, sizeof(double), rows*cols, fp);
    fclose(fp);
}

// matrixMultiplyXYZ

// 1
// IJK
void matrixMultiplyIJK(double* A, double* B, double* C, int mtx_A_rows, int mtx_A_cols, int mtx_B_cols) {
    for (int i = 0; i < mtx_A_rows; i++) {
        for (int j = 0; j < mtx_B_cols; j++) {
            // C[i * mtx_B_cols + j] = 0.0;
            for (int k = 0; k < mtx_A_cols; k++) {
                C[i * mtx_B_cols + j] += A[i * mtx_A_cols + k] * B[k * mtx_B_cols + j];
            }
        }
    }
}

// 2
// IKJ
void matrixMultiplyIKJ(double* A, double* B, double* C, int mtx_A_rows, int mtx_A_cols, int mtx_B_cols) {
    
    for (int i = 0; i < mtx_A_rows; i++) {
        for (int k = 0; k < mtx_A_cols; k++) {
            for (int j = 0; j < mtx_B_cols; j++) {
                C[i * mtx_B_cols + j] += A[i * mtx_A_cols + k] * B[k * mtx_B_cols + j];
            }
        }
    }
}



// 3
// JIK
void matrixMultiplyJIK(double* A, double* B, double* C, int mtx_A_rows, int mtx_A_cols, int mtx_B_cols) {
    for (int j = 0; j < mtx_B_cols; j++) {
        for (int i = 0; i < mtx_A_rows; i++) {
            // C[i * mtx_B_cols + j] = 0.0;
            for (int k = 0; k < mtx_A_cols; k++) {
                C[i * mtx_B_cols + j] += A[i * mtx_A_cols + k] * B[k * mtx_B_cols + j];
            }
        }
    }
}


// 4
// JKI
void matrixMultiplyJKI(double* A, double* B, double* C, int mtx_A_rows, int mtx_A_cols, int mtx_B_cols) {
    for (int j = 0; j < mtx_B_cols; j++) {
        for (int k = 0; k < mtx_A_cols; k++) {
            for (int i = 0; i < mtx_A_rows; i++) {
                C[i * mtx_B_cols + j] += A[i * mtx_A_cols + k] * B[k * mtx_B_cols + j];
            }
        }
    }
}


// 5
// KIJ
void matrixMultiplyKIJ(double* A, double* B, double* C, int mtx_A_rows, int mtx_A_cols, int mtx_B_cols) {
    for (int k = 0; k < mtx_A_cols; k++) {
        for (int i = 0; i < mtx_A_rows; i++) {
            for (int j = 0; j < mtx_B_cols; j++) {
                C[i * mtx_B_cols + j] += A[i * mtx_A_cols + k] * B[k * mtx_B_cols + j];
            }
        }
    }
}

// 6
// KJI
void matrixMultiplyKJI(double* A, double* B, double* C, int mtx_A_rows, int mtx_A_cols, int mtx_B_cols) {
    for (int k = 0; k < mtx_A_cols; k++) {
        for (int j = 0; j < mtx_B_cols; j++) {
            for (int i = 0; i < mtx_A_rows; i++) {
                C[i * mtx_B_cols + j] += A[i * mtx_A_cols + k] * B[k * mtx_B_cols + j];
            }
        }
    }
}




int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "error in no.of arguments !!!!!!" << std::endl;
        return 1;
    }


    // <type> <mtx_A_rows> <mtx_A_cols> <mtx_B_cols> <input_path> <output_path> 

    int combination = stoi(argv[1]);
    int mtx_A_rows = stoi(argv[2]);
    int mtx_A_cols = stoi(argv[3]);
    int mtx_B_rows = mtx_A_cols;
    int mtx_B_cols = stoi(argv[4]);

    //input_path  =  path  where  you  can  find  both  the  input  matrices  binary  file (mtx_A.bin and mtx_B.bin)
    string input_path = argv[5];
    string output_path = argv[6];

    // 0 = IJK, 1 = IKJ, 2=JIK, 3=JKI, 4=KIJ, 5=KJI

    // creating the matrix A, B

    double* A = new double[mtx_A_rows * mtx_A_cols];
    double* B = new double[mtx_B_rows * mtx_B_cols];

    readMatrix((input_path + "/mtx_A.bin").c_str(), A, mtx_A_rows, mtx_A_cols);
    readMatrix((input_path + "/mtx_B.bin").c_str(), B, mtx_B_rows, mtx_B_cols);
    // print the matrix A, B

    
    
    double* C = new double[mtx_A_rows * mtx_B_cols];
    memset(C, 0, sizeof(double) * mtx_A_rows * mtx_B_cols);

    if (combination == 0) {
        matrixMultiplyIJK(A, B, C, mtx_A_rows, mtx_A_cols, mtx_B_cols);
        printf("IJK\n");
    }
    else if (combination == 1) {
        matrixMultiplyIKJ(A, B, C, mtx_A_rows, mtx_A_cols, mtx_B_cols);
    }
    else if (combination == 2) {
        matrixMultiplyJIK(A, B, C, mtx_A_rows, mtx_A_cols, mtx_B_cols);
    }
    else if (combination == 3) {
        matrixMultiplyJKI(A, B, C, mtx_A_rows, mtx_A_cols, mtx_B_cols);
    }
    else if (combination == 4) {
        matrixMultiplyKIJ(A, B, C, mtx_A_rows, mtx_A_cols, mtx_B_cols);
    }
    else if (combination == 5) {
        matrixMultiplyKJI(A, B, C, mtx_A_rows, mtx_A_cols, mtx_B_cols);
    }
    // write the matrix C
    
    writeMatrix((output_path + "/mtx_C.bin").c_str(), C, mtx_A_rows, mtx_B_cols);

}

