#include <stdio.h>

int main() {
    double inp_mtx[6]; // Student has to create this dynamically based on input dimensions
    FILE *fp = fopen("./input_path/mtx_A.bin", "rb"); // Student has to open the file from the input path passed in command line argument
    fread(inp_mtx, sizeof(double), 6, fp); // Student has to read the number of double types based on input dimensions
    fclose(fp);

    double out_mtx[] = {4., 3., 5., 7.}; // Student result matrix
    fp = fopen("./output_path/mtx_C.bin", "wb"); // Student has to open the file from the output path passed in command line argument
    fwrite(out_mtx, sizeof(double), 4, fp); // Depending on the calculated result, the number of double types to write should change 
    fclose(fp);

    return 0;
}