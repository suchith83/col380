#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <omp.h>
#include "check.h"
#include <iomanip>

using namespace std;

bool black_box() {
    return true;
}

vector<vector<int>> multiply_blocks(vector<vector<int>>& block1,
                                    vector<vector<int>>& block2, int m) {
    vector<vector<int>> product(m, vector<int>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                product[i][j] += block1[i][k] * block2[k][j];
            }
        }
    }
    return product;
}

void removeMultiplesOf5(map<pair<int, int>, vector<vector<int>>>& matrixBlocks) {
    for (auto it = matrixBlocks.begin(); it != matrixBlocks.end(); ) {
        vector<vector<int>>& block = it->second;
        bool isBlockNonZero = false;

        for (auto& row : block) {
            for (auto& value : row) {
                if (value % 5 == 0) {
                    value = 0;
                }
                if (value != 0) {
                    isBlockNonZero = true;
                }
            }
        }

        if (!isBlockNonZero) {
            it = matrixBlocks.erase(it);
        } else {
            ++it;
        }
    }
}

bool is_square(map<pair<int, int>, vector<vector<int>>>& matrix1,
               map<pair<int, int>, vector<vector<int>>>& matrix2, int m) {
    removeMultiplesOf5(matrix1);
    map<pair<int, int>, vector<vector<int>>> squared_result;

    for (auto& [block_pos1, block1] : matrix1) {
        for (auto& [block_pos2, block2] : matrix1) {
            if (block_pos1.second == block_pos2.first) {
                vector<vector<int>> product = multiply_blocks(block1, block2, m);
                if (squared_result.find({block_pos1.first, block_pos2.second}) != squared_result.end()) {
                    vector<vector<int>>& existing_block = squared_result[{block_pos1.first, block_pos2.second}];
                    for (int i = 0; i < m; ++i) {
                        for (int j = 0; j < m; ++j) {
                            existing_block[i][j] += product[i][j];
                        }
                    }
                } else {
                    squared_result[{block_pos1.first, block_pos2.second}] = product;
                }
            }
        }
    }
    

    if (squared_result.size() != matrix2.size()) {
        return false;
    }

    for (auto& [block_pos, block] : squared_result) {
        if (matrix2.find(block_pos) == matrix2.end()) {
            return false;
        }

        vector<vector<int>>& block2 = matrix2.at(block_pos);
        if (block != block2) {
            return false;
        }
    }

    return true;
}

bool has_non_zero_element(vector<vector<int>>& block) {
    for (auto& row : block)
        for (int val : row)
            if (val != 0)
                return true;
    return false;
}

int count_non_zero_blocks(map<pair<int, int>, vector<vector<int>>>& blocks) {
    int non_zero_count = 0;

    for (auto& entry : blocks) {
        vector<vector<int>>& block = entry.second;

        if (has_non_zero_element(block))
            non_zero_count++;
    }

    return non_zero_count;
}

void print_matrix_map(map<pair<int, int>, vector<vector<int>>>& matrix_map) {
    for (auto& entry : matrix_map) {
        cout << "Block (" << entry.first.first << ", " << entry.first.second << "):\n";
        for (auto& row : entry.second) {
            for (int val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

int main() {
    int n = 100000; 
    int m = 16;     
    int k = 2;      
    vector<int> b_values = {64, 256, 1024, 4096};
    vector<int> cores = {2, 4, 8, 16, 32, 40}; 

    
    ofstream file("data.csv");
    if (!file.is_open()) {
        cerr << "Error: Could not open data.csv for writing!\n";
        return 1;
    }

    file << "b,2 cores,4 cores,8 cores,16 cores,32 cores,40 cores\n";

    for (int b : b_values) {
        file << b; 
        for (int num_threads : cores) {
            omp_set_num_threads(num_threads);

            map<pair<int, int>, vector<vector<int>>> blocks = generate_matrix(n, m, b);

            double start = omp_get_wtime();
            vector<float> s = matmul(blocks, n, m, k);
            double end = omp_get_wtime();
            cout << "b=" << b << ", Threads=" << num_threads << ", Time=" << (end - start) * 1000.0 << " ms" << endl;
            double time_taken = (end - start) * 1000.0; 


            // Store time in CSV file
            file << "," << fixed << setprecision(2) << time_taken;
            // cout << "b=" << b << ", Threads=" << num_threads << ", Time=" << time_taken << " ms" << endl;
        }
        file << "\n"; 
    }
    file.close();
    return 0;
}
