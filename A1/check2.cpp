#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include "check.h"

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
    int b = 2;
    int k = 2;

    srand(time(0));

    map<pair<int, int>, vector<vector<int>>> blocks = generate_matrix(n, m, b);
    if(count_non_zero_blocks(blocks)==blocks.size() && blocks.size()>=b)
        cout<<"You have generated the matrix correctly\n";
    else
        cout<<"You have NOT generated the matrix correctly\n";

    map<pair<int, int>, vector<vector<int>>> sparse_matrix;

    // Insert blocks into the map
    sparse_matrix[{0, 0}] = {
        {1, 0, 2},
        {0, 3, 0},
        {2, 0, 4}
    };

    sparse_matrix[{0, 2}] = {
        {0, 0, 0},
        {0, 3, 1},
        {0, 0, 5}
    };

    sparse_matrix[{2, 0}] = {
        {0, 0, 0},
        {0, 3, 0},
        {0, 1, 5}
    };

    map<pair<int, int>, vector<vector<int>>> original_blocks = sparse_matrix;

    // Test Case 1: Small Sparse Matrix (n = 6, m = 2, k = 2)
    map<pair<int, int>, vector<vector<int>>> sparse_matrix_1;

    // Insert blocks into the map
    sparse_matrix_1[{0, 0}] = {
        {1, 2},
        {0, 3}
    };

    sparse_matrix_1[{0, 1}] = {
        {0, 4},
        {5, 0}
    };

    sparse_matrix_1[{1, 0}] = {
        {6, 0},
        {0, 7}
    };

    map<pair<int, int>, vector<vector<int>>> original_blocks_1 = sparse_matrix_1;

    // Test Case 2: Medium Sparse Matrix (n = 9, m = 3, k = 3)
    map<pair<int, int>, vector<vector<int>>> sparse_matrix_2;

    // Insert blocks into the map
    sparse_matrix_2[{0, 0}] = {
        {1, 0, 2},
        {0, 3, 0},
        {2, 0, 4}
    };

    sparse_matrix_2[{0, 2}] = {
        {0, 0, 0},
        {0, 3, 1},
        {0, 0, 5}
    };

    sparse_matrix_2[{2, 0}] = {
        {0, 0, 0},
        {0, 3, 0},
        {0, 1, 5}
    };

    map<pair<int, int>, vector<vector<int>>> original_blocks_2 = sparse_matrix_2;

    // Test Case 3: Large Sparse Matrix (n = 12, m = 4, k = 2)
    map<pair<int, int>, vector<vector<int>>> sparse_matrix_3;

    // Insert blocks into the map
    sparse_matrix_3[{0, 0}] = {
        {2, 0, 0, 3},
        {0, 5, 0, 0},
        {0, 0, 6, 0},
        {7, 0, 0, 0}
    };

    sparse_matrix_3[{1, 1}] = {
        {0, 0, 8, 0},
        {9, 0, 0, 0},
        {0, 10, 0, 0},
        {0, 0, 0, 11}
    };

    sparse_matrix_3[{2, 0}] = {
        {12, 0, 0, 0},
        {0, 13, 0, 0},
        {0, 0, 14, 0},
        {0, 0, 0, 15}
    };

    map<pair<int, int>, vector<vector<int>>> original_blocks_3 = sparse_matrix_3;

    // Test Case 4: Very Sparse Matrix (n = 16, m = 4, k = 2)
    map<pair<int, int>, vector<vector<int>>> sparse_matrix_4;

    // Insert blocks into the map
    sparse_matrix_4[{0, 0}] = {
        {0, 0, 0, 0},
        {0, 5, 0, 0},
        {0, 0, 0, 0},
        {7, 0, 0, 0}
    };

    sparse_matrix_4[{3, 2}] = {
        {0, 0, 8, 0},
        {0, 0, 0, 0},
        {0, 10, 0, 0},
        {0, 0, 0, 0}
    };

    sparse_matrix_4[{2, 3}] = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 15}
    };

    map<pair<int, int>, vector<vector<int>>> original_blocks_4 = sparse_matrix_4;

    
    // vector<float> s = matmul(sparse_matrix, n, 3, k);
    
    // vector<float> s = matmul(sparse_matrix, 9, 3, 3);
    // cout << "s2: n: 9, m: 3, k: 2" << endl;
    // for (int i = 0; i < s.size(); i++) {
    //     cout << s[i] << " ";
    // }
    // cout << endl;
    // test cases
    // vector<float> s1 = matmul(sparse_matrix_1, 6, 2, 2);
    // vector<float> s2 = matmul(sparse_matrix_2, 9, 3, 2);
    // vector<float> s3 = matmul(sparse_matrix_3, 12, 4, 2);
    vector<float> s4 = matmul(sparse_matrix_4, 16, 4, 2);

    // print s1, s2, s3, s4 by mentioning n, m, k
    // cout << "s1: n: 6, m: 2, k: 2" << endl;
    // for (int i = 0; i < s1.size(); i++) {
    //     cout << s1[i] << " ";
    // }
    // cout << endl;
    // cout << "s2: n: 9, m: 3, k: 2" << endl;
    // for (int i = 0; i < s2.size(); i++) {
    //     cout << s2[i] << " ";
    // }
    // cout << endl;
    // cout << "s3: n: 12, m: 4, k: 2" << endl;
    // for (int i = 0; i < s3.size(); i++) {
    //     cout << s3[i] << " ";
    // }
    // cout << endl;
    cout << "s4: n: 16, m: 4, k: 2" << endl;
    for (int i = 0; i < s4.size(); i++) {
        cout << s4[i] << " ";
    }
    cout << endl;



    // for k = 2, check with is_suqare also
    // bool res1 = is_square(original_blocks_1, sparse_matrix_1, 2);
    // bool res3 = is_square(original_blocks_3, sparse_matrix_3, 4);
    bool res4 = is_square(original_blocks_4, sparse_matrix_4, 4);

    // print all the res's

    // cout << "res1: " << res1 << endl;
    // cout << "res3: " << res3 << endl;
    cout << "res4: " << res4 << endl;
    // bool res2 = is_square(original_blocks_2, sparse_matrix_2, 3);
    // cout<<"hi\n";	
    bool res = is_square(original_blocks, sparse_matrix, 3);
    if(res4)
        cout<<"Your function computed the square correctly\n";
    else
        cout<<"Your function did NOT compute the square correctly\n";
    cout << "Size of S = " << s4.size()<<endl;

    return 0;
}