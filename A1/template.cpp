#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <omp.h>
#include "check.h"

using namespace std;

map<pair<int, int>, vector<vector<int>>> generate_matrix(int n, int m, int b) {
    map<pair<int, int>, vector<vector<int>>> matrix_map;
    vector<pair<int, int>> block_positions;
    int num_blocks = (n / m) * (n / m);
    while(block_positions.size() < b) {
        int row = rand() % (n / m);
        int col = rand() % (n / m);
        pair<int, int> block_pos = make_pair(row, col);
        if (find(block_positions.begin(), block_positions.end(), block_pos) == block_positions.end()) {
            block_positions.push_back(block_pos);
        }
    }
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (auto [row, col] : block_positions) {
                #pragma omp task shared(matrix_map) if(black_box())
                {
                    vector<vector<int>> block(m, vector<int>(m));
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m; j++) {
                            block[i][j] = rand();
                        }
                    }
                    #pragma omp critical
                    {
                        matrix_map[{row, col}] = block;
                    }
                    
                }
            }
        }
    }
    return matrix_map;
}

vector<float> matmul(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    vector<float> row_statistics(n, 0.0f);
    vector<int> row_elements(n, 0);
    map<pair<int, int>, vector<vector<int>>> result_map;
    bool has_odd_exponent = (k % 2 != 0);

    map<pair<int, int>, vector<vector<int>>> filtered_blocks;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (auto it = blocks.begin(); it != blocks.end(); ++it) {
                #pragma omp task shared(filtered_blocks) if(black_box())
                {
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
                    if (isBlockNonZero) {
                        #pragma omp critical
                        {
                            filtered_blocks[it->first] = block;
                        }
                    }
                }
            }
        }
    }

    blocks = move(filtered_blocks);
    result_map = blocks;
    int exp = k;
    while (exp > 1) {
        map<pair<int, int>, vector<vector<int>>> tmp_map;

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (auto& [blockA_pos, blockA] : result_map) {
                    for (auto& [blockB_pos, blockB] : result_map) {
                        if (blockA_pos.second == blockB_pos.first) {
                            #pragma omp task shared(tmp_map) if(black_box())
                            {
                                vector<vector<int>> product(m, vector<int>(m, 0));
                                for (int i = 0; i < m; ++i) {
                                    for (int j = 0; j < m; ++j) {
                                        for (int p = 0; p < m; ++p) {
                                            product[i][j] += blockA[i][p] * blockB[p][j];
                                            if (k == 2 && blockA[i][p] * blockB[p][j] != 0) {
                                                #pragma omp critical
                                                {
                                                    row_statistics[blockA_pos.first * m + i]++;
                                                }
                                            }
                                        }
                                    }
                                }

                                #pragma omp critical
                                {
                                    if (tmp_map.find({blockA_pos.first, blockB_pos.second}) != tmp_map.end()) {
                                        vector<vector<int>>& existing_block = tmp_map[{blockA_pos.first, blockB_pos.second}];
                                        for (int i = 0; i < m; i++) {
                                            for (int j = 0; j < m; j++) {
                                                existing_block[i][j] += product[i][j];
                                            }
                                        }
                                    } else {
                                        tmp_map[{blockA_pos.first, blockB_pos.second}] = product;
                                    }
                                    
                                }
                            }
                        }
                    }
                }
            }
        }
        result_map = tmp_map;
        exp /= 2;
    }

    if (has_odd_exponent) {
        map<pair<int, int>, vector<vector<int>>> final_map;
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (auto& [blockA_pos, blockA] : result_map) {
                    for (auto& [blockB_pos, blockB] : blocks) {
                        if (blockA_pos.second == blockB_pos.first) {
                            #pragma omp task shared(final_map) if(black_box())
                            {
                                vector<vector<int>> product(m, vector<int>(m, 0));
                                for (int i = 0; i < m; ++i) {
                                    for (int j = 0; j < m; ++j) {
                                        for (int p = 0; p < m; ++p) {
                                            product[i][j] += blockA[i][p] * blockB[p][j];
                                        }
                                    }
                                }

                                #pragma omp critical
                                {
                                    if (final_map.find({blockA_pos.first, blockB_pos.second}) != final_map.end()) {
                                        vector<vector<int>>& existing_block = final_map[{blockA_pos.first, blockB_pos.second}];
                                        for (int i = 0; i < m; i++) {
                                            for (int j = 0; j < m; j++) {
                                                existing_block[i][j] += product[i][j];
                                            }
                                        }
                                    } else {
                                        final_map[{blockA_pos.first, blockB_pos.second}] = product;
                                    }
                                    
                                }
                            }
                        }
                    }
                }
            }
        }
        result_map = final_map;
    }

    blocks = result_map;

    for(const auto& block : result_map) {
        int block_row = block.first.first;
        for (int i = 0; i < m; i++) {
            int row_ind = block_row * m + i;
            row_elements[row_ind] += m;
        }
    }

    if (k == 2) {
        for (int i = 0; i < n; i++) {
            if (row_elements[i] > 0) {
                row_statistics[i] /= row_elements[i];
            } else {
                row_statistics[i] = 0.0f;
            }
        }
    }

    return (k == 2) ? row_statistics : vector<float>();
}