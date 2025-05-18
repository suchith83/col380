#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <stdexcept>

using namespace std;

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

// Reads a single k×k–blocked matrix file and returns a map from
// (rowBlock, colBlock) → block data.
map<pair<int,int>, vector<vector<int>>> read_matrix(const string& filename, int k, int& N, int& first_matrix_height, int& last_matrix_width) {
    ifstream inFile(filename);
    if (!inFile) {
        throw runtime_error("Error: Cannot open file " + filename);
    }

    int height, width, numBlocks;
    if (!(inFile >> height >> width >> numBlocks)) {
        throw runtime_error("Error: Invalid header in file " + filename);
    }

    cout << "Reading matrix from file: " << filename << "\n"
         << "  Height: " << height
         << ", Width: " << width
         << ", Number of blocks: " << numBlocks << "\n";
    //if the filename is matrix1, set first_matrix_width
    if (filename == "matrix1") {
        first_matrix_height = height;
    }
    //if the filename is matrixN, set last_matrix_height
    cout<<filename<<endl;
    cout<<"matrix" + to_string(N)<<endl;
    if (filename == "matrix" + to_string(N)) {
        last_matrix_width = width;
    }
    map<pair<int,int>, vector<vector<int>>> matrix;
    int printed_blocks = 0;
    for (int b = 0; b < numBlocks; ++b) {
        int rowPos, colPos;
        inFile >> rowPos >> colPos;

        vector<vector<int>> block(k, vector<int>(k));
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                inFile >> block[i][j];

        if (printed_blocks < 5) {
            print_block({rowPos, colPos}, block);
            ++printed_blocks;
        }

        matrix[{rowPos, colPos}] = move(block);
    }

    return matrix;
}

// Writes the given k×k–blocked matrix into a file named "result"
// using the provided height and width.
void write_matrix(const map<pair<int,int>, vector<vector<int>>>& matrix,
                  int height, int width, int k) {
    int numBlocks = matrix.size();
    ofstream outFile("result");
    if (!outFile) {
        cerr << "Error: Cannot open result file.\n";
        return;
    }

    // header: total height, total width, number of blocks
    outFile << height << " " << width << " " << numBlocks << "\n";

    // each block: rowPos, colPos, then k×k entries
    for (const auto& entry : matrix) {
        int rowPos = entry.first.first;
        int colPos = entry.first.second;
        outFile << rowPos << " " << colPos << "\n";

        const auto& block = entry.second;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                outFile << block[i][j] << " ";
            }
            outFile << "\n";
        }
    }

    cout << "Wrote matrix1 to file: result\n";
}

// Reads "size" to get N and k, then loads all N matrices into a vector.
// Exits on any I/O error.
vector< map<pair<int,int>, vector<vector<int>>> >
load_all_matrices(int& k, int& N, int& first_matrix_width, int& last_matrix_height) {
    ifstream sizeFile("size");
    if (!sizeFile) {
        cerr << "Error: Cannot open size file.\n";
        exit(1);
    }
    sizeFile >> N >> k;
    sizeFile.close();

    vector< map<pair<int,int>, vector<vector<int>>> > matrices(N);
    for (int idx = 0; idx < N; ++idx) {
        string filename = "matrix" + to_string(idx + 1);
        try {
            matrices[idx] = read_matrix(filename, k, N, first_matrix_width, last_matrix_height);
        }
        catch (const exception& e) {
            cerr << e.what() << "\n";
            exit(1);
        }
    }
    return matrices;
}

int main() {
    int k, N, last_matrix_height, first_matrix_width;
    auto matrices = load_all_matrices(k, N, first_matrix_width, last_matrix_height);



    write_matrix(matrices[0], first_matrix_width, last_matrix_height, k);

    // (optional) print the captured values
    // cout << "first_matrix_width = " << first_matrix_width << "\n"
    //      << "last_matrix_height = " << last_matrix_height << "\n";

    return 0;
}