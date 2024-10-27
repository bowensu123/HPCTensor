#include <armadillo>
#include <iostream>
#include <chrono>
#include <vector>
#include <tuple>

using namespace std;
using namespace arma;
using namespace std::chrono;
mat generate_low_rank_matrix(int n, int r) {
    mat U = randu<mat>(n, r);  
    mat V = randu<mat>(n, r);  
    return U * V.t();           
}

tuple<vector<int>, mat, mat> maxvol(const mat &A, int rank, int num_blocks, bool use_parallel, int num_threads = 1) {
    int n = A.n_rows;
    int m = A.n_cols;

    vector<int> row_idx(rank);  
    vector<int> rest_of_rows(n);
    iota(rest_of_rows.begin(), rest_of_rows.end(), 0);  

    mat A_new = A;  // Is necessary?

    // Compute block sizes for dividing the matrix into blocks
    int block_size = n / num_blocks;
    vector<mat> blocks(num_blocks);
    for (int b = 0; b < num_blocks; ++b) {
        int start = b * block_size;
        int end = (b == num_blocks - 1) ? n : (b + 1) * block_size;
        blocks[b] = A_new.rows(start, end - 1);  
    }

    for (int i = 0; i < rank; ++i) {
        vec norms(n, fill::zeros);  
        int maxIdx = -1;            
        double maxNorm = -1;         

        // Compute row norms in parallel
        #pragma omp parallel for num_threads(num_threads) shared(norms, maxIdx, maxNorm) if (use_parallel)
        for (int b = 0; b < num_blocks; ++b) {
            for (size_t j = 0; j < blocks[b].n_rows; ++j) {
                int globalRowIdx = b * block_size + j;
                norms(globalRowIdx) = accu(square(blocks[b].row(j)));

                // Atomic update for maxNorm and maxIdx
                #pragma omp critical
                {
                    if (norms(globalRowIdx) > maxNorm) {
                        maxNorm = norms(globalRowIdx);
                        maxIdx = globalRowIdx;
                    }
                }
            }
        }

        // Extract the row with maximal norm
        int block_idx = maxIdx / block_size;
        int local_idx = maxIdx % block_size;
        rowvec max_row = blocks[block_idx].row(local_idx);

        // Update all blocks in parallel using the shared max_row
        #pragma omp parallel for num_threads(num_threads) shared(max_row) if (use_parallel)
        for (int b = 0; b < num_blocks; ++b) {
            for (size_t j = 0; j < blocks[b].n_rows; ++j) {
                blocks[b].row(j) -= (dot(blocks[b].row(j), max_row) / dot(max_row, max_row)) * max_row;
            }
        }

        // Store the selected row index
        row_idx[i] = maxIdx;

        // Remove the selected row from the tracking list
        auto it = find(rest_of_rows.begin(), rest_of_rows.end(), maxIdx);
        if (it != rest_of_rows.end()) {
            rest_of_rows.erase(it);
        }

        cout << "Selected row: " << maxIdx << ", Remaining rows: " << rest_of_rows.size() << endl;
    }

    // Construct the output matrices
    uvec arma_row_indices = conv_to<uvec>::from(row_idx);
    mat selected_rows = A.rows(arma_row_indices);
    mat A_inv = pinv(selected_rows);

    return make_tuple(row_idx, selected_rows, A_inv);
}

vector<double> benchmark(const mat &A, int rank, const vector<int> &thread_counts) {
    vector<double> timings;  // Store execution times for each thread count

    for (int threads : thread_counts) {
        int num_blocks = threads;  // Match number of blocks to number of threads
        cout << "Testing with " << threads << " threads..." << endl;

        auto start = high_resolution_clock::now();
        auto [row_idx, selected_rows, A_inv] = maxvol(A, rank, num_blocks, true, threads);
        auto end = high_resolution_clock::now();

        double duration = duration_cast<milliseconds>(end - start).count();
        timings.push_back(duration);
        cout << "Execution time with " << threads << " threads: " << duration << " ms" << endl;
    }
    return timings;
}

int main() {
    vector<int> matrix_sizes = {128, 256, 512, 1024,2048,4098,8196};  
    vector<int> thread_counts = {1, 2, 4, 8,16,32,64,128};  
    int rank = 10;  

    ofstream outfile("timing_results.csv");
    outfile << "Matrix Size,1 Thread,2 Threads,4 Threads,8 Threads\n";
    outfile.close();

    // Run benchmark for each matrix size
    for (int size : matrix_sizes) {
        cout << "\nTesting matrix size: " << size << "x" << size << endl;
        mat A = generate_low_rank_matrix(size, rank); 
        vector<double> timings = benchmark(A, rank, thread_counts);
        ofstream outfile("timing_results.csv", ios::app);
        outfile << size;
        for (double time : timings) {
            outfile << "," << time;
        }
        outfile << "\n";
        outfile.close();
    }

    return 0;
}