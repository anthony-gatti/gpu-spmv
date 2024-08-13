#include <cuda_runtime.h>
#include <iostream>
#include "kernels.cuh"
#include "shared_kernels.cuh"

// CUDA kernel declarations
__global__ void csr_spmv_kernel(int *row_ptr, int *col_ind, double *values, double *x, double *y, int num_rows);
__global__ void ellpack_rm_spmv_kernel(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row);
__global__ void ellpack_cm_spmv_kernel(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row);
__global__ void csr_spmv_kernel_shared(int *row_ptr, int *col_ind, double *values, double *x, double *y, int num_rows);
__global__ void ellpack_rm_spmv_kernel_shared(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row);
__global__ void ellpack_cm_spmv_kernel_shared(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row);

// Utility functions
void read_matrix(const char* file_path, int*& row_ptr, int*& col_ind, double*& values, int& num_rows, int& num_cols);
void csr_spmv(int *row_ptr, int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, float& time);
void csr_spmv_shared(int *row_ptr, int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, float& time);
void ellpack_rm_spmv(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row, float& time);
void ellpack_cm_spmv(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row, float& time);
void ellpack_rm_spmv_shared(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row, float& time);
void ellpack_cm_spmv_shared(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row, float& time);
void csr_to_ellpack_rm(int *row_ptr, int *col_ind, double *values, int num_rows, int num_cols, int *&ell_col_ind, double *&ell_values, int &max_cols_per_row);
void csr_to_ellpack_cm(int *row_ptr, int *col_ind, double *values, int num_rows, int num_cols, int *&ell_col_ind, double *&ell_values, int &max_cols_per_row);

int main(int argc, char** argv) {
    cudaSetDevice(1);
    // Read matrix from SuiteSparse collection
    int *row_ptr, *col_ind, num_rows, num_cols;
    double *values, *x, *y;

    read_matrix("./nlpkkt80.mtx", row_ptr, col_ind, values, num_rows, num_cols);

    // Allocate memory for vectors x and y
    x = new double[num_cols];
    y = new double[num_rows];
    // Initialize vector x with some values
    for (int i = 0; i < num_cols; ++i) x[i] = 1.0;

    float time_csr, time_rm, time_cm, time_csr_shared, time_rm_shared, time_cm_shared;
    float time_csr_total, time_rm_total, time_cm_total, time_csr_shared_total, time_rm_shared_total, time_cm_shared_total;
    float time_csr_average, time_rm_average, time_cm_average, time_csr_shared_average, time_rm_shared_average, time_cm_shared_average;

    // Perform SpMV with CSR format
    time_csr_total = 0;
    for(int i = 0; i < 100; i++) {
        csr_spmv(row_ptr, col_ind, values, x, y, num_rows, num_cols, time_csr);
        time_csr_total += time_csr;
    }
    
    // Print the resulting vector y
    // std::cout << "Resulting vector y (CSR):" << std::endl;
    // for (int i = 0; i < num_rows; ++i) {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Convert CSR to ELLPACK-RM format
    int *ell_col_ind;
    double *ell_values;
    int max_cols_per_row;
    csr_to_ellpack_rm(row_ptr, col_ind, values, num_rows, num_cols, ell_col_ind, ell_values, max_cols_per_row);

    // Perform SpMV with ELLPACK-RM format
    time_rm_total = 0;
    for(int i = 0; i < 100; i++) {
        ellpack_rm_spmv(ell_col_ind, ell_values, x, y, num_rows, num_cols, max_cols_per_row, time_rm);
        time_rm_total += time_rm;
    }

    // Print the resulting vector y
    // std::cout << "Resulting vector y (ELLPACK-RM):" << std::endl;
    // for (int i = 0; i < num_rows; ++i) {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;
    
    csr_to_ellpack_cm(row_ptr, col_ind, values, num_rows, num_cols, ell_col_ind, ell_values, max_cols_per_row);

    // Perform SpMV with ELLPACK-CM format
    time_cm_total = 0;
    for(int i = 0; i < 100; i++) {
        ellpack_cm_spmv(ell_col_ind, ell_values, x, y, num_rows, num_cols, max_cols_per_row, time_cm);
        time_cm_total += time_cm;
    }

    // Print the resulting vector y
    // std::cout << "Resulting vector y (ELLPACK-CM):" << std::endl;
    // for (int i = 0; i < num_rows; ++i) {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;

    // Perform SpMV with CSR format
    time_csr_shared_total = 0;
    for(int i = 0; i < 100; i++) {
        csr_spmv_shared(row_ptr, col_ind, values, x, y, num_rows, num_cols, time_csr_shared);
        time_csr_shared_total += time_csr_shared;
    }

    // Print the resulting vector y
    // std::cout << "Resulting vector y (CSR-SHARED):" << std::endl;
    // for (int i = 0; i < num_rows; ++i) {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;

    csr_to_ellpack_rm(row_ptr, col_ind, values, num_rows, num_cols, ell_col_ind, ell_values, max_cols_per_row);

    // Perform SpMV with ELLPACK-RM format
    time_rm_shared_total = 0;
    for(int i = 0; i < 100; i++) {
        ellpack_rm_spmv_shared(ell_col_ind, ell_values, x, y, num_rows, num_cols, max_cols_per_row, time_rm_shared);
        time_rm_shared_total += time_rm_shared;
    }

    // Print the resulting vector y
    // std::cout << "Resulting vector y (ELLPACK-RM-SHARED):" << std::endl;
    // for (int i = 0; i < num_rows; ++i) {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;

    csr_to_ellpack_cm(row_ptr, col_ind, values, num_rows, num_cols, ell_col_ind, ell_values, max_cols_per_row);

    // Perform SpMV with ELLPACK-CM format
    time_cm_shared_total = 0;
    for(int i = 0; i < 100; i++) {
        ellpack_cm_spmv_shared(ell_col_ind, ell_values, x, y, num_rows, num_cols, max_cols_per_row, time_cm_shared);
        time_cm_shared_total += time_cm_shared;
    }
    
    // Print the resulting vector y
    // std::cout << "Resulting vector y (ELLPACK-CM-SHARED):" << std::endl;
    // for (int i = 0; i < num_rows; ++i) {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;
    time_csr_average = time_csr_total / 100;
    time_rm_average = time_rm_total / 100;
    time_cm_average = time_cm_total / 100;
    time_csr_shared_average = time_csr_shared_total / 100;
    time_rm_shared_average = time_rm_shared_total / 100;
    time_cm_shared_average = time_cm_shared_total / 100;

    // Print execution times
    std::cout << "Average CSR execution time: " << time_csr_average << " ms" << std::endl;
    std::cout << "Average Row major ELLPACK execution time: " << time_rm_average << " ms" << std::endl;
    std::cout << "Average Column major ELLPACK execution time: " << time_cm_average << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Average CSR with shared memory execution time: " << time_csr_shared_average << " ms" << std::endl;
    std::cout << "Average Row major ELLPACK with shared memory execution time: " << time_rm_shared_average << " ms" << std::endl;
    std::cout << "Average Column major ELLPACK with shared memory execution time: " << time_cm_shared_average << " ms" << std::endl;

    // Clean up
    delete[] row_ptr;
    delete[] col_ind;
    delete[] values;
    delete[] x;
    delete[] y;
    delete[] ell_col_ind;
    delete[] ell_values;

    return 0;
}

void csr_spmv(int *row_ptr, int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, float& time) {
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, row_ptr[num_rows] * sizeof(int));
    cudaMalloc(&d_values, row_ptr[num_rows] * sizeof(double));
    cudaMalloc(&d_x, num_cols * sizeof(double));
    cudaMalloc(&d_y, num_rows * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, row_ptr[num_rows] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, row_ptr[num_rows] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    csr_spmv_kernel<<<numBlocks, blockSize>>>(d_row_ptr, d_col_ind, d_values, d_x, d_y, num_rows);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(y, d_y, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void csr_to_ellpack_rm(int *row_ptr, int *col_ind, double *values, int num_rows, int num_cols, int *&ell_col_ind, double *&ell_values, int &max_cols_per_row) {
    max_cols_per_row = 0;
    for (int i = 0; i < num_rows; ++i) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        if (row_length > max_cols_per_row) {
            max_cols_per_row = row_length;
        }
    }

    ell_col_ind = new int[num_rows * max_cols_per_row];
    ell_values = new double[num_rows * max_cols_per_row];

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < max_cols_per_row; ++j) {
            int csr_index = row_ptr[i] + j;
            if (csr_index < row_ptr[i + 1]) {
                ell_col_ind[i * max_cols_per_row + j] = col_ind[csr_index];
                ell_values[i * max_cols_per_row + j] = values[csr_index];
            } else {
                ell_col_ind[i * max_cols_per_row + j] = -1;
                ell_values[i * max_cols_per_row + j] = 0.0;
            }
        }
    }
}

void csr_to_ellpack_cm(int *row_ptr, int *col_ind, double *values, int num_rows, int num_cols, int *&ell_col_ind, double *&ell_values, int &max_cols_per_row) {
    max_cols_per_row = 0;
    for (int i = 0; i < num_rows; ++i) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        if (row_length > max_cols_per_row) {
            max_cols_per_row = row_length;
        }
    }

    ell_col_ind = new int[num_rows * max_cols_per_row];
    ell_values = new double[num_rows * max_cols_per_row];

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < max_cols_per_row; ++j) {
            int csr_index = row_ptr[i] + j;
            if (csr_index < row_ptr[i + 1]) {
                ell_col_ind[j * num_rows + i] = col_ind[csr_index];
                ell_values[j * num_rows + i] = values[csr_index];
            } else {
                ell_col_ind[j * num_rows + i] = -1;
                ell_values[j * num_rows + i] = 0.0;
            }
        }
    }
}

void ellpack_rm_spmv(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row, float& time) {
    int *d_col_ind;
    double *d_values, *d_x, *d_y;
    cudaMalloc(&d_col_ind, num_rows * max_cols_per_row * sizeof(int));
    cudaMalloc(&d_values, num_rows * max_cols_per_row * sizeof(double));
    cudaMalloc(&d_x, num_cols * sizeof(double));
    cudaMalloc(&d_y, num_rows * sizeof(double));

    cudaMemcpy(d_col_ind, col_ind, num_rows * max_cols_per_row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, num_rows * max_cols_per_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    ellpack_rm_spmv_kernel<<<numBlocks, blockSize>>>(d_col_ind, d_values, d_x, d_y, num_rows, num_cols, max_cols_per_row);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(y, d_y, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ellpack_cm_spmv(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row, float& time) {
    int *d_col_ind;
    double *d_values, *d_x, *d_y;
    cudaMalloc(&d_col_ind, num_rows * max_cols_per_row * sizeof(int));
    cudaMalloc(&d_values, num_rows * max_cols_per_row * sizeof(double));
    cudaMalloc(&d_x, num_cols * sizeof(double));
    cudaMalloc(&d_y, num_rows * sizeof(double));

    cudaMemcpy(d_col_ind, col_ind, num_rows * max_cols_per_row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, num_rows * max_cols_per_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    ellpack_cm_spmv_kernel<<<numBlocks, blockSize>>>(d_col_ind, d_values, d_x, d_y, num_rows, num_cols, max_cols_per_row);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(y, d_y, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void csr_spmv_shared(int *row_ptr, int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, float& time) {
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_x, *d_y;
    
    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, row_ptr[num_rows] * sizeof(int));
    cudaMalloc(&d_values, row_ptr[num_rows] * sizeof(double));
    cudaMalloc(&d_x, num_cols * sizeof(double));
    cudaMalloc(&d_y, num_rows * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, row_ptr[num_rows] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, row_ptr[num_rows] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size_x = (num_rows + block_size - 1) / block_size;
    int grid_size_y = 1;
    dim3 grid_dim(grid_size_x, grid_size_y);
    dim3 block_dim(block_size);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    csr_spmv_kernel_shared<<<grid_dim, block_dim>>>(d_row_ptr, d_col_ind, d_values, d_x, d_y, num_rows);
    
    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(y, d_y, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void ellpack_rm_spmv_shared(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row, float& time) {
    int *d_col_ind;
    double *d_values, *d_x, *d_y;
    
    size_t col_ind_size = num_rows * max_cols_per_row * sizeof(int);
    size_t values_size = num_rows * max_cols_per_row * sizeof(double);
    size_t x_size = num_cols * sizeof(double);
    size_t y_size = num_rows * sizeof(double);
    
    cudaMalloc(&d_col_ind, col_ind_size);
    cudaMalloc(&d_values, values_size);
    cudaMalloc(&d_x, x_size);
    cudaMalloc(&d_y, y_size);

    cudaMemcpy(d_col_ind, col_ind, num_rows * max_cols_per_row * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_values, values, num_rows * max_cols_per_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size_x = (num_rows + block_size - 1) / block_size;
    int grid_size_y = 1;
    dim3 grid_dim(grid_size_x, grid_size_y);
    dim3 block_dim(block_size);

    int tile_size = 6144;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    ellpack_rm_spmv_kernel_shared<<<grid_dim, block_dim>>>(d_col_ind, d_values, d_x, d_y, num_rows, max_cols_per_row, tile_size);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(y, d_y, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void ellpack_cm_spmv_shared(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row, float& time) {
    int *d_col_ind;
    double *d_values, *d_x, *d_y;
    
    cudaMalloc(&d_col_ind, num_rows * max_cols_per_row * sizeof(int));
    cudaMalloc(&d_values, num_rows * max_cols_per_row * sizeof(double));
    cudaMalloc(&d_x, num_cols * sizeof(double));
    cudaMalloc(&d_y, num_rows * sizeof(double));

    cudaMemcpy(d_col_ind, col_ind, num_rows * max_cols_per_row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, num_rows * max_cols_per_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(double), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size_x = (num_rows + block_size - 1) / block_size;
    int grid_size_y = 1;
    dim3 grid_dim(grid_size_x, grid_size_y);
    dim3 block_dim(block_size);

    int tile_size = 6144;
    
    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    ellpack_cm_spmv_kernel_shared<<<grid_dim, block_dim>>>(d_col_ind, d_values, d_x, d_y, num_rows, max_cols_per_row, tile_size);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(y, d_y, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}