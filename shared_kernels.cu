#include "shared_kernels.cuh"
#include <stdio.h>

__global__ void csr_spmv_kernel_shared(int *row_ptr, int *col_ind, double *values, double *x, double *y, int num_rows) {
    __shared__ double shared_x[6144];
    int tile_size = 6144;

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    for (int tile = 0; tile < num_rows; tile += tile_size) {
        // Load tile of x into shared memory
        int tile_end = min(tile + tile_size, num_rows);
        for (int i = threadIdx.x + tile; i < tile_end; i += blockDim.x) {
            if ((i - tile) < tile_size) {
                shared_x[i - tile] = x[i];
            }
        }
        __syncthreads();

        if (row < num_rows) {
            double prod = 0.0;
            int row_start = row_ptr[row];
            int row_end = row_ptr[row + 1];

            for (int i = row_start; i < row_end; i++) {
                int col = col_ind[i];
                if (col >= tile && col < tile_end && (col - tile) < tile_size) {
                    prod += values[i] * shared_x[col - tile];
                }
            }
            if (tile == 0) {
                y[row] = prod;
            } else {
                y[row] += prod;
            }
        }
        __syncthreads();
    }
}

__global__ void ellpack_rm_spmv_kernel_shared(int *col_ind, double *values, double *x, double *y, int num_rows, int max_cols_per_row, int tile_size) {
    __shared__ double shared_x[6144];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_offset = blockIdx.y * blockDim.y;

    for (int tile = 0; tile < num_rows; tile += tile_size) {
        // Load tile of x into shared memory
        int tile_end = min(tile + tile_size, num_rows);
        for (int i = threadIdx.x + tile; i < tile_end; i += blockDim.x) {
            if ((i - tile) < tile_size) {
                shared_x[i - tile] = x[i];
            }
        }
        __syncthreads();

        if ((tile_offset + row) < num_rows) {
            double prod = 0;
            for (int i = 0; i < max_cols_per_row; i++) {
                int col = col_ind[(tile_offset + row) * max_cols_per_row + i];
                if (col >= tile && col < tile_end && (col - tile) < tile_size) {
                    prod += values[(tile_offset + row) * max_cols_per_row + i] * shared_x[col - tile];
                }
            }
            if (tile == 0) {
                y[(tile_offset + row)] = prod;
            } else {
                y[(tile_offset + row)] += prod;
            }
        }
        __syncthreads();
    }
}

__global__ void ellpack_cm_spmv_kernel_shared(int *col_ind, double *values, double *x, double *y, int num_rows, int max_cols_per_row, int tile_size) {
    __shared__ double shared_x[6144];

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    for (int tile = 0; tile < num_rows; tile += tile_size) {
        // Load tile of x into shared memory
        int tile_end = min(tile + tile_size, num_rows);
        for (int i = threadIdx.x + tile; i < tile_end; i += blockDim.x) {
            if ((i - tile) < tile_size) {
                shared_x[i - tile] = x[i];
            }
        }
        __syncthreads();

        if (row < num_rows) {
            double prod = 0;
            for (int i = 0; i < max_cols_per_row; i++) {
                int col = col_ind[i * num_rows + row];
                if (col != -1 && col >= tile && col < tile_end) {
                    prod += values[i * num_rows + row] * shared_x[col - tile];
                }
            }
            if (tile == 0) {
                y[row] = prod;
            } else {
                y[row] += prod;
            }
        }
        __syncthreads();
    }
}