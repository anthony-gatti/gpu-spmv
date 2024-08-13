__global__ void csr_spmv_kernel(int *row_ptr, int *col_ind, double *values, double *x, double *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        double dot = 0;

        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x[col_ind[i]];
        }
        y[row] = dot;
    }
}

__global__ void ellpack_rm_spmv_kernel(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        double dot = 0;
        for (int i = 0; i < max_cols_per_row; i++) {
            int col = col_ind[row * max_cols_per_row + i];
            if (col < num_cols) {
                dot += values[row * max_cols_per_row + i] * x[col];
            }
        }
        y[row] = dot;
    }
}

__global__ void ellpack_cm_spmv_kernel(int *col_ind, double *values, double *x, double *y, int num_rows, int num_cols, int max_cols_per_row) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        double dot = 0;
        for (int i = 0; i < max_cols_per_row; i++) {
            int col = col_ind[i * num_rows + row];
            if (col < num_cols) {
                dot += values[i * num_rows + row] * x[col];
            }
        }
        y[row] = dot;
    }
}