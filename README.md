# GPU-Accelerated Sparse Matrix-Vector Multiplication (SpMV)

## Overview
This project implements Sparse Matrix-Vector Multiplication (SpMV) using CUDA, leveraging the power of GPU acceleration. The primary goal is to compare the performance of different storage formats: Compressed Sparse Row (CSR), ELLPACK Row-Major, and ELLPACK Column-Major, and to analyze the impact of optimization techniques such as shared memory.

## Objectives
- Implement SpMV using CUDA.
- Compare performance across different storage formats (CSR, ELLPACK Row-Major, ELLPACK Column-Major).
- Optimize performance using shared memory.
- Perform timing analysis to determine the fastest format.

## Implementation Details
The project involves converting HIP code for SpMV into CUDA and implementing the necessary kernels and main program. The steps include:
1. **Reading in a Matrix:** The matrix is read into CSR format and converted to other storage formats if necessary.
2. **Allocating Variables:** Necessary variables for the kernel are allocated.
3. **Copying Data:** Relevant data is copied to the device.
4. **Timing:** The execution time is measured for performance analysis.
5. **SpMV Computation:** Multiplication is performed.
6. **Copying Results:** Data is copied back to the host for analysis.

## Code Structure
- **kernels.cu:** Contains CUDA kernels for CSR, ELLPACK Row-Major, and ELLPACK Column-Major SpMV.
- **read_mtx.cu:** Handles reading matrices from the SuiteSparse Matrix Collection.
- **main.cu:** Drives the SpMV computation, performs timing analysis, and experiments with shared memory.

## Performance Analysis
### Test Setup
- **Matrix Used for Benchmarking:** nlpkkt80
  - Dimensions: 1,062,400 x 1,062,400
  - Non-zero elements: 28,192,672
- **GPU Architecture:** NVIDIA A5000
  - Memory Bandwidth: 768 GB/s

### Results
#### Without Shared Memory
- **CSR Execution Time:** 0.667904 ms
- **Row Major ELLPACK Execution Time:** 0.967031 ms
- **Column Major ELLPACK Execution Time:** 0.551673 ms

#### With Shared Memory
- **CSR Execution Time:** 66.0662 ms
- **Row Major ELLPACK Execution Time:** 77.7513 ms
- **Column Major ELLPACK Execution Time:** 76.1583 ms

### Bandwidth Analysis
- **50% Bandwidth:** 587.347 µs
- **Peak Bandwidth:** 293.67 µs

### FLOP Analysis
- **50% Bandwidth:** 96.0 Gflops
- **Peak Bandwidth:** 192.0 Gflops

### Results
#### Without Shared Memory
- **CSR Execution Time:** 0.667904 ms
- **Row Major ELLPACK Execution Time:** 0.967031 ms
- **Column Major ELLPACK Execution Time:** 0.551673 ms

#### With Shared Memory
- **CSR Execution Time:** 66.0662 ms
- **Row Major ELLPACK Execution Time:** 77.7513 ms
- **Column Major ELLPACK Execution Time:** 76.1583 ms

### Bandwidth Analysis
- **50% Bandwidth:** 587.347 µs
- **Peak Bandwidth:** 293.67 µs

### FLOP Analysis
- **50% Bandwidth:** 96.0 Gflops
- **Peak Bandwidth:** 192.0 Gflops

### Bandwidth Analysis of Results
#### Without Shared Memory
- **CSR:**
  - **Execution Time:** 0.667904 ms
  - **Achieved Bandwidth:** \( \frac{28,192,672 \times 8 \times 2}{0.667904 \times 10^6} \approx 674.57 \) GB/s
- **Row Major ELLPACK:**
  - **Execution Time:** 0.967031 ms
  - **Achieved Bandwidth:** \( \frac{28,192,672 \times 8 \times 2}{0.967031 \times 10^6} \approx 465.97 \) GB/s
- **Column Major ELLPACK:**
  - **Execution Time:** 0.551673 ms
  - **Achieved Bandwidth:** \( \frac{28,192,672 \times 8 \times 2}{0.551673 \times 10^6} \approx 816.64 \) GB/s

#### With Shared Memory
- **CSR:**
  - **Execution Time:** 66.0662 ms
  - **Achieved Bandwidth:** \( \frac{28,192,672 \times 8 \times 2}{66.0662 \times 10^6} \approx 6.76 \) GB/s
- **Row Major ELLPACK:**
  - **Execution Time:** 77.7513 ms
  - **Achieved Bandwidth:** \( \frac{28,192,672 \times 8 \times 2}{77.7513 \times 10^6} \approx 5.74 \) GB/s
- **Column Major ELLPACK:**
  - **Execution Time:** 76.1583 ms
  - **Achieved Bandwidth:** \( \frac{28,192,672 \times 8 \times 2}{76.1583 \times 10^6} \approx 5.87 \) GB/s

### FLOP Analysis of Results
#### Without Shared Memory
- **CSR:**
  - **Execution Time:** 0.667904 ms
  - **Achieved FLOPs:** \( \frac{28,192,672 \times 2}{0.667904 \times 10^6} \approx 84.38 \) GFLOPs
- **Row Major ELLPACK:**
  - **Execution Time:** 0.967031 ms
  - **Achieved FLOPs:** \( \frac{28,192,672 \times 2}{0.967031 \times 10^6} \approx 58.27 \) GFLOPs
- **Column Major ELLPACK:**
  - **Execution Time:** 0.551673 ms
  - **Achieved FLOPs:** \( \frac{28,192,672 \times 2}{0.551673 \times 10^6} \approx 102.13 \) GFLOPs

#### With Shared Memory
- **CSR:**
  - **Execution Time:** 66.0662 ms
  - **Achieved FLOPs:** \( \frac{28,192,672 \times 2}{66.0662 \times 10^6} \approx 0.85 \) GFLOPs
- **Row Major ELLPACK:**
  - **Execution Time:** 77.7513 ms
  - **Achieved FLOPs:** \( \frac{28,192,672 \times 2}{77.7513 \times 10^6} \approx 0.72 \) GFLOPs
- **Column Major ELLPACK:**
  - **Execution Time:** 76.1583 ms
  - **Achieved FLOPs:** \( \frac{28,192,672 \times 2}{76.1583 \times 10^6} \approx 0.74 \) GFLOPs

## Shared Memory Optimization
Shared memory is used to reduce global memory accesses by loading frequently accessed data into fast, low-latency memory. However, due to limited shared memory size (6144 double precision elements), memory tiling is employed to process data in tiles. Despite these efforts, shared memory implementations showed significant slowdowns compared to non-shared memory versions.

## How to Run the Code
1. **Download a Testing Matrix**
   Go to https://sparse.tamu.edu/ and download a matrix to use in the code.
2. **Clone the Repository:**
   ```bash
   git clone <gpu-spmv>
   cd <gpu-spmv>
   ```
3. **Compile the Code**
   ```bash
   nvcc -o spmv main.cu kernels.cu read_mtx.cu
   ```
4. **Run the Program**
   ```bash
   ./spmv
   ```