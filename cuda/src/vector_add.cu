#include <stdio.h>
#include <stdint.h>
#include "cuda_call.h"
#include <stdexcept>


// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int n)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += gridDim.x * blockDim.x) {
        C[idx] = A[idx] + B[idx];
    }
}

/*int main()
{
    const size_t n = 100 * 1000 * 1000;
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[n];  // allocate space for vectors in host memory
    h_B = new float[n];
    h_C = new float[n];
    for (size_t i = 0; i < n; i++) {  // initialize vectors in host memory
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }
    CUDACall( cudaMalloc(&d_A, n * sizeof(float)) ); // allocate device space for vector A
    CUDACall( cudaMalloc(&d_B, n * sizeof(float)) ); // allocate device space for vector B
    CUDACall( cudaMalloc(&d_C, n * sizeof(float)) ); // allocate device space for vector C
    
    CUDACall( cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDACall( cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice) );
    
    size_t block_size = 1;
    size_t block_count = 1;
    vadd<<<block_count, block_size>>>(d_A, d_B, d_C, n);
    
    CUDACall( cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost) );

    for (size_t i = 0; i < n; i++) {
        if (h_A[i] + h_B[i] != h_C[i]) {
            throw new std::runtime_error("error");
        }
    }

    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);
    return 0;
}*/
  