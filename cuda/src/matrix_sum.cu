#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const size_t DSIZE = 32768;      // matrix side dimension
const int BLOCK_SIZE = 256;      // CUDA maximum is 1024

// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t n){

  int idx = threadIdx.x + blockIdx.x * blockDim.x; // create typical 1D thread index from built-in variables
  if (idx < n){
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
      sum += A[idx * DSIZE + i];         // write a for loop that will cause the thread to iterate across a row, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
}}


// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds){

  int idx = threadIdx.x + blockIdx.x * blockDim.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[i * DSIZE + idx];         // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
}}

// blocks of size (BS, BS)
// grid of size (GS, 1)
// we can have BS* GS < n
#define BS 32
__global__ void column_sums_blocks(const float* A, float* sums, size_t n)
{
    __shared__ float block[BS][BS];

    for (uint xBlock = blockDim.x * blockIdx.x; xBlock < n; xBlock += gridDim.x * blockDim.x) {
        float sum = 0;
        for (uint yBlock = blockDim.y * blockIdx.y; yBlock < n; yBlock += blockDim.y) {
            // load the block
            {
                uint x = xBlock + threadIdx.x;
                uint y = yBlock + threadIdx.y;
                if (x < n && y < n) {
                    block[threadIdx.y][threadIdx.x] = A[x + n * y];
                }
            }
            __syncthreads();

            /*// sum the columns of the block
            for (uint k = 1; k < BS; k *= 2) {
                if (2*k*threadIdx.y < BS) {
                    block[2*k*threadIdx.y][threadIdx.x] += block[2*k*threadIdx.y + k][threadIdx.x];
                }
                __syncthreads();
            }*/

            // accumulate in local variable
            if (threadIdx.y == 0) {
                for (uint ofs = 0; ofs < blockDim.y && yBlock + ofs < n; ofs++) {
                    sum += block[ofs][threadIdx.x];
                }
            }
            __syncthreads();
        }
        if (threadIdx.y == 0) {
            sums[xBlock + threadIdx.x] = sum;
        }
    }
}



bool validate(float *data, size_t sz){
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz); return false;}
    return true;
}

int main(){

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE*DSIZE];  // allocate space for data in host memory
  h_sums = new float[DSIZE]();
    
  for (int i = 0; i < DSIZE*DSIZE; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
    
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sums, DSIZE*sizeof(float)); // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
    
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  {  
    //cuda processing sequence step 1 is complete
    row_sums<<<(DSIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");
    //cuda processing sequence step 2 is complete
        
    // copy vector sums from device to host:
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
        
    //cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
        
    if (!validate(h_sums, DSIZE)) return -1; 
    printf("row sums correct!\n");
  }  

  {  
    cudaMemset(d_sums, 0, DSIZE*sizeof(float));
        
    column_sums<<<(DSIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");
    //cuda processing sequence step 2 is complete
        
    // copy vector sums from device to host:
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    //cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
        
    if (!validate(h_sums, DSIZE)) return -1; 
    printf("column sums correct!\n");
  }

  {     
    cudaMemset(d_sums, 0, DSIZE*sizeof(float));
        
    dim3 block_num(512, 1);
    dim3 block_size(32, 32);
    column_sums_blocks<<<block_num, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");
    //cuda processing sequence step 2 is complete
        
    // copy vector sums from device to host:
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    //cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
        
    if (!validate(h_sums, DSIZE)) return -1; 
    printf("column sums correct!\n");
  }

  return 0;
}
  