#include <cuda.h>

#define CUDACall(ans) { CudaAssert((ans), __FILE__, __LINE__); }
inline void CudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDACheck %s:%d:\n\t%s\n", file, line, cudaGetErrorString(code));
      if (abort) exit(code);
   }
}