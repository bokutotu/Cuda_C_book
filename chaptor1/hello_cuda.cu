#include <stdio.h>
#include "cuda_runtime.h"

__global__ void hello() {
  uint block_id = blockIdx.x;
  uint thread_id = threadIdx.x;
  printf("Hello from GPU block idx %d thread is %d\n", block_id, thread_id);
}

int main() {
  hello<<< 1, 10 >>>();
  cudaDeviceReset();
}
