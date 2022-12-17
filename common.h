#include "cuda_runtime.h"

#define CHECK(call)                                                            \
{                                                                              \
  const cudaError_t error = call;                                              \
  if (error != cudaSuccess)                                                    \
  {                                                                            \
     printf("Error: %s:%d,  ", __FILE__, __LINE__);                            \
     printf("code:%d reason: %s\n", error, cudaGetErrorString(error));         \
     exit(1);                                                                  \
  }                                                                            \
}                                                                              \

void checkResult(float *hostRef, float *gpuRef, int size) {
  double epsilon = 1e-3;

  for (int i = 0; i < size; i++ ) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      printf("arrays do not match\n");
      return ;
    }
  }

  printf("arrays match\n");
  return ;
}

void initArray(float *hostArray, int size) {
  srand((unsigned int)time(NULL));
  for (int i = 0; i < size; i++ )
    hostArray[i] = (float)rand()/(float)RAND_MAX;
  return ;
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void setUpDevice(int devId) {
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, devId));
  printf("Uing Device %d: %s\n", devId, deviceProp.name);
  CHECK(cudaSetDevice(devId));
}
