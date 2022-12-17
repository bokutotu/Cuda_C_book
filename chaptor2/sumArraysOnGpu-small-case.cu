#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "../common.h"

__global__ void sumArrysOnGpu(float *deviceArrayA, float* deviceArrayB, float* deviceArrayC) {
  int idx = threadIdx.x;
  deviceArrayC[idx] = deviceArrayA[idx] + deviceArrayB[idx];
  return ;
}

void sumArrayOnCpu(float *hostArryA, float *hostArryB, float *hostArryC, int size) {
  for (int i = 0; i < size; i++ )
    hostArryC[i] = hostArryA[i] + hostArryB[i];
  return ;
}

int main() {
  CHECK(cudaSetDevice(0));
  int numElms = 32;
  int numBits = numElms * sizeof(float);
  float *hostArrayA, *hostArrayB, *hostArrayC, *resArray;

  hostArrayA = (float *)malloc(numBits);
  hostArrayB = (float *)malloc(numBits);
  hostArrayC = (float *)malloc(numBits);
  resArray   = (float *)malloc(numBits);

  initArray(hostArrayA, numElms);
  initArray(hostArrayB, numElms);

  sumArrayOnCpu(hostArrayA, hostArrayB, hostArrayC, numElms);

  float *deviceArrayA, *deviceArrayB, *deviceArrayC;

  CHECK(cudaMalloc(&deviceArrayA, numBits));
  CHECK(cudaMalloc(&deviceArrayB, numBits));
  CHECK(cudaMalloc(&deviceArrayC, numBits));

  CHECK(cudaMemcpy(deviceArrayA, hostArrayA, numBits, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceArrayB, hostArrayB, numBits, cudaMemcpyHostToDevice));

  sumArrysOnGpu <<< 1, numElms >>> (deviceArrayA, deviceArrayB, deviceArrayC);
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(resArray, deviceArrayC, numBits, cudaMemcpyDeviceToHost));
  checkResult(resArray, hostArrayC, numElms);
}
