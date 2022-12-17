#include <stdio.h>
#include <sys/time.h>

#include "cuda_runtime.h"

#include "../common.h"

__global__ void sumArrysOnGpu(float *deviceArrayA, float* deviceArrayB, float* deviceArrayC, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  deviceArrayC[idx] = deviceArrayA[idx] + deviceArrayB[idx];
  return ;
}

void sumArrayOnCpu(float *hostArryA, float *hostArryB, float *hostArryC, int size) {
  for (int i = 0; i < size; i++ )
    hostArryC[i] = hostArryA[i] + hostArryB[i];
  return ;
}

int main() {
  setUpDevice(0);

  int nElms = 1 << 24;
  printf("Vector size: %d", nElms);

  int nBites = nElms * sizeof(float);

  float *deviceArrayA, *deviceArrayB, *deviceArrayC, 
        *hostArrayA, *hostArrayB, *hostArrayC,
        *resArray;

  hostArrayA = (float *)malloc(nBites);
  hostArrayB = (float *)malloc(nBites);
  hostArrayC = (float *)malloc(nBites);
  resArray   = (float *)malloc(nBites);

  CHECK(cudaMalloc(&deviceArrayA, nBites));
  CHECK(cudaMalloc(&deviceArrayB, nBites));
  CHECK(cudaMalloc(&deviceArrayC, nBites));

  initArray(hostArrayA, nElms);
  initArray(hostArrayB, nElms);

  sumArrayOnCpu(hostArrayA, hostArrayB, hostArrayC, nElms);

  CHECK(cudaMemcpy(deviceArrayA, hostArrayA, nBites, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceArrayB, hostArrayB, nBites, cudaMemcpyHostToDevice));

  double cpuStart = cpuSecond();
  sumArrayOnCpu(hostArrayA, hostArrayB, hostArrayC, nElms);
  double cpuEnd = cpuSecond();
  printf("Cpu Sum second is: %f\n", cpuEnd - cpuStart);

  int iLen = 1024;
  dim3 block(iLen);
  dim3 grid((nElms + block.x - 1)/block.x);

  double gpuStart = cpuSecond();
  sumArrysOnGpu <<< grid, block >>> (deviceArrayA, deviceArrayB, deviceArrayC, nElms);
  printf("sumArrayOnGpu <<< %d, %d >>> (~~~)", grid.x, block.x);
  double gpuEnd = cpuSecond();
  printf("gpu time: %f\n", gpuEnd - gpuStart);

  CHECK(cudaMemcpy(resArray, deviceArrayC, nBites, cudaMemcpyDeviceToHost));
  checkResult(resArray, hostArrayC, nElms);
}
