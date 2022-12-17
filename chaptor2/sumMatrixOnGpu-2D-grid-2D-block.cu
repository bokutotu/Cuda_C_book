#include <stdio.h>
#include <sys/time.h>

#include "cuda_runtime.h"

#include "../common.h"

__global__ void sumArrysOnGpu(float *deviceArrayA, float* deviceArrayB, float* deviceArrayC, int size, int nx, int ny) {
  int idxX = threadIdx.x + blockDim.x * blockIdx.x;
  int idxY = threadIdx.y + blockDim.y * blockIdx.y;

  int idx = idxY * ny + idxX;

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

  int nx = 1 << 14;
  int ny = 1 << 14;
  int nElms = nx * ny;
  printf("Vector size: %d", nElms);

  int nBites = nx * ny * sizeof(float);

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

  dim3 block(32, 16);
  dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1) / block.y);

  double gpuStart = cpuSecond();
  sumArrysOnGpu <<< grid, block >>> (deviceArrayA, deviceArrayB, deviceArrayC, nElms, nx, ny);
  printf("sumArrayOnGpu <<< (%d, %d), (%d, %d) >>> (~~~)", grid.x, grid.y, block.x, block.y);
  double gpuEnd = cpuSecond();
  printf("gpu time: %f\n", gpuEnd - gpuStart);

  CHECK(cudaMemcpy(resArray, deviceArrayC, nBites, cudaMemcpyDeviceToHost));
  checkResult(resArray, hostArrayC, nElms);
}
