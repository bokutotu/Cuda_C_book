#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void sumArraysOnHost(int* h_A, int *h_B, int* h_C, int size) {
  for (int i = 0; i < size; i++)
    h_C[i] = h_A[i] + h_B[i];
  return ;
}

void initData(int* hostArray, int size) {
  time_t t;
  srand((unsigned int) time(&t));

  for (int i = 0; i < size; i++ )
    hostArray[i] = (int)(rand());
  return ;
}

int main() {
  int *h_A, *h_B, *h_C;
  int numElms = 1 << 10;
  int numBites = numElms * sizeof(int);
  h_A = (int*)malloc(numBites);
  h_B = (int*)malloc(numBites);
  h_C = (int*)malloc(numBites);

  initData(h_A, numElms);
  initData(h_B, numElms);

  sumArraysOnHost(h_A, h_B, h_C, numElms);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
