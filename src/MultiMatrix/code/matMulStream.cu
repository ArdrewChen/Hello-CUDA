#include <cuda_runtime.h>
#include "MultiMatrix.h"

__global__ void matMulKernel(const float *A, const float *B, float *C, int M, int K, int N, int rowOffset) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = rowOffset + blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = sum;
}

void matMulMarixStream(dim3 gridDim, dim3 blockDim, const float *d_A, const float *d_B, float *d_C, int M, int K, int N, cudaStream_t stream){
    matMulKernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, K, N, N);
}