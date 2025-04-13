#include <cuda_runtime.h>
#include "MultiMatrix.h"

// 使用共享内存优化的矩阵乘法 kernel
__global__ void matMulSharedKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    // block 内线程的行列索引
    int tx = threadIdx.x;  // 列
    int ty = threadIdx.y;  // 行

    // 确定 C 中的元素 (row, col)
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // 用于缓存 A 和 B 的 tile（共享内存）
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // 以 tile 为单位迭代 A 和 B
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // 每个线程加载 A 和 B 各自 tile 的一个元素到共享内存
        if (row < M && t * BLOCK_SIZE + tx < K)
            tile_A[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        else
            tile_A[ty][tx] = 0.0f;

        if (col < N && t * BLOCK_SIZE + ty < K)
            tile_B[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        else
            tile_B[ty][tx] = 0.0f;

        __syncthreads();  // 确保共享内存 tile_A、tile_B 装载完毕

        // 对 tile 执行局部乘加
        for (int i = 0; i < BLOCK_SIZE; ++i)
            sum += tile_A[ty][i] * tile_B[i][tx];

        __syncthreads();  // 准备下一轮 tile 加载
    }

    // 写入结果 C
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matMulShared(dim3 gridDim, dim3 blockDim, const float *d_A, const float *d_B, float *d_C, int M, int K, int N){
    matMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
}