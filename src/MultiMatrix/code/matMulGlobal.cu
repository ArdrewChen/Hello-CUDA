#include <cuda_runtime.h>

// CUDA kernel：每个线程负责计算 C 中的一个元素
__global__ void matMulGlobalKernel(const float *A, const float *B, float *C, int M, int K, int N) {
    // 当前线程计算的 C 元素索引（row, col）
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列

    // 线程索引可能超出矩阵范围，需要判断
    if (row < M && col < N) {
        float sum = 0.0f;

        // 计算 A 的第 row 行 和 B 的第 col 列 的点积
        for (int i = 0; i < K; ++i) {
            float a = A[row * K + i];     // A[row][i]
            float b = B[i * N + col];     // B[i][col]
            sum += a * b;
        }

        // 将计算结果写回 C 矩阵
        C[row * N + col] = sum;
    }
}

void matMulGlobal(dim3 gridDim, dim3 blockDim, const float *d_A, const float *d_B, float *d_C, int M, int K, int N){
       // 调用 kernel 执行矩阵乘法
       matMulGlobalKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
}