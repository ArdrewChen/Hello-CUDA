#include <iostream>
#include <cuda_runtime.h>
#include "MultiMatrix.h"

int main()
{
    // 定义任意尺寸（不一定方阵）
    int M = 3; // A 的行数
    int K = 2; // A 的列数 / B 的行数
    int N = 3; // B 的列数

    // 分配主机内存（Host memory）
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // 初始化矩阵 A (M x K)
    printf("Matrix A:\n");
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = 1.0f; // 赋值为 1.0
        printf("%f ", h_A[i]);
        if ((i + 1) % K == 0)
            printf("\n");
    }

    // 初始化矩阵 B (K x N)
    printf("Matrix B:\n");
    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = 1.0f; // 赋值为 1.0
        printf("%f ", h_B[i]);
        if ((i + 1) % N == 0)
            printf("\n");
    }

    // 分配设备内存（Device memory）
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    // 定义 block 和 grid 的尺寸
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);           // 每个 block 16x16 个线程
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,  // 处理 N 列
                 (M + blockDim.y - 1) / blockDim.y); // 处理 M 行

    // 使用全局内存完成矩阵相乘
    matMulGlobal(gridDim, blockDim, d_A, d_B, d_C, M, K, N);

    // 使用共享内存完成矩阵相乘
    matMulShared(gridDim, blockDim, d_A, d_B, d_C, M, K, N);

    // 等待设备完成
    cudaDeviceSynchronize();

    // 将结果拷贝回主机
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // // 使用多流并发完成矩阵乘法
    // int rowsPerStream = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    // cudaStream_t streams[NUM_STREAMS];

    // // 为每个流分配内存
    // float *d_As[NUM_STREAMS], *d_Cs[NUM_STREAMS];

    // for (int i = 0; i < NUM_STREAMS; i++)
    // {
    //     int rows = (i == NUM_STREAMS - 1) ? (M - rowsPerStream * i) : rowsPerStream;
    //     size_t size_A_part = rows * K * sizeof(float);
    //     size_t size_C_part = rows * N * sizeof(float);

    //     cudaMalloc((void **)&d_As[i], size_A_part);
    //     cudaMalloc((void **)&d_Cs[i], size_C_part);
    //     cudaStreamCreate(&streams[i]);
    //     // 异步传输 A 分块数据
    //     cudaMemcpyAsync(d_As[i], h_A + i * rowsPerStream * K, size_A_part, cudaMemcpyHostToDevice, streams[i]);
    //     dim3 gridDims((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //     matMulMarixStream(gridDims, blockDim, d_As[i], d_B, d_Cs[i], i * rowsPerStream, K, N, streams[i]);

    //     // 异步传回结果 C 分块
    //     cudaMemcpyAsync(h_C + i * rowsPerStream * N, d_Cs[i], size_C_part, cudaMemcpyDeviceToHost, streams[i]);
    // }

    // // 等待所有流完成
    // for (int i = 0; i < NUM_STREAMS; ++i)
    // {
    //     cudaStreamSynchronize(streams[i]);
    //     cudaStreamDestroy(streams[i]);
    //     cudaFree(d_As[i]);
    //     cudaFree(d_Cs[i]);
    // }

    // 打印结果矩阵 C (M x N)
    printf("Matrix C (Result):\n");
    for (int i = 0; i < M * N; ++i)
    {
        printf("%f ", h_C[i]);
        if ((i + 1) % N == 0)
            printf("\n");
    }

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}