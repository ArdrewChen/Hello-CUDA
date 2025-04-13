#include <cuda_runtime.h>
#define BLOCK_SIZE 16
#define NUM_STREAMS 4

/**
 * \brief 使用全局内存进行矩阵乘法计算
 * \param gridDim grid尺寸
 * \param blockDim block尺寸
 * \param d_A 矩阵A
 * \param d_B 矩阵B
 * \param d_C 矩阵C
 * \param M 行数
 * \param K 矩阵A的列数，矩阵B的行数
 * \param N 矩阵B的列数，矩阵C的列数
 * \return void
* \note 矩阵乘法计算公式：Cij = sum(Aik * Bkj)
 */
void matMulGlobal(dim3 gridDim, dim3 blockDim, const float *d_A, const float *d_B, float *d_C, int M, int K, int N);

/**
 * \brief 使用共享内存矩阵乘法计算
 * \param gridDim 网格尺寸
 * \param blockDim block尺寸
 * \param d_A 矩阵A
 * \param d_B 矩阵B
 * \param d_C 矩阵C
 * \param M 行数
 * \param K 矩阵A的列数，矩阵B的行数
 * \param N 矩阵B的列数，矩阵C的列数
 * \return void
* \note 矩阵乘法计算公式：Cij = sum()
 */
void matMulShared(dim3 gridDim, dim3 blockDim, const float *d_A, const float *d_B, float *d_C, int M, int K, int N);

/**
 * \brief 使用多流并发进行矩阵乘法计算
 * \param gridDim grid尺寸
 * \param blockDim block尺寸
 * \param d_A 矩阵A
 * \param d_B 矩阵B
 * \param d_C 矩阵C
 * \param M 行数
 * \param K 矩阵A的列数，矩阵B的行数
 * \param N 矩阵B的列数，矩阵C的列数
 * \param stream 流 
 * \return void
*\note 矩阵乘法计算公式：Cij = sum()
 */
void matMulMarixStream(dim3 gridDim, dim3 blockDim, const float *d_A, const float *d_B, float *d_C, int M, int K, int N, cudaStream_t stream);