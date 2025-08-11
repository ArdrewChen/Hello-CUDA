/***
 * 矩阵乘法CPU和GPU实现
 */
#include <iostream>
#include <math.h>

#define BLOCK_SIZE 16

__global__ void gpu_mul_matrix(int *d_a, int *d_b, int *d_c, int matrix_size)
{
  int y = blockDim.y * blockIdx.y + threadIdx.y; // 矩阵的行位置
  int x = blockDim.x * blockIdx.x + threadIdx.x; // 矩阵的列位置
  if (y < matrix_size && x < matrix_size)
  {
    int tmp = 0;
    for (int step = 0; step < matrix_size; step++)
    {
      tmp = tmp + d_a[y * matrix_size + step] * d_b[step * matrix_size + x];
    }
    d_c[y * matrix_size + x] = tmp;
  }
}

void cpu_mul_matrix(int *h_a, int *h_b, int *h_cc, int matrix_size)
{
  for (int i = 0; i < matrix_size; i++)
  {
    for (int j = 0; j < matrix_size; j++)
    {
      int tmp = 0;
      for (int step = 0; step < matrix_size; step++)
      {
        tmp = tmp + h_a[i * matrix_size + step] * h_b[j + step * matrix_size];
      }
      h_cc[j + i * matrix_size] = tmp;
    }
  }
}

int main()
{
  int matrix_size = 1000;
  int *h_a, *h_b, *h_c, *h_cc;
  cudaMallocHost((void **)&h_a, sizeof(int) * matrix_size * matrix_size);
  cudaMallocHost((void **)&h_b, sizeof(int) * matrix_size * matrix_size);
  cudaMallocHost((void **)&h_c, sizeof(int) * matrix_size * matrix_size);
  cudaMallocHost((void **)&h_cc, sizeof(int) * matrix_size * matrix_size);
  for (int y = 0; y < matrix_size; y++)
  {
    for (int x = 0; x < matrix_size; x++)
    {
      h_a[y * matrix_size + x] = rand() % 1024;
      h_b[y * matrix_size + x] = rand() % 1024;
    }
  }
  int *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, sizeof(int) * matrix_size * matrix_size);
  cudaMalloc((void **)&d_b, sizeof(int) * matrix_size * matrix_size);
  cudaMalloc((void **)&d_c, sizeof(int) * matrix_size * matrix_size);

  cudaMemcpy(d_a, h_a, sizeof(int) * matrix_size * matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * matrix_size * matrix_size, cudaMemcpyHostToDevice);

  unsigned int grid_rows = (matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  dim3 dimGrid(grid_rows, grid_cols);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  gpu_mul_matrix<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, matrix_size);
  cudaMemcpy(h_c, d_c, sizeof(int) * matrix_size * matrix_size, cudaMemcpyDeviceToHost);

  cpu_mul_matrix(h_a, h_b, h_cc, matrix_size);

  bool result_flag = true;
  for (int i = 0; i < matrix_size; i++)
  {
    for (int j = 0; j < matrix_size; j++)
    {
      if (fabs(h_cc[i * matrix_size + j] - h_c[i * matrix_size + j]) > 0.000001)
      {
        std::cout << "error position:" << i * matrix_size + j << std::endl;
        result_flag = false;
      }
    }
  }
  std::cout << "Result: " << result_flag << std::endl;
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFreeHost(h_cc);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}