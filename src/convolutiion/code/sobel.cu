#include <opencv2/opencv.hpp>
#include <iostream>
#include "error.cuh"

__global__ void sobel_gpu(unsigned char *in_gpu, unsigned char *out_gpu, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y; // 对应输入输出矩阵的大小，每个线程对应一个元素

    int index = y * width + x;

    int gx = 0;
    int gy = 0;

    // 对应卷积核的计算的相对位置
    unsigned char x0, x1, x2, x3, x5, x6, x7, x8; // x4在两处卷积核的结果都是0

    // 卷积边缘不处理
    if (x > 0 && x < height - 1 && y > 0 && y < width - 1)
    {
        x0 = in[(y - 1) * width + x - 1];
        x1 = in[(y - 1) * width + x];
        x2 = in[(y - 1) * width + x + 1];
        x3 = in[y * width + x - 1];
        x5 = in[y * width + x + 1];
        x6 = in[(y + 1) * width + x - 1];
        x7 = in[(y + 1) * width + x];
        x8 = in[(y + 1) * width + x + 1];

        gx = x0 - x3 + 2 * x3 - 2 * x5 + x6 - x8;
        gy = x0 + 2 * x1 + x2 - x6 - 2 * x7 - x8;

        out_gpu[index] = (abs(gx) + abs(gy)) / 2;
    }
}

int main()
{
    Mat img = imread("../../../images/image.png");
    int height = img.rows;
    int width = img.cols;

    Mat gaussImg;
    GaussianBlur(img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    Mat dst_gpu(height, weight, CV_8UC1, Scalar(0));

    int memsize = height * widght * sizeof(unsigned char);

    unsigned char *in_gpu;
    unsigned char *out_gpu;

    cudaMalloc(in_gpu, memsize);
    cudaMalloc(out_gpu, memsize);

    dim3 thread_per_block(32, 32);
    dim3 block_per_grid((height + thread_per_block.x - 1) / thread_per_block.x, (width + thread_per_block.y - 1) / thread_per_block.y);

    cudaMemcpy(in_gpu, gaussImg.data, memsize, cudaMemcpyHostToDevice);

    sobel_gpu<<<block_per_grid, thread_per_block>>>(in_gpu, out_gpu, height, width);

    cudaMemcpy(dst_gpu.data, out_gpu, memsize, cudaMemcpyDeviceToDevice);

    imwrite("save.png", dst_gpu);

    cudaFree(in_gpu);
    cudaFree(out_gpu);
    return 0;
}