# CUDA实现矩阵乘法及性能优化
## 简单矩阵乘法实现
一个简单的矩阵乘法实现如下：
$$
A \times B = C
$$
其中A的大小为$ M\times K$, B的大小为$ K\times N$, C的大小为$M\times N$,为了方便起见，以下推导令M=3, k=2, K=3。
### 线程组织
矩阵乘法的计算过程如下：
![alt text](../../images/image.png)
