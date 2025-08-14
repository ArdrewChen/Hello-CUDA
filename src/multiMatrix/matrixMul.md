# CUDA实现矩阵乘法及性能优化
## 简单矩阵乘法实现
一个简单的矩阵乘法实现如下：
$$
A \times B = C
$$
方便起见，以下均用方阵，具体展开如下：

```cpp
                        b00 b01 b02 b03
                        b10 b11 b12 b13
                        b20 b21 b22 b23
                        b30 b31 b32 b33

a00 a01 a02 a03         c00 c01 c02 c03
a10 a11 a12 a13         c10 c11 c12 c13    
a20 a21 a22 a23         c20 c21 c22 c23
a30 a31 a32 a33         c30 c31 c32 c33

c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33

index = y * size + x
step 0 -> 3
a_index = y * size + step;
b_index = step * size + x;
```

具体来说，思路如下: 矩阵对应位置进行乘加运算，每次双层循环计算一个位置的值。

## cpu矩阵乘法实现

tmp用于存储乘加运算产生的中间结果

```cpp
  for(int i=0; i<matrix_size; i++){
    for(int j=0; j<matrix_size; j==){
      int tmp = 0;
      for(int step=0; step<matrix_size; step++){
        tmp = tmp + h_a[i*matrix_size+step] * h_b[j+step*matrix*step];
      }
      h_c[j+i*matrix_size] = tmp;
    }
  }
```

## gpu实现矩阵乘法及性能优化

### 基本思路

gpu的并行优化点为矩阵双层的循环，使用block和thread index代替，保持每一个线程计算结果矩阵的一个结果，少去两个循环，保持性能优化。

### cuda中矩阵行列对应关系

这其中一个值得注意的点是：

blockDim.x和blockDim.y哪个对应的是矩阵的行哪个是矩阵的列？

这没有硬性规定，但为了获得最佳性能，有一个非常强烈的约定。

**blockDim.x, threadIdx.x 和 blockIdx.x 通常对应 矩阵的列 (Column)。**

**blockDim.y, threadIdx.y 和 blockIdx.y 通常对应 矩阵的行 (Row)。**

原因：

性能核心：合并内存访问 (Coalesced Memory Access)
为了理解这个约定，需要知道两个关键概念：

行主序（Row-Major Order）: 在C/C++中，二维矩阵在内存中是按“行”连续存储的。也就是说，matrix[0][0]的旁边是matrix[0][1]，然后是matrix[0][2]，以此类推。同一行的元素在内存地址上是紧挨着的。

内存地址: ... [row 0, col 0] [row 0, col 1] [row 0, col 2] ... [row 1, col 0] ...

合并内存访问（Coalesced Memory Access）: 这是CUDA性能优化的黄金法则。当一个Warp（一个包含32个线程的执行单元）中的线程同时访问连续的内存地址时，GPU可以将这32次独立的访问合并成一次或几次大的内存事务。这极大地提高了内存带宽的利用率，是获得高性能的关键。

现在，让我们把这两个概念结合起来。

线程在一个Warp内是按照它们的threadIdx.x值（以及threadIdx.y和threadIdx.z）来组织的。threadIdx.x连续的线程（例如，ID从0到31）最有可能在同一个Warp中。

当我们遵循"x -> 列, y -> 行"的约定:

考虑一个Warp中的32个线程，它们的threadIdx.x分别是0, 1, 2, ..., 31，并且它们的threadIdx.y是相同的。

它们计算出的列索引col将是连续的（例如 base_col + 0, base_col + 1, ..., base_col + 31）。

它们计算出的行索引row是相同的。

当它们去访问内存matrix[row * width + col]时，它们访问的内存地址是 base_address + 0, base_address + 1, base_address + 2, ...。

这正是连续的内存地址！ GPU硬件可以完美地将这些访问合并成一次高效的事务。

如果我们颠倒约定:

Warp中32个线程的threadIdx.x是0, 1, 2, ..., 31，但我们把它们映射到行。

那么它们的行索引row是连续的，而列索引col是相同的。

当它们访问内存matrix[row * width + col]时，它们访问的地址是：

线程0: (base_row + 0) * width + col

线程1: (base_row + 1) * width + col

线程2: (base_row + 2) * width + col

...

这些内存地址之间相隔了width个字节。它们不是连续的！这种访问模式叫做跨步访问（Strided Access）。

硬件无法合并这些访问，必须为每个线程都执行一次独立的、低效的内存事务，导致性能急剧下降。

### 线程组织

```cpp
int y = blockDim.y * blockIdx.y + threadIdx.y;     // 矩阵的行位置
int x = blockDim.x * blockIdx.x + threadIdx.x;     // 矩阵的列位置
...
```
### 利用shared memory进行矩阵乘优化

每次求取结果矩阵每一行或者每一列的数据，都要读取A或B矩阵的某一行和某一列读取多次，造成内存多次读取。

因此可以考虑的优化点是，将这些数据从gloabl memory放到shared memeory中，读取的时候可以直接从shared memory读取到寄存器

具体来说， 将一个block中的数据，将需要共用的数据加载到shared memory，为了防止shared memory大小溢出，还需要考虑分块

```cpp
                        b00 b01 b02 b03
                        b10 b11 b12 b13
                        b20 b21 b22 b23
                        b30 b31 b32 b33

a00 a01 a02 a03         c00 c01 c02 c03
a10 a11 a12 a13         c10 c11 c12 c13     block(1, 0) -> shared memory
a20 a21 a22 a23         c20 c21 c22 c23     c20 c21
a30 a31 a32 a33         c30 c31 c32 c33     c30 c31

                             b00 b01->  sub_b_step_0
                             b10 b11

                             b20 b21->  sub_b_step_1
                             b30 b31
sub_a_step_0 sub_a_step_1    sub_c
a20 a21      a22 a23         c20 c21
a30 a31      a32 a33         c30 c31

sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;
```

实际操作中，可以考虑的进行如下操作

```cpp
for(int step =0; step < N/block_size; step++ )
     load sub_a_step to shared memory;
     load sub_b_step to shared memory;
     tmp += sub_a_step_on_sharedmemory * sub_b_step_on_sharedmemory;
sub_c = tmp;
```
