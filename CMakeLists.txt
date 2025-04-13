cmake_minimum_required(VERSION 3.10)
project(HelloCuda VERSION 1.0 LANGUAGES CXX CUDA)

# 设置程序依赖路径，此处需要根据实际生产环境进行修改
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)                               # cuda安装路径

# 设置cuda编译选项
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)          # 显式的指定cuda编译器
set(CMAKE_CUDA_ARCHITECTURES OFF)                                   # 让CMake和nvcc自动检测当前可用的GPU架构

# 设置项目文件路径
set(BIN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)                               # 编译输出结果
set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)                          # 外部头文件

# 设置子项目名称
set(TEST1 MultiMatrix)

# 设置项目依赖
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)        # 链接cuda头文件

# 设置CUDA编译
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)                                           # 告诉系统要使用CUDA语言，绝不可以省略
    message(STATUS "CUDA support enabled.")

    if (POLICY CMP0146)                                             # 忽略CMP0146警告
        cmake_policy(SET CMP0146 OLD)
    endif ()
    include(FindCUDA)
    set(CUDA_ARCH_LIST Auto CACHE STRING "List of CUDA architectures (e.g. Pascal, Volta, etc)\
     or compute capability version (6.1, 7.0, etc) to generate code for. Set to Auto for \
     automatic detection (default).")
    cuda_select_nvcc_arch_flags(CUDA_ARC_FLAGS ${CUDA_ARC_LIST})
    list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
else ()
    message(WARNING "CUDA Support disable.")
endif ()

# 添加编译选项 -Wno-deprecated-declarations
add_compile_options(-Wno-deprecated-declarations)

# 添加子目录
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/MultiMatrix)