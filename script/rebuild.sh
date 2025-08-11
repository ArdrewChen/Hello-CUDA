#!/bin/bash
# 文件名：rebuild.sh
# 用途：安全清理并重建 build 目录，并执行cmake编译

# 启用严格错误检查
# set -euo pipefail

# 若build文件存在，删除build目录
if [ -d "../build" ]; then
    echo "→ 删除旧 build 目录"
    rm -rf ../build
fi

# 创建新的build目录
echo "→ 创建新 build 目录"
mkdir ../build

# 进入build目录
echo "→ 进入 build 目录"
cd ../build

# 执行cmake
echo "→ 执行cmake"
cmake ..
