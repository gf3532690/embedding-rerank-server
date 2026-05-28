#!/bin/bash
# Embedding & Rerank Server CPU 打包脚本（Mac/Linux）
# Mac ARM64 原生构建，无需 QEMU 模拟，速度快
set -e

OUT="deploy-cpu"
mkdir -p "$OUT"

echo "=== Embedding & Rerank Server CPU 打包 ==="

# 构建镜像
echo "[1] 构建 CPU 镜像..."
docker build -f Dockerfile.cpu -t embedding-rerank-server:cpu .

# 导出镜像
echo "[2] 导出镜像..."
docker save embedding-rerank-server:cpu -o "$OUT/embedding-rerank-server-cpu.tar"

# 复制配置文件
echo "[3] 复制配置文件..."
cp docker-compose-cpu.yml "$OUT/docker-compose.yml"
cp config-cpu.yaml "$OUT/config-cpu.yaml"

echo "=== 完成 ==="
du -sh "$OUT/"
echo ""
echo "部署步骤:"
echo "  1. scp deploy-cpu/* user@server:/path/to/embedding-rerank-server/"
echo "  2. docker load -i embedding-rerank-server-cpu.tar"
echo "  3. docker compose up -d"
