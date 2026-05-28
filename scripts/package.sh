#!/bin/bash
# Embedding & Rerank Server GPU 打包脚本（Mac/Linux）
set -e

OUT="deploy"
mkdir -p "$OUT"

echo "=== Embedding & Rerank Server GPU 打包 ==="

# 构建镜像
echo "[1] 构建 GPU 镜像..."
docker build -t embedding-rerank-server:latest .

# 导出镜像
echo "[2] 导出镜像..."
docker save embedding-rerank-server:latest -o "$OUT/embedding-rerank-server.tar"

# 复制配置文件
echo "[3] 复制配置文件..."
cp docker-compose.yml "$OUT/docker-compose.yml"
cp config.yaml "$OUT/config.yaml"

echo "=== 完成 ==="
du -sh "$OUT/"
echo ""
echo "部署步骤:"
echo "  1. scp deploy/* user@gpu-server:/weique/jmarag/embedding-rerank-server/"
echo "  2. docker load -i embedding-rerank-server.tar"
echo "  3. docker compose up -d"
