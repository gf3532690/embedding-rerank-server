#!/bin/bash
# ============================================================
# Embedding & Rerank Server v2.0 打包脚本（CPU 模式）
# ============================================================
# 构建 CPU 版镜像（ONNX Runtime），用于无 GPU 的服务器
# 支持 AMD64 和 ARM64 平台
set -e

VERSION="2.0.0"
IMAGE_NAME="embedding-rerank-server"
IMAGE_TAG="cpu"
OUT="deploy-cpu"

# 目标平台（默认当前平台，可通过参数覆盖）
PLATFORM="${1:-linux/$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')}"

echo "=== Embedding & Rerank Server v${VERSION} CPU 打包 ==="
echo "  镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  平台: ${PLATFORM}"
echo ""

# 清理旧输出
rm -rf "$OUT"
mkdir -p "$OUT"

# [1] 构建镜像
echo "[1/4] 构建 CPU 镜像 (平台: ${PLATFORM})..."
docker build --platform "${PLATFORM}" -f Dockerfile.cpu -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo "  构建成功"

# [2] 导出镜像
echo "[2/4] 导出镜像 tar..."
docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "$OUT/${IMAGE_NAME}-cpu.tar"
echo "  导出完成"

# [3] 复制配置和编排文件
echo "[3/4] 复制部署文件..."
cp docker-compose-cpu.yml "$OUT/docker-compose.yml"
cp config.yaml "$OUT/config.yaml"
echo "  config-cpu.yaml (CPU 优化配置)"
echo "  docker-compose.yml (零配置)"

# [4] 生成部署说明
echo "[4/4] 生成部署说明..."
cat > "$OUT/README.txt" << EOF
# Embedding & Rerank Server v${VERSION} (CPU)
# 部署日期: $(date '+%Y-%m-%d %H:%M')
# 目标平台: ${PLATFORM}
#
# 部署步骤:
#   1. docker load -i ${IMAGE_NAME}-cpu.tar
#   2. docker compose up -d
#   3. curl http://localhost:7997/health
#
# v2.0 特性:
#   - 自动探测 CPU 核数，配置最优线程数和并发
#   - Token-level batching，按 token 数合批
#   - ONNX Runtime 推理加速
#   - /metrics Prometheus 监控端点
#   - 队列过深返回 429 (背压机制)
#
# CPU 模式注意事项:
#   - 确保模型目录有 onnx/ 子目录
#   - OMP_NUM_THREADS 自动设置，无需手动配置
#   - 如需覆盖参数，编辑 config-cpu.yaml
EOF

# 完成
echo ""
echo "=== 打包完成 ==="
du -sh "$OUT/"
echo ""
echo "部署步骤:"
echo "  1. scp -r deploy-cpu/* user@server:/opt/embedding-rerank-server/"
echo "  2. ssh server"
echo "  3. cd /opt/embedding-rerank-server"
echo "  4. docker load -i ${IMAGE_NAME}-cpu.tar"
echo "  5. rm ${IMAGE_NAME}-cpu.tar"
echo "  6. docker compose up -d"
echo "  7. curl http://localhost:7997/health"
echo ""
echo "提示: 默认构建当前平台架构，如需指定平台:"
echo "  ./scripts/package-cpu.sh linux/arm64"
echo "  ./scripts/package-cpu.sh linux/amd64"
