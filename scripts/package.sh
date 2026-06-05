#!/bin/bash
# ============================================================
# Embedding & Rerank Server v2.0 打包脚本（AMD64 + GPU）
# ============================================================
# 构建自适应引擎镜像并导出，用于离线部署到 GPU 服务器
# 特性：零配置启动，自动探测硬件
set -e

VERSION="2.1.0"
IMAGE_NAME="embedding-rerank-server"
IMAGE_TAG="latest"
OUT="deploy"

echo "=== Embedding & Rerank Server v${VERSION} GPU 打包 ==="
echo "  镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  平台: linux/amd64 (GPU)"
echo ""

# 清理旧输出
rm -rf "$OUT"
mkdir -p "$OUT"

# [1] 构建镜像
echo "[1/4] 构建 GPU 镜像..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo "  构建成功"

# [2] 导出镜像
echo "[2/4] 导出镜像 tar..."
docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "$OUT/${IMAGE_NAME}.tar"
echo "  导出完成"

# [3] 复制配置和编排文件
echo "[3/4] 复制部署文件..."
cp docker-compose.yml "$OUT/docker-compose.yml"
cp config.yaml "$OUT/config.yaml"
echo "  config.yaml (所有参数默认 auto)"
echo "  docker-compose.yml (零配置)"

# [4] 生成部署说明
echo "[4/4] 生成部署说明..."
cat > "$OUT/README.txt" << EOF
# Embedding & Rerank Server v${VERSION} (GPU)
# 部署日期: $(date '+%Y-%m-%d %H:%M')
#
# 部署步骤:
#   1. docker load -i ${IMAGE_NAME}.tar
#   2. docker compose up -d
#   3. curl http://localhost:7997/health
#
# 核心特性:
#   - 零配置启动，自动探测 GPU 显存并配置最优参数
#   - Token-level batching，按 token 数合批
#   - 优先级队列，查询优先于入库
#   - 向量缓存 + batch 内去重 + 动态 max_length
#   - /metrics Prometheus 监控端点
#   - 队列过深返回 429 (背压机制)
#
# 如需手动覆盖参数，编辑 config.yaml 中对应值即可
EOF

# 完成
echo ""
echo "=== 打包完成 ==="
du -sh "$OUT/"
echo ""
echo "部署步骤:"
echo "  1. scp -r deploy/* user@gpu-server:/opt/embedding-rerank-server/"
echo "  2. ssh gpu-server"
echo "  3. cd /opt/embedding-rerank-server"
echo "  4. docker load -i ${IMAGE_NAME}.tar"
echo "  5. rm ${IMAGE_NAME}.tar"
echo "  6. docker compose up -d"
echo "  7. curl http://localhost:7997/health"
echo "  8. curl http://localhost:7997/metrics  # Prometheus 监控"
