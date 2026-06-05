# ============================================================
# Embedding & Rerank Server v2.0 打包脚本（CPU 模式）
# ============================================================
# 构建 CPU 版镜像（ONNX Runtime），用于无 GPU 的服务器
# 支持 AMD64 和 ARM64 平台
$ErrorActionPreference = "Stop"

$VERSION = "2.1.0"
$IMAGE_NAME = "embedding-rerank-server"
$IMAGE_TAG = "cpu"
$OUT = "deploy-cpu"

# 目标平台（默认 ARM64，可通过参数覆盖）
$PLATFORM = if ($args[0]) { $args[0] } else { "linux/arm64" }

Write-Host "=== Embedding & Rerank Server v$VERSION CPU 打包 ===" -ForegroundColor Green
Write-Host "  镜像: ${IMAGE_NAME}:${IMAGE_TAG}" -ForegroundColor Gray
Write-Host "  平台: $PLATFORM" -ForegroundColor Gray
Write-Host ""

# 清理旧输出
if (Test-Path $OUT) {
    Remove-Item -Recurse -Force $OUT
}
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# [1] 构建镜像
Write-Host "[1/4] 构建 CPU 镜像 (平台: $PLATFORM)..." -ForegroundColor Cyan
docker build --platform $PLATFORM -f Dockerfile.cpu -t "${IMAGE_NAME}:${IMAGE_TAG}" .
if ($LASTEXITCODE -ne 0) {
    Write-Host "构建失败！" -ForegroundColor Red
    Write-Host "  提示: ARM64 跨平台构建需要 docker buildx" -ForegroundColor Yellow
    exit 1
}
Write-Host "  构建成功" -ForegroundColor Green

# [2] 导出镜像
Write-Host "[2/4] 导出镜像 tar..." -ForegroundColor Cyan
docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "$OUT\${IMAGE_NAME}-cpu.tar"
Write-Host "  导出完成" -ForegroundColor Green

# [3] 复制配置和编排文件
Write-Host "[3/4] 复制部署文件..." -ForegroundColor Cyan
Copy-Item docker-compose-cpu.yml "$OUT\docker-compose.yml"
Copy-Item config.yaml "$OUT\config.yaml"
Write-Host "  config.yaml (自适应配置，CPU/GPU 通用)" -ForegroundColor Gray
Write-Host "  docker-compose.yml (零配置)" -ForegroundColor Gray

# [4] 生成部署说明
Write-Host "[4/4] 生成部署说明..." -ForegroundColor Cyan
$deployNote = @"
# Embedding & Rerank Server v$VERSION (CPU)
# 部署日期: $(Get-Date -Format "yyyy-MM-dd HH:mm")
# 目标平台: $PLATFORM
#
# 部署步骤:
#   1. docker load -i ${IMAGE_NAME}-cpu.tar
#   2. docker compose up -d
#   3. curl http://localhost:7997/health
#
# 核心特性:
#   - 自动探测 CPU 核数，配置最优线程数和并发
#   - Token-level batching，按 token 数合批
#   - ONNX Runtime O3 优化 + 纯 numpy 后处理
#   - 向量缓存 + batch 内去重 + 动态 max_length
#   - /metrics Prometheus 监控端点
#   - 队列过深返回 429 (背压机制)
#
# CPU 模式注意事项:
#   - 确保模型目录有 onnx/ 子目录
#   - OMP_NUM_THREADS 自动设置，无需手动配置
#   - 如需覆盖参数，编辑 config.yaml
"@
$deployNote | Out-File -Encoding utf8 "$OUT\README.txt"

# 完成
Write-Host ""
Write-Host "=== 打包完成 ===" -ForegroundColor Green
$size = [math]::Round((Get-ChildItem -Recurse $OUT | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
Write-Host "  输出目录: $OUT\" -ForegroundColor White
Write-Host "  总大小: $size GB" -ForegroundColor White
Write-Host ""
Write-Host "部署步骤:" -ForegroundColor Yellow
Write-Host "  1. scp -r deploy-cpu/* user@server:/opt/embedding-rerank-server/"
Write-Host "  2. ssh server"
Write-Host "  3. cd /opt/embedding-rerank-server"
Write-Host "  4. docker load -i ${IMAGE_NAME}-cpu.tar"
Write-Host "  5. rm ${IMAGE_NAME}-cpu.tar"
Write-Host "  6. docker compose up -d"
Write-Host "  7. curl http://localhost:7997/health"
Write-Host ""
Write-Host "提示: 默认构建 ARM64，如需 AMD64 请运行:" -ForegroundColor Gray
Write-Host "  .\scripts\package-cpu.ps1 linux/amd64" -ForegroundColor Gray
