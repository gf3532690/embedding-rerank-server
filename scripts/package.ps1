# ============================================================
# Embedding & Rerank Server v2.0 打包脚本（AMD64 + GPU）
# ============================================================
# 构建自适应引擎镜像并导出，用于离线部署到 GPU 服务器
# 特性：零配置启动，自动探测硬件
$ErrorActionPreference = "Stop"

$VERSION = "2.1.0"
$IMAGE_NAME = "embedding-rerank-server"
$IMAGE_TAG = "latest"
$OUT = "deploy"

Write-Host "=== Embedding & Rerank Server v$VERSION GPU 打包 ===" -ForegroundColor Green
Write-Host "  镜像: ${IMAGE_NAME}:${IMAGE_TAG}" -ForegroundColor Gray
Write-Host "  平台: linux/amd64 (GPU)" -ForegroundColor Gray
Write-Host ""

# 清理旧输出
if (Test-Path $OUT) {
    Remove-Item -Recurse -Force $OUT
}
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# [1] 构建镜像
Write-Host "[1/4] 构建 GPU 镜像..." -ForegroundColor Cyan
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${IMAGE_TAG}" .
if ($LASTEXITCODE -ne 0) {
    Write-Host "构建失败！" -ForegroundColor Red
    exit 1
}
Write-Host "  构建成功" -ForegroundColor Green

# [2] 导出镜像
Write-Host "[2/4] 导出镜像 tar..." -ForegroundColor Cyan
docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "$OUT\${IMAGE_NAME}.tar"
Write-Host "  导出完成" -ForegroundColor Green

# [3] 复制配置和编排文件
Write-Host "[3/4] 复制部署文件..." -ForegroundColor Cyan
Copy-Item docker-compose.yml "$OUT\docker-compose.yml"
Copy-Item config.yaml "$OUT\config.yaml"
Write-Host "  config.yaml (所有参数默认 auto)" -ForegroundColor Gray
Write-Host "  docker-compose.yml (零配置)" -ForegroundColor Gray

# [4] 生成部署说明
Write-Host "[4/4] 生成部署说明..." -ForegroundColor Cyan
$deployNote = @"
# Embedding & Rerank Server v$VERSION (GPU)
# 部署日期: $(Get-Date -Format "yyyy-MM-dd HH:mm")
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
Write-Host "  1. scp -r deploy/* user@gpu-server:/opt/embedding-rerank-server/"
Write-Host "  2. ssh gpu-server"
Write-Host "  3. cd /opt/embedding-rerank-server"
Write-Host "  4. docker load -i ${IMAGE_NAME}.tar"
Write-Host "  5. rm ${IMAGE_NAME}.tar"
Write-Host "  6. docker compose up -d"
Write-Host "  7. curl http://localhost:7997/health"
Write-Host "  8. curl http://localhost:7997/metrics  # Prometheus 监控"
