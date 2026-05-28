# Embedding & Rerank Server 打包脚本（AMD64 CPU 模式）
# 无 GPU 依赖，镜像更小（~2GB vs GPU 版 ~6GB）
$ErrorActionPreference = "Stop"
$OUT = "deploy-cpu"

Write-Host "=== Embedding & Rerank Server CPU 打包 ===" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# 构建镜像
Write-Host "`n[1] 构建 CPU 镜像..." -ForegroundColor Cyan
docker build --platform linux/arm64 -f Dockerfile.cpu -t embedding-rerank-server:cpu .
if ($LASTEXITCODE -ne 0) { Write-Host "构建失败！" -ForegroundColor Red; exit 1 }

# 导出镜像
Write-Host "`n[2] 导出镜像..." -ForegroundColor Cyan
docker save embedding-rerank-server:cpu -o "$OUT\embedding-rerank-server-cpu.tar"

# 复制配置文件
Write-Host "`n[3] 复制配置文件..." -ForegroundColor Cyan
Copy-Item docker-compose-cpu.yml "$OUT\docker-compose.yml"
Copy-Item config-cpu.yaml "$OUT\config-cpu.yaml"

Write-Host "`n=== 完成 ===" -ForegroundColor Green
$size = [math]::Round((Get-ChildItem -Recurse $OUT | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
Write-Host "输出: $OUT\ ($size GB)"
Write-Host ""
Write-Host "部署步骤:" -ForegroundColor Yellow
Write-Host "  1. scp deploy-cpu/* user@server:/path/to/embedding-rerank-server/"
Write-Host "  2. docker load -i embedding-rerank-server-cpu.tar"
Write-Host "  3. docker compose up -d"
