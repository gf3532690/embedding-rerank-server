# Embedding & Rerank Server 打包脚本（AMD64 + GPU）
# 本地构建镜像并导出，用于离线部署到无外网的 GPU 服务器
$ErrorActionPreference = "Stop"
$OUT = "deploy"

Write-Host "=== Embedding & Rerank Server 打包 ===" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# 构建镜像（利用缓存，仅依赖变更时重装）
Write-Host "`n[1] 构建服务镜像..." -ForegroundColor Cyan
docker build -t embedding-rerank-server:latest .
if ($LASTEXITCODE -ne 0) { Write-Host "构建失败！" -ForegroundColor Red; exit 1 }

# 导出镜像
Write-Host "`n[2] 导出镜像..." -ForegroundColor Cyan
docker save embedding-rerank-server:latest -o "$OUT\embedding-rerank-server.tar"

# 复制配置文件
Write-Host "`n[3] 复制配置文件..." -ForegroundColor Cyan
Copy-Item docker-compose.yml "$OUT\docker-compose.yml"
Copy-Item config.yaml "$OUT\config.yaml"

Write-Host "`n=== 完成 ===" -ForegroundColor Green
$size = [math]::Round((Get-ChildItem -Recurse $OUT | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
Write-Host "输出: $OUT\ ($size GB)"
Write-Host ""
Write-Host "部署步骤:" -ForegroundColor Yellow
Write-Host "  1. scp deploy/* user@gpu-server:/weique/jmarag/embedding-rerank-server/"
Write-Host "  2. ssh gpu-server"
Write-Host "  3. cd /weique/jmarag/embedding-rerank-server"
Write-Host "  4. docker load -i embedding-rerank-server.tar"
Write-Host "  5. rm embedding-rerank-server.tar"
Write-Host "  6. docker compose up -d"
