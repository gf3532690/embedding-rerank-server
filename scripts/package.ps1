# Embedding & Rerank Server 打包脚本（AMD64 + GPU）
$ErrorActionPreference = "Stop"
$OUT = "deploy"

Write-Host "=== Embedding & Rerank Server 打包 ===" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# 构建镜像
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
Write-Host "  1. scp deploy/* user@gpu-server:/opt/embedding-rerank-server/"
Write-Host "  2. ssh gpu-server"
Write-Host "  3. docker load -i embedding-rerank-server.tar"
Write-Host "  4. 编辑 config.yaml 确认模型路径"
Write-Host "  5. docker compose up -d"
Write-Host ""
Write-Host "ARM 服务器 aladdin .env 配置:" -ForegroundColor Yellow
Write-Host "  EMBED_BASE_URL=http://<gpu-server-ip>:7997/v1"
Write-Host "  RERANK_BASE_URL=http://<gpu-server-ip>:7998/v1"
