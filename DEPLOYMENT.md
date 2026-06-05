# 部署指南

## 架构概览

| 部署模式 | 适用场景 | 推理引擎 | 镜像标签 |
|---------|---------|---------|---------|
| GPU (AMD64) | 有 NVIDIA GPU 的服务器 | PyTorch + torch.compile | `embedding-rerank-server:latest` |
| CPU (ARM64/AMD64) | 无 GPU 服务器 | ONNX Runtime | `embedding-rerank-server:cpu` |

两种模式共用同一份 `config.yaml`（所有参数 `auto`，启动时自适应）。

## 环境要求

| 项目 | 要求 |
|------|------|
| Docker | 20.10+ |
| GPU 模式 | NVIDIA 驱动 + nvidia-container-toolkit；Flash Attention 需 Ampere+ 架构（A100/A10/3090/4090） |
| CPU 模式 | 无特殊要求；模型目录需有 `onnx/` 子目录 |
| 内存 | 建议 ≥16GB（含向量缓存约 800MB） |

## 打包（离线部署）

### Windows (PowerShell)

```powershell
# GPU 版（构建 + 导出 tar + 生成部署包）
.\scripts\package.ps1

# CPU 版（默认 ARM64，可指定平台）
.\scripts\package-cpu.ps1                # ARM64
.\scripts\package-cpu.ps1 linux/amd64    # AMD64
```

### Mac / Linux (Shell)

```bash
chmod +x scripts/*.sh
./scripts/package.sh                      # GPU 版
./scripts/package-cpu.sh                  # CPU 版（自动识别当前平台）
./scripts/package-cpu.sh linux/arm64      # 指定 ARM64
```

打包产物在 `deploy/`（GPU）或 `deploy-cpu/`（CPU）目录，包含：
- `embedding-rerank-server.tar` — 镜像
- `docker-compose.yml` — 编排文件
- `config.yaml` — 配置
- `README.txt` — 部署说明（自动生成）

## 模型准备

### 目录结构

```
/models/
├── bge-m3/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   ├── sparse_linear.pt
│   ├── colbert_linear.pt
│   └── onnx/                  # CPU 模式必需
│       ├── model.onnx
│       ├── model.onnx_data
│       ├── tokenizer.json
│       └── config.json
└── bge-reranker-v2-m3/
    ├── model.safetensors
    ├── tokenizer.json
    ├── config.json
    └── onnx/                  # CPU 模式必需
        ├── model.onnx
        ├── tokenizer.json
        └── config.json
```

### 导出 ONNX 模型（CPU 模式需要）

```bash
pip install optimum[onnxruntime]

# Embedding
python -m optimum.exporters.onnx --model /path/to/bge-m3 \
    --task feature-extraction /path/to/bge-m3/onnx

# Reranker
python -m optimum.exporters.onnx --model /path/to/bge-reranker-v2-m3 \
    --task text-classification /path/to/bge-reranker-v2-m3/onnx
```

## 部署步骤

### GPU 模式

```bash
# 1. 传输部署包
scp -r deploy/* user@gpu-server:/opt/embedding-rerank-server/

# 2. 加载镜像
ssh gpu-server
cd /opt/embedding-rerank-server
docker load -i embedding-rerank-server.tar
rm embedding-rerank-server.tar

# 3. 启动（自动探测 GPU，零配置）
docker compose up -d

# 4. 验证
curl http://localhost:7997/health
# {"status":"ready","models_loaded":{"embedding":true},"queue_depth":0,...}
```

### CPU 模式

```bash
# 1. 传输部署包
scp -r deploy-cpu/* user@server:/opt/embedding-rerank-server/

# 2. 加载镜像
docker load -i embedding-rerank-server-cpu.tar

# 3. 确保模型目录有 onnx/ 子目录

# 4. 启动（自动探测核数，自动配置线程数）
docker compose up -d

# 5. 验证
curl http://localhost:7997/health
```

## 配置说明

### 自动配置（推荐）

默认所有性能参数为 `auto`，启动时自动探测填充：

| 参数 | GPU 自动值 | CPU 自动值 |
|------|-----------|-----------|
| device | cuda | cpu |
| engine | pytorch | onnx |
| fp16 | true | false |
| max_batch_size | 按显存 16–256 | 按核数 8–64 |
| max_batch_tokens | 按显存 4096–65536 | 8192 |
| max_concurrency | 1 | 核数/2 |
| OMP_NUM_THREADS | 不设置 | 自动设置 |

### 手动覆盖

在 config.yaml 中指定具体值即可覆盖自动探测：

```yaml
batching:
  max_batch_size: 32        # 覆盖
  max_batch_tokens: 8192    # 覆盖
  max_concurrency: 2        # 覆盖
```

### 环境变量覆盖（优先级最高）

| 变量 | 说明 | 示例 |
|------|------|------|
| MODE | 运行模式 | embed / rerank / all |
| PORT | 服务端口 | 7997 |
| EMBED_MODEL_PATH | Embedding 模型路径 | /models/bge-m3 |
| RERANK_MODEL_PATH | Rerank 模型路径 | /models/bge-reranker-v2-m3 |
| DEVICE | 强制设备 | cuda / cpu |
| ENGINE | 强制引擎 | pytorch / onnx |
| MAX_BATCH_SIZE | batch 大小 | 64 / auto |
| MAX_BATCH_TOKENS | token budget | 16384 / auto |
| MAX_CONCURRENCY | 并发数 | 1 / auto |
| BACKPRESSURE_THRESHOLD | 背压阈值 | 3200 / auto |

## 监控

### /health

```json
{
  "status": "ready",
  "models_loaded": {"embedding": true, "rerank": true},
  "queue_depth": 12,
  "estimated_wait_seconds": 0.6,
  "version": "2.1.0"
}
```

### /metrics（Prometheus）

```
ers_requests_total{endpoint="/v1/embeddings",status_code="200"} 1234
ers_request_duration_seconds_bucket{endpoint="/v1/embeddings",le="0.1"} 1100
ers_batch_size_bucket{model_type="dense",le="32"} 500
ers_batch_tokens_bucket{model_type="dense",le="16384"} 480
ers_queue_depth{batcher_type="dense"} 5
ers_tokens_per_second{model_type="dense"} 12500
```

## 背压机制

队列深度超过阈值（默认 max_batch_size × 50）时：
- 返回 HTTP 429 Too Many Requests
- 响应头 `Retry-After: <seconds>`
- 响应体含 `queue_depth` 和 `estimated_wait_seconds`

客户端应据此决定重试或切换备用服务。

## Aladdin 对接

`.env` 配置：

```bash
EMBED_BASE_URL=http://<server-ip>:7997/v1
RERANK_BASE_URL=http://<server-ip>:7998/v1
```

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| 启动崩溃 `NumPy 1.x cannot be run in NumPy 2.x` | numpy 被装成 2.x | 确认 requirements 锁定 `numpy<2.0.0`，重新构建 |
| HTTP 429 | 队列过深，服务过载 | 等 Retry-After 后重试，或扩容 |
| CUDA OOM | GPU 显存不足 | config.yaml 手动调小 max_batch_size / max_batch_tokens |
| 启动慢（GPU） | torch.compile 首次编译 | 正常，首次启动慢 30–60s；编译缓存在 /tmp 复用 |
| ONNX 模型加载失败 | 缺 onnx/ 子目录 | 按上文导出 ONNX 模型 |
| Flash Attention 未启用 | 未安装或 GPU 架构不支持 | 仅 Ampere+ 支持；不影响功能，仅长文本变慢 |
| Worker health check 失败 | 服务未就绪 | 检查 /health；GPU 模式 start_period 设为 120s |
| /metrics 返回提示文本 | prometheus_client 未安装 | 已在 requirements 中，重新构建即可 |

## 镜像构建说明

- **GPU 镜像**：基于 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`，torch 由基础镜像提供。Flash Attention 安装失败会自动降级（Dockerfile 内置 fallback）。
- **CPU 镜像**：基于 `python:3.11-slim`，安装 CPU 版 torch + onnxruntime。
- 构建上下文只 COPY `app/` 和 `config.yaml`，文档/spec 不进镜像。
