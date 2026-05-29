# 部署指南

## 架构概览

| 部署模式 | 适用场景 | 推理引擎 | 镜像标签 |
|---------|---------|---------|---------|
| GPU (AMD64) | 独立 GPU 服务器 | PyTorch + torch.compile | `embedding-rerank-server:latest` |
| CPU (ARM64/AMD64) | 无 GPU 服务器 | ONNX Runtime | `embedding-rerank-server:cpu` |

## v2.0 变更

- **零配置启动**：所有性能参数默认 `auto`，自动探测硬件
- **去掉 request_timeout**：服务端永不超时，由客户端控制
- **Token-level batching**：按 token 数合批，替代固定条数
- **背压机制**：队列过深返回 429，替代超时拒绝
- **Prometheus 监控**：新增 `/metrics` 端点
- **docker-compose 简化**：去掉 MAX_BATCH_SIZE、DEVICE 等环境变量（自动探测）

## 打包

### Windows (PowerShell)

```powershell
# GPU 版
.\scripts\package.ps1

# CPU 版
.\scripts\package-cpu.ps1
```

### Mac / Linux (Shell)

```bash
# GPU 版
chmod +x scripts/package.sh
./scripts/package.sh

# CPU 版
chmod +x scripts/package-cpu.sh
./scripts/package-cpu.sh
```

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
│   └── onnx/                  # CPU 模式需要
│       ├── model.onnx
│       ├── model.onnx_data
│       ├── tokenizer.json
│       └── config.json
└── bge-reranker-v2-m3/
    ├── model.safetensors
    ├── tokenizer.json
    ├── config.json
    └── onnx/                  # CPU 模式需要
        ├── model.onnx
        ├── tokenizer.json
        └── config.json
```

### 导出 ONNX 模型

```bash
pip install optimum[onnxruntime]

# Embedding
python -m optimum.exporters.onnx --model /path/to/bge-m3 --task feature-extraction /path/to/bge-m3/onnx

# Reranker
python -m optimum.exporters.onnx --model /path/to/bge-reranker-v2-m3 --task text-classification /path/to/bge-reranker-v2-m3/onnx
```

## 部署步骤

### GPU 模式

```bash
# 1. 传输文件
scp deploy/* user@gpu-server:/opt/embedding-rerank-server/

# 2. 加载镜像
docker load -i embedding-rerank-server.tar

# 3. 启动（自动探测 GPU，零配置）
docker compose up -d

# 4. 验证
curl http://localhost:7997/health
# 响应示例：{"status":"ready","models_loaded":{"embedding":true},"queue_depth":0,"estimated_wait_seconds":0.0}
```

### CPU 模式

```bash
# 1. 传输文件
scp deploy-cpu/* user@arm-server:/opt/embedding-rerank-server/

# 2. 加载镜像
docker load -i embedding-rerank-server-cpu.tar

# 3. 确保模型目录有 onnx/ 子目录

# 4. 启动（自动探测 CPU 核数，自动配置线程数）
docker compose up -d

# 5. 验证
curl http://localhost:7997/health
```

## 配置说明

### 自动配置（推荐）

v2.0 默认所有性能参数为 `auto`，启动时自动探测：

| 参数 | GPU 自动值 | CPU 自动值 |
|------|-----------|-----------|
| device | cuda | cpu |
| engine | pytorch | onnx |
| fp16 | true | false |
| max_batch_size | 根据显存 (16-256) | 根据核数 (8-64) |
| max_batch_tokens | 根据显存 (4096-65536) | 8192 |
| max_concurrency | 1 | 核数/2 |
| OMP_NUM_THREADS | - | 自动设置 |

### 手动覆盖

如需精细控制，可在 config.yaml 中指定具体值：

```yaml
batching:
  max_batch_size: 32        # 覆盖自动探测值
  max_batch_tokens: 8192    # 覆盖自动探测值
  max_concurrency: 2        # 覆盖自动探测值
```

### 环境变量覆盖

| 变量 | 说明 | 示例 |
|------|------|------|
| MODE | 运行模式 | embed / rerank / all |
| PORT | 服务端口 | 7997 |
| EMBED_MODEL_PATH | Embedding 模型路径 | /models/bge-m3 |
| RERANK_MODEL_PATH | Rerank 模型路径 | /models/bge-reranker-v2-m3 |
| DEVICE | 强制指定设备 | cuda / cpu |
| ENGINE | 强制指定引擎 | pytorch / onnx |
| MAX_BATCH_SIZE | 强制指定 batch 大小 | 64 / auto |
| MAX_BATCH_TOKENS | 强制指定 token budget | 16384 / auto |
| MAX_CONCURRENCY | 强制指定并发数 | 1 / auto |

## 监控

### /health 端点

```json
{
  "status": "ready",
  "models_loaded": {"embedding": true, "rerank": true},
  "queue_depth": 12,
  "estimated_wait_seconds": 0.6,
  "version": "2.0.0"
}
```

### /metrics 端点（Prometheus）

```
# 请求计数
ers_requests_total{endpoint="/v1/embeddings",status_code="200"} 1234

# 延迟分布
ers_request_duration_seconds_bucket{endpoint="/v1/embeddings",le="0.1"} 1100

# Batch 大小
ers_batch_size_bucket{model_type="dense",le="32"} 500

# 队列深度
ers_queue_depth{batcher_type="dense"} 5

# 吞吐量
ers_tokens_per_second{model_type="dense"} 12500
```

## 背压机制

当队列深度超过阈值（默认 max_batch_size × 50）时：

- 返回 HTTP 429 Too Many Requests
- 响应头包含 `Retry-After: <seconds>`
- 响应体包含 `queue_depth` 和 `estimated_wait_seconds`

客户端应根据 Retry-After 决定是否重试或切换备用服务。

## Aladdin 对接

`.env` 配置：

```bash
EMBED_BASE_URL=http://<embedding-server-ip>:7997/v1
RERANK_BASE_URL=http://<embedding-server-ip>:7998/v1
```

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| HTTP 429 | 队列过深，服务过载 | 等待 Retry-After 后重试，或扩容 |
| CUDA OOM | GPU 显存不足 | 手动设置较小的 max_batch_size |
| 启动慢 | 模型加载 + 预热 | 正常现象，等待 /health 返回 ready |
| Worker health check 失败 | 服务未就绪 | 检查 /health 端点 |
