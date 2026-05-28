# 部署指南

## 架构概览

| 部署模式 | 适用场景 | 推理引擎 | 镜像标签 |
|---------|---------|---------|---------|
| GPU (AMD64) | 独立 GPU 服务器 | PyTorch + torch.compile | `embedding-rerank-server:latest` |
| CPU (ARM64) | ARM 服务器，无 GPU | ONNX Runtime | `embedding-rerank-server:cpu` |

## 打包

### Windows (PowerShell)

```powershell
# GPU 版
.\scripts\package.ps1

# CPU 版（ARM64 跨平台构建，较慢）
.\scripts\package-cpu.ps1
```

### Mac / Linux (Shell)

```bash
# GPU 版
chmod +x scripts/package.sh
./scripts/package.sh

# CPU 版（Mac ARM64 原生构建，快）
chmod +x scripts/package-cpu.sh
./scripts/package-cpu.sh
```

## 模型准备

### 目录结构

```
/models/
├── bge-m3/
│   ├── model.safetensors      # PyTorch 权重
│   ├── tokenizer.json
│   ├── config.json
│   ├── sparse_linear.pt       # Sparse 向量权重
│   ├── colbert_linear.pt
│   └── onnx/                  # ONNX 模型（CPU 模式需要）
│       ├── model.onnx
│       ├── model.onnx_data
│       ├── tokenizer.json
│       └── config.json
└── bge-reranker-v2-m3/
    ├── model.safetensors
    ├── tokenizer.json
    ├── config.json
    └── onnx/                  # ONNX 模型（CPU 模式需要）
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

### 打包模型传输

```bash
# 分别打包
tar -cf bge-m3-onnx.tar -C /path/to/bge-m3 onnx
tar -cf bge-reranker-v2-m3-onnx.tar -C /path/to/bge-reranker-v2-m3 onnx

# 服务器上解压
cd /models/bge-m3 && tar -xf bge-m3-onnx.tar
cd /models/bge-reranker-v2-m3 && tar -xf bge-reranker-v2-m3-onnx.tar
```

## 部署步骤

### GPU 模式（AMD64 服务器）

```bash
# 1. 传输文件
scp deploy/* user@gpu-server:/opt/embedding-rerank-server/

# 2. 加载镜像
docker load -i embedding-rerank-server.tar
rm embedding-rerank-server.tar

# 3. 启动
docker compose up -d

# 4. 验证
curl http://localhost:7997/health
curl http://localhost:7998/health
```

### CPU 模式（ARM64 服务器）

```bash
# 1. 传输文件
scp deploy-cpu/* user@arm-server:/opt/embedding-rerank-server/

# 2. 加载镜像
docker load -i embedding-rerank-server-cpu.tar
rm embedding-rerank-server-cpu.tar

# 3. 确保模型目录有 onnx/ 子目录

# 4. 启动
docker compose up -d

# 5. 验证
curl http://localhost:7997/health
curl http://localhost:7998/health
```

## 配置说明

### config.yaml (GPU)

```yaml
server:
  host: "0.0.0.0"
  port: 7997
  workers: 1
  request_timeout: 120

mode: "all"

models:
  embed:
    path: "/models/bge-m3"
    device: "cuda"
    fp16: true
    max_length: 1024
    engine: "pytorch"
  rerank:
    path: "/models/bge-reranker-v2-m3"
    device: "cuda"
    fp16: true
    max_length: 512
    engine: "pytorch"

batching:
  max_batch_size: 64
  max_wait_ms: 10
  max_concurrency: 1
```

### config-cpu.yaml (CPU + ONNX)

```yaml
server:
  host: "0.0.0.0"
  port: 7997
  workers: 1
  request_timeout: 600

mode: "all"

models:
  embed:
    path: "/models/bge-m3"
    device: "cpu"
    fp16: false
    max_length: 1024
    engine: "onnx"
  rerank:
    path: "/models/bge-reranker-v2-m3"
    device: "cpu"
    fp16: false
    max_length: 512
    engine: "onnx"

batching:
  max_batch_size: 32
  max_wait_ms: 50
  max_concurrency: 4
```

### docker-compose 环境变量

| 变量 | 说明 | GPU 推荐 | CPU 推荐 |
|------|------|---------|---------|
| MODE | 运行模式 | embed / rerank | embed / rerank |
| PORT | 服务端口 | 7997 / 7998 | 7997 / 7998 |
| OMP_NUM_THREADS | 单次推理线程数 | - | 24 |
| MKL_NUM_THREADS | MKL 线程数 | - | 24 |
| NVIDIA_VISIBLE_DEVICES | GPU 编号 | 0-7 | - |

### docker-compose 资源限制（CPU 模式）

```yaml
deploy:
  resources:
    limits:
      cpus: "48"    # embedding
      # cpus: "8"   # reranker
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/embeddings` | POST | Dense embedding (OpenAI 兼容) |
| `/v1/embed_sparse` | POST | Sparse embedding |
| `/v1/rerank` | POST | 文档重排序 |

## Aladdin 对接

ARM 服务器 `.env` 配置：

```bash
EMBED_BASE_URL=http://<embedding-server-ip>:7997/v1
RERANK_BASE_URL=http://<embedding-server-ip>:7998/v1
EMBED_SPARSE_ENABLED=false  # CPU 模式建议关闭 sparse
```

前端配置页面设置 timeout >= 600（CPU 模式推理较慢）。

## 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| 504 Gateway Timeout | 推理超时 | 调大 request_timeout 或减小 max_batch_size |
| CUDA OOM | GPU 显存不足 | 降低 max_batch_size / max_length，或指定空闲 GPU |
| Worker health check 失败 | 服务未就绪或网络不通 | 检查 /health 端点，确认 .env 地址正确 |
| torch.int4 错误 | onnxruntime 版本过高 | 锁定 onnxruntime==1.19.2 |
