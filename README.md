# Embedding & Rerank Server

基于 FlagEmbedding 的**自适应推理服务**，提供 OpenAI 兼容的 API 接口。
专为 Aladdin RAG 系统设计，支持 BGE-M3 的 Dense + Sparse 双路 Embedding 和文档 Rerank。

当前版本：**v2.1**

## 核心特性

- **零配置启动** — 自动探测硬件（GPU/CPU、显存、核数），所有参数默认 `auto`，CPU/GPU 共用一套配置
- **Token-level Continuous Batching** — 按总 token 数合批，短文本多塞、长文本少塞，利用率最大化
- **优先级队列** — 单条查询优先于批量入库，避免实时查询被入库阻塞
- **向量缓存** — LRU 缓存，相同文本不重复推理，命中 0ms 返回
- **无超时** — 服务端永不主动拒绝请求；队列过深返回 429 + Retry-After（背压）
- **多项推理优化** — 动态 max_length、batch 内去重、pipeline 重叠、Flash Attention、ONNX O3
- **Prometheus 监控** — `/metrics` 暴露请求数、延迟、batch 大小、队列深度、吞吐量

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/embeddings` | POST | OpenAI 兼容 dense embedding |
| `/v1/embed_sparse` | POST | BGE-M3 sparse embedding (lexical weights) |
| `/v1/rerank` | POST | 文档重排序（Cohere 格式） |
| `/health` | GET | 模型状态 + 队列深度 + 预估等待时间 |
| `/metrics` | GET | Prometheus 格式监控指标 |

## 部署架构

生产环境推荐 Embedding 和 Rerank 分进程部署（共享镜像）：

```
embedding (port 7997) ─── GPU/CPU ─── bge-m3
reranker  (port 7998) ─── GPU/CPU ─── bge-reranker-v2-m3
```

两种部署模式：

| 模式 | 镜像 | 引擎 | 适用 |
|------|------|------|------|
| GPU | `embedding-rerank-server:latest` | PyTorch + torch.compile | 有 NVIDIA GPU 的 AMD64 服务器 |
| CPU | `embedding-rerank-server:cpu` | ONNX Runtime | 无 GPU 的 ARM64/AMD64 服务器 |

## 快速开始

```bash
# 构建镜像（GPU）
docker build -t embedding-rerank-server:latest .

# 启动（自动探测硬件，零配置）
docker compose up -d

# 验证
curl http://localhost:7997/health
curl http://localhost:7998/health

# 查看监控
curl http://localhost:7997/metrics
```

打包离线部署见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 配置

所有配置在 `config.yaml`，支持环境变量覆盖。**核心理念：不配就自动探测，手动配置作为覆盖项。**

```yaml
# 所有性能参数默认 "auto"，CPU/GPU 通用
models:
  embed:
    device: "auto"    # 有 GPU 用 cuda，无 GPU 用 cpu
    engine: "auto"    # GPU 用 pytorch，CPU 用 onnx
    fp16: "auto"      # GPU 开启，CPU 关闭

batching:
  max_batch_size: "auto"          # 根据显存/核数自动决定
  max_batch_tokens: "auto"        # 根据显存估算 token budget
  max_concurrency: "auto"         # GPU=1, CPU=核数/2
  backpressure_threshold: "auto"  # max_batch_size × 50

cache:
  enabled: true                   # 向量缓存
  max_size: 100000                # 每类型缓存条数（约 400MB）
```

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 架构设计、模块职责、关键决策、性能优化清单
- [DEPLOYMENT.md](DEPLOYMENT.md) — 打包、部署、配置、监控、故障排查
- [OPTIMIZATION_ROADMAP.md](OPTIMIZATION_ROADMAP.md) — 优化路线图（已实现 + 未来方向）

## 技术栈

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.11 | |
| FastAPI | 0.115.0 | Web 框架 |
| FlagEmbedding | 1.2.11 | BGE-M3 推理（dense + sparse） |
| transformers | 4.44.2 | 锁定，FlagEmbedding 依赖 |
| numpy | <2.0.0 | **必须锁定**，老栈不兼容 numpy 2.x |
| torch | 2.6.0 | GPU 由基础镜像提供 |
| onnxruntime | 1.19.2 | CPU 推理 |
| optimum | 1.22.0 | ONNX 模型加载 |

> ⚠️ numpy 必须锁定 `<2.0.0`。onnxruntime 1.19.2 / FlagEmbedding 1.2.11 等老栈
> 的部分 wheel 针对 numpy 1.x ABI 编译，装 numpy 2.x 会在启动时崩溃。
