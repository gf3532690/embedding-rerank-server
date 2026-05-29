# Embedding & Rerank Server

基于 FlagEmbedding 的**自适应推理服务**，提供 OpenAI 兼容的 API 接口。
专为 Aladdin RAG 系统设计，支持 Dense + Sparse 双路 Embedding 和文档 Rerank。

## 核心特性

- **零配置启动** — 自动探测硬件（GPU/CPU、显存、核数），所有参数默认 `auto`
- **Token-level Continuous Batching** — 按总 token 数合批，短文本多塞、长文本少塞，GPU 利用率最大化
- **优先级队列** — 单条查询优先于批量入库，避免实时查询被阻塞
- **无超时** — 服务端永不主动拒绝请求，请求多久都等
- **背压机制** — 队列过深时返回 HTTP 429 + Retry-After，客户端可决策重试或切换
- **Prometheus 监控** — `/metrics` 端点暴露请求数、延迟、batch 大小、队列深度、吞吐量

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/embeddings` | POST | OpenAI 兼容 dense embedding |
| `/v1/embed_sparse` | POST | BGE-M3 sparse embedding (lexical weights) |
| `/v1/rerank` | POST | 文档重排序 |
| `/health` | GET | 模型状态 + 队列深度 + 预估等待时间 |
| `/metrics` | GET | Prometheus 格式监控指标 |

## 部署架构

生产环境推荐 Embedding 和 Rerank 分进程部署（共享镜像）：

```
embedding (port 7997) ─── GPU/CPU ─── bge-m3
reranker  (port 7998) ─── GPU/CPU ─── bge-reranker-v2-m3
```

## 快速开始

```bash
# 构建镜像
docker build -t embedding-rerank-server:latest .

# 启动（自动探测硬件，零配置）
docker compose up -d

# 验证
curl http://localhost:7997/health
curl http://localhost:7998/health

# 查看监控
curl http://localhost:7997/metrics
```

## 配置

所有配置在 `config.yaml` 中，支持环境变量覆盖。

**核心理念：不配就自动探测，手动配置作为覆盖项。**

```yaml
# 所有性能参数默认 "auto"
batching:
  max_batch_size: "auto"      # 根据显存/核数自动决定
  max_batch_tokens: "auto"    # 根据显存估算 token budget
  max_concurrency: "auto"     # GPU=1, CPU=核数/2
  backpressure_threshold: "auto"  # max_batch_size × 50

models:
  embed:
    device: "auto"    # 有 GPU 用 cuda，无 GPU 用 cpu
    engine: "auto"    # GPU 用 pytorch，CPU 用 onnx
    fp16: "auto"      # GPU 开启，CPU 关闭
```

详见 [DEPLOYMENT.md](DEPLOYMENT.md) 和 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 自适应逻辑

启动时自动执行：

1. **硬件探测** — GPU 型号/显存、CPU 核数、可用内存
2. **参数填充** — 将所有 `"auto"` 替换为探测到的最优值
3. **日志输出** — 打印最终配置，方便排查

详见 [ARCHITECTURE.md](ARCHITECTURE.md)。
