# 架构说明

## 自适应推理引擎 v2.1

本服务借鉴 [Infinity](https://github.com/michaelfeil/infinity) 和 TEI (Text Embeddings Inference) 的设计思路，
在 Python 生态内实现了零配置、自适应、高吞吐的 BGE-M3 推理服务。

核心目标：**客户端零配置、服务端自适应任何硬件、CPU/GPU 单一镜像逻辑统一**。

## 架构概览

```
                          HTTP Request
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  FastAPI (app/main.py)                         │
        │  - ORJSONResponse + uvloop（高性能 IO）         │
        │  - metrics 中间件（请求计数 + 延迟）            │
        │  - 优先级判断：单条=0(查询), 多条=1(入库)       │
        └──────────────────────────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  EmbeddingCache (app/cache.py)                 │
        │  - LRU 缓存，命中直接返回（0ms）                │
        │  - 只把未命中的文本送入 batcher                 │
        └──────────────────────────────────────────────┘
                               │ (cache miss)
                               ▼
        ┌──────────────────────────────────────────────┐
        │  TokenBatcher (app/batcher.py)                 │
        │  - 优先级队列 (heapq)                          │
        │  - Token budget 合批 (max_batch_tokens)         │
        │  - Pipeline 重叠（收集下一批 ∥ 当前推理）        │
        │  - Speculative Batching（QPS 自适应等待）       │
        │  - Batch 内去重                                 │
        │  - 无超时；队列过深抛 BackpressureError → 429    │
        └──────────────────────────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  GPU Semaphore (max_concurrency)               │
        │  - GPU: 1 (串行)   CPU: 核数/2 (并行)           │
        └──────────────────────────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  推理引擎 (app/models.py)                       │
        │  - GPU: FlagEmbedding(PyTorch) + torch.compile  │
        │  - CPU: PipelinedDenseEngine(ONNX + numpy)      │
        │  - 动态 max_length（padding 到 batch 实际长度）  │
        └──────────────────────────────────────────────┘
                               │
                               ▼
                           Response
                               │
                               ▼ (写回缓存)
                       EmbeddingCache
```

## 模块职责

### app/hardware.py — 硬件探测与自动配置

启动时执行一次（`auto_configure`），探测 GPU/显存/CPU 核数/内存，
然后把配置里所有 `"auto"` 值替换为实际最优值：

| 配置项 | GPU 自动值 | CPU 自动值 |
|--------|-----------|-----------|
| device | cuda | cpu |
| engine | pytorch | onnx |
| fp16 | true | false |
| quantization | none | none |
| max_batch_size | 按显存 16–256 | 按核数 8–64 |
| max_batch_tokens | 显存GB × 2048（4096–65536） | 8192 |
| max_concurrency | 1（串行） | 核数/2（并行） |
| backpressure_threshold | max_batch_size × 50 | 同左 |
| OMP_NUM_THREADS | 不设置 | 核数/并发数 |

> 设计取舍：`max_batch_size` 用经验值表而非运行时 dummy 探测（TEI 同款做法）。
> 运行时探测会拖慢启动 10–30 秒，且显存波动时不稳定。

### app/cache.py — 向量缓存

- `LRUCache`：线程安全（`OrderedDict` + `threading.Lock`），O(1) 读写
- `EmbeddingCache`：管理 dense / sparse 两个独立缓存
- Key = 文本 MD5 hash（省内存，不存原文）
- API 层先批量查缓存，只把未命中的送入 batcher，命中部分 0ms 返回
- 命中率统计可查（`stats` 属性）

### app/batcher.py — Token-level Continuous Batching

| 特性 | 说明 |
|------|------|
| 按 token 合批 | `max_batch_tokens` 控制每批总 token，短文本多塞、长文本少塞 |
| 优先级队列 | heapq，priority=0 查询优先，priority=1 入库 |
| Pipeline 重叠 | worker 在推理当前批时，用 `asyncio.Task` 预收集下一批 |
| Speculative Batching | 按实时 QPS 调整等待：高负载等满凑大批，低负载 1ms 立即发 |
| Batch 内去重 | 相同文本只推理一次，结果共享回所有请求 |
| 无超时 | 服务端永不主动拒绝；队列过深抛 `BackpressureError` |

### app/models.py — 推理引擎

- GPU（pytorch）：FlagEmbedding 原生 + `torch.compile(max-autotune)` + 可选 Flash Attention
- CPU（onnx）：`PipelinedDenseEngine`，ONNX Runtime O3 图优化 + 线程精调
- 动态 max_length：每批 padding 到实际最大长度（对齐 8 的倍数），短文本批省大量算力
- INT8 量化：CPU 用 torch 动态量化，GPU 走 ONNX 量化模型（`onnx_int8/` 目录）
- `count_tokens()` 供 batcher 精确计算 token 数
- Sparse embedding 始终用 FlagEmbedding（ONNX 不输出 lexical weights）

### app/pipeline.py — ONNX Dense 推理引擎

- tokenize → ONNX forward → **纯 numpy** mean pooling + L2 normalize
- ONNX 路径不依赖 torch，避免 tensor 创建/转换开销
- `return_tensors="np"` 直接输出 numpy，零中间拷贝

### app/metrics.py — Prometheus 监控

| 指标 | 类型 | 说明 |
|------|------|------|
| ers_requests_total | Counter | 请求计数（端点 + 状态码） |
| ers_request_duration_seconds | Histogram | 端到端延迟 |
| ers_inference_duration_seconds | Histogram | 纯推理延迟 |
| ers_batch_size | Histogram | 每批条数 |
| ers_batch_tokens | Histogram | 每批 token 数 |
| ers_queue_depth | Gauge | 当前队列深度 |
| ers_tokens_per_second | Gauge | 吞吐量 |

prometheus_client 未安装时自动降级（`/metrics` 返回提示，不影响服务）。

### app/config.py — 配置

- 所有数值型字段支持 `"auto"`（`Union[int, str]` / `Union[bool, str]`）
- 环境变量 > config.yaml > 默认值（auto）
- 加载后由 `auto_configure()` 解析 auto 值

## 关键设计决策

### 为什么按 token 数合批？

固定条数合批：64 条短文本（各 10 token）只占 640 token，GPU 空闲；
64 条长文本（各 8192 token）共 52 万 token，OOM。

按 token 合批：`max_batch_tokens=16384` 时，短文本一批塞 1600+ 条，
长文本一批只塞 2 条。每批计算量恒定，利用率稳定。

### 为什么去掉超时？

旧设计大批量入库时排队可能超 timeout，超时后客户端重试加剧拥堵（雪崩）。
新设计服务端永不超时，靠背压（429 + Retry-After）让客户端自己决策重试/切换。

### 为什么用优先级队列？

用户查询（1 条）和后台入库（每次 100 条）并发时，若无优先级，
查询会排在入库后面，延迟 10s+。优先级队列让查询插队，延迟 <100ms，
同时入库吞吐不受影响。

### GPU 串行 vs CPU 并行

- GPU：`max_concurrency=1`。GPU 本身并行计算，多任务同跑只会争抢显存 OOM，串行 + 大 batch 吞吐最高。
- CPU：`max_concurrency=核数/2`。CPU 多核真并行，每个推理占 `OMP_NUM_THREADS` 个核。

### Pipeline 重叠为何放在 batcher 层

FlagEmbedding 的 `encode()` 是黑盒（tokenize+forward 一体），无法拆 3 阶段线程。
所以在 batcher 层做等价 overlap：worker 推理当前批时，用独立 asyncio.Task 收集下一批，
隐藏 batch 收集的 `max_wait_ms` 延迟。实现 20 行，拿到约 80% 的 pipeline 收益。

## 性能优化清单（v2.1）

| 优化 | 适用 | 收益 |
|------|------|------|
| 向量缓存（LRU） | GPU + CPU | 重复文本 0ms |
| 动态 max_length | GPU + CPU | 短文本批省 5–10x 算力 |
| Batch 内去重 | GPU + CPU | 批量入库减 10–50% 推理量 |
| Pipeline 重叠 | GPU + CPU | 整体吞吐 +15–20% |
| Speculative Batching | GPU + CPU | 低负载降延迟，高负载提吞吐 |
| Flash Attention | GPU | 长文本 2x、显存减半 |
| torch.compile max-autotune | GPU | 首次慢，稳态 +10–20% |
| ONNX O3 + 线程精调 | CPU | 推理 +10–30% |
| 纯 numpy 后处理 | CPU | 后处理 +3–8% |
| INT8 量化（可选） | GPU + CPU | 吞吐 ~2x，需预导出量化模型 |
| uvloop + orjson | GPU + CPU | 高并发 IO +5–10% |

## 与 TEI / Infinity 对比

| 维度 | TEI | Infinity | 本服务 |
|------|-----|----------|--------|
| 语言 | Rust | Python | Python |
| Dense / Rerank | ✅ | ✅ | ✅ |
| Sparse（lexical） | ❌ | ❌ | ✅ |
| Token-level batching | ✅ | ❌（按条数） | ✅ |
| 业务优先级队列 | ❌ | ❌ | ✅ |
| 向量缓存 | ❌ | 磁盘 | 内存 LRU |
| 背压 429 + Retry-After | ✅ | 部分 | ✅ |
| OpenAI 兼容 API | 部分 | ✅ | ✅ |
| 多模态 | ❌ | ✅ | ❌（不需要） |
| 零配置 | ✅ | 部分 | ✅ |

在 BGE-M3 + RAG 这个垂直场景，本服务功能最全、部署最简；
纯推理吞吐约为 TEI（Rust）的 70–85%，是 Python 生态内的实践上限。
