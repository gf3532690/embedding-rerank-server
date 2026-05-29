# 架构说明

## 自适应推理引擎 v2.0

本服务借鉴 [Infinity](https://github.com/michaelfeil/infinity) 和 TEI (Text Embeddings Inference) 的设计思路，
实现了零配置、自适应的推理服务。

## 架构概览

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────┐
│  FastAPI (app/main.py)                          │
│  - 优先级判断：单条=0(查询), 多条=1(入库)         │
│  - 背压检查：队列过深返回 429                     │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  TokenBatcher (app/batcher.py)                  │
│  - 优先级队列 (heapq)                            │
│  - Token budget 合批 (max_batch_tokens)          │
│  - 无超时，无限等待                               │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  GPU Semaphore                                  │
│  - GPU: max_concurrency=1 (串行)                 │
│  - CPU: max_concurrency=核数/2 (并行)            │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Model Inference (app/models.py)                │
│  - PyTorch (GPU) / ONNX Runtime (CPU)           │
│  - 不再内部分批，由 batcher 控制                  │
└─────────────────────────────────────────────────┘
     │
     ▼
  Response
```

## 模块职责

### app/hardware.py — 硬件探测

启动时执行一次，探测：
- GPU 是否可用、型号、显存
- CPU 核数
- 可用系统内存

然后根据探测结果填充所有 `"auto"` 配置值：

| 配置项 | GPU 自动值 | CPU 自动值 |
|--------|-----------|-----------|
| device | cuda | cpu |
| engine | pytorch | onnx |
| fp16 | true | false |
| max_batch_size | 根据显存 (16-256) | 根据核数 (8-64) |
| max_batch_tokens | 显存GB × 2048 | 8192 |
| max_concurrency | 1 | 核数/2 |
| OMP_NUM_THREADS | 不设置 | 核数/并发数 |

### app/batcher.py — Token-level Continuous Batching

核心改进：

1. **按 token 数合批**（而非固定条数）
   - 设定 `max_batch_tokens`（如 16384）
   - 短文本（50 tokens）一批可塞 300+ 条
   - 长文本（8192 tokens）一批只塞 2 条
   - 减少 padding 浪费，GPU 利用率最大化

2. **优先级队列**
   - `priority=0`：单条查询请求（实时性要求高）
   - `priority=1`：批量入库请求（吞吐优先）
   - 使用 heapq 实现，O(log n) 插入/弹出

3. **无超时**
   - 去掉 `request_timeout`
   - 服务端永不主动拒绝请求
   - 请求多久都等，直到处理完返回

4. **背压机制**
   - 队列深度 > `backpressure_threshold` 时抛出 `BackpressureError`
   - API 层捕获后返回 HTTP 429 + `Retry-After` 头
   - 阈值默认 = `max_batch_size × 50`（约 50 批的积压）

### app/models.py — 推理引擎

- 去掉内部 `batch_size` 限制（由 batcher 控制）
- 提供 `count_tokens()` 方法供 batcher 精确计算 token 数
- 保持 ONNX/PyTorch 双引擎

### app/metrics.py — Prometheus 监控

指标列表：

| 指标名 | 类型 | 说明 |
|--------|------|------|
| ers_requests_total | Counter | 请求计数（按端点、状态码） |
| ers_request_duration_seconds | Histogram | 端到端延迟 |
| ers_inference_duration_seconds | Histogram | 纯推理延迟 |
| ers_batch_size | Histogram | 每批条数 |
| ers_batch_tokens | Histogram | 每批 token 数 |
| ers_queue_depth | Gauge | 当前队列深度 |
| ers_tokens_per_second | Gauge | 吞吐量 |

### app/config.py — 配置

- 所有数值型配置支持 `"auto"` 字符串
- 加载后调用 `hardware.auto_configure()` 替换为实际值
- 启动日志输出最终配置值

## 关键设计决策

### 为什么按 token 数合批？

固定条数合批的问题：
- 64 条短文本（每条 10 tokens）→ 总 640 tokens，GPU 空闲
- 64 条长文本（每条 8192 tokens）→ 总 524288 tokens，OOM

按 token 数合批：
- `max_batch_tokens=16384` 时：
  - 短文本：一批塞 1600+ 条
  - 长文本：一批只塞 2 条
- GPU 每批的计算量恒定，利用率稳定

### 为什么去掉超时？

旧设计的问题：
- 大批量入库时，排队时间可能超过 timeout
- 超时后客户端重试，加剧拥堵（雪崩效应）

新设计：
- 服务端永不超时，保证每个请求最终都能处理
- 通过背压机制（429）让客户端知道服务繁忙
- 客户端可以决策：等待、重试、或切换备用服务

### 为什么用优先级队列？

场景：用户正在查询，同时后台在批量入库。

没有优先级：
- 入库请求（每次 100 条）占满队列
- 查询请求（1 条）排在后面，延迟 10s+

有优先级：
- 查询请求 priority=0，入库请求 priority=1
- 查询请求插队到入库请求前面
- 查询延迟 < 100ms，入库吞吐不受影响

### GPU 串行 vs CPU 并行

- **GPU 模式**：`max_concurrency=1`
  - GPU 本身是并行计算设备
  - 多个推理任务同时跑会争抢显存，导致 OOM
  - 串行推理 + 大 batch = 最高吞吐

- **CPU 模式**：`max_concurrency=核数/2`
  - CPU 多核可以真正并行
  - 每个推理任务占用 `OMP_NUM_THREADS` 个核
  - 并发数 × 线程数 ≈ 总核数

## 与 Infinity 的对比

| 特性 | Infinity | 本服务 |
|------|----------|--------|
| 语言 | Python | Python |
| Batching | 按条数 + 长度排序 | 按 token 数 |
| 优先级 | 按文本长度排序 | 按请求类型（查询/入库） |
| 队列 | 自定义 FIFO + 排序 | heapq 优先级队列 |
| Pipeline | 3 阶段（pre/core/post） | 2 阶段（tokenize/inference） |
| 背压 | queue_size 限制 | 429 + Retry-After |
| 配置 | CLI 参数 | YAML + auto |
