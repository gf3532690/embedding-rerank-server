# 自适应推理引擎改造

## 背景

当前 embedding-rerank-server 需要手动配置大量参数（batch_size、concurrency、timeout、OMP_THREADS 等），
不同硬件环境需要不同配置，配错就会超时或 OOM。TEI 和 Infinity 做到了开箱即用、零配置。

本次改造目标：借鉴 TEI 和 Infinity 的设计思路，让服务自动适配任何硬件环境，客户端零配置。

## 参考实现

- Infinity 源码位于 `c:\newHLSWorkspace\infinity`
- TEI 设计思路（Rust 实现，参考其架构而非代码）

## 需求列表

### 1. 自动感知硬件能力
- 启动时探测 CPU 核数、GPU 显存
- GPU 模式：根据显存自动决定 max_batch_size
- CPU 模式：根据核数自动决定 concurrency 和 OMP_NUM_THREADS
- 自动选择推理引擎（有 GPU 用 pytorch，无 GPU 用 onnx）
- 自动探测安全 max_batch_size（启动时用 dummy 数据逐步增大直到 OOM，取安全值）

### 2. 无限队列 + 无超时
- 去掉 request_timeout，服务端永不主动拒绝请求
- batcher 队列无上限，按推理能力消费
- 请求多久都等，直到处理完返回结果

### 3. Token-level continuous batching
- 不按固定条数合批，按总 token 数合批
- 设定 max_batch_tokens（如 16384），短文本多塞几条，长文本少塞几条
- 减少 padding 浪费，GPU/CPU 利用率最大化

### 4. 优先级队列
- 单条请求（查询场景）优先于批量请求（入库场景）
- 避免大批量入库阻塞实时查询的响应

### 5. Tokenizer 和推理 pipeline 重叠
- 当前批在 GPU/CPU 推理时，下一批同时进行 tokenize
- 隐藏 tokenize 延迟，提升整体吞吐

### 6. 运维端点
- `/health` 返回队列深度和预估等待时间
- `/metrics` Prometheus 格式端点：请求数、延迟分布、batch 大小、吞吐量、GPU/CPU 利用率
- 启动日志输出探测到的硬件参数和自动配置值

### 7. 自适应并发
- CPU 模式下根据实时负载动态调整推理线程数
- 空闲时多开线程加速，繁忙时收缩避免 CPU 争抢

### 8. Graceful degradation
- 队列过深时（如 >1000 条待处理）返回 HTTP 429 + Retry-After 头 + 预估等待时间
- 让客户端知道服务繁忙，可以决策是否重试或切换备用服务

### 9. config.yaml 支持 auto
- 所有性能参数支持 `"auto"` 值，不配就自动探测
- 手动配置作为覆盖项，给需要精细控制的场景用
- 示例：
  ```yaml
  batching:
    max_batch_size: auto
    max_concurrency: auto
  ```

## 约束

- 保持现有 API 接口不变（/v1/embeddings, /v1/embed_sparse, /v1/rerank, /health）
- 保持 GPU 和 CPU 双模式支持
- 保持 ONNX 和 PyTorch 双引擎支持
- 不引入新的外部依赖（prometheus_client 除外）
- 向后兼容：现有 config.yaml 手动配置仍然生效
