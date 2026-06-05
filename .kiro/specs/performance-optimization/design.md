# 性能优化设计

## 三轮迭代架构

```
第1轮: 减少无效计算          第2轮: Pipeline 并行化        第3轮: 硬件极限压榨
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ 向量缓存 (LRU)   │     │ Tokenizer 重叠       │     │ INT8 量化         │
│ 动态 max_length  │     │ 3阶段 pipeline       │     │ 多 GPU replica    │
│ 请求去重         │     │ Speculative batching │     │ TensorRT 后端     │
│ Flash Attention  │     │ ONNX 线程调优        │     │ 多进程 (CPU)      │
│ ONNX O3 优化     │     │                     │     │                  │
└─────────────────┘     └─────────────────────┘     └──────────────────┘
     +50-100%                  再+20-30%                  再+50-100%
```

## 第 1 轮设计

### 向量缓存

```python
# 在 batcher submit 之前拦截
request → cache_check → hit? → 直接返回
                      → miss? → 进队列 → 推理 → 写缓存 → 返回
```

- 使用 `cachetools.LRUCache`，key = text 的 MD5 hash
- Dense 和 Sparse 分别缓存
- 可配置 cache_size（默认 100,000 条，约 400MB）
- config.yaml 新增 `cache.enabled` 和 `cache.max_size`

### 动态 max_length

```python
# 在 model forward 之前
actual_max = max(token_count for item in batch) + 2  # +2 for [CLS][SEP]
actual_max = min(actual_max, model_max_length)
inputs = tokenizer(texts, padding=True, max_length=actual_max, truncation=True)
```

- 需要在 batcher 层传递每个 item 的 token_count 到 model 层
- 或者在 model 层重新计算（tokenizer 已有）

### 请求去重

```python
# 在 batcher _process_batch 中
unique_map = {}  # text → index in unique_list
for item in batch:
    if item.data not in unique_map:
        unique_map[item.data] = len(unique_list)
        unique_list.append(item.data)

results = await process_fn(unique_list)  # 只推理去重后的

# 映射回原始位置
for item in batch:
    item.future.set_result(results[unique_map[item.data]])
```

### Flash Attention

- 仅 GPU 模式，Ampere+ 架构
- 安装 `flash-attn` 后 transformers 自动启用
- 需要在 Dockerfile 中加入安装步骤
- config.yaml 新增 `models.embed.use_flash_attention: auto`

### ONNX O3 优化

- 在 `_load_onnx()` 中配置 SessionOptions
- 设置 `graph_optimization_level = ORT_ENABLE_ALL`
- 配置 `intra_op_num_threads` 和 `inter_op_num_threads`

## 第 2 轮设计

### 3 阶段 Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Preprocess   │    │ Forward      │    │ Postprocess  │
│ (CPU thread) │───→│ (GPU/CPU)    │───→│ (CPU thread) │
│ tokenize     │    │ model forward│    │ normalize    │
│ move to GPU  │    │              │    │ 分发结果      │
└──────────────┘    └──────────────┘    └──────────────┘
       ↑                                       │
       └───── 队列 ←── batcher ←── 请求 ←──────┘
```

### Speculative Batching

```python
# 根据最近 1 秒的 QPS 动态调整等待时间
if qps > 100: wait = max_wait_ms      # 高负载，等满凑大 batch
elif qps > 10: wait = max_wait_ms / 2  # 中负载
else: wait = 1                          # 低负载，立即发车
```

## 第 3 轮设计

### INT8 量化

- GPU: `bitsandbytes` load_in_8bit 或 GPTQ
- CPU: ONNX Runtime 动态量化（`onnxruntime.quantization`）
- config.yaml 新增 `models.embed.quantization: "auto" / "int8" / "none"`

### 多 GPU Replica

- config.yaml 新增 `models.embed.device_ids: [0, 1]`
- 每个 device 加载一份模型，独立推理线程
- batcher round-robin 分发到不同 replica
