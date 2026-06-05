# 优化路线图

基于 TEI、Infinity、vLLM、Triton Inference Server 等主流推理服务的设计趋势，
针对两种部署场景（GPU / CPU）分别整理优化方向。

---

## 实现状态（v2.1）

下列优化**已全部实现**：

| 优化 | 适用 | 文件 |
|------|------|------|
| ✅ 向量缓存（LRU） | GPU + CPU | app/cache.py |
| ✅ 动态 max_length 裁剪 | GPU + CPU | app/models.py, app/pipeline.py |
| ✅ Batch 内去重 | GPU + CPU | app/batcher.py |
| ✅ Pipeline 重叠 | GPU + CPU | app/batcher.py |
| ✅ Speculative Batching | GPU + CPU | app/batcher.py |
| ✅ Flash Attention | GPU | app/models.py + requirements |
| ✅ torch.compile max-autotune | GPU | app/models.py |
| ✅ ONNX O3 + 线程精调 | CPU | app/models.py |
| ✅ 纯 numpy 后处理 | CPU | app/pipeline.py |
| ✅ INT8 量化（可选） | GPU + CPU | app/models.py |
| ✅ uvloop + orjson | GPU + CPU | app/main.py |

**未实现（投入产出比低，按需启用）**：
- 多 GPU Replica — 用 docker-compose 多容器水平扩展更简单
- TensorRT 后端 — 需额外导出 engine，与 GPU 型号绑定
- CPU 多进程绕 GIL — ONNX Runtime 本身释放 GIL，收益有限

**已达 Python 生态优化上限**。继续提升只能换语言（Rust）或换更小的模型。

下面是各优化项的详细设计说明（供后续维护参考）。

---

## GPU 部署场景优化

适用于：有 NVIDIA GPU 的 AMD64 服务器，PyTorch 推理引擎。

### P0 — 立即可做（低成本高收益）

#### 1. Flash Attention 集成

- **原理**：将 Self-Attention 的显存复杂度从 O(n²) 降到 O(n)，速度提升 2x（长序列时）
- **收益**：长文本（>1k tokens）推理速度翻倍，显存减半
- **工作量**：0.5h（装包即可，transformers 自动启用）
- **实现**：
  ```bash
  pip install flash-attn --no-build-isolation
  ```
  FlagEmbedding 底层用 transformers，安装后自动检测并启用，零代码改动。
- **注意**：需要 Ampere 及以上架构 GPU（A100/A10/3090/4090），不支持 V100

#### 2. 向量缓存（Embedding Cache）

- **原理**：相同文本不重复推理，直接返回缓存结果
- **收益**：命中缓存时延迟 0ms，减少 GPU 负载
- **工作量**：2h
- **实现思路**：
  ```python
  from cachetools import LRUCache
  import hashlib

  _cache = LRUCache(maxsize=100_000)  # 约 400MB（1024维 × 4字节 × 10万条）

  def get_or_compute(text: str) -> list[float]:
      key = hashlib.md5(text.encode()).hexdigest()
      if key in _cache:
          return _cache[key]
      embedding = model.encode_dense([text])[0]
      _cache[key] = embedding
      return embedding
  ```
- **适用场景**：知识库更新时大量文本不变、热门查询重复率高

#### 3. 动态 max_length 裁剪

- **原理**：每批 padding 到实际最大长度，而非模型最大长度
- **收益**：短文本批次计算量减少 5-10x
- **工作量**：1h
- **实现思路**：
  ```python
  # 当前：padding 到 model_max_length (8192)
  # 优化后：padding 到 batch 内实际最大长度
  actual_max = min(max(len(tokenizer.encode(t)) for t in batch), model_max_length)
  inputs = tokenizer(batch, padding=True, max_length=actual_max, truncation=True)
  ```
- **注意**：需要在 tokenize 阶段先扫一遍长度，与 pipeline 重叠配合使用效果最佳

### P1 — 短期优化（中等投入中等收益）

#### 4. Tokenizer Pipeline 真正重叠

- **原理**：当前批在 GPU 推理时，下一批同时在 CPU 上 tokenize
- **收益**：整体吞吐 +15%（隐藏 tokenize 延迟）
- **工作量**：4h
- **实现思路**：
  ```
  Thread/Task 1 (CPU): 从队列取 batch → tokenize → 放入 feature_queue
  Thread/Task 2 (GPU): 从 feature_queue 取 → forward → 放入 result_queue
  Main loop: 从 result_queue 取 → 分发结果
  ```
- **参考**：Infinity 的 `_preprocess_batch` / `_core_batch` / `_postprocess_batch` 三线程设计

#### 5. 3 阶段 Pipeline（完整版）

- **原理**：将推理拆为 preprocess → forward → postprocess，各阶段并行
- **收益**：整体吞吐 +20-30%，GPU 利用率接近 100%
- **工作量**：8h
- **实现思路**：
  ```python
  # 3 个线程 + 2 个队列
  preprocess_thread: tokenize + move to GPU → feature_queue
  forward_thread: model forward → result_queue
  postprocess_thread: normalize + 分发
  ```
- **注意**：需要将 FlagEmbedding 的 encode 拆成 encode_pre / encode_core / encode_post

#### 6. 多 GPU Replica

- **原理**：同一模型在多张 GPU 上各加载一份，batcher 轮询分发
- **收益**：N 卡 ≈ N 倍吞吐
- **工作量**：8h
- **实现思路**：
  ```yaml
  models:
    embed:
      replicas: 2
      device_ids: [0, 1]
  ```
  每个 replica 有独立的推理线程和信号量，batcher 用 round-robin 分发。

### P2 — 长期方向（高投入高收益）

#### 7. INT8/FP8 量化推理

- **原理**：将模型权重从 FP16 量化到 INT8，计算量减半
- **收益**：吞吐 ~2x，显存减半，精度损失 <1%（MTEB 评测）
- **工作量**：4h
- **实现思路**：
  ```bash
  # 导出量化模型
  optimum-cli export onnx --model bge-m3 --task feature-extraction --fp16 --optimize O3
  # 或使用 bitsandbytes
  model = AutoModel.from_pretrained("bge-m3", load_in_8bit=True)
  ```

#### 8. TensorRT 加速

- **原理**：NVIDIA 专用推理优化器，算子融合 + 内核自动调优
- **收益**：比 PyTorch 快 2-4x（取决于模型和 batch size）
- **工作量**：16h（需要导出 TensorRT engine）
- **实现思路**：
  ```bash
  # 导出 ONNX → TensorRT
  trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
  ```
- **注意**：TensorRT engine 与 GPU 型号绑定，不同卡需要重新编译

#### 9. Speculative Batching（自适应等待）

- **原理**：根据实时 QPS 动态调整 max_wait_ms
- **收益**：低负载时延迟更低，高负载时 batch 更大
- **工作量**：4h
- **实现思路**：
  ```python
  # 高 QPS → 多等一会凑大 batch
  # 低 QPS → 立即发车
  if recent_qps > 100:
      effective_wait = max_wait_ms
  elif recent_qps > 10:
      effective_wait = max_wait_ms // 2
  else:
      effective_wait = 1  # 几乎立即发
  ```

---

## CPU 部署场景优化

适用于：无 GPU 的 ARM64/AMD64 服务器，ONNX Runtime 推理引擎。

### P0 — 立即可做（低成本高收益）

#### 1. 向量缓存（同 GPU 方案）

- **收益**：CPU 推理慢（单条 50-200ms），缓存命中直接 0ms，收益比 GPU 场景更大
- **工作量**：2h
- **注意**：CPU 内存通常充裕，缓存可以开更大（50 万条）

#### 2. 动态 max_length 裁剪（同 GPU 方案）

- **收益**：CPU 模式下 padding 浪费更严重（ONNX 不像 GPU 那样并行计算 padding 部分）
- **工作量**：1h
- **注意**：CPU 模式 max_length 已经限制为 1024，但实际文本可能只有 100 tokens

#### 3. ONNX Graph 优化（O3 级别）

- **原理**：ONNX Runtime 的图优化器可以融合算子、消除冗余计算
- **收益**：推理速度 +10-30%
- **工作量**：1h（导出时指定优化级别）
- **实现思路**：
  ```python
  from optimum.onnxruntime import ORTOptimizer, OptimizationConfig

  optimizer = ORTOptimizer.from_pretrained(model_path)
  optimization_config = OptimizationConfig(
      optimization_level=99,  # O3 最高级别
      optimize_for_gpu=False,
  )
  optimizer.optimize(save_dir=output_path, optimization_config=optimization_config)
  ```

### P1 — 短期优化

#### 4. ONNX Runtime 多线程精细调优

- **原理**：ONNX Runtime 有 intra_op（单算子内并行）和 inter_op（算子间并行）两个线程池
- **收益**：合理配置可提升 20-50%
- **工作量**：2h
- **实现思路**：
  ```python
  import onnxruntime as ort

  sess_options = ort.SessionOptions()
  sess_options.intra_op_num_threads = cpu_cores // concurrency  # 单次推理用的核数
  sess_options.inter_op_num_threads = 2  # 算子间并行度
  sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 小 batch 用串行更快
  ```
- **注意**：当前通过 optimum 加载，需要传入 session_options

#### 5. Tokenizer Pipeline 重叠（CPU 版）

- **原理**：CPU 推理时间长（50-200ms/batch），tokenize 重叠收益更明显
- **收益**：整体吞吐 +20-30%
- **工作量**：4h
- **实现**：与 GPU 版相同，但 CPU 模式下 tokenize 和推理都在 CPU 上，需要注意线程分配

#### 6. 多进程推理（绕过 GIL）

- **原理**：Python GIL 限制单进程内的真正并行，CPU 模式下影响更大
- **收益**：多核利用率从 ~60% 提升到 ~90%
- **工作量**：8h
- **实现思路**：
  ```python
  # 方案 A：多 worker 进程（uvicorn --workers N）
  # 每个 worker 独立加载模型，独立推理
  # 缺点：内存翻倍

  # 方案 B：ProcessPoolExecutor
  # 主进程负责调度，子进程负责推理
  from concurrent.futures import ProcessPoolExecutor
  executor = ProcessPoolExecutor(max_workers=concurrency)
  ```
- **注意**：ONNX Runtime 本身释放了 GIL（C++ 实现），所以当前的 ThreadPoolExecutor 方案在 ONNX 模式下其实已经能并行

### P2 — 长期方向

#### 7. INT8 量化（ONNX）

- **原理**：ONNX Runtime 原生支持 INT8 量化推理
- **收益**：速度 +50-100%，内存减半
- **工作量**：4h
- **实现思路**：
  ```bash
  # 动态量化（无需校准数据）
  python -m onnxruntime.quantization.quantize \
      --input model.onnx \
      --output model_int8.onnx \
      --per_channel
  ```
- **注意**：ARM64 上 INT8 支持取决于 CPU 是否有 NEON/SVE 指令集

#### 8. 模型蒸馏（小模型替代）

- **原理**：用 bge-m3 蒸馏出更小的模型（如 bge-small），牺牲少量精度换取速度
- **收益**：推理速度 3-5x，内存 1/4
- **工作量**：数天（需要训练）
- **适用**：如果 CPU 场景对精度要求不高（如内部知识库检索）

#### 9. 请求合并去重

- **原理**：批量入库时，同一批请求中可能有重复文本
- **收益**：减少实际推理量
- **工作量**：2h
- **实现思路**：
  ```python
  # 在 batcher 层面去重
  unique_texts = list(set(texts))
  unique_embeddings = model.encode(unique_texts)
  # 然后映射回原始位置
  ```

---

## 两种场景通用优化

| 优化项 | GPU 收益 | CPU 收益 | 工作量 |
|--------|---------|---------|--------|
| 向量缓存 | 中 | 高 | 2h |
| 动态 max_length | 高 | 高 | 1h |
| Tokenizer pipeline 重叠 | 中 | 高 | 4h |
| 请求去重 | 低 | 中 | 2h |
| Speculative batching | 中 | 中 | 4h |
| Prometheus 指标完善（实际接入 batcher） | - | - | 2h |

---

## 实施建议

### 第一阶段（1-2 天）
- [GPU] Flash Attention（0.5h）
- [通用] 向量缓存（2h）
- [通用] 动态 max_length 裁剪（1h）
- [CPU] ONNX Graph O3 优化（1h）

### 第二阶段（1 周）
- [通用] Tokenizer pipeline 真正重叠（4h）
- [CPU] ONNX Runtime 线程精细调优（2h）
- [GPU] 3 阶段 pipeline（8h）

### 第三阶段（长期）
- [GPU] 多 GPU replica（8h）
- [GPU] INT8 量化 / TensorRT（4-16h）
- [CPU] INT8 量化 ONNX（4h）
- [CPU] 多进程推理（8h）
