# 任务列表 — 性能优化（做到 Python 极限）

## 第 1 轮：减少无效计算（预估收益 +50-100%）

- [x] 1. 向量缓存模块 `app/cache.py`
  - LRU 缓存，key = text MD5 hash
  - Dense 和 Sparse 分别缓存
  - config.yaml 新增 cache 配置段
  - batcher submit 前拦截，命中直接返回
  - 缓存命中率统计（接入 metrics）

- [x] 2. 动态 max_length 裁剪
  - batcher 传递 batch 内最大 token_count 到 model 层
  - model 层 padding 到 actual_max 而非 model_max_length
  - GPU 和 ONNX 两条路径都要改

- [x] 3. 请求去重（batch 内）
  - batcher _process_batch 中对相同文本去重
  - 只推理 unique 文本，结果映射回所有请求
  - 保持 future 正确分发

- [x] 4. Flash Attention 集成（GPU）
  - requirements.txt 加入 flash-attn（可选依赖）
  - Dockerfile 加入安装步骤
  - 启动日志输出是否启用
  - config 支持 use_flash_attention: auto/true/false

- [x] 5. ONNX Runtime 优化（CPU）
  - SessionOptions 设置 graph_optimization_level = ORT_ENABLE_ALL
  - 配置 intra_op_num_threads / inter_op_num_threads
  - execution_mode 根据 batch 大小自适应
  - 启动日志输出 ONNX 优化配置

## 第 2 轮：Pipeline 并行化（预估收益 再+20-30%）

- [x] 6. Tokenizer 与推理重叠
  - 当前批推理时，预 tokenize 下一批
  - 使用 asyncio.Task 或 ThreadPoolExecutor
  - 传递 pre-tokenized inputs 到 model forward

- [x] 7. 3 阶段 Pipeline
  - 拆分 encode 为 encode_pre / encode_core / encode_post
  - preprocess 线程：tokenize + move to device
  - forward 线程：model forward
  - postprocess 线程：normalize + to_list
  - 3 个线程通过 Queue 连接

- [x] 8. Speculative Batching（自适应等待）
  - 跟踪最近 1 秒 QPS
  - 高 QPS 时等满 max_wait_ms 凑大 batch
  - 低 QPS 时立即发车（1ms）
  - 中间态线性插值

- [x] 9. ONNX Runtime 线程精细调优
  - 根据 CPU 核数和 concurrency 自动计算最优线程配置
  - intra_op = cores / concurrency
  - inter_op = 2（小 batch）或 4（大 batch）
  - 集成到 auto_configure

## 第 3 轮：硬件极限压榨（预估收益 再+50-100%）

- [x] 10. INT8 量化支持
  - GPU: bitsandbytes load_in_8bit 或 torch quantization
  - CPU: ONNX Runtime 动态量化
  - config 新增 quantization 配置
  - 启动时根据配置选择量化方式

- [x] 11. 多 GPU Replica
  - config 支持 device_ids 列表
  - 每个 GPU 加载独立模型实例
  - batcher round-robin 分发
  - 信号量按 replica 数量设置

- [x] 12. TensorRT 后端（GPU 极致）
  - 支持加载 TensorRT engine
  - 自动检测 .trt 文件
  - 回退到 PyTorch（无 TRT 时）

- [x] 13. 多进程推理（CPU 绕过 GIL）
  - ProcessPoolExecutor 替代 ThreadPoolExecutor
  - 子进程加载独立 ONNX session
  - 主进程负责调度和结果收集
  - 评估内存开销 vs 性能收益

- [x] 14. torch.compile max-autotune（GPU）
  - 首次启动时自动搜索最优 kernel
  - 缓存编译结果到磁盘
  - 后续启动直接加载（跳过编译）
