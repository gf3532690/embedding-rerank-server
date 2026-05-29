# 设计方案

## 架构概览

```
HTTP Request → Priority Queue → Token Batcher → Inference Engine → Response
                                     ↑                    ↑
                              Tokenizer Pipeline    Hardware Auto-Config
```

## 模块划分

### 1. `app/hardware.py` — 硬件探测模块
- `detect_device()` → "cuda" / "cpu"
- `detect_gpu_memory()` → 显存 GB
- `detect_cpu_cores()` → 核数
- `probe_max_batch_size(model, device)` → 安全 batch_size
- `auto_configure(config)` → 填充所有 "auto" 值

### 2. `app/batcher.py` — 重写为 Token-level Batcher
- 按 token 数合批而非固定条数
- 优先级队列（priority=0 查询优先，priority=1 批量入库）
- 无超时，无限队列
- Tokenizer pipeline 重叠（预 tokenize 下一批）

### 3. `app/metrics.py` — 监控指标模块
- Prometheus Counter/Histogram/Gauge
- 请求数、延迟、batch 大小、队列深度、吞吐量

### 4. `app/models.py` — 推理引擎（保持现有，微调）
- 接收已 tokenized 的输入（减少重复 tokenize）
- 支持动态 batch_size（由 batcher 决定）

### 5. `app/main.py` — 入口（调整）
- lifespan 中调用 hardware.auto_configure()
- 注册 /metrics 端点
- /health 返回队列状态

### 6. `app/config.py` — 配置（调整）
- 所有数值型配置支持 "auto" 字符串
- auto_configure() 负责解析并替换为实际值

## 开发顺序

1. `hardware.py` — 硬件探测 + auto_configure
2. `batcher.py` — 重写为 token-level + 优先级队列 + 无超时
3. `models.py` — 适配新 batcher 接口
4. `main.py` — 集成新模块，去掉旧超时逻辑
5. `metrics.py` — 监控端点
6. `config.py` — 支持 auto 值
7. 测试 + 文档更新

## 关键设计决策

- **单推理线程（GPU）/ 多推理线程（CPU）** — GPU 模式只有一个推理在跑（GPU 本身串行），CPU 模式根据核数开多个
- **Token budget** — 每批的总 token 数上限，而非条数上限。短文本一批可以塞 64 条，长文本一批可能只有 8 条
- **优先级** — API 层根据请求的 input 数量判断：1 条 = 查询（priority=0），多条 = 入库（priority=1）
- **背压** — 队列深度 > 阈值时返回 429，阈值 = max_batch_size × 50（约 50 批的积压）
