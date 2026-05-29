# 任务列表

## 开发顺序（按依赖关系排列）

- [ ] 1. 创建 `app/hardware.py` — 硬件探测模块
  - 探测 GPU/CPU、显存、核数
  - 实现 probe_max_batch_size（dummy 推理试探安全值）
  - 实现 auto_configure（填充所有 auto 值）

- [ ] 2. 重写 `app/batcher.py` — Token-level continuous batching
  - 按总 token 数合批（max_batch_tokens）
  - 优先级队列（单条优先于批量）
  - 去掉 request_timeout（无限等待）
  - Tokenizer pipeline 重叠（预 tokenize 下一批）
  - 背压机制（队列过深返回 429）

- [ ] 3. 调整 `app/models.py` — 适配新 batcher
  - 接收已 tokenized 输入
  - 去掉内部 batch_size 限制（由 batcher 控制）
  - 保持 ONNX/PyTorch 双引擎

- [ ] 4. 调整 `app/main.py` — 集成新模块
  - lifespan 中调用 auto_configure
  - 去掉 asyncio.wait_for 超时逻辑
  - API 层根据 input 数量设置优先级
  - /health 返回队列深度和预估等待时间

- [ ] 5. 创建 `app/metrics.py` — Prometheus 监控
  - 请求计数器（按端点分）
  - 延迟直方图
  - Batch 大小直方图
  - 队列深度 Gauge
  - 吞吐量（tokens/sec）
  - 注册 /metrics 端点

- [ ] 6. 调整 `app/config.py` — 支持 auto 值
  - 数值型字段支持 "auto" 字符串
  - load_config 后调用 auto_configure 替换 auto 为实际值
  - 启动日志输出最终配置值

- [ ] 7. 更新配置文件
  - config.yaml 默认值改为 auto
  - config-cpu.yaml 同步
  - docker-compose 去掉不需要的环境变量

- [ ] 8. 更新文档
  - README.md 更新
  - DEPLOYMENT.md 更新
  - 新增 ARCHITECTURE.md 说明自适应逻辑

- [ ] 9. 测试验证
  - GPU 模式启动验证
  - CPU 模式启动验证
  - 大批量请求不超时验证
  - 优先级队列验证（查询不被入库阻塞）
  - /metrics 端点验证
