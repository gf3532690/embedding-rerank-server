# 性能优化需求 — 做到 Python 极限

## 背景

v2.0 自适应引擎已完成基础架构（token-level batching、优先级队列、背压机制）。
本轮优化目标：在 Python 生态内把推理性能压榨到极限，覆盖 GPU 和 CPU 两种部署场景。

## 现状

- ONNX Runtime 已实现（CPUExecutionProvider），但未做图优化和线程调优
- 无向量缓存，相同文本重复推理
- padding 到固定 max_length，短文本浪费严重
- Tokenizer 和推理串行执行，未重叠
- 无请求去重

## 优化目标

### GPU 场景
- 长文本推理速度翻倍（Flash Attention）
- 短文本批次计算量减少 5-10x（动态 max_length）
- 重复文本 0ms 返回（缓存）
- GPU 利用率接近 100%（pipeline 并行）
- 支持 INT8 量化（吞吐再翻倍）

### CPU 场景
- ONNX 图优化 O3（+10-30%）
- 线程精细调优（+20-50%）
- 重复文本 0ms 返回（缓存）
- 动态 max_length（减少无效计算）
- pipeline 并行隐藏 tokenize 延迟

## 约束

- 保持 API 接口不变
- 保持向后兼容（现有配置仍生效）
- 不引入重量级新依赖（flash-attn、cachetools 可以）
- 每轮迭代独立可交付、可验证
