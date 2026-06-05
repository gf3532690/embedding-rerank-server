# Embedding & Rerank Server v2.0.0 (CPU)
# 部署日期: 2026-05-29 14:26
# 目标平台: linux/arm64
#
# 部署步骤:
#   1. docker load -i embedding-rerank-server-cpu.tar
#   2. docker compose up -d
#   3. curl http://localhost:7997/health
#
# v2.0 特性:
#   - 自动探测 CPU 核数，配置最优线程数和并发
#   - Token-level batching，按 token 数合批
#   - ONNX Runtime 推理加速
#   - /metrics Prometheus 监控端点
#   - 队列过深返回 429 (背压机制)
#
# CPU 模式注意事项:
#   - 确保模型目录有 onnx/ 子目录
#   - OMP_NUM_THREADS 自动设置，无需手动配置
#   - 如需覆盖参数，编辑 config.yaml
