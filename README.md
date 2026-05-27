# Embedding & Rerank Server

基于 FlagEmbedding 的模型推理服务，提供 OpenAI 兼容的 API 接口。
专为 Aladdin RAG 系统设计，支持 Dense + Sparse 双路 Embedding 和文档 Rerank。

## 功能

- `/v1/embeddings` — OpenAI 兼容 dense embedding
- `/v1/embed_sparse` — BGE-M3 sparse embedding (lexical weights)
- `/v1/rerank` — 文档重排序
- `/health` — 模型就绪状态检查（供 aladdin 启动时探测）

## 性能优化

- **Dynamic Batching**：请求自动合批，凑够 batch 立即推理，未凑够超时也发
- **FP16 推理**：显存减半，速度翻倍
- **长度排序 Padding 优化**：同 batch 内按文本长度排序，减少无效计算
- **模型预热**：启动时 dummy 推理，避免首次请求延迟高

## 部署架构

生产环境推荐 Embedding 和 Rerank 分进程部署（共享镜像）：

```
embedding (port 7997) ─── GPU ─── bge-m3
reranker  (port 7998) ─── GPU ─── bge-reranker-v2-m3
```

## 快速开始

```bash
# 构建镜像
docker build -t embedding-rerank-server:latest .

# 启动（需要 nvidia-docker）
docker compose up -d

# 验证
curl http://localhost:7997/health
curl http://localhost:7998/health
```

## 配置

所有配置在 `config.yaml` 中，支持环境变量覆盖。
详见 config.yaml 注释。
