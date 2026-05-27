"""Embedding & Rerank Server 入口

基于 FlagEmbedding 的模型推理服务，提供：
- /v1/embeddings — OpenAI 兼容 dense embedding
- /v1/embed_sparse — sparse embedding (lexical weights)
- /v1/rerank — 文档重排序
- /health — 模型就绪状态检查
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.batcher import DynamicBatcher
from app.config import AppConfig, load_config
from app.models import EmbedModel, RerankModel
from app.schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    HealthResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    SparseEmbeddingRequest,
)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# 全局状态
config: AppConfig = None
embed_model: EmbedModel = None
rerank_model: RerankModel = None
dense_batcher: DynamicBatcher = None
sparse_batcher: DynamicBatcher = None


async def _batch_dense(texts: list[str]) -> list[list[float]]:
    """Dense embedding 批量推理（在线程池中执行避免阻塞事件循环）"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_model.encode_dense, texts)


async def _batch_sparse(texts: list[str]) -> list[list[dict]]:
    """Sparse embedding 批量推理"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_model.encode_sparse, texts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时释放资源"""
    global config, embed_model, rerank_model, dense_batcher, sparse_batcher

    config = load_config()
    logger.info("Server mode: %s", config.mode)
    logger.info("Batching config: max_batch_size=%d, max_wait_ms=%d",
                config.batching.max_batch_size, config.batching.max_wait_ms)

    # 加载 Embedding 模型
    if config.mode in ("embed", "all"):
        embed_model = EmbedModel(
            model_path=config.models.embed.path,
            device=config.models.embed.device,
            fp16=config.models.embed.fp16,
            max_length=config.models.embed.max_length,
        )
        embed_model.load()

        # 初始化 batcher
        dense_batcher = DynamicBatcher(
            process_fn=_batch_dense,
            max_batch_size=config.batching.max_batch_size,
            max_wait_ms=config.batching.max_wait_ms,
        )
        sparse_batcher = DynamicBatcher(
            process_fn=_batch_sparse,
            max_batch_size=config.batching.max_batch_size,
            max_wait_ms=config.batching.max_wait_ms,
        )
        dense_batcher.start()
        sparse_batcher.start()

        if config.warmup.enabled:
            embed_model.warmup(config.warmup.samples)

    # 加载 Rerank 模型
    if config.mode in ("rerank", "all"):
        rerank_model = RerankModel(
            model_path=config.models.rerank.path,
            device=config.models.rerank.device,
            fp16=config.models.rerank.fp16,
            max_length=config.models.rerank.max_length,
        )
        rerank_model.load()

        if config.warmup.enabled:
            rerank_model.warmup()

    logger.info("All models loaded, server ready.")
    yield

    # 关闭
    if dense_batcher:
        await dense_batcher.stop()
    if sparse_batcher:
        await sparse_batcher.stop()
    logger.info("Server shutdown complete.")


app = FastAPI(
    title="Embedding & Rerank Server",
    description="基于 FlagEmbedding 的模型推理服务，OpenAI 兼容接口",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Health Check =====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口

    返回模型加载状态，供 aladdin 后端在启动时确认服务可用。
    status:
    - ready: 所有配置的模型已加载完成
    - loading: 模型正在加载中
    - error: 模型加载失败
    """
    models_loaded = {}

    if config.mode in ("embed", "all"):
        models_loaded["embedding"] = embed_model.ready if embed_model else False
    if config.mode in ("rerank", "all"):
        models_loaded["rerank"] = rerank_model.ready if rerank_model else False

    all_ready = all(models_loaded.values()) if models_loaded else False

    return HealthResponse(
        status="ready" if all_ready else "loading",
        models_loaded=models_loaded,
    )


# ===== Embedding Endpoints =====

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI 兼容的 dense embedding 接口"""
    if not embed_model or not embed_model.ready:
        raise HTTPException(status_code=503, detail="Embedding model not ready")

    # 统一为 list
    texts = request.input if isinstance(request.input, list) else [request.input]

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # 通过 batcher 提交（每个文本作为独立请求进入队列）
    # 但这里一次请求可能包含多条文本，直接作为一个 batch 提交更高效
    futures = [dense_batcher.submit(text) for text in texts]
    embeddings = await asyncio.gather(*futures)

    data = [
        EmbeddingData(embedding=emb, index=i)
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=config.models.embed.path,
        usage=EmbeddingUsage(prompt_tokens=sum(len(t) for t in texts)),
    )


@app.post("/v1/embed_sparse")
async def create_sparse_embeddings(request: SparseEmbeddingRequest):
    """Sparse embedding 接口

    请求格式：{"inputs": ["text1", "text2"], "model": "..."}
    返回格式：[[{"index": token_id, "value": weight}, ...], ...]
    """
    if not embed_model or not embed_model.ready:
        raise HTTPException(status_code=503, detail="Embedding model not ready")

    texts = request.inputs if isinstance(request.inputs, list) else [request.inputs]

    if not texts:
        raise HTTPException(status_code=400, detail="Inputs cannot be empty")

    futures = [sparse_batcher.submit(text) for text in texts]
    sparse_vecs = await asyncio.gather(*futures)

    return sparse_vecs


# ===== Rerank Endpoint =====

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """文档重排序接口"""
    if not rerank_model or not rerank_model.ready:
        raise HTTPException(status_code=503, detail="Rerank model not ready")

    if not request.documents:
        raise HTTPException(status_code=400, detail="Documents cannot be empty")

    # Rerank 不走 batcher（每次请求本身就是一个完整的 query-docs 对）
    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(
        None, rerank_model.compute_scores, request.query, request.documents
    )

    # 构建结果并按分数降序排列
    results = [
        RerankResult(index=i, relevance_score=score)
        for i, score in enumerate(scores)
    ]
    results.sort(key=lambda x: x.relevance_score, reverse=True)

    # 截取 top_n
    if request.top_n:
        results = results[: request.top_n]

    return RerankResponse(
        results=results,
        model=config.models.rerank.path,
    )





if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        workers=cfg.server.workers,
    )
