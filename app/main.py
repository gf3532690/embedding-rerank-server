"""Embedding & Rerank Server 入口

基于 FlagEmbedding 的自适应推理服务，提供：
- /v1/embeddings — OpenAI 兼容 dense embedding
- /v1/embed_sparse — sparse embedding (lexical weights)
- /v1/rerank — 文档重排序
- /health — 模型就绪状态 + 队列深度 + 预估等待时间
- /metrics — Prometheus 监控端点

核心特性：
- 自动硬件探测，零配置启动
- Token-level continuous batching
- 优先级队列（查询优先于入库）
- 背压机制（队列过深返回 429）
- 无超时，永不主动拒绝请求
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.batcher import BackpressureError, TokenBatcher
from app.config import AppConfig, load_config
from app.hardware import auto_configure
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
dense_batcher: TokenBatcher = None
sparse_batcher: TokenBatcher = None
# GPU 推理信号量：控制同时执行的推理任务数，避免显存叠加 OOM
_gpu_semaphore: asyncio.Semaphore = None


async def _batch_dense(texts: list[str]) -> list[list[float]]:
    """Dense embedding 批量推理（在线程池中执行避免阻塞事件循环）"""
    async with _gpu_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, embed_model.encode_dense, texts)


async def _batch_sparse(texts: list[str]) -> list[list[dict]]:
    """Sparse embedding 批量推理"""
    async with _gpu_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, embed_model.encode_sparse, texts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时释放资源"""
    global config, embed_model, rerank_model, dense_batcher, sparse_batcher, _gpu_semaphore

    # 加载配置
    config = load_config()

    # 自动配置：根据硬件探测结果填充所有 "auto" 值
    auto_configure(config)

    _gpu_semaphore = asyncio.Semaphore(config.batching.max_concurrency)
    logger.info("Server mode: %s", config.mode)
    logger.info(
        "Batching config: max_batch_size=%d, max_batch_tokens=%d, "
        "max_concurrency=%d, backpressure_threshold=%d",
        config.batching.max_batch_size,
        config.batching.max_batch_tokens,
        config.batching.max_concurrency,
        config.batching.backpressure_threshold,
    )

    # 加载 Embedding 模型
    if config.mode in ("embed", "all"):
        embed_model = EmbedModel(
            model_path=config.models.embed.path,
            device=config.models.embed.device,
            fp16=config.models.embed.fp16,
            max_length=config.models.embed.max_length,
            engine=config.models.embed.engine,
        )
        embed_model.load()

        # 初始化 TokenBatcher（使用模型的 tokenizer 进行精确 token 计数）
        dense_batcher = TokenBatcher(
            process_fn=_batch_dense,
            tokenize_fn=embed_model.count_tokens,
            max_batch_tokens=config.batching.max_batch_tokens,
            max_batch_size=config.batching.max_batch_size,
            max_wait_ms=config.batching.max_wait_ms,
            backpressure_threshold=config.batching.backpressure_threshold,
        )
        sparse_batcher = TokenBatcher(
            process_fn=_batch_sparse,
            tokenize_fn=embed_model.count_tokens,
            max_batch_tokens=config.batching.max_batch_tokens,
            max_batch_size=config.batching.max_batch_size,
            max_wait_ms=config.batching.max_wait_ms,
            backpressure_threshold=config.batching.backpressure_threshold,
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
            engine=config.models.rerank.engine,
        )
        rerank_model.load()

        if config.warmup.enabled:
            rerank_model.warmup()

    # 注册 metrics 端点（如果 prometheus_client 可用）
    _register_metrics(app)

    logger.info("All models loaded, server ready.")
    yield

    # 关闭
    if dense_batcher:
        await dense_batcher.stop()
    if sparse_batcher:
        await sparse_batcher.stop()
    logger.info("Server shutdown complete.")


def _register_metrics(app: FastAPI) -> None:
    """尝试注册 Prometheus metrics 端点"""
    try:
        from app.metrics import get_metrics_response, setup_metrics
        setup_metrics()

        @app.get("/metrics")
        async def metrics_endpoint():
            return get_metrics_response()

        logger.info("Prometheus /metrics endpoint registered")
    except ImportError:
        logger.info("prometheus_client not installed, /metrics endpoint disabled")
    except Exception as e:
        logger.warning("Failed to register /metrics endpoint: %s", e)


app = FastAPI(
    title="Embedding & Rerank Server",
    description="自适应推理服务：自动硬件探测、Token-level batching、优先级队列",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 背压异常处理 =====

@app.exception_handler(BackpressureError)
async def backpressure_handler(request: Request, exc: BackpressureError):
    """队列过深时返回 429 + Retry-After"""
    retry_after = max(int(exc.estimated_wait), 1)
    return JSONResponse(
        status_code=429,
        content={
            "error": "Server overloaded",
            "detail": str(exc),
            "queue_depth": exc.queue_depth,
            "estimated_wait_seconds": exc.estimated_wait,
        },
        headers={"Retry-After": str(retry_after)},
    )


# ===== Health Check =====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口

    返回模型加载状态、队列深度和预估等待时间。
    """
    models_loaded = {}

    if config.mode in ("embed", "all"):
        models_loaded["embedding"] = embed_model.ready if embed_model else False
    if config.mode in ("rerank", "all"):
        models_loaded["rerank"] = rerank_model.ready if rerank_model else False

    all_ready = all(models_loaded.values()) if models_loaded else False

    # 队列状态
    queue_depth = 0
    estimated_wait = 0.0
    if dense_batcher:
        queue_depth = dense_batcher.queue_depth
        estimated_wait = dense_batcher.estimated_wait_time()

    return HealthResponse(
        status="ready" if all_ready else "loading",
        models_loaded=models_loaded,
        queue_depth=queue_depth,
        estimated_wait_seconds=estimated_wait,
    )


# ===== Embedding Endpoints =====

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI 兼容的 dense embedding 接口

    优先级规则：
    - 单条输入 → priority=0（查询场景，低延迟）
    - 多条输入 → priority=1（入库场景，高吞吐）
    """
    if not embed_model or not embed_model.ready:
        raise HTTPException(status_code=503, detail="Embedding model not ready")

    # 统一为 list
    texts = request.input if isinstance(request.input, list) else [request.input]

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # 根据输入数量设置优先级
    priority = 0 if len(texts) == 1 else 1

    # 通过 TokenBatcher 提交（无超时，永不主动拒绝）
    try:
        futures = [dense_batcher.submit(text, priority=priority) for text in texts]
        embeddings = await asyncio.gather(*futures)
    except BackpressureError:
        raise  # 由 exception_handler 处理

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

    # 根据输入数量设置优先级
    priority = 0 if len(texts) == 1 else 1

    try:
        futures = [sparse_batcher.submit(text, priority=priority) for text in texts]
        sparse_vecs = await asyncio.gather(*futures)
    except BackpressureError:
        raise

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
    async with _gpu_semaphore:
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
    auto_configure(cfg)
    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        workers=cfg.server.workers,
    )
