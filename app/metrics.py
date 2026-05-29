"""Prometheus 监控指标模块

提供以下指标：
- 请求计数器（按端点分）
- 延迟直方图
- Batch 大小直方图
- 队列深度 Gauge
- 吞吐量（tokens/sec）

通过 /metrics 端点暴露 Prometheus 格式数据。

@author hardware-adaptive-engine
@since 2.0.0
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 延迟导入 prometheus_client，不可用时优雅降级
_prometheus_available = False
_REQUESTS_TOTAL = None
_REQUEST_DURATION = None
_BATCH_SIZE_HISTOGRAM = None
_BATCH_TOKENS_HISTOGRAM = None
_QUEUE_DEPTH = None
_TOKENS_PER_SECOND = None
_INFERENCE_DURATION = None

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _prometheus_available = True
except ImportError:
    logger.info("prometheus_client not installed, metrics will be disabled")


# 自定义 registry（避免与其他库冲突）
_registry: Optional["CollectorRegistry"] = None


def setup_metrics() -> None:
    """初始化 Prometheus 指标

    在应用启动时调用一次。
    """
    global _registry, _REQUESTS_TOTAL, _REQUEST_DURATION, _BATCH_SIZE_HISTOGRAM
    global _BATCH_TOKENS_HISTOGRAM, _QUEUE_DEPTH, _TOKENS_PER_SECOND, _INFERENCE_DURATION

    if not _prometheus_available:
        return

    _registry = CollectorRegistry()

    # 请求计数器（按端点和状态码分）
    _REQUESTS_TOTAL = Counter(
        "ers_requests_total",
        "Total number of requests",
        ["endpoint", "status_code"],
        registry=_registry,
    )

    # 请求延迟直方图（端到端，包含排队时间）
    _REQUEST_DURATION = Histogram(
        "ers_request_duration_seconds",
        "Request duration in seconds (end-to-end)",
        ["endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        registry=_registry,
    )

    # 推理延迟直方图（纯推理时间，不含排队）
    _INFERENCE_DURATION = Histogram(
        "ers_inference_duration_seconds",
        "Inference duration in seconds (pure model inference)",
        ["model_type"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        registry=_registry,
    )

    # Batch 大小直方图
    _BATCH_SIZE_HISTOGRAM = Histogram(
        "ers_batch_size",
        "Number of items per batch",
        ["model_type"],
        buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        registry=_registry,
    )

    # Batch token 数直方图
    _BATCH_TOKENS_HISTOGRAM = Histogram(
        "ers_batch_tokens",
        "Total tokens per batch",
        ["model_type"],
        buckets=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        registry=_registry,
    )

    # 队列深度 Gauge
    _QUEUE_DEPTH = Gauge(
        "ers_queue_depth",
        "Current queue depth",
        ["batcher_type"],
        registry=_registry,
    )

    # 吞吐量（tokens/sec）
    _TOKENS_PER_SECOND = Gauge(
        "ers_tokens_per_second",
        "Tokens processed per second (rolling average)",
        ["model_type"],
        registry=_registry,
    )

    logger.info("Prometheus metrics initialized")


def get_metrics_response():
    """生成 Prometheus 格式的 metrics 响应"""
    from fastapi.responses import Response

    if not _prometheus_available or _registry is None:
        return Response(
            content="# prometheus_client not available\n",
            media_type="text/plain",
        )

    # 更新队列深度（从 batcher 获取实时值）
    _update_queue_depth()

    content = generate_latest(_registry)
    return Response(content=content, media_type="text/plain; charset=utf-8")


def _update_queue_depth() -> None:
    """从全局 batcher 更新队列深度指标"""
    if _QUEUE_DEPTH is None:
        return
    try:
        from app.main import dense_batcher, sparse_batcher
        if dense_batcher:
            _QUEUE_DEPTH.labels(batcher_type="dense").set(dense_batcher.queue_depth)
        if sparse_batcher:
            _QUEUE_DEPTH.labels(batcher_type="sparse").set(sparse_batcher.queue_depth)
    except (ImportError, AttributeError):
        pass


# ===== 指标记录辅助函数 =====

def record_request(endpoint: str, status_code: int) -> None:
    """记录请求计数"""
    if _REQUESTS_TOTAL:
        _REQUESTS_TOTAL.labels(endpoint=endpoint, status_code=str(status_code)).inc()


def record_request_duration(endpoint: str, duration: float) -> None:
    """记录请求延迟"""
    if _REQUEST_DURATION:
        _REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)


def record_inference_duration(model_type: str, duration: float) -> None:
    """记录推理延迟"""
    if _INFERENCE_DURATION:
        _INFERENCE_DURATION.labels(model_type=model_type).observe(duration)


def record_batch_size(model_type: str, size: int) -> None:
    """记录 batch 大小"""
    if _BATCH_SIZE_HISTOGRAM:
        _BATCH_SIZE_HISTOGRAM.labels(model_type=model_type).observe(size)


def record_batch_tokens(model_type: str, tokens: int) -> None:
    """记录 batch token 数"""
    if _BATCH_TOKENS_HISTOGRAM:
        _BATCH_TOKENS_HISTOGRAM.labels(model_type=model_type).observe(tokens)


def record_throughput(model_type: str, tokens_per_sec: float) -> None:
    """记录吞吐量"""
    if _TOKENS_PER_SECOND:
        _TOKENS_PER_SECOND.labels(model_type=model_type).set(tokens_per_sec)
