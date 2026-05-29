"""配置加载模块

从 config.yaml 读取配置，支持环境变量覆盖。
所有性能参数支持 "auto" 值 — 不配就自动探测，手动配置作为覆盖项。

@author hardware-adaptive-engine
@since 2.0.0
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ===== 类型定义 =====
# 支持 "auto" 字符串或实际数值
AutoInt = Union[int, str]
AutoStr = Union[str]
AutoBool = Union[bool, str]


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 7997
    workers: int = 1


class EmbedModelConfig(BaseModel):
    """Embedding 模型配置

    所有字段支持 "auto"：
    - device: auto → 自动探测 cuda/cpu
    - engine: auto → cuda 用 pytorch，cpu 用 onnx
    - fp16: auto → cuda 开启，cpu 关闭
    """
    path: str = "/models/bge-m3"
    # 设备：cuda / cpu / auto
    device: AutoStr = "auto"
    # 是否使用 fp16：true / false / auto
    fp16: AutoBool = "auto"
    # 最大输入 token 长度
    max_length: int = 8192
    # 推理引擎：pytorch / onnx / auto
    engine: AutoStr = "auto"


class RerankModelConfig(BaseModel):
    """Rerank 模型配置"""
    path: str = "/models/bge-reranker-v2-m3"
    device: AutoStr = "auto"
    fp16: AutoBool = "auto"
    max_length: int = 512
    engine: AutoStr = "auto"


class ModelsConfig(BaseModel):
    """模型配置"""
    embed: EmbedModelConfig = EmbedModelConfig()
    rerank: RerankModelConfig = RerankModelConfig()


class BatchingConfig(BaseModel):
    """Batching 配置

    所有数值型字段支持 "auto"：
    - max_batch_size: auto → 根据显存/核数自动决定
    - max_batch_tokens: auto → 根据显存估算 token budget
    - max_concurrency: auto → GPU=1, CPU=核数/2
    - max_wait_ms: 最大等待时间（毫秒）
    - backpressure_threshold: auto → max_batch_size × 50
    """
    # 每批最大条数
    max_batch_size: AutoInt = "auto"
    # 每批最大总 token 数（token-level batching 核心参数）
    max_batch_tokens: AutoInt = "auto"
    # 最大等待时间（毫秒），超时即使未凑够也推理
    max_wait_ms: int = 10
    # 推理并发数
    max_concurrency: AutoInt = "auto"
    # 背压阈值：队列深度超过此值返回 429
    backpressure_threshold: AutoInt = "auto"


class WarmupConfig(BaseModel):
    """预热配置"""
    enabled: bool = True
    samples: int = 8


class AppConfig(BaseModel):
    """应用总配置"""
    server: ServerConfig = ServerConfig()
    mode: str = "all"
    models: ModelsConfig = ModelsConfig()
    batching: BatchingConfig = BatchingConfig()
    warmup: WarmupConfig = WarmupConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """加载配置文件，支持环境变量覆盖

    优先级：环境变量 > config.yaml > 默认值（"auto"）

    加载后需要调用 hardware.auto_configure(config) 将 "auto" 替换为实际值。
    """
    config_data = {}

    # 确定配置文件路径
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")

    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    # 构建配置对象
    config = AppConfig(**config_data)

    # 环境变量覆盖（高优先级）
    if os.environ.get("MODE"):
        config.mode = os.environ["MODE"]
    if os.environ.get("PORT"):
        config.server.port = int(os.environ["PORT"])
    if os.environ.get("WORKERS"):
        config.server.workers = int(os.environ["WORKERS"])
    if os.environ.get("EMBED_MODEL_PATH"):
        config.models.embed.path = os.environ["EMBED_MODEL_PATH"]
    if os.environ.get("RERANK_MODEL_PATH"):
        config.models.rerank.path = os.environ["RERANK_MODEL_PATH"]
    if os.environ.get("DEVICE"):
        config.models.embed.device = os.environ["DEVICE"]
        config.models.rerank.device = os.environ["DEVICE"]
    if os.environ.get("ENGINE"):
        config.models.embed.engine = os.environ["ENGINE"]
        config.models.rerank.engine = os.environ["ENGINE"]
    if os.environ.get("MAX_BATCH_SIZE"):
        config.batching.max_batch_size = _parse_auto_int(os.environ["MAX_BATCH_SIZE"])
    if os.environ.get("MAX_BATCH_TOKENS"):
        config.batching.max_batch_tokens = _parse_auto_int(os.environ["MAX_BATCH_TOKENS"])
    if os.environ.get("MAX_WAIT_MS"):
        config.batching.max_wait_ms = int(os.environ["MAX_WAIT_MS"])
    if os.environ.get("MAX_CONCURRENCY"):
        config.batching.max_concurrency = _parse_auto_int(os.environ["MAX_CONCURRENCY"])
    if os.environ.get("BACKPRESSURE_THRESHOLD"):
        config.batching.backpressure_threshold = _parse_auto_int(
            os.environ["BACKPRESSURE_THRESHOLD"]
        )

    logger.info("Configuration loaded from: %s", config_path)
    return config


def _parse_auto_int(value: str) -> Union[int, str]:
    """解析可能为 "auto" 的整数值"""
    if value.lower().strip() == "auto":
        return "auto"
    return int(value)
