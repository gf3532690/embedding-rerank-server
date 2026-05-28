"""配置加载模块

从 config.yaml 读取配置，支持环境变量覆盖。
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7997
    workers: int = 1


class EmbedModelConfig(BaseModel):
    path: str = "/models/bge-m3"
    device: str = "cuda"
    fp16: bool = True
    max_length: int = 8192


class RerankModelConfig(BaseModel):
    path: str = "/models/bge-reranker-v2-m3"
    device: str = "cuda"
    fp16: bool = True
    max_length: int = 512


class ModelsConfig(BaseModel):
    embed: EmbedModelConfig = EmbedModelConfig()
    rerank: RerankModelConfig = RerankModelConfig()


class BatchingConfig(BaseModel):
    max_batch_size: int = 64
    max_wait_ms: int = 10
    # GPU 推理并发数：1 表示串行（显存紧张时用），>1 允许并行推理
    max_concurrency: int = 1


class WarmupConfig(BaseModel):
    enabled: bool = True
    samples: int = 8


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    mode: str = "all"
    models: ModelsConfig = ModelsConfig()
    batching: BatchingConfig = BatchingConfig()
    warmup: WarmupConfig = WarmupConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """加载配置文件，支持环境变量覆盖

    优先级：环境变量 > config.yaml > 默认值
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
    if os.environ.get("MAX_BATCH_SIZE"):
        config.batching.max_batch_size = int(os.environ["MAX_BATCH_SIZE"])
    if os.environ.get("MAX_WAIT_MS"):
        config.batching.max_wait_ms = int(os.environ["MAX_WAIT_MS"])
    if os.environ.get("MAX_CONCURRENCY"):
        config.batching.max_concurrency = int(os.environ["MAX_CONCURRENCY"])

    return config
