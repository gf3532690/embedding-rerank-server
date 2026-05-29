"""API 请求/响应模型定义

兼容 OpenAI embeddings API 格式。
"""

from typing import Optional

from pydantic import BaseModel, Field


# ===== Embedding =====

class EmbeddingRequest(BaseModel):
    """OpenAI 兼容的 embedding 请求"""
    input: str | list[str] = Field(..., description="待编码文本，支持单条或批量")
    model: Optional[str] = Field(None, description="模型名称（忽略，使用配置中的模型）")
    encoding_format: Optional[str] = Field("float", description="返回格式：float")


class EmbeddingData(BaseModel):
    """单条 embedding 结果"""
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """OpenAI 兼容的 embedding 响应"""
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = EmbeddingUsage()


# ===== Sparse Embedding =====

class SparseEmbeddingRequest(BaseModel):
    """Sparse embedding 请求"""
    inputs: str | list[str] = Field(..., description="待编码文本")
    model: Optional[str] = Field(None, description="模型名称")


class SparseEntry(BaseModel):
    """稀疏向量中的单个 token"""
    index: int
    value: float


# 响应格式：list[list[SparseEntry]]
# 即 [[{"index": 123, "value": 0.8}, ...], ...]


# ===== Rerank =====

class RerankRequest(BaseModel):
    """Rerank 请求"""
    query: str = Field(..., description="查询文本")
    documents: list[str] = Field(..., description="候选文档列表")
    top_n: Optional[int] = Field(None, description="返回前 N 个结果，默认全部返回")
    model: Optional[str] = Field(None, description="模型名称")


class RerankResult(BaseModel):
    """单条 rerank 结果"""
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    """Rerank 响应"""
    results: list[RerankResult]
    model: str


# ===== Health =====

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态：ready / loading / error")
    models_loaded: dict[str, bool] = Field(default_factory=dict, description="各模型加载状态")
    queue_depth: int = Field(0, description="当前队列中待处理的请求数")
    estimated_wait_seconds: float = Field(0.0, description="预估等待时间（秒）")
    version: str = "2.0.0"
