"""模型加载与推理模块

封装 FlagEmbedding 的模型加载和推理逻辑，
提供批量推理接口供 DynamicBatcher 调用。
"""

import logging
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EmbedModel:
    """BGE-M3 Embedding 模型封装

    支持 Dense + Sparse 双路输出。
    """

    def __init__(self, model_path: str, device: str = "cuda", fp16: bool = True, max_length: int = 8192):
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.max_length = max_length
        self._model = None
        self._ready = False

    def load(self) -> None:
        """加载模型到指定设备"""
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading embedding model from %s ...", self.model_path)
        start = time.time()

        self._model = BGEM3FlagModel(
            self.model_path,
            use_fp16=self.fp16,
            device=self.device,
        )

        elapsed = time.time() - start
        logger.info("Embedding model loaded in %.1fs", elapsed)
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        """批量生成 dense embedding

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量为 float list
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        # 按长度排序减少 padding 浪费
        indexed_texts = list(enumerate(texts))
        indexed_texts.sort(key=lambda x: len(x[1]), reverse=True)

        sorted_texts = [t for _, t in indexed_texts]
        output = self._model.encode(
            sorted_texts,
            batch_size=min(len(sorted_texts), 32),
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vecs = output["dense_vecs"]
        if isinstance(dense_vecs, torch.Tensor):
            dense_vecs = dense_vecs.cpu().numpy()

        # 恢复原始顺序
        results = [None] * len(texts)
        for i, (orig_idx, _) in enumerate(indexed_texts):
            results[orig_idx] = dense_vecs[i].tolist()

        return results

    def encode_sparse(self, texts: list[str]) -> list[list[dict]]:
        """批量生成 sparse embedding (lexical weights)

        Args:
            texts: 文本列表

        Returns:
            稀疏向量列表，每个为 [{"index": token_id, "value": weight}, ...]
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        output = self._model.encode(
            texts,
            batch_size=min(len(texts), 32),
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        lexical_weights = output["lexical_weights"]
        results = []
        for weights in lexical_weights:
            # weights 是 dict[str, float]，key 是 token_id 字符串
            sparse_vec = [
                {"index": int(token_id), "value": float(weight)}
                for token_id, weight in weights.items()
                if weight > 0
            ]
            # 按 value 降序排列
            sparse_vec.sort(key=lambda x: x["value"], reverse=True)
            results.append(sparse_vec)

        return results

    def warmup(self, n_samples: int = 8) -> None:
        """模型预热，避免首次请求延迟高"""
        if not self._model:
            return
        logger.info("Warming up embedding model with %d samples...", n_samples)
        dummy_texts = ["预热测试文本"] * n_samples
        self.encode_dense(dummy_texts)
        self.encode_sparse(dummy_texts)
        logger.info("Embedding model warmup complete")


class RerankModel:
    """BGE Reranker 模型封装"""

    def __init__(self, model_path: str, device: str = "cuda", fp16: bool = True, max_length: int = 512):
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.max_length = max_length
        self._model = None
        self._ready = False

    def load(self) -> None:
        """加载 rerank 模型"""
        from FlagEmbedding import FlagReranker

        logger.info("Loading rerank model from %s ...", self.model_path)
        start = time.time()

        self._model = FlagReranker(
            self.model_path,
            use_fp16=self.fp16,
            device=self.device,
        )

        elapsed = time.time() - start
        logger.info("Rerank model loaded in %.1fs", elapsed)
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def compute_scores(self, query: str, documents: list[str]) -> list[float]:
        """计算 query 与每个 document 的相关性分数

        Args:
            query: 查询文本
            documents: 候选文档列表

        Returns:
            分数列表，与 documents 一一对应
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        pairs = [[query, doc] for doc in documents]
        scores = self._model.compute_score(
            pairs,
            max_length=self.max_length,
            batch_size=len(pairs),
        )

        # compute_score 单条时返回 float，多条返回 list
        if isinstance(scores, (int, float)):
            scores = [scores]

        return [float(s) for s in scores]

    def warmup(self, n_samples: int = 4) -> None:
        """模型预热"""
        if not self._model:
            return
        logger.info("Warming up rerank model with %d samples...", n_samples)
        dummy_docs = ["预热测试文档"] * n_samples
        self.compute_scores("预热测试查询", dummy_docs)
        logger.info("Rerank model warmup complete")
