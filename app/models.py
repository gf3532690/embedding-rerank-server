"""模型加载与推理模块

支持两种推理引擎：
- pytorch: 使用 FlagEmbedding 原生推理（GPU 推荐）
- onnx: 使用 ONNX Runtime 推理（CPU 推荐，快 2-3x）
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
    engine="onnx" 时 Dense 用 ONNX Runtime，Sparse 仍用 FlagEmbedding。
    """

    def __init__(self, model_path: str, device: str = "cuda", fp16: bool = True,
                 max_length: int = 8192, engine: str = "pytorch"):
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.max_length = max_length
        self.engine = engine
        self._model = None  # FlagEmbedding model (for pytorch + sparse)
        self._onnx_session = None  # ONNX Runtime session (for dense)
        self._tokenizer = None  # tokenizer (for onnx mode)
        self._ready = False

    def load(self) -> None:
        """加载模型"""
        logger.info("Loading embedding model from %s (engine=%s)...", self.model_path, self.engine)
        start = time.time()

        if self.engine == "onnx":
            self._load_onnx()
        else:
            self._load_pytorch()

        elapsed = time.time() - start
        logger.info("Embedding model loaded in %.1fs", elapsed)
        self._ready = True

    def _load_pytorch(self) -> None:
        """PyTorch 模式加载（FlagEmbedding 原生）"""
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
            self.model_path,
            use_fp16=self.fp16,
            device=self.device,
        )

        # torch.compile 优化（仅 GPU）
        if self.device != "cpu":
            try:
                self._model.model = torch.compile(self._model.model, mode="reduce-overhead")
                logger.info("torch.compile applied (reduce-overhead mode)")
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode: %s", e)

    def _load_onnx(self) -> None:
        """ONNX 模式加载（optimum + onnxruntime）"""
        import os
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        onnx_path = os.path.join(self.model_path, "onnx")
        if not os.path.exists(onnx_path):
            # 如果没有 onnx 子目录，尝试直接从模型目录加载
            onnx_path = self.model_path

        self._tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        self._onnx_session = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path,
            provider="CPUExecutionProvider",
        )
        logger.info("ONNX model loaded from %s", onnx_path)

        # 同时加载 FlagEmbedding 用于 sparse（ONNX 不支持 sparse 输出）
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                self.model_path,
                use_fp16=False,
                device="cpu",
            )
            logger.info("FlagEmbedding model loaded for sparse support")
        except Exception as e:
            logger.warning("FlagEmbedding load failed, sparse will be unavailable: %s", e)

    @property
    def ready(self) -> bool:
        return self._ready

    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        """批量生成 dense embedding"""
        if self.engine == "onnx":
            return self._encode_dense_onnx(texts)
        return self._encode_dense_pytorch(texts)

    def _encode_dense_onnx(self, texts: list[str]) -> list[list[float]]:
        """ONNX Runtime dense 推理"""
        if not self._onnx_session or not self._tokenizer:
            raise RuntimeError("ONNX model not loaded")

        # 空文本过滤
        non_empty_indices = []
        non_empty_texts = []
        for i, t in enumerate(texts):
            if t and t.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(t)

        if not non_empty_texts:
            return [[0.0] * 1024 for _ in texts]

        # 分批推理（避免内存峰值过高）
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(non_empty_texts), batch_size):
            batch = non_empty_texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            outputs = self._onnx_session(**inputs)
            # Mean pooling over token embeddings
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.detach().numpy().tolist())

        # 填回原始位置
        dim = len(all_embeddings[0])
        results = [[0.0] * dim] * len(texts)
        for i, orig_idx in enumerate(non_empty_indices):
            results[orig_idx] = all_embeddings[i]

        return results

    def _encode_dense_pytorch(self, texts: list[str]) -> list[list[float]]:
        """PyTorch dense 推理（FlagEmbedding 原生）"""
        if not self._model:
            raise RuntimeError("Model not loaded")

        # 空文本过滤
        non_empty_indices = []
        non_empty_texts = []
        for i, t in enumerate(texts):
            if t and t.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(t)

        if not non_empty_texts:
            dim = 1024
            return [[0.0] * dim for _ in texts]

        # 按长度排序减少 padding 浪费
        indexed_texts = list(enumerate(non_empty_texts))
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

        # 恢复排序前的顺序
        sorted_results = [None] * len(non_empty_texts)
        for i, (orig_idx, _) in enumerate(indexed_texts):
            sorted_results[orig_idx] = dense_vecs[i].tolist()

        # 填回原始位置
        dim = len(sorted_results[0])
        results = [[0.0] * dim] * len(texts)
        for i, orig_idx in enumerate(non_empty_indices):
            results[orig_idx] = sorted_results[i]

        return results

    def encode_sparse(self, texts: list[str]) -> list[list[dict]]:
        """批量生成 sparse embedding（始终用 FlagEmbedding PyTorch）"""
        if not self._model:
            # ONNX 模式下如果 FlagEmbedding 加载失败，返回空
            return [[] for _ in texts]

        # 空文本过滤
        non_empty_indices = []
        non_empty_texts = []
        for i, t in enumerate(texts):
            if t and t.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(t)

        if not non_empty_texts:
            return [[] for _ in texts]

        output = self._model.encode(
            non_empty_texts,
            batch_size=min(len(non_empty_texts), 32),
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        lexical_weights = output["lexical_weights"]
        sparse_results = []
        for weights in lexical_weights:
            sparse_vec = [
                {"index": int(token_id), "value": float(weight)}
                for token_id, weight in weights.items()
                if weight > 0
            ]
            sparse_vec.sort(key=lambda x: x["value"], reverse=True)
            sparse_results.append(sparse_vec[:200])

        # 填回原始位置
        results = [[] for _ in texts]
        for i, orig_idx in enumerate(non_empty_indices):
            results[orig_idx] = sparse_results[i]

        return results

    def warmup(self, n_samples: int = 8) -> None:
        """模型预热"""
        if not self._ready:
            return
        logger.info("Warming up embedding model with %d samples...", n_samples)
        dummy_texts = ["预热测试文本"] * n_samples
        self.encode_dense(dummy_texts)
        if self._model:
            self.encode_sparse(dummy_texts)
        logger.info("Embedding model warmup complete")


class RerankModel:
    """BGE Reranker 模型封装"""

    def __init__(self, model_path: str, device: str = "cuda", fp16: bool = True,
                 max_length: int = 512, engine: str = "pytorch"):
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.max_length = max_length
        self.engine = engine
        self._model = None
        self._onnx_session = None
        self._tokenizer = None
        self._ready = False

    def load(self) -> None:
        """加载 rerank 模型"""
        logger.info("Loading rerank model from %s (engine=%s)...", self.model_path, self.engine)
        start = time.time()

        if self.engine == "onnx":
            self._load_onnx()
        else:
            self._load_pytorch()

        elapsed = time.time() - start
        logger.info("Rerank model loaded in %.1fs", elapsed)
        self._ready = True

    def _load_pytorch(self) -> None:
        """PyTorch 模式"""
        from FlagEmbedding import FlagReranker
        self._model = FlagReranker(
            self.model_path,
            use_fp16=self.fp16,
            device=self.device,
        )

    def _load_onnx(self) -> None:
        """ONNX 模式"""
        import os
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        onnx_path = os.path.join(self.model_path, "onnx")
        if not os.path.exists(onnx_path):
            onnx_path = self.model_path

        self._tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        self._onnx_session = ORTModelForSequenceClassification.from_pretrained(
            onnx_path,
            provider="CPUExecutionProvider",
        )
        logger.info("ONNX rerank model loaded from %s", onnx_path)

    @property
    def ready(self) -> bool:
        return self._ready

    def compute_scores(self, query: str, documents: list[str]) -> list[float]:
        """计算 query 与每个 document 的相关性分数"""
        if self.engine == "onnx":
            return self._compute_scores_onnx(query, documents)
        return self._compute_scores_pytorch(query, documents)

    def _compute_scores_pytorch(self, query: str, documents: list[str]) -> list[float]:
        """PyTorch 推理"""
        if not self._model:
            raise RuntimeError("Model not loaded")
        pairs = [[query, doc] for doc in documents]
        scores = self._model.compute_score(
            pairs,
            max_length=self.max_length,
            batch_size=len(pairs),
        )
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(s) for s in scores]

    def _compute_scores_onnx(self, query: str, documents: list[str]) -> list[float]:
        """ONNX Runtime 推理"""
        if not self._onnx_session or not self._tokenizer:
            raise RuntimeError("ONNX rerank model not loaded")

        pairs = [[query, doc] for doc in documents]
        scores = []
        batch_size = 16
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            outputs = self._onnx_session(**inputs)
            # Cross-encoder 输出 logits，取第一列作为相关性分数
            logits = outputs.logits
            if logits.shape[-1] == 1:
                batch_scores = logits.squeeze(-1).detach().numpy().tolist()
            else:
                batch_scores = logits[:, 0].detach().numpy().tolist()
            scores.extend(batch_scores)

        return scores

    def warmup(self, n_samples: int = 4) -> None:
        """模型预热"""
        if not self._ready:
            return
        logger.info("Warming up rerank model with %d samples...", n_samples)
        dummy_docs = ["预热测试文档"] * n_samples
        self.compute_scores("预热测试查询", dummy_docs)
        logger.info("Rerank model warmup complete")
