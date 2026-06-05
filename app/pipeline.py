"""ONNX Dense Embedding 推理引擎

封装 ONNX 模式下的 dense embedding 推理：tokenize → forward → postprocess。

关键优化：
1. 动态 max_length — padding 到 batch 内实际最大长度，而非 model_max_length
2. 纯 numpy postprocess — mean pooling + L2 normalize 全用 numpy，
   避免 torch tensor 的创建/转换开销（ONNX 路径不需要 torch）

注：真正的 tokenize/forward pipeline 重叠在 batcher 层实现
（当前批推理时，batcher 已在收集并预处理下一批），
所以这里不再单独起线程做 pipeline，保持单批同步执行最简洁高效。

@author performance-optimization
@since 2.1.0
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PipelinedDenseEngine:
    """ONNX Dense Embedding 引擎

    Args:
        tokenizer: HuggingFace tokenizer
        onnx_session: optimum ORTModelForFeatureExtraction 实例
        max_length: 模型最大输入长度
    """

    def __init__(self, tokenizer, onnx_session, max_length: int = 8192):
        self._tokenizer = tokenizer
        self._onnx_session = onnx_session
        self._max_length = max_length

        logger.info(
            "PipelinedDenseEngine initialized: max_length=%d (numpy postprocess)",
            max_length,
        )

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """执行 dense embedding 推理（单批）

        tokenize → ONNX forward → 纯 numpy mean pooling + L2 normalize
        """
        # Stage 1: Preprocess (tokenize)
        non_empty_indices, non_empty_texts = self._filter_empty(texts)
        if not non_empty_texts:
            return [[0.0] * 1024 for _ in texts]

        # 动态 max_length：padding 到 batch 实际最大长度
        actual_max = self._compute_actual_max(non_empty_texts)

        inputs = self._tokenizer(
            non_empty_texts,
            padding=True,
            truncation=True,
            max_length=actual_max,
            return_tensors="np",  # 直接输出 numpy，避免 torch 转换
        )

        # Stage 2: Forward (ONNX Runtime, C++ 实现，释放 GIL)
        outputs = self._onnx_session(**{k: v for k, v in inputs.items()})

        # Stage 3: Postprocess — 纯 numpy（零 torch 依赖）
        token_embeddings = self._to_numpy(outputs.last_hidden_state)
        attention_mask = self._to_numpy(inputs["attention_mask"])

        # Mean pooling: sum(token_emb * mask) / sum(mask)
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms

        all_embeddings = embeddings.tolist()

        # 填回原始位置（空文本对应零向量）
        dim = len(all_embeddings[0])
        results = [[0.0] * dim for _ in texts]
        for i, orig_idx in enumerate(non_empty_indices):
            results[orig_idx] = all_embeddings[i]

        return results

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        """统一转为 numpy array（兼容 torch tensor / list / ndarray）"""
        if isinstance(arr, np.ndarray):
            return arr
        if hasattr(arr, "numpy"):
            return arr.numpy()
        return np.array(arr)

    @staticmethod
    def _filter_empty(texts: list[str]) -> tuple[list[int], list[str]]:
        """过滤空文本，返回 (非空索引, 非空文本)"""
        non_empty_indices = []
        non_empty_texts = []
        for i, t in enumerate(texts):
            if t and t.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(t)
        return non_empty_indices, non_empty_texts

    def _compute_actual_max(self, texts: list[str]) -> int:
        """计算 batch 内实际最大 token 数（对齐到 8 的倍数）"""
        max_tokens = 0
        for text in texts:
            try:
                n = len(self._tokenizer.encode(text, add_special_tokens=False))
                max_tokens = max(max_tokens, n)
            except Exception:
                return self._max_length
        actual = min(max_tokens + 2, self._max_length)
        return ((actual + 7) // 8) * 8
