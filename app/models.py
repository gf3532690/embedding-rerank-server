"""模型加载与推理模块

支持两种推理引擎：
- pytorch: 使用 FlagEmbedding 原生推理（GPU 推荐）
- onnx: 使用 ONNX Runtime 推理（CPU 推荐，快 2-3x）

适配 TokenBatcher：
- 去掉内部 batch_size 限制（由 batcher 控制批大小）
- 保持 ONNX/PyTorch 双引擎
- 提供 tokenize 方法供 batcher 估算 token 数

@author hardware-adaptive-engine
@since 2.0.0
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
                 max_length: int = 8192, engine: str = "pytorch",
                 quantization: str = "none"):
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.max_length = max_length
        self.engine = engine
        self.quantization = quantization
        self._model = None  # FlagEmbedding model (for pytorch + sparse)
        self._onnx_session = None  # ONNX Runtime session (for dense)
        self._tokenizer = None  # tokenizer (for onnx mode + token counting)
        self._pipeline_engine = None  # 3-stage pipeline (ONNX mode)
        self._ready = False

    def load(self) -> None:
        """加载模型"""
        logger.info("Loading embedding model from %s (engine=%s, device=%s)...",
                    self.model_path, self.engine, self.device)
        start = time.time()

        if self.engine == "onnx":
            self._load_onnx()
        else:
            self._load_pytorch()

        # 确保 tokenizer 可用（用于 token 计数）
        self._ensure_tokenizer()

        elapsed = time.time() - start
        logger.info("Embedding model loaded in %.1fs", elapsed)
        self._ready = True

    def _load_pytorch(self) -> None:
        """PyTorch 模式加载（FlagEmbedding 原生）

        支持 INT8 量化（通过 bitsandbytes 或 torch 动态量化）。
        """
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
            self.model_path,
            use_fp16=self.fp16,
            device=self.device,
        )

        # INT8 量化（GPU: bitsandbytes, CPU: torch 动态量化）
        if self.quantization == "int8":
            self._apply_quantization()

        # 检测 Flash Attention 是否可用
        self._log_flash_attention_status()

        # torch.compile 优化（仅 GPU，且未量化时）
        # max-autotune: 自动搜索最优 kernel 配置，首次慢但后续最快
        # 编译缓存存储到 /tmp/torch_compile_cache，重启后复用
        if self.device != "cpu" and self.quantization == "none":
            try:
                import os
                # 启用编译缓存（避免每次重启都重新编译）
                cache_dir = os.environ.get("TORCH_COMPILE_CACHE", "/tmp/torch_compile_cache")
                os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)

                self._model.model = torch.compile(
                    self._model.model,
                    mode="max-autotune",
                    fullgraph=False,  # 允许 graph break（兼容性更好）
                )
                logger.info(
                    "torch.compile applied (max-autotune mode, cache=%s)",
                    cache_dir,
                )
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode: %s", e)

    def _apply_quantization(self) -> None:
        """应用 INT8 量化

        GPU: 尝试 bitsandbytes（需要安装）
        CPU: 使用 torch 动态量化（内置，无需额外依赖）
        """
        if self.device == "cpu":
            # CPU: torch 动态量化（对 Linear 层量化）
            try:
                self._model.model = torch.quantization.quantize_dynamic(
                    self._model.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                logger.info("INT8 dynamic quantization applied (CPU, torch built-in)")
            except Exception as e:
                logger.warning("CPU INT8 quantization failed: %s", e)
        else:
            # GPU: 尝试 bitsandbytes
            try:
                import bitsandbytes  # noqa: F401
                # bitsandbytes 需要在模型加载时指定，这里做 post-hoc 量化
                # 对于 FlagEmbedding，post-hoc 量化效果有限
                # 更好的方式是在加载时用 load_in_8bit=True（需要改 FlagEmbedding 源码）
                logger.info(
                    "bitsandbytes detected but FlagEmbedding doesn't support load_in_8bit. "
                    "Using FP16 instead. For INT8, use ONNX engine with quantized model."
                )
            except ImportError:
                logger.info(
                    "bitsandbytes not installed, skipping GPU INT8. "
                    "Install with: pip install bitsandbytes"
                )

    def _load_onnx(self) -> None:
        """ONNX 模式加载（optimum + onnxruntime）

        包含 O3 图优化和线程精细调优。
        支持 INT8 量化模型（查找 onnx_int8/ 目录）。
        """
        import os
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        onnx_path = os.path.join(self.model_path, "onnx")
        if not os.path.exists(onnx_path):
            onnx_path = self.model_path

        # INT8 量化模型优先
        if self.quantization == "int8":
            int8_path = os.path.join(self.model_path, "onnx_int8")
            if os.path.exists(int8_path):
                onnx_path = int8_path
                logger.info("Using INT8 quantized ONNX model from %s", int8_path)
            else:
                logger.info(
                    "INT8 model not found at %s, using default ONNX model",
                    int8_path,
                )

        self._tokenizer = AutoTokenizer.from_pretrained(onnx_path)

        # 配置 ONNX Runtime SessionOptions（性能优化）
        session_options = self._create_ort_session_options()

        self._onnx_session = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path,
            provider="CPUExecutionProvider",
            session_options=session_options,
        )
        logger.info("ONNX model loaded from %s (O3 optimized)", onnx_path)

        # 初始化 3 阶段 Pipeline 引擎
        from app.pipeline import PipelinedDenseEngine
        self._pipeline_engine = PipelinedDenseEngine(
            tokenizer=self._tokenizer,
            onnx_session=self._onnx_session,
            max_length=self.max_length,
        )
        logger.info("3-stage pipeline engine initialized")

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

    def _create_ort_session_options(self):
        """创建优化的 ONNX Runtime SessionOptions

        - graph_optimization_level = ORT_ENABLE_ALL (O3)
        - intra_op_num_threads: 单次推理内部并行线程数
        - inter_op_num_threads: 算子间并行度
        - execution_mode: 根据场景选择串行或并行
        """
        import onnxruntime as ort
        from app.hardware import detect_cpu_cores

        cpu_cores = detect_cpu_cores()

        sess_options = ort.SessionOptions()

        # O3 最高级别图优化（算子融合、常量折叠、冗余消除）
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 线程配置：
        # intra_op = 单个算子内部的并行线程数（如矩阵乘法）
        # inter_op = 不同算子之间的并行度
        # 经验值：intra_op 占大部分核，inter_op 给 2-4 个
        intra_threads = max(cpu_cores - 2, 1)
        inter_threads = min(2, cpu_cores)

        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = inter_threads

        # 执行模式：ORT_SEQUENTIAL 对小 batch 更快（减少线程调度开销）
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 启用内存优化
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        logger.info(
            "ONNX SessionOptions: optimization=O3, intra_threads=%d, "
            "inter_threads=%d, execution=SEQUENTIAL",
            intra_threads,
            inter_threads,
        )

        return sess_options

    def _ensure_tokenizer(self) -> None:
        """确保 tokenizer 可用（用于 token 计数）"""
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("Tokenizer loaded for token counting")
        except Exception as e:
            logger.warning("Failed to load tokenizer for counting, will use estimation: %s", e)

    @staticmethod
    def _log_flash_attention_status() -> None:
        """检测并记录 Flash Attention 状态"""
        try:
            import flash_attn  # noqa: F401
            logger.info(
                "Flash Attention %s detected — enabled for long sequence acceleration",
                getattr(flash_attn, "__version__", "unknown"),
            )
        except ImportError:
            logger.info(
                "Flash Attention not installed — long sequences will use standard attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量

        供 TokenBatcher 使用，用于按 token budget 合批。

        Args:
            text: 输入文本

        Returns:
            token 数量
        """
        if not text or not text.strip():
            return 1
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # 回退：按字符估算（中文 1 字 ≈ 1.5 tokens，英文 4 chars ≈ 1 token）
        return max(len(text) // 3, 1)

    def _compute_actual_max_length(self, texts: list[str]) -> int:
        """计算 batch 内实际最大 token 数，用于动态 padding

        避免短文本 batch padding 到 model_max_length（如 8192）造成巨大计算浪费。
        例如：一批全是 50 token 的短文本，padding 到 56 而非 8192，计算量差 146 倍。

        Args:
            texts: 文本列表

        Returns:
            实际应使用的 max_length（含特殊 token，对齐到 8 的倍数）
        """
        if self._tokenizer is None:
            return self.max_length

        max_tokens = 0
        for text in texts:
            try:
                n_tokens = len(self._tokenizer.encode(text, add_special_tokens=False))
                max_tokens = max(max_tokens, n_tokens)
            except Exception:
                return self.max_length

        # +2 for [CLS] + [SEP]，对齐到 8 的倍数（GPU tensor core 友好）
        actual = min(max_tokens + 2, self.max_length)
        actual = ((actual + 7) // 8) * 8
        return min(actual, self.max_length)

    @property
    def ready(self) -> bool:
        return self._ready

    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        """批量生成 dense embedding

        由 TokenBatcher 控制批大小，此处不再内部分批。

        Args:
            texts: 文本列表（批大小由 batcher 决定）

        Returns:
            embedding 列表
        """
        if self.engine == "onnx":
            return self._encode_dense_onnx(texts)
        return self._encode_dense_pytorch(texts)

    def _encode_dense_onnx(self, texts: list[str]) -> list[list[float]]:
        """ONNX Runtime dense 推理 — 使用 3 阶段 Pipeline 引擎

        Pipeline 引擎内部处理：tokenize → forward → postprocess
        包含动态 max_length 优化。
        """
        if self._pipeline_engine:
            return self._pipeline_engine.encode_batch(texts)

        # Fallback: 无 pipeline 时直接推理
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

        # 动态 max_length：计算 batch 内实际最大 token 数，padding 到该值
        # 避免短文本 batch padding 到 model_max_length 造成巨大浪费
        actual_max_length = self._compute_actual_max_length(non_empty_texts)

        # 直接推理整个 batch（大小由 batcher 控制）
        inputs = self._tokenizer(
            non_empty_texts,
            padding=True,
            truncation=True,
            max_length=actual_max_length,
            return_tensors="pt",
        )
        outputs = self._onnx_session(**inputs)
        # Mean pooling over token embeddings
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings = embeddings.detach().numpy().tolist()

        # 填回原始位置
        dim = len(all_embeddings[0])
        results = [[0.0] * dim for _ in texts]
        for i, orig_idx in enumerate(non_empty_indices):
            results[orig_idx] = all_embeddings[i]

        return results

    def _encode_dense_pytorch(self, texts: list[str]) -> list[list[float]]:
        """PyTorch dense 推理（FlagEmbedding 原生）

        不再内部分批 — 由 TokenBatcher 保证每批 token 总量在安全范围内。
        """
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
        # batch_size 设为整个列表大小（由 batcher 控制外部批大小）
        # 动态 max_length：使用 batch 内实际最大长度
        actual_max_length = self._compute_actual_max_length(sorted_texts)
        output = self._model.encode(
            sorted_texts,
            batch_size=len(sorted_texts),
            max_length=actual_max_length,
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
        results = [[0.0] * dim for _ in texts]
        for i, orig_idx in enumerate(non_empty_indices):
            results[orig_idx] = sorted_results[i]

        return results

    def encode_sparse(self, texts: list[str]) -> list[list[dict]]:
        """批量生成 sparse embedding（始终用 FlagEmbedding PyTorch）

        Args:
            texts: 文本列表（批大小由 batcher 决定）

        Returns:
            sparse embedding 列表
        """
        if not self._model:
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
            batch_size=len(non_empty_texts),
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
        dummy_texts = ["预热测试文本，用于初始化模型推理路径。"] * n_samples
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
        logger.info("Loading rerank model from %s (engine=%s, device=%s)...",
                    self.model_path, self.engine, self.device)
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
        """ONNX 模式（含 O3 图优化）"""
        import os
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        from app.hardware import detect_cpu_cores

        onnx_path = os.path.join(self.model_path, "onnx")
        if not os.path.exists(onnx_path):
            onnx_path = self.model_path

        self._tokenizer = AutoTokenizer.from_pretrained(onnx_path)

        # SessionOptions 优化
        cpu_cores = detect_cpu_cores()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(cpu_cores - 2, 1)
        sess_options.inter_op_num_threads = min(2, cpu_cores)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        self._onnx_session = ORTModelForSequenceClassification.from_pretrained(
            onnx_path,
            provider="CPUExecutionProvider",
            session_options=sess_options,
        )
        logger.info("ONNX rerank model loaded from %s (O3 optimized)", onnx_path)

    @property
    def ready(self) -> bool:
        return self._ready

    def compute_scores(self, query: str, documents: list[str]) -> list[float]:
        """计算 query 与每个 document 的相关性分数

        不再内部分批 — rerank 本身每次请求就是一个完整的 query-docs 对。
        """
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
        # 直接推理整个 batch
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        outputs = self._onnx_session(**inputs)
        logits = outputs.logits
        if logits.shape[-1] == 1:
            scores = logits.squeeze(-1).detach().numpy().tolist()
        else:
            scores = logits[:, 0].detach().numpy().tolist()

        if isinstance(scores, float):
            scores = [scores]
        return scores

    def warmup(self, n_samples: int = 4) -> None:
        """模型预热"""
        if not self._ready:
            return
        logger.info("Warming up rerank model with %d samples...", n_samples)
        dummy_docs = ["预热测试文档，用于初始化模型推理路径。"] * n_samples
        self.compute_scores("预热测试查询", dummy_docs)
        logger.info("Rerank model warmup complete")
