"""向量缓存模块

LRU 缓存，避免相同文本重复推理。
- Dense embedding 和 Sparse embedding 分别缓存
- Key = 文本 MD5 hash（节省内存，避免存储原始文本）
- 支持配置开关和容量限制
- 接入 Prometheus metrics（命中率统计）

@author performance-optimization
@since 2.1.0
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LRUCache:
    """线程安全的 LRU 缓存

    使用 OrderedDict 实现，O(1) 读写。

    Args:
        max_size: 最大缓存条数
        name: 缓存名称（用于日志和 metrics）
    """

    def __init__(self, max_size: int = 100_000, name: str = "default"):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._name = name
        self._lock = threading.Lock()

        # 统计
        self._hits: int = 0
        self._misses: int = 0

    @property
    def size(self) -> int:
        """当前缓存条数"""
        return len(self._cache)

    @property
    def hits(self) -> int:
        """缓存命中次数"""
        return self._hits

    @property
    def misses(self) -> int:
        """缓存未命中次数"""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，命中时移到末尾（最近使用）

        Args:
            key: 缓存 key

        Returns:
            缓存值，未命中返回 None
        """
        with self._lock:
            if key in self._cache:
                # 移到末尾（最近使用）
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """写入缓存，超出容量时淘汰最久未使用的

        Args:
            key: 缓存 key
            value: 缓存值
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                self._cache[key] = value
                if len(self._cache) > self._max_size:
                    # 淘汰最久未使用的（头部）
                    self._cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


class EmbeddingCache:
    """Embedding 向量缓存管理器

    管理 Dense 和 Sparse 两个独立的 LRU 缓存。

    Args:
        enabled: 是否启用缓存
        max_size: 每个缓存的最大条数
    """

    def __init__(self, enabled: bool = True, max_size: int = 100_000):
        self.enabled = enabled
        self._dense_cache = LRUCache(max_size=max_size, name="dense")
        self._sparse_cache = LRUCache(max_size=max_size, name="sparse")

        if enabled:
            logger.info(
                "Embedding cache enabled: max_size=%d per type (dense + sparse)",
                max_size,
            )
        else:
            logger.info("Embedding cache disabled")

    @staticmethod
    def _hash_text(text: str) -> str:
        """计算文本的 MD5 hash 作为缓存 key"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ===== Dense Cache =====

    def get_dense(self, text: str) -> Optional[list[float]]:
        """获取 dense embedding 缓存"""
        if not self.enabled:
            return None
        key = self._hash_text(text)
        return self._dense_cache.get(key)

    def put_dense(self, text: str, embedding: list[float]) -> None:
        """写入 dense embedding 缓存"""
        if not self.enabled:
            return
        key = self._hash_text(text)
        self._dense_cache.put(key, embedding)

    # ===== Sparse Cache =====

    def get_sparse(self, text: str) -> Optional[list[dict]]:
        """获取 sparse embedding 缓存"""
        if not self.enabled:
            return None
        key = self._hash_text(text)
        return self._sparse_cache.get(key)

    def put_sparse(self, text: str, sparse_vec: list[dict]) -> None:
        """写入 sparse embedding 缓存"""
        if not self.enabled:
            return
        key = self._hash_text(text)
        self._sparse_cache.put(key, sparse_vec)

    # ===== Batch Operations =====

    def get_dense_batch(self, texts: list[str]) -> tuple[list[Optional[list[float]]], list[int]]:
        """批量查询 dense 缓存

        Returns:
            (cached_results, miss_indices)
            - cached_results: 与 texts 等长的列表，命中的位置有值，未命中为 None
            - miss_indices: 未命中的索引列表
        """
        if not self.enabled:
            return [None] * len(texts), list(range(len(texts)))

        cached_results = []
        miss_indices = []
        for i, text in enumerate(texts):
            result = self.get_dense(text)
            cached_results.append(result)
            if result is None:
                miss_indices.append(i)

        return cached_results, miss_indices

    def get_sparse_batch(self, texts: list[str]) -> tuple[list[Optional[list[dict]]], list[int]]:
        """批量查询 sparse 缓存"""
        if not self.enabled:
            return [None] * len(texts), list(range(len(texts)))

        cached_results = []
        miss_indices = []
        for i, text in enumerate(texts):
            result = self.get_sparse(text)
            cached_results.append(result)
            if result is None:
                miss_indices.append(i)

        return cached_results, miss_indices

    # ===== Stats =====

    @property
    def stats(self) -> dict:
        """缓存统计信息"""
        return {
            "enabled": self.enabled,
            "dense": {
                "size": self._dense_cache.size,
                "hits": self._dense_cache.hits,
                "misses": self._dense_cache.misses,
                "hit_rate": f"{self._dense_cache.hit_rate:.2%}",
            },
            "sparse": {
                "size": self._sparse_cache.size,
                "hits": self._sparse_cache.hits,
                "misses": self._sparse_cache.misses,
                "hit_rate": f"{self._sparse_cache.hit_rate:.2%}",
            },
        }
