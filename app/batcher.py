"""Token-level Continuous Batching 模块

核心特性：
1. 按总 token 数合批（max_batch_tokens），而非固定条数
2. 优先级队列（priority=0 查询优先，priority=1 批量入库）
3. 无超时，无限等待
4. Pipeline 重叠：当前批推理时，同时收集下一批
5. Speculative Batching：根据实时 QPS 动态调整等待时间
6. Batch 内去重：相同文本只推理一次
7. 背压机制：队列过深返回 HTTP 429

@author performance-optimization
@since 2.1.0
"""

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedItem:
    """优先级队列中的单个请求

    排序规则：priority 小的优先，相同 priority 按入队时间排序
    """

    priority: int
    enqueue_time: float = field(compare=True)
    data: Any = field(compare=False)
    token_count: int = field(compare=False, default=0)
    future: asyncio.Future = field(compare=False, default=None)


class TokenBatcher:
    """Token-level Continuous Batching 处理器

    Args:
        process_fn: 批量处理函数，接收 list[Any] 返回 list[Any]
        tokenize_fn: tokenize 函数，接收 str 返回 token 数量
        max_batch_tokens: 每批最大总 token 数
        max_batch_size: 每批最大条数（硬上限）
        max_wait_ms: 最大等待时间（毫秒）
        backpressure_threshold: 队列深度超过此值触发背压
    """

    def __init__(
        self,
        process_fn: Callable[[list[Any]], Coroutine[Any, Any, list[Any]]],
        tokenize_fn: Optional[Callable[[str], int]] = None,
        max_batch_tokens: int = 16384,
        max_batch_size: int = 64,
        max_wait_ms: int = 10,
        backpressure_threshold: int = 3200,
        name: str = "batch",
    ):
        self.process_fn = process_fn
        self.tokenize_fn = tokenize_fn or self._default_tokenize
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.backpressure_threshold = backpressure_threshold
        self.name = name  # 用于 metrics 标签（dense/sparse）

        # 优先级队列
        self._queue: list[PrioritizedItem] = []
        self._queue_lock = asyncio.Lock()
        self._queue_event = asyncio.Event()

        # 统计
        self._total_processed: int = 0
        self._total_batches: int = 0

        # QPS 跟踪（Speculative Batching）
        self._recent_submit_times: list[float] = []
        self._qps_window: float = 1.0

        # Worker
        self._worker_task: Optional[asyncio.Task] = None

    @staticmethod
    def _default_tokenize(text: str) -> int:
        """默认 tokenize 估算：字符数 / 3"""
        if not text:
            return 1
        return max(len(text) // 3, 1)

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def total_processed(self) -> int:
        return self._total_processed

    @property
    def total_batches(self) -> int:
        return self._total_batches

    @property
    def is_overloaded(self) -> bool:
        return len(self._queue) > self.backpressure_threshold

    @property
    def current_qps(self) -> float:
        """最近 1 秒的 QPS"""
        return float(len(self._recent_submit_times))

    def estimated_wait_time(self) -> float:
        """估算当前队列等待时间（秒）"""
        if self._total_batches == 0:
            return 0.0
        avg_batch_time = 0.05
        pending_batches = max(len(self._queue) / max(self.max_batch_size, 1), 1)
        return pending_batches * avg_batch_time

    def start(self) -> None:
        """启动后台 worker"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info(
                "TokenBatcher started: max_batch_tokens=%d, max_batch_size=%d, "
                "max_wait_ms=%d, backpressure=%d",
                self.max_batch_tokens,
                self.max_batch_size,
                self.max_wait_ms,
                self.backpressure_threshold,
            )

    async def stop(self) -> None:
        """停止后台 worker"""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, data: Any, priority: int = 0) -> Any:
        """提交请求，等待结果

        Raises:
            BackpressureError: 队列过深
        """
        if self.is_overloaded:
            raise BackpressureError(
                queue_depth=len(self._queue),
                estimated_wait=self.estimated_wait_time(),
            )

        loop = asyncio.get_running_loop()
        token_count = self.tokenize_fn(data) if isinstance(data, str) else 1

        item = PrioritizedItem(
            priority=priority,
            enqueue_time=time.monotonic(),
            data=data,
            token_count=token_count,
            future=loop.create_future(),
        )

        async with self._queue_lock:
            heapq.heappush(self._queue, item)
        self._queue_event.set()

        # QPS 跟踪
        now = time.monotonic()
        self._recent_submit_times.append(now)
        cutoff = now - self._qps_window
        while self._recent_submit_times and self._recent_submit_times[0] < cutoff:
            self._recent_submit_times.pop(0)

        return await item.future

    # ===== Pipeline Worker =====

    async def _worker_loop(self) -> None:
        """Pipeline 重叠 Worker

        核心优化：当前批在推理时，同时收集下一批。
        这样 batch 收集的延迟（max_wait_ms）与推理并行，
        整体吞吐提升 15-30%。

        流程：
          collect batch₁ → [inference batch₁ | collect batch₂] → [inference batch₂ | collect batch₃] → ...
        """
        # 预收集第一批
        pending_collect: Optional[asyncio.Task] = None

        while True:
            try:
                # 获取当前要处理的 batch
                if pending_collect is not None:
                    batch = await pending_collect
                    pending_collect = None
                else:
                    batch = await self._collect_batch()

                if not batch:
                    continue

                # 启动下一批的预收集（与当前推理并行）
                # 只在队列有内容时预收集，避免空转
                if self._queue:
                    pending_collect = asyncio.create_task(self._collect_batch())

                # 执行当前批推理
                await self._process_batch(batch)

            except asyncio.CancelledError:
                if pending_collect and not pending_collect.done():
                    pending_collect.cancel()
                    try:
                        await pending_collect
                    except (asyncio.CancelledError, Exception):
                        pass
                # 处理剩余
                remaining: list[PrioritizedItem] = []
                async with self._queue_lock:
                    while self._queue:
                        remaining.append(heapq.heappop(self._queue))
                if remaining:
                    await self._process_batch(remaining)
                raise
            except Exception as e:
                logger.error("Worker loop error: %s", e, exc_info=True)

    # ===== Speculative Batching =====

    async def _collect_batch(self) -> list[PrioritizedItem]:
        """收集一个 batch（Speculative Batching）

        根据实时 QPS 动态调整等待时间：
        - 高 QPS (>50): 等满 max_wait_ms，凑大 batch 提高吞吐
        - 中 QPS (10-50): 等一半
        - 低 QPS (<10): 1ms 立即发车，降低延迟
        """
        batch: list[PrioritizedItem] = []
        batch_tokens: int = 0

        # 等待队列有请求
        await self._queue_event.wait()

        # 取第一个
        async with self._queue_lock:
            if not self._queue:
                self._queue_event.clear()
                return batch
            first_item = heapq.heappop(self._queue)
            if not self._queue:
                self._queue_event.clear()

        batch.append(first_item)
        batch_tokens += first_item.token_count

        # 自适应等待时间
        effective_wait_ms = self._compute_adaptive_wait()
        deadline = time.monotonic() + effective_wait_ms / 1000.0

        while batch_tokens < self.max_batch_tokens and len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            async with self._queue_lock:
                if not self._queue:
                    self._queue_event.clear()
                    break
                next_item = self._queue[0]
                if batch_tokens + next_item.token_count > self.max_batch_tokens:
                    break
                heapq.heappop(self._queue)
                if not self._queue:
                    self._queue_event.clear()

            batch.append(next_item)
            batch_tokens += next_item.token_count

        return batch

    def _compute_adaptive_wait(self) -> float:
        """根据 QPS 计算自适应等待时间（毫秒）"""
        qps = len(self._recent_submit_times)
        if qps > 50:
            return float(self.max_wait_ms)
        elif qps > 10:
            return float(self.max_wait_ms) / 2.0
        else:
            return 1.0

    # ===== Batch Processing (with dedup) =====

    async def _process_batch(self, batch: list[PrioritizedItem]) -> None:
        """执行批量推理（含 batch 内去重）"""
        if not batch:
            return

        # Batch 内去重
        unique_map: dict[str, int] = {}
        unique_list: list[Any] = []
        item_to_unique: list[int] = []

        for item in batch:
            text = item.data
            if text not in unique_map:
                unique_map[text] = len(unique_list)
                unique_list.append(text)
            item_to_unique.append(unique_map[text])

        batch_size = len(batch)
        total_tokens = sum(item.token_count for item in batch)

        try:
            import time as _time
            _t0 = _time.monotonic()
            results = await self.process_fn(unique_list)
            _inference_time = _time.monotonic() - _t0

            if len(results) != len(unique_list):
                raise ValueError(
                    f"Process fn returned {len(results)} results "
                    f"for {len(unique_list)} unique texts (batch={batch_size})"
                )

            for i, item in enumerate(batch):
                if not item.future.done():
                    item.future.set_result(results[item_to_unique[i]])

            self._total_processed += batch_size
            self._total_batches += 1

            # 记录 Prometheus 指标（可用时）
            self._record_metrics(batch_size, total_tokens, _inference_time)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Batch[%s]: size=%d, unique=%d, tokens=%d, qps=%.0f, wait=%.1fms, infer=%.1fms",
                    self.name,
                    batch_size,
                    len(unique_list),
                    total_tokens,
                    self.current_qps,
                    self._compute_adaptive_wait(),
                    _inference_time * 1000,
                )

        except Exception as e:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

    def _record_metrics(self, batch_size: int, total_tokens: int, inference_time: float) -> None:
        """记录 Prometheus 指标（metrics 不可用时静默跳过）"""
        try:
            from app import metrics
            metrics.record_batch_size(self.name, batch_size)
            metrics.record_batch_tokens(self.name, total_tokens)
            metrics.record_inference_duration(self.name, inference_time)
            if inference_time > 0:
                metrics.record_throughput(self.name, total_tokens / inference_time)
        except Exception:
            pass


class BackpressureError(Exception):
    """队列过深时抛出，API 层捕获后返回 429"""

    def __init__(self, queue_depth: int, estimated_wait: float):
        self.queue_depth = queue_depth
        self.estimated_wait = estimated_wait
        super().__init__(
            f"Server overloaded: {queue_depth} requests pending, "
            f"estimated wait {estimated_wait:.1f}s"
        )
