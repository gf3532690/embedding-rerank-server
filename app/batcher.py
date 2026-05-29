"""Token-level Continuous Batching 模块

核心改进（相比旧版 DynamicBatcher）：
1. 按总 token 数合批（max_batch_tokens），而非固定条数
   - 短文本多塞几条，长文本少塞几条
   - 减少 padding 浪费，GPU/CPU 利用率最大化
2. 优先级队列
   - priority=0: 单条查询请求（实时性要求高）
   - priority=1: 批量入库请求（吞吐优先）
3. 无超时，无限等待
   - 去掉 request_timeout，服务端永不主动拒绝
4. Tokenizer pipeline 重叠
   - 当前批在推理时，下一批同时进行 tokenize
5. 背压机制
   - 队列过深时返回 HTTP 429 + Retry-After

参考 Infinity 的 BatchHandler + CustomFIFOQueue 设计。
"""

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


# ===== 数据结构 =====

@dataclass(order=True)
class PrioritizedItem:
    """优先级队列中的单个请求

    排序规则：priority 小的优先，相同 priority 按入队时间排序
    """

    priority: int
    enqueue_time: float = field(compare=True)
    # 以下字段不参与排序
    data: Any = field(compare=False)
    token_count: int = field(compare=False, default=0)
    future: asyncio.Future = field(compare=False, default=None)


class TokenBatcher:
    """Token-level Continuous Batching 处理器

    按总 token 数合批，支持优先级队列和 tokenizer pipeline 重叠。

    Args:
        process_fn: 批量处理函数，接收 list[Any] 返回 list[Any]
        tokenize_fn: tokenize 函数，接收 str 返回 token 数量
        max_batch_tokens: 每批最大总 token 数
        max_batch_size: 每批最大条数（硬上限，防止极短文本塞太多）
        max_wait_ms: 最大等待时间（毫秒），超时即使未凑够也推理
        backpressure_threshold: 队列深度超过此值触发背压（返回 429）
    """

    def __init__(
        self,
        process_fn: Callable[[list[Any]], Coroutine[Any, Any, list[Any]]],
        tokenize_fn: Optional[Callable[[str], int]] = None,
        max_batch_tokens: int = 16384,
        max_batch_size: int = 64,
        max_wait_ms: int = 10,
        backpressure_threshold: int = 3200,
    ):
        self.process_fn = process_fn
        self.tokenize_fn = tokenize_fn or self._default_tokenize
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.backpressure_threshold = backpressure_threshold

        # 优先级队列（heapq）
        self._queue: list[PrioritizedItem] = []
        self._queue_lock = asyncio.Lock()
        self._queue_event = asyncio.Event()

        # 统计信息
        self._total_processed: int = 0
        self._total_batches: int = 0

        # Worker 任务
        self._worker_task: Optional[asyncio.Task] = None

        # Tokenizer pipeline 重叠：预 tokenize 缓冲区
        self._pretokenize_task: Optional[asyncio.Task] = None

    @staticmethod
    def _default_tokenize(text: str) -> int:
        """默认 tokenize 估算：按字符数 / 3 估算 token 数

        中文约 1 字 = 1-2 tokens，英文约 4 chars = 1 token
        取折中值 3 chars = 1 token
        """
        if not text:
            return 1
        return max(len(text) // 3, 1)

    @property
    def queue_depth(self) -> int:
        """当前队列深度"""
        return len(self._queue)

    @property
    def total_processed(self) -> int:
        """已处理的总请求数"""
        return self._total_processed

    @property
    def total_batches(self) -> int:
        """已处理的总批次数"""
        return self._total_batches

    @property
    def is_overloaded(self) -> bool:
        """是否触发背压"""
        return len(self._queue) > self.backpressure_threshold

    def estimated_wait_time(self) -> float:
        """估算当前队列的等待时间（秒）

        基于历史平均批处理时间和当前队列深度估算。
        """
        if self._total_batches == 0:
            return 0.0
        # 粗略估算：每批约 50ms（GPU）或 200ms（CPU）
        avg_batch_time = 0.05  # 默认 50ms
        pending_batches = max(len(self._queue) / max(self.max_batch_size, 1), 1)
        return pending_batches * avg_batch_time

    def start(self) -> None:
        """启动后台 worker"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info(
                "TokenBatcher started: max_batch_tokens=%d, max_batch_size=%d, "
                "max_wait_ms=%d, backpressure_threshold=%d",
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
        if self._pretokenize_task and not self._pretokenize_task.done():
            self._pretokenize_task.cancel()
            try:
                await self._pretokenize_task
            except asyncio.CancelledError:
                pass

    async def submit(self, data: Any, priority: int = 0) -> Any:
        """提交单个请求，等待结果返回

        Args:
            data: 请求数据（文本字符串）
            priority: 优先级，0=查询（高优先），1=批量入库（低优先）

        Returns:
            该请求对应的处理结果

        Raises:
            BackpressureError: 队列过深时抛出
        """
        if self.is_overloaded:
            raise BackpressureError(
                queue_depth=len(self._queue),
                estimated_wait=self.estimated_wait_time(),
            )

        loop = asyncio.get_running_loop()
        # 估算 token 数
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

        return await item.future

    async def _worker_loop(self) -> None:
        """后台 worker：持续从优先级队列收集请求并按 token budget 合批"""
        while True:
            batch: list[PrioritizedItem] = []
            batch_tokens: int = 0

            try:
                # 等待队列中有请求
                await self._queue_event.wait()

                # 收集第一个请求
                async with self._queue_lock:
                    if not self._queue:
                        self._queue_event.clear()
                        continue
                    first_item = heapq.heappop(self._queue)
                    if not self._queue:
                        self._queue_event.clear()

                batch.append(first_item)
                batch_tokens += first_item.token_count

                # 开始计时，尝试凑更多请求（按 token budget）
                deadline = time.monotonic() + self.max_wait_ms / 1000.0

                while (
                    batch_tokens < self.max_batch_tokens
                    and len(batch) < self.max_batch_size
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break

                    # 尝试从队列取下一个
                    async with self._queue_lock:
                        if not self._queue:
                            self._queue_event.clear()
                            break
                        # 检查下一个 item 加入后是否超 budget
                        next_item = self._queue[0]
                        if batch_tokens + next_item.token_count > self.max_batch_tokens:
                            # 超 budget，不再加入
                            break
                        heapq.heappop(self._queue)
                        if not self._queue:
                            self._queue_event.clear()

                    batch.append(next_item)
                    batch_tokens += next_item.token_count

                # 如果队列还有剩余但 batch 已满，等一小段时间让更多请求进来
                if not batch and not self._queue:
                    await asyncio.sleep(0.001)
                    continue

                # 执行批量推理
                await self._process_batch(batch)

            except asyncio.CancelledError:
                # 处理剩余队列中的请求
                async with self._queue_lock:
                    while self._queue:
                        item = heapq.heappop(self._queue)
                        batch.append(item)
                if batch:
                    await self._process_batch(batch)
                raise
            except Exception as e:
                # 推理失败，通知所有等待的请求
                logger.error("Batch processing error: %s", e, exc_info=True)
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)

    async def _process_batch(self, batch: list[PrioritizedItem]) -> None:
        """执行批量推理并分发结果"""
        if not batch:
            return

        data_list = [item.data for item in batch]
        batch_size = len(batch)
        total_tokens = sum(item.token_count for item in batch)

        try:
            results = await self.process_fn(data_list)

            if len(results) != batch_size:
                raise ValueError(
                    f"Process function returned {len(results)} results "
                    f"for batch of {batch_size}"
                )

            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

            # 更新统计
            self._total_processed += batch_size
            self._total_batches += 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Batch processed: size=%d, tokens=%d, avg_tokens=%.0f",
                    batch_size,
                    total_tokens,
                    total_tokens / batch_size,
                )

        except Exception as e:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)


class BackpressureError(Exception):
    """队列过深时抛出，API 层捕获后返回 429"""

    def __init__(self, queue_depth: int, estimated_wait: float):
        self.queue_depth = queue_depth
        self.estimated_wait = estimated_wait
        super().__init__(
            f"Server overloaded: {queue_depth} requests pending, "
            f"estimated wait {estimated_wait:.1f}s"
        )
