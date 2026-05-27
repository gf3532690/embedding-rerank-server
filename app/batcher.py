"""Dynamic Batching 模块

实现请求自动合批：
- 凑够 max_batch_size 立即发车
- 未凑够但超过 max_wait_ms 也发车
- 保证每个请求都能拿到自己的结果
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """队列中的单个请求"""
    data: Any
    future: asyncio.Future = field(init=False)
    enqueue_time: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        # future 由 submit() 方法显式设置，这里仅占位
        pass


class DynamicBatcher:
    """动态合批处理器

    将多个异步请求合并成一个 batch 送入推理函数，
    然后将结果拆分回各个请求。

    Args:
        process_fn: 批量处理函数，接收 list[Any] 返回 list[Any]
        max_batch_size: 凑够此数量立即推理
        max_wait_ms: 最大等待时间（毫秒），超时即使未凑够也推理
    """

    def __init__(
        self,
        process_fn: Callable[[list[Any]], Coroutine[Any, Any, list[Any]]],
        max_batch_size: int = 64,
        max_wait_ms: int = 10,
    ):
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        """启动后台 worker"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info(
                "DynamicBatcher started: max_batch_size=%d, max_wait_ms=%d",
                self.max_batch_size,
                self.max_wait_ms,
            )

    async def stop(self) -> None:
        """停止后台 worker"""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, data: Any) -> Any:
        """提交单个请求，等待结果返回

        Args:
            data: 请求数据（会被放入 batch 列表中）

        Returns:
            该请求对应的处理结果
        """
        loop = asyncio.get_running_loop()
        item = BatchItem(data=data, future=loop.create_future())
        await self._queue.put(item)
        return await item.future

    async def _worker_loop(self) -> None:
        """后台 worker：持续从队列收集请求并批量处理"""
        while True:
            batch: list[BatchItem] = []

            try:
                # 等待第一个请求到达（阻塞）
                first_item = await self._queue.get()
                batch.append(first_item)

                # 开始计时，尝试凑更多请求
                deadline = time.monotonic() + self.max_wait_ms / 1000.0

                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # 执行批量推理
                await self._process_batch(batch)

            except asyncio.CancelledError:
                # 处理剩余队列中的请求
                while not self._queue.empty():
                    try:
                        item = self._queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break
                if batch:
                    await self._process_batch(batch)
                raise
            except Exception as e:
                # 推理失败，通知所有等待的请求
                logger.error("Batch processing error: %s", e)
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)

    async def _process_batch(self, batch: list[BatchItem]) -> None:
        """执行批量推理并分发结果"""
        if not batch:
            return

        data_list = [item.data for item in batch]
        try:
            results = await self.process_fn(data_list)

            if len(results) != len(batch):
                raise ValueError(
                    f"Process function returned {len(results)} results "
                    f"for batch of {len(batch)}"
                )

            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

        except Exception as e:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
