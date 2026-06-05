"""硬件探测与自动配置模块

启动时自动探测硬件能力（GPU/CPU、显存、核数），
并根据探测结果填充所有 "auto" 配置值。

参考 Infinity 的设计思路：零配置、开箱即用。
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """硬件探测结果"""

    # 设备类型: "cuda" / "cpu"
    device: str
    # GPU 显存 (GB)，CPU 模式为 0
    gpu_memory_gb: float
    # GPU 名称
    gpu_name: str
    # CPU 核数
    cpu_cores: int
    # 可用内存 (GB)
    available_memory_gb: float


def detect_device() -> str:
    """探测可用设备

    Returns:
        "cuda" 如果有可用 GPU，否则 "cpu"
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def detect_gpu_memory() -> float:
    """探测 GPU 显存 (GB)

    Returns:
        显存大小 (GB)，无 GPU 返回 0.0
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device_id).total_mem
            return total_memory / (1024 ** 3)
    except (ImportError, RuntimeError) as e:
        logger.warning("GPU memory detection failed: %s", e)
    return 0.0


def detect_gpu_name() -> str:
    """探测 GPU 名称"""
    try:
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            return torch.cuda.get_device_name(device_id)
    except (ImportError, RuntimeError):
        pass
    return "N/A"


def detect_cpu_cores() -> int:
    """探测 CPU 核数

    Returns:
        可用 CPU 核数
    """
    # 优先使用 os.sched_getaffinity（容器内准确）
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    # 回退到 os.cpu_count
    return os.cpu_count() or 4


def detect_available_memory() -> float:
    """探测可用系统内存 (GB)"""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        # psutil 不可用时，返回保守估计
        return 8.0


def detect_hardware() -> HardwareInfo:
    """执行完整硬件探测

    Returns:
        HardwareInfo 包含所有探测结果
    """
    device = detect_device()
    gpu_memory = detect_gpu_memory() if device == "cuda" else 0.0
    gpu_name = detect_gpu_name() if device == "cuda" else "N/A"
    cpu_cores = detect_cpu_cores()
    available_memory = detect_available_memory()

    info = HardwareInfo(
        device=device,
        gpu_memory_gb=gpu_memory,
        gpu_name=gpu_name,
        cpu_cores=cpu_cores,
        available_memory_gb=available_memory,
    )

    logger.info("=" * 60)
    logger.info("Hardware Detection Results:")
    logger.info("  Device: %s", info.device)
    if info.device == "cuda":
        logger.info("  GPU: %s (%.1f GB VRAM)", info.gpu_name, info.gpu_memory_gb)
    logger.info("  CPU cores: %d", info.cpu_cores)
    logger.info("  Available memory: %.1f GB", info.available_memory_gb)
    logger.info("=" * 60)

    return info


def _compute_max_batch_tokens(gpu_memory_gb: float, max_length: int) -> int:
    """根据显存估算 max_batch_tokens

    经验公式：每 GB 显存约支持 2048 tokens 的 batch budget
    （基于 bge-m3 模型的实测数据）

    Args:
        gpu_memory_gb: GPU 显存 (GB)
        max_length: 模型最大输入长度

    Returns:
        max_batch_tokens 值
    """
    if gpu_memory_gb <= 0:
        # CPU 模式，给保守值
        return 8192

    # 每 GB 显存约 2048 tokens budget（bge-m3 经验值）
    tokens_per_gb = 2048
    estimated = int(gpu_memory_gb * tokens_per_gb)

    # 限制在合理范围内
    return max(4096, min(estimated, 65536))


def auto_configure(config) -> None:
    """根据硬件探测结果，填充配置中所有 "auto" 值

    修改 config 对象（in-place），将 "auto" 替换为实际探测值。

    Args:
        config: AppConfig 实例
    """
    hw = detect_hardware()

    # --- 设备自动选择 ---
    if _is_auto(config.models.embed.device):
        config.models.embed.device = hw.device
    if _is_auto(config.models.rerank.device):
        config.models.rerank.device = hw.device

    # --- 推理引擎自动选择 ---
    if _is_auto(config.models.embed.engine):
        config.models.embed.engine = "pytorch" if hw.device == "cuda" else "onnx"
    if _is_auto(config.models.rerank.engine):
        config.models.rerank.engine = "pytorch" if hw.device == "cuda" else "onnx"

    # --- fp16 自动选择 ---
    if _is_auto(config.models.embed.fp16):
        config.models.embed.fp16 = hw.device == "cuda"
    if _is_auto(config.models.rerank.fp16):
        config.models.rerank.fp16 = hw.device == "cuda"

    # --- 量化自动选择 ---
    # auto: GPU 不量化（保持精度，靠 fp16 + Flash Attention），CPU 也默认不量化
    # （INT8 量化需要预先导出量化模型，不能凭空开启，所以 auto = none）
    if _is_auto(config.models.embed.quantization):
        config.models.embed.quantization = "none"
    if _is_auto(config.models.rerank.quantization):
        config.models.rerank.quantization = "none"

    # --- Batching 参数 ---
    if _is_auto(config.batching.max_batch_size):
        if hw.device == "cuda":
            # GPU: 根据显存估算
            config.batching.max_batch_size = _estimate_batch_size_from_vram(hw.gpu_memory_gb)
        else:
            # CPU: 根据核数
            config.batching.max_batch_size = min(max(hw.cpu_cores * 2, 8), 64)

    if _is_auto(config.batching.max_batch_tokens):
        config.batching.max_batch_tokens = _compute_max_batch_tokens(
            hw.gpu_memory_gb, config.models.embed.max_length
        )

    if _is_auto(config.batching.max_concurrency):
        if hw.device == "cuda":
            # GPU 串行推理（GPU 本身并行）
            config.batching.max_concurrency = 1
        else:
            # CPU 多线程推理
            config.batching.max_concurrency = max(hw.cpu_cores // 2, 1)

    # --- 背压阈值 ---
    if _is_auto(config.batching.backpressure_threshold):
        config.batching.backpressure_threshold = config.batching.max_batch_size * 50

    # --- OMP_NUM_THREADS (CPU 模式) ---
    if hw.device == "cpu":
        omp_threads = os.environ.get("OMP_NUM_THREADS")
        if not omp_threads:
            optimal_threads = max(hw.cpu_cores // config.batching.max_concurrency, 1)
            os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
            logger.info("Set OMP_NUM_THREADS=%d", optimal_threads)

    # --- 输出最终配置 ---
    logger.info("=" * 60)
    logger.info("Auto-configured values:")
    logger.info("  Device: %s", config.models.embed.device)
    logger.info("  Engine: embed=%s, rerank=%s",
                config.models.embed.engine, config.models.rerank.engine)
    logger.info("  FP16: embed=%s, rerank=%s",
                config.models.embed.fp16, config.models.rerank.fp16)
    logger.info("  Batching: max_batch_size=%d, max_batch_tokens=%d, max_concurrency=%d",
                config.batching.max_batch_size,
                config.batching.max_batch_tokens,
                config.batching.max_concurrency)
    logger.info("  Backpressure threshold: %d", config.batching.backpressure_threshold)
    logger.info("=" * 60)


def _is_auto(value) -> bool:
    """判断配置值是否为 "auto" """
    if isinstance(value, str):
        return value.lower().strip() == "auto"
    return False


def _estimate_batch_size_from_vram(gpu_memory_gb: float) -> int:
    """根据显存估算 max_batch_size

    经验值（基于 bge-m3 模型）：
    - 4GB VRAM → batch_size 16
    - 8GB VRAM → batch_size 32
    - 16GB VRAM → batch_size 64
    - 24GB VRAM → batch_size 128
    """
    if gpu_memory_gb <= 4:
        return 16
    elif gpu_memory_gb <= 8:
        return 32
    elif gpu_memory_gb <= 16:
        return 64
    elif gpu_memory_gb <= 24:
        return 128
    else:
        return 256
