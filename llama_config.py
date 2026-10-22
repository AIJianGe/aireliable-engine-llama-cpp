"""llama-server 启动参数自动配置。

根据模型文件大小 + GPU 显存自动生成 llama-server 启动参数：
    - NGL（GPU 层数）：按显存档位映射
    - CTX（上下文长度）：8×8 二维查表（模型档 × 显存档）
    - cache-type-k/v：默认 q4_0 量化（节省显存）
    - mmproj：多模态投影文件路径

迁移自参考实现 `启动 llama-server.bat` 的 `:APPLY_CTX` 段。
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .model_scanner import VariantInfo

logger = logging.getLogger(__name__)


# ===== 模型档分级（按参数量，单位：十亿参数）=====

def get_model_tier(params_b: int) -> int:
    """按参数量分级（1-8 档）。

    Args:
        params_b: 模型参数量（十亿参数）

    Returns:
        int: 1-8 档
            1: ≤1B
            2: 1-2B
            3: 2-4B
            4: 4-8B
            5: 8-14B
            6: 14-32B
            7: 32-70B（含 Qwen3.5/3.6-35B-A3B MoE）
            8: >70B
    """
    if params_b <= 1:
        return 1
    if params_b <= 2:
        return 2
    if params_b <= 4:
        return 3
    if params_b <= 8:
        return 4
    if params_b <= 14:
        return 5
    if params_b <= 32:
        return 6
    if params_b <= 70:
        return 7
    return 8


# ===== 显存档分级（按 GB）=====

def get_vram_bucket(vram_gb: int) -> int:
    """按显存大小分级（1-8 档）。

    Args:
        vram_gb: GPU 显存（GB）

    Returns:
        int: 1-8 档
            1: ≤8GB
            2: 8-12GB
            3: 12-16GB
            4: 16-24GB
            5: 24-32GB
            6: 32-48GB
            7: 48-64GB
            8: >64GB
    """
    if vram_gb <= 8:
        return 1
    if vram_gb <= 12:
        return 2
    if vram_gb <= 16:
        return 3
    if vram_gb <= 24:
        return 4
    if vram_gb <= 32:
        return 5
    if vram_gb <= 48:
        return 6
    if vram_gb <= 64:
        return 7
    return 8


# ===== CTX 二维查表（8 模型档 × 8 显存档 = 64 组合）=====
# 0 = 该档位显存不足，禁止启动（OOM 防护）

CTX_TABLE: dict[tuple[int, int], int] = {
    # Tier 1: ≤1B
    (1, 1): 65536,  (1, 2): 131072, (1, 3): 262144, (1, 4): 524288,
    (1, 5): 524288, (1, 6): 524288, (1, 7): 524288, (1, 8): 524288,
    # Tier 2: 1-2B
    (2, 1): 32768,  (2, 2): 65536,  (2, 3): 131072, (2, 4): 262144,
    (2, 5): 524288, (2, 6): 524288, (2, 7): 524288, (2, 8): 524288,
    # Tier 3: 2-4B
    (3, 1): 16384,  (3, 2): 32768,  (3, 3): 65536,  (3, 4): 131072,
    (3, 5): 262144, (3, 6): 524288, (3, 7): 524288, (3, 8): 524288,
    # Tier 4: 4-8B
    (4, 1): 8192,   (4, 2): 16384,  (4, 3): 32768,  (4, 4): 65536,
    (4, 5): 131072, (4, 6): 262144, (4, 7): 524288, (4, 8): 524288,
    # Tier 5: 8-14B
    (5, 1): 0,      (5, 2): 4096,   (5, 3): 16384,  (5, 4): 32768,
    (5, 5): 65536,  (5, 6): 262144, (5, 7): 524288, (5, 8): 524288,
    # Tier 6: 14-32B
    (6, 1): 0,      (6, 2): 0,      (6, 3): 4096,   (6, 4): 16384,
    (6, 5): 32768,  (6, 6): 131072, (6, 7): 262144, (6, 8): 524288,
    # Tier 7: 32-70B（含 Qwen3.5/3.6-35B-A3B MoE）
    (7, 1): 0,      (7, 2): 0,      (7, 3): 0,      (7, 4): 0,
    (7, 5): 4096,   (7, 6): 196608, (7, 7): 327680, (7, 8): 524288,
    # Tier 8: >70B
    (8, 1): 0,      (8, 2): 0,      (8, 3): 0,      (8, 4): 0,
    (8, 5): 0,      (8, 6): 0,      (8, 7): 4096,   (8, 8): 65536,
}


# NGL（GPU 层数）默认值
# 999 = 全部上 GPU（-ngl 999 在 llama-server 中表示全 offload）
NGL_BY_VRAM_BUCKET: dict[int, int] = {
    1: 20, 2: 30, 3: 40, 4: 60,
    5: 999, 6: 999, 7: 999, 8: 999,
}


# ===== 参数量估算 =====

def estimate_params_b(file_size_bytes: int) -> int:
    """估算模型参数量（十亿参数）。

    基于 Q4_K_M 量化经验：1B params ≈ 0.6GB
    不同量化级别会有偏差，但用于档位分级足够。

    Args:
        file_size_bytes: 模型文件字节数

    Returns:
        int: 估算的参数量（十亿参数，至少 1）
    """
    size_gb = file_size_bytes / (1024 ** 3)
    params_b = int(size_gb * 10 / 6)
    return max(1, params_b)


# ===== 异常 =====

class VramInsufficientError(Exception):
    """显存不足异常（CTX 查表返回 0）。"""

    def __init__(self, model_tier: int, vram_bucket: int, gpu_vram_gb: Optional[int]):
        self.model_tier = model_tier
        self.vram_bucket = vram_bucket
        self.gpu_vram_gb = gpu_vram_gb
        super().__init__(
            f"当前显存 {gpu_vram_gb}GB 不足以加载此模型"
            f"（Tier {model_tier}），请选更小量化版本或换更大显存"
        )


# ===== 启动参数数据类 =====

@dataclass
class LlamaServerParams:
    """llama-server 启动参数。"""

    model_path: Path
    ngl: int                                          # GPU 层数（0=纯 CPU，999=全部）
    ctx: int                                          # 上下文长度
    port: int                                         # 隐藏端口
    host: str = "127.0.0.1"
    cache_type_k: str = "q4_0"                        # K 缓存量化类型
    cache_type_v: str = "q4_0"                        # V 缓存量化类型
    alias: str = ""                                   # variant.name 或 alias
    mmproj_path: Optional[Path] = None                # 多模态投影文件
    extra_args: list[str] = field(default_factory=list)
    # 元信息（用于日志/状态查询）
    model_tier: int = 0
    vram_bucket: int = 0
    gpu_vram_gb: Optional[int] = None
    params_b: int = 0


# ===== 参数生成器 =====

class LlamaConfigGenerator:
    """根据模型 + GPU 自动生成 llama-server 启动参数。"""

    def __init__(self, llama_bins_dir: Path):
        """
        Args:
            llama_bins_dir: llama-bins 二进制目录
        """
        self.llama_bins_dir = Path(llama_bins_dir)

    def generate(
        self,
        variant: VariantInfo,
        gpu_vram_gb: Optional[int],
        internal_port: int,
        mmproj_path: Optional[Path] = None,
        user_overrides: Optional[dict] = None,
    ) -> LlamaServerParams:
        """生成启动参数。

        Args:
            variant: 扫描得到的 variant 信息
            gpu_vram_gb: GPU 显存（GB），None 表示无 GPU
            internal_port: 隐藏端口
            mmproj_path: 多模态投影文件路径（覆盖 variant 自带的）
            user_overrides: 用户自定义参数覆盖

        Returns:
            LlamaServerParams: 启动参数

        Raises:
            VramInsufficientError: 显存不足以加载此模型
        """
        if variant.path is None or not variant.path.exists():
            raise FileNotFoundError(f"variant 模型文件不存在: {variant.variant_id}")

        # 估算参数量
        params_b = estimate_params_b(variant.file_size)
        model_tier = get_model_tier(params_b)

        # 显存档位
        if gpu_vram_gb is None or gpu_vram_gb <= 0:
            # 无 GPU，强制 CPU 模式
            logger.warning(
                "未检测到 GPU，使用 CPU 模式（性能会很慢）: variant=%s",
                variant.variant_id,
            )
            ngl = 0
            ctx = 8192  # CPU 模式保守值
            vram_bucket = 0
        else:
            vram_bucket = get_vram_bucket(gpu_vram_gb)
            ctx = CTX_TABLE.get((model_tier, vram_bucket), 0)
            if ctx == 0:
                # OOM 防护
                logger.error(
                    "显存不足: variant=%s, tier=%d, vram=%dGB(bucket=%d), ctx=0",
                    variant.variant_id, model_tier, gpu_vram_gb, vram_bucket,
                )
                raise VramInsufficientError(model_tier, vram_bucket, gpu_vram_gb)
            ngl = NGL_BY_VRAM_BUCKET[vram_bucket]

        # mmproj
        actual_mmproj = mmproj_path or variant.mmproj_path

        # alias 优先用短别名
        alias = variant.alias or variant.variant_id

        params = LlamaServerParams(
            model_path=variant.path,
            ngl=ngl,
            ctx=ctx,
            port=internal_port,
            host="127.0.0.1",
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            alias=alias,
            mmproj_path=actual_mmproj,
            model_tier=model_tier,
            vram_bucket=vram_bucket,
            gpu_vram_gb=gpu_vram_gb,
            params_b=params_b,
        )

        # 应用用户覆盖
        if user_overrides:
            params = self._apply_overrides(params, user_overrides)

        logger.info(
            "生成参数: variant=%s, params_b=%d, tier=%d, vram=%sGB(bucket=%d), "
            "ngl=%d, ctx=%d, mmproj=%s",
            variant.variant_id, params_b, model_tier,
            gpu_vram_gb, vram_bucket, ngl, ctx,
            actual_mmproj.name if actual_mmproj else "无",
        )

        return params

    def to_command(self, params: LlamaServerParams) -> list[str]:
        """转成命令行参数列表（不含可执行文件路径）。

        Returns:
            list[str]: 命令行参数列表
        """
        cmd: list[str] = [
            "-m", str(params.model_path),
            "--host", params.host,
            "--port", str(params.port),
            "-ngl", str(params.ngl),
            "-c", str(params.ctx),
            "--cache-type-k", params.cache_type_k,
            "--cache-type-v", params.cache_type_v,
            "-a", params.alias,
        ]

        if params.mmproj_path is not None:
            cmd.extend(["--mmproj", str(params.mmproj_path)])

        # 额外参数
        cmd.extend(params.extra_args)

        return cmd

    def to_full_command(self, params: LlamaServerParams) -> list[str]:
        """完整命令（含可执行文件路径）。"""
        server_bin = self._get_server_bin()
        return [str(server_bin)] + self.to_command(params)

    def _get_server_bin(self) -> Path:
        """获取 llama-server 可执行文件路径。"""
        if sys.platform == "win32":
            return self.llama_bins_dir / "llama-server.exe"
        return self.llama_bins_dir / "llama-server"

    @staticmethod
    def _apply_overrides(params: LlamaServerParams, overrides: dict) -> LlamaServerParams:
        """应用用户自定义参数覆盖。"""
        if "ngl" in overrides:
            params.ngl = int(overrides["ngl"])
        if "ctx" in overrides:
            params.ctx = int(overrides["ctx"])
        if "cache_type_k" in overrides:
            params.cache_type_k = str(overrides["cache_type_k"])
        if "cache_type_v" in overrides:
            params.cache_type_v = str(overrides["cache_type_v"])
        if "extra_args" in overrides:
            params.extra_args = list(overrides["extra_args"])
        return params

    @staticmethod
    def detect_gpu_vram() -> Optional[int]:
        """通过 nvidia-smi 检测 GPU 显存（GB）。

        Returns:
            int | None: 显存 GB 数（无 GPU 或检测失败返回 None）
        """
        try:
            # 查询第一块 GPU 的总显存（单位 MiB）
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5.0,
                creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
            )
            if result.returncode != 0:
                logger.warning("nvidia-smi 调用失败: %s", result.stderr.strip())
                return None

            first_line = result.stdout.strip().split("\n")[0].strip()
            vram_mib = int(first_line)
            vram_gb = vram_mib // 1024
            logger.info("检测到 GPU 显存: %d MiB ≈ %d GB", vram_mib, vram_gb)
            return vram_gb
        except FileNotFoundError:
            logger.info("未找到 nvidia-smi，可能是无 NVIDIA GPU 或非 Windows/Linux")
            return None
        except Exception as e:
            logger.warning("GPU 显存检测失败: %s", e)
            return None
