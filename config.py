"""Llama.cpp 引擎配置（v0.3.0 router mode 重构版）。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

v0.3.0 变更：
    - 删除 Gradio 相关字段（theme_path / file_storage_dir）
    - 新增 core_url（读 AIRELIABLE_CORE_URL，平台 API 调用）
    - 新增 models_dir（router mode 的 --models-dir 参数）
    - 新增 models_max（router mode 最大并存模型数，默认 4）
    - 新增 models_autoload（是否启动时自动加载，默认 False）
    - 输出根目录统一为 output/text/（v0.2.5+ 规范）

环境变量优先级（高 → 低）：
    命令行参数 > 环境变量 > 默认值

关键环境变量（由平台 engine_manager._build_subprocess_env 注入）：
    AIRELIABLE_CORE_URL          平台 API 地址（http://127.0.0.1:8001）
    AIRELIABLE_MODELS_ROOT       models 根目录
    AIRELIABLE_MODEL_LAYOUT_BASE 当前引擎的 model_layout.base（如 llm/checkpoints/llama）
    AIRELIABLE_LORA_DIR          LoRA 目录
    AIRELIABLE_UI_DIR            UI 目录（v0.3.0 反代模式下未使用，保留兼容）
    LLAMA_BINS_DIR               llama-bins 二进制目录
    LLAMA_INTERNAL_PORT_START    隐藏端口范围起始（默认 8100）
    LLAMA_INTERNAL_PORT_END      隐藏端口范围结束（默认 8200）
    LLAMA_MODELS_MAX             router mode 最大并存模型数（默认 4）
    LLAMA_MODELS_AUTOLOAD        是否启动时自动加载首个模型（默认 0=false）
    AIRELIABLE_DEVICE_POLICY     设备策略（v0.3.5+ 平台注入：auto/gpu/gpu+cpu/cpu）
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from aireliable_engine import EngineConfig as BaseEngineConfig

logger = logging.getLogger(__name__)

# v0.3.5+ 设备策略合法取值（平台注入 AIRELIABLE_DEVICE_POLICY 环境变量）
#   auto    ：引擎自行决定（现状行为：llama-server 默认 ngl=auto + fit=on）
#   gpu     ：强制全 GPU（-ngl all --fit off，显存不足直接报错）
#   gpu+cpu ：GPU 优先，显存不足时层数溢出 CPU（-ngl auto --fit on，
#             llama-server 原生按实际显存自动决定 offload 层数）
#   cpu     ：强制 CPU（-ngl 0 --fit off）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


# ===== 常量 =====

ENGINE_ID = "llama_cpp_9cdb76f8"
ENGINE_NAME = "Llama.cpp"
CAPABILITY = "chat"
ENGINE_TYPE = "llm"

# 默认端口（与 config/engines/llama_cpp_9cdb76f8.yaml 一致）
DEFAULT_PORT = 7002

# 模型目录布局（相对 MODELS_ROOT）
# v0.2.1：优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE / AIRELIABLE_LORA_DIR
# v0.3.0：models_dir 直接由 AIRELIABLE_MODELS_ROOT + AIRELIABLE_MODEL_LAYOUT_BASE 拼接
_DEFAULT_CHECKPOINT_SUBDIR = "llm/checkpoints/llama/"
_DEFAULT_LORA_SUBDIR = "llm/loras/llama/"               # v0.1 不启用，目录留空


def _resolve_checkpoint_subdir() -> str:
    """解析 checkpoint 子目录（v0.2.1：优先读环境变量）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_CHECKPOINT_SUBDIR


def _resolve_lora_subdir() -> str:
    """解析 lora 子目录（v0.2.1：优先读环境变量）。"""
    env_val = os.environ.get("AIRELIABLE_LORA_DIR")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_LORA_SUBDIR


# 运行时常量（在模块加载时根据环境变量解析）
CHECKPOINT_SUBDIR = _resolve_checkpoint_subdir()
LORA_SUBDIR = _resolve_lora_subdir()

# v0.2.15+ 日志统一到 logs/engines/ 和 logs/engines/llama_cpp_server/（项目顶层）
# 旧的 LOG_SUBDIR = "logs/"（相对 data_root）已废弃

# llama-bins 二进制目录默认值（外部路径，由 registry.yaml env 注入）
DEFAULT_LLAMA_BINS_DIR = "D:/myProInstallDir/VLLM/llama/llama-bins"

# 隐藏端口范围（llama-server 内部监听，仅 127.0.0.1）
DEFAULT_INTERNAL_PORT_START = 8100
DEFAULT_INTERNAL_PORT_END = 8200

# 子进程启动超时（秒）
# 注：28GB+ 的大模型加载到 GPU 通常需要 60-120 秒
DEFAULT_SERVER_TIMEOUT = 120.0

# 健康检查间隔（秒）
DEFAULT_HEALTH_CHECK_INTERVAL = 5.0

# router mode 默认参数
DEFAULT_MODELS_MAX = 4                                       # 最大并存模型数
DEFAULT_PARALLEL_SLOTS = 1                                   # parallel/server slots（对齐 start_llama_server.bat --parallel 1，
                                                             #   避免默认 -1=auto 在单卡单用户场景下生成 4 槽导致 KV ×4 预占，
                                                             #   触发 auto fit 判定显存不足降 offload 层数）
DEFAULT_CTX = 8192                                           # 全局默认上下文长度
DEFAULT_CACHE_TYPE_K = "q4_0"                                # K 缓存量化
DEFAULT_CACHE_TYPE_V = "q4_0"                                # V 缓存量化


@dataclass
class EngineConfig(BaseEngineConfig):
    """引擎运行配置（v0.3.0 router mode 版）。

    通过环境变量 + 命令行参数构造。
    """

    # ===== 基础（身份 / 端口默认值由本引擎覆盖，其余公共字段见基类）=====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT

    # ===== 推导路径 =====
    checkpoint_dir: Path = field(init=False)         # GGUF 模型目录（router mode 的 --models-dir）
    lora_dir: Path = field(init=False)
    log_dir: Path = field(init=False)                # 引擎 Python 进程日志目录
    server_log_dir: Path = field(init=False)         # llama-server 子进程日志目录（v0.2.15+）

    # ===== llama-bins 二进制 =====
    llama_bins_dir: Path = field(default_factory=lambda: Path(DEFAULT_LLAMA_BINS_DIR))

    # ===== 隐藏端口范围 =====
    internal_port_start: int = DEFAULT_INTERNAL_PORT_START
    internal_port_end: int = DEFAULT_INTERNAL_PORT_END
    # v0.3.5+ 固定内部端口（优先使用，若设置则忽略端口范围）
    internal_port: Optional[int] = None

    # ===== 子进程管理 =====
    server_timeout: float = DEFAULT_SERVER_TIMEOUT
    health_check_interval: float = DEFAULT_HEALTH_CHECK_INTERVAL

    # ===== router mode 参数（v0.3.0 新增）=====
    models_max: int = DEFAULT_MODELS_MAX              # 最大并存模型数
    # llama-server --models-autoload 默认 enabled（按需自动加载）。
    # 此处默认 True：让 router mode 收到 chat 请求时自动加载指定模型，
    # 无需 engine 层手动调 /models/load。
    # 若设为 False，会传 --no-models-autoload，此时必须手动调 /models/load。
    models_autoload: bool = True
    ctx: int = DEFAULT_CTX                            # 全局默认上下文长度
    cache_type_k: str = DEFAULT_CACHE_TYPE_K
    cache_type_v: str = DEFAULT_CACHE_TYPE_V
    # v1.0+ 并行槽数（等价 bat --parallel 1）：影响 KV 预分配数量。
    # 来源优先级：命令行 --parallel > 环境变量 LLAMA_PARALLEL_SLOTS > DEFAULT_PARALLEL_SLOTS(=1)
    # 平台通过 engine_manager._build_engine_subprocess_env 注入 LLAMA_PARALLEL_SLOTS
    # （读 config/engines/<engine_id>.runtime.json.parallel_slots，如果存在）。
    parallel_slots: int = DEFAULT_PARALLEL_SLOTS

    # ===== 默认 variant（v0.3.0 新增）=====
    # 优先级：manifest.models.default_variant > 此处 config.default_variant > None
    # 用途：反代路由 auto-load 时预加载此 variant；BaseEngine.infer auto-load 时也用此值。
    # 平台启动引擎子进程时通过 LLAMA_DEFAULT_VARIANT 环境变量注入（可选），
    # 引擎也会自行读取 manifest.yaml 的 default_variant 字段作为主来源。
    default_variant: Optional[str] = None

    # ===== per-variant ctx 覆盖（v0.3.3+ 新增）=====
    # 用户通过前端 UI 滑块调整后，平台会通过 LLAMA_VARIANT_CTX_OVERRIDES 环境变量注入。
    # 格式：JSON 字符串，key=variant.name，value=ctx_size（int）
    # 示例：{"qwen3.6-35b-a3b-q6-k": 65536, "qwen3.5-35b-a3b-q4-k-m": 32768}
    # 优先级：variant_ctx_overrides[variant_name] > variant.ctx.recommended > self.ctx
    variant_ctx_overrides: dict[str, int] = field(default_factory=dict)

    # ===== GPU 检测（None = 自动检测，int = 强制指定显存 GB）=====
    gpu_vram_gb: Optional[int] = None

    # ===== 设备策略（v0.3.5+ 新增）=====
    # auto / gpu / gpu+cpu / cpu，见模块顶部 DEVICE_POLICIES 注释。
    # 平台通过 AIRELIABLE_DEVICE_POLICY 环境变量注入（UI 设置 > 默认 gpu，v1.0+ 改为 gpu 默认等价 bat --n-gpu-layers 999）
    # 映射到 llama-server 的 -ngl / --fit 参数（见 llama_process._build_router_mode_command）。
    #   gpu     → -ngl all --fit off（对应 bat：全 GPU，显存不足直接 OOM 报错；避免 auto fit 保守降 offload 层）
    #   gpu+cpu → -ngl auto --fit on --fit-target 512 --fit-ctx 4096（对应新版 llm 增强：GPU 优先 + 显存安全余量 + ctx不下限）
    #   cpu     → -ngl 0 --fit off
    #   auto    → 不干预（llama-server 默认 ngl=auto+fit=on，保守档，不再作平台默认）
    # v1.0+ 默认从 auto 改为 gpu（对齐 bat --n-gpu-layers 999 语义）
    device_policy: str = "gpu"

    def __post_init__(self) -> None:
        """推导子路径。"""
        # 模型相关
        self.checkpoint_dir = self.models_root / CHECKPOINT_SUBDIR
        self.lora_dir = self.models_root / LORA_SUBDIR

        # 数据相关
        # v0.2.15+ 日志统一到 logs/engines/（引擎 Python 进程）和
        # logs/engines/llama_cpp_server/（llama-server 子进程）
        self.log_dir = self.project_root / "logs" / "engines"
        # llama-server 子进程日志单独分目录（避免与引擎 Python 日志混淆）
        self.server_log_dir = self.project_root / "logs" / "engines" / "llama_cpp_server"

        # 确保运行时目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # 输出根目录（output/{text,...}/）会在引擎调 get_output_path(category, ...)
        # 时按需创建，这里只确保根目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)

        # 平台 core URL（若未显式传入，从环境变量读取）
        if self.core_url is None:
            self.core_url = os.environ.get("AIRELIABLE_CORE_URL")

    @property
    def llama_server_path(self) -> Path:
        """llama-server 可执行文件路径。

        Windows: llama-bins/llama-server.exe
        Linux/Mac: llama-bins/llama-server
        """
        import sys
        if sys.platform == "win32":
            return self.llama_bins_dir / "llama-server.exe"
        return self.llama_bins_dir / "llama-server"

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "checkpoint_dir": str(self.checkpoint_dir),
            "lora_dir": str(self.lora_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "llama_bins_dir": str(self.llama_bins_dir),
            "llama_server_path": str(self.llama_server_path),
            "internal_port_range": f"{self.internal_port_start}-{self.internal_port_end}",
            "server_timeout": self.server_timeout,
            "gpu_vram_gb": self.gpu_vram_gb,
            "device_policy": self.device_policy,
            "models_max": self.models_max,
            "models_autoload": self.models_autoload,
            "ctx": self.ctx,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "core_url": self.core_url,
            "variant_ctx_overrides": self.variant_ctx_overrides,
        }


def parse_args() -> EngineConfig:
    """从命令行参数 + 环境变量构造 EngineConfig。

    优先级：命令行 > 环境变量 > 默认值
    """
    parser = argparse.ArgumentParser(description="Llama.cpp 引擎服务（v0.3.0 router mode）")

    # 网络参数
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    parser.add_argument("--host", type=str, default=None, help="监听地址")

    # 路径参数
    parser.add_argument(
        "--models-root", type=str, default=None,
        help="模型根目录（默认: ./models 或 AIRELIABLE_MODELS_ROOT 环境变量）",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="运行时数据根目录（默认: ./data 或 DATA_ROOT 环境变量）",
    )
    parser.add_argument(
        "--project-root", type=str, default=None,
        help="项目根目录（默认: 当前目录）",
    )

    # llama-bins 路径
    parser.add_argument(
        "--llama-bins-dir", type=str, default=None,
        help="llama-bins 二进制目录（默认: 环境变量 LLAMA_BINS_DIR）",
    )

    # 隐藏端口范围
    parser.add_argument(
        "--internal-port-start", type=int, default=None,
        help="隐藏端口范围起始（默认: 8100）",
    )
    parser.add_argument(
        "--internal-port-end", type=int, default=None,
        help="隐藏端口范围结束（默认: 8200）",
    )
    parser.add_argument(
        "--internal-port", type=int, default=None,
        help="固定内部端口（优先使用，若设置则忽略端口范围，默认: 8100）",
    )

    # 子进程超时
    parser.add_argument(
        "--server-timeout", type=float, default=None,
        help="llama-server 启动超时秒数（默认: 120）",
    )

    # GPU 显存（强制指定）
    parser.add_argument(
        "--gpu-vram-gb", type=int, default=None,
        help="强制指定 GPU 显存 GB（默认: 自动检测 nvidia-smi）",
    )

    # 设备策略（v0.3.5+ 新增）
    parser.add_argument(
        "--device-policy", type=str, default=None,
        choices=list(DEVICE_POLICIES),
        help=(
            "设备策略（默认: 环境变量 AIRELIABLE_DEVICE_POLICY 或 gpu；对齐 bat --n-gpu-layers 999）。"
            "auto=引擎自行决定；gpu=强制全 GPU；"
            "gpu+cpu=GPU 优先显存不足溢出 CPU；cpu=强制 CPU"
        ),
    )

    # router mode 参数（v0.3.0 新增）
    parser.add_argument(
        "--models-max", type=int, default=None,
        help=f"router mode 最大并存模型数（默认: {DEFAULT_MODELS_MAX}）",
    )
    parser.add_argument(
        "--models-autoload", action="store_true", default=None,
        help="启动时自动加载首个模型（默认: 不自动加载，由用户在 UI 选择）",
    )
    parser.add_argument(
        "--parallel", type=int, default=None, dest="parallel",
        help=(
            f"router server slots/并行槽数（默认: 环境变量 LLAMA_PARALLEL_SLOTS "
            f"或 {DEFAULT_PARALLEL_SLOTS}）。对齐 bat 脚本 --parallel 1：设为 1 可避免 "
            "默认 auto=4 槽 KV ×4 预占，触发 fit 判定显存不足降 offload 层数。"
        ),
    )
    parser.add_argument(
        "--ctx", type=int, default=None,
        help=f"全局默认上下文长度（默认: {DEFAULT_CTX}）",
    )
    parser.add_argument(
        "--cache-type-k", type=str, default=None,
        help=f"K 缓存量化类型（默认: {DEFAULT_CACHE_TYPE_K}）",
    )
    parser.add_argument(
        "--cache-type-v", type=str, default=None,
        help=f"V 缓存量化类型（默认: {DEFAULT_CACHE_TYPE_V}）",
    )

    # per-variant ctx 覆盖（v0.3.3+ 新增）
    # 支持两种格式：
    #   1. JSON 字符串：--variant-ctx '{"qwen3.6-q6": 65536}'
    #   2. 重复的 key=value：--variant-ctx qwen3.6-q6=65536 --variant-ctx qwen3.5-q4=32768
    parser.add_argument(
        "--variant-ctx", action="append", default=None,
        help=(
            "per-variant ctx 覆盖（可重复指定）。"
            "格式 1：JSON 字符串 '{\"qwen3.6-q6\": 65536}'；"
            "格式 2：key=value 'qwen3.6-q6=65536'。"
            "也可通过环境变量 LLAMA_VARIANT_CTX_OVERRIDES（JSON 字符串）注入。"
        ),
    )

    # 日志
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="日志级别（DEBUG/INFO/WARNING/ERROR）",
    )

    args = parser.parse_args()

    # 合并环境变量
    def _env_or(arg_val: str | None, env_key: str, default: str | None = None) -> str | None:
        if arg_val is not None:
            return arg_val
        return os.environ.get(env_key, default)

    port = int(args.port) if args.port else int(os.environ.get("ENGINE_PORT", DEFAULT_PORT))
    host = args.host or "127.0.0.1"
    models_root = Path(_env_or(args.models_root, "AIRELIABLE_MODELS_ROOT", "models"))
    data_root = Path(_env_or(args.data_root, "DATA_ROOT", "data"))
    project_root = Path(args.project_root or ".")

    # llama-bins 目录
    llama_bins_dir_str = _env_or(args.llama_bins_dir, "LLAMA_BINS_DIR", DEFAULT_LLAMA_BINS_DIR)
    llama_bins_dir = Path(llama_bins_dir_str)

    # 隐藏端口范围
    internal_port_start = (
        args.internal_port_start
        if args.internal_port_start is not None
        else int(os.environ.get("LLAMA_INTERNAL_PORT_START", DEFAULT_INTERNAL_PORT_START))
    )
    internal_port_end = (
        args.internal_port_end
        if args.internal_port_end is not None
        else int(os.environ.get("LLAMA_INTERNAL_PORT_END", DEFAULT_INTERNAL_PORT_END))
    )

    # 子进程超时
    server_timeout = (
        args.server_timeout
        if args.server_timeout is not None
        else float(os.environ.get("LLAMA_SERVER_TIMEOUT", DEFAULT_SERVER_TIMEOUT))
    )

    # GPU 显存
    gpu_vram_gb = args.gpu_vram_gb
    if gpu_vram_gb is None and os.environ.get("GPU_VRAM_GB"):
        try:
            gpu_vram_gb = int(os.environ["GPU_VRAM_GB"])
        except ValueError:
            pass

    # v1.0+ 设备策略：命令行 > 环境变量 AIRELIABLE_DEVICE_POLICY（平台注入）> 默认 gpu
    # 映射到 llama-server 参数：
    #   gpu     → -ngl all --fit off（对齐 bat --n-gpu-layers 999，强制全 GPU；OOM 直接报错）
    #   gpu+cpu → -ngl auto --fit on --fit-target 512 --fit-ctx 4096
    #             （新版 llama.cpp 机制：GPU 优先，fit 自动调 offload 层数 + 留 512MiB 安全余量
    #              + 保证 ctx ≥ 4096；对应 bat 脚本的 GPU 优先但允许溢出 CPU）
    #   cpu     → -ngl 0 --fit off（纯 CPU）
    #   auto    → 不干预（llama-server 默认 ngl=auto + fit=on；不再是平台默认）
    device_policy = (
        args.device_policy
        or os.environ.get("AIRELIABLE_DEVICE_POLICY", "gpu").strip()
        or "gpu"
    )
    if device_policy not in DEVICE_POLICIES:
        logger.warning(
            "非法设备策略 %r（合法值：%s），回退 gpu。"
            "来源：命令行 --device-policy 或环境变量 AIRELIABLE_DEVICE_POLICY。",
            device_policy, DEVICE_POLICIES,
        )
        device_policy = "gpu"
    logger.info("设备策略：%s（由 AIRELIABLE_DEVICE_POLICY / --device-policy 生效）", device_policy)

    # router mode 参数（v0.3.0 新增）
    models_max = (
        args.models_max
        if args.models_max is not None
        else int(os.environ.get("LLAMA_MODELS_MAX", str(DEFAULT_MODELS_MAX)))
    )
    # parallel_slots（v1.0+，对齐 bat --parallel 1）
    # 优先级：命令行 --parallel > 环境变量 LLAMA_PARALLEL_SLOTS > DEFAULT_PARALLEL_SLOTS(=1)
    parallel_slots = DEFAULT_PARALLEL_SLOTS
    if args.parallel is not None:
        parallel_slots = int(args.parallel)
    elif os.environ.get("LLAMA_PARALLEL_SLOTS"):
        try:
            parallel_slots = int(os.environ["LLAMA_PARALLEL_SLOTS"])
        except ValueError:
            logger.warning(
                "LLAMA_PARALLEL_SLOTS=%r 非整数，回退默认 %d。",
                os.environ["LLAMA_PARALLEL_SLOTS"], DEFAULT_PARALLEL_SLOTS,
            )
    if parallel_slots <= 0:
        parallel_slots = DEFAULT_PARALLEL_SLOTS  # 0 或负数=auto（bat 里无此用法），回退 1
    if parallel_slots != 4:  # 非历史默认值时打印日志
        logger.info("parallel_slots=%d（来源：--parallel / LLAMA_PARALLEL_SLOTS）", parallel_slots)
    # models_autoload 优先级：命令行 > 环境变量 > 默认 True
    # 默认 True：router mode 按需自动加载模型（用户在 UI 切换模型时无需手动调 /models/load）
    # 若设为 False，必须手动调 POST /models/load 才能使用模型，UI 体验差
    if args.models_autoload is True:
        models_autoload = True
    else:
        models_autoload = os.environ.get("LLAMA_MODELS_AUTOLOAD", "1").lower() in ("1", "true", "yes")

    ctx = (
        args.ctx
        if args.ctx is not None
        else int(os.environ.get("LLAMA_CTX", str(DEFAULT_CTX)))
    )
    cache_type_k = (
        args.cache_type_k
        if args.cache_type_k is not None
        else os.environ.get("LLAMA_CACHE_TYPE_K", DEFAULT_CACHE_TYPE_K)
    )
    cache_type_v = (
        args.cache_type_v
        if args.cache_type_v is not None
        else os.environ.get("LLAMA_CACHE_TYPE_V", DEFAULT_CACHE_TYPE_V)
    )
    # default_variant：优先环境变量 LLAMA_DEFAULT_VARIANT（可选，引擎也会自行读 manifest）
    default_variant = os.environ.get("LLAMA_DEFAULT_VARIANT") or None

    # per-variant ctx 覆盖（v0.3.3+ 新增）
    # 优先级：命令行 --variant-ctx > 环境变量 LLAMA_VARIANT_CTX_OVERRIDES > 空
    # 支持两种 CLI 格式：JSON 字符串 或 重复的 key=value
    variant_ctx_overrides: dict[str, int] = {}

    # 1. 先读环境变量（JSON 字符串）
    env_json = os.environ.get("LLAMA_VARIANT_CTX_OVERRIDES", "").strip()
    if env_json:
        try:
            import json
            parsed = json.loads(env_json)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, int) and v > 0:
                        variant_ctx_overrides[str(k)] = v
                    else:
                        logger.warning("LLAMA_VARIANT_CTX_OVERRIDES 项 %s=%r 非 int 或 <=0，已忽略", k, v)
            else:
                logger.warning("LLAMA_VARIANT_CTX_OVERRIDES 不是 JSON 对象，已忽略")
        except json.JSONDecodeError as e:
            logger.warning("LLAMA_VARIANT_CTX_OVERRIDES JSON 解析失败: %s", e)

    # 2. 命令行参数覆盖（优先级最高）
    if args.variant_ctx:
        for item in args.variant_ctx:
            # 格式 1：JSON 字符串（以 { 开头）
            if item.strip().startswith("{"):
                try:
                    import json
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if isinstance(v, int) and v > 0:
                                variant_ctx_overrides[str(k)] = v
                            else:
                                logger.warning("--variant-ctx JSON 项 %s=%r 非 int 或 <=0，已忽略", k, v)
                    else:
                        logger.warning("--variant-ctx JSON 不是对象，已忽略: %s", item)
                except json.JSONDecodeError as e:
                    logger.warning("--variant-ctx JSON 解析失败: %s, raw=%s", e, item)
            # 格式 2：key=value
            elif "=" in item:
                k, _, v_str = item.partition("=")
                k = k.strip()
                try:
                    v = int(v_str.strip())
                    if v > 0:
                        variant_ctx_overrides[k] = v
                    else:
                        logger.warning("--variant-ctx %s=%s 值 <=0，已忽略", k, v_str)
                except ValueError:
                    logger.warning("--variant-ctx %s=%s 值非整数，已忽略", k, v_str)
            else:
                logger.warning("--variant-ctx 参数格式无法识别（需 JSON 或 key=value）: %s", item)

    return EngineConfig(
        port=port,
        host=host,
        models_root=models_root,
        data_root=data_root,
        project_root=project_root,
        llama_bins_dir=llama_bins_dir,
        internal_port_start=internal_port_start,
        internal_port_end=internal_port_end,
        server_timeout=server_timeout,
        gpu_vram_gb=gpu_vram_gb,
        device_policy=device_policy,
        models_max=models_max,
        parallel_slots=parallel_slots,
        models_autoload=models_autoload,
        ctx=ctx,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        default_variant=default_variant,
        variant_ctx_overrides=variant_ctx_overrides,
        log_level=args.log_level or os.environ.get("LOG_LEVEL", "INFO"),
    )
