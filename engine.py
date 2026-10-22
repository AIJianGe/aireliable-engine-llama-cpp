"""Llama.cpp 引擎推理核心（v0.3.0 router mode 重构版）。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法，封装 llama-server router mode 子进程。

8 端点契约：
    - /capabilities    返回扫描到的 GGUF variant 列表
    - /load            启动 router mode 子进程（不预加载模型）
    - /unload          停止子进程（释放所有模型显存）
    - /infer           转发到 llama-server /v1/chat/completions（字节流透传）
    - /status          返回引擎状态（子进程 / 已加载模型 / 端口）
    - /lora/load       v0.1 不支持，返回 501
    - /lora/unload     v0.1 不支持，返回 501

v0.3.0 关键变更：
    - 继承 SDK CustomEngine（替代本地 BaseEngine）
    - /load 改为启动 router mode 子进程，不预加载任何 variant
    - /infer 改为字节流透传（不再用 ChatAdapter 转 AIReliable 进度帧）
    - /load 中加文件完整性校验（不下载，只校验 + 抛错指引）
    - 模型扫描走平台 API 优先 + 本地 fallback
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    make_error_frame,
    make_final_frame,
)

from .config import EngineConfig
from .llama_process import LlamaProcessManager
from .model_scanner import ModelScanner, VariantInfo
from .schemas import (
    InferInputs,
    InferRequest,
    LlamaStatusResponse,
    LlamaVariantInfo,
)

logger = logging.getLogger(__name__)


def _parse_vram_str_to_bytes(vram_str: str) -> int:
    """v0.3.3+ 解析显存字符串为字节数。

    支持格式：
        - "48GB" / "48gb" / "48 GB" → 48 * 1024^3
        - "2048MB" / "2048mb" → 2048 * 1024^2
        - "1024" → 1024 字节（纯数字按字节处理）

    Args:
        vram_str: 显存字符串

    Returns:
        int: 字节数；解析失败返回 0
    """
    if not vram_str:
        return 0
    s = vram_str.strip().upper().replace(" ", "")
    try:
        if s.endswith("GB"):
            return int(float(s[:-2]) * 1024 ** 3)
        if s.endswith("MB"):
            return int(float(s[:-2]) * 1024 ** 2)
        if s.endswith("KB"):
            return int(float(s[:-2]) * 1024)
        return int(float(s))
    except (ValueError, IndexError):
        return 0


class LlamaCppEngine(CustomEngine):
    """Llama.cpp 引擎（v0.3.0 router mode + 反代 UI）。

    全自定义模式（integrated 范式），直接控制 llama-server router mode 子进程，
    不走工作流节点系统。UI 通过反向代理 llama-server 自带 SvelteKit Web UI
    （由 server.py 的反代路由处理）。
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # 核心组件
        self.process_manager = LlamaProcessManager(config)
        # v0.3.0：注入 engine=self，启用平台 API 声明式检测
        # （平台不可达时 fallback 到本地扫描）
        self.scanner = ModelScanner(config.checkpoint_dir, engine=self)

        # 启动时扫描模型目录（同步初始化为空，异步扫描在 lifespan 中触发）
        self._available_variants: list[VariantInfo] = []

        # v0.3.9+（方案 C2）扫描自愈状态：
        #   _last_scan_mode     最近一次扫描来源（"platform" / "fallback"）
        #   _last_scan_time     最近一次扫描完成时间（time.time()）
        #   _scan_rescue_attempts 进程内自愈重扫已尝试次数（上限 2，
        #                        防平台长期离线时每次 /load 卡重试链）
        self._last_scan_mode: str = ""
        self._last_scan_time: float = 0.0
        self._scan_rescue_attempts: int = 0

        # __init__ 阶段平台 API 可能尚未就绪，先用 fallback 本地扫描
        # 保证引擎实例化不失败；后续 lifespan 会在平台 API 可用后再次异步刷新
        self._refresh_scan()

        # v0.1 不支持 LoRA
        self._available_loras: list[str] = []

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 default_variant（优先 manifest，回退 config）
        # 用途：BaseEngine.infer auto-load + 反代路由 auto-load
        self.default_variant = self._read_default_variant()

    # ============ 扫描 ============

    def _read_default_variant(self) -> Optional[str]:
        """读取默认 variant 名。

        优先级：
            1. manifest.yaml 的 models.default_variant 字段（主来源）
            2. config.default_variant（环境变量 LLAMA_DEFAULT_VARIANT 注入）
            3. None

        manifest 路径推导：engine.py 所在目录 / manifest.yaml
        """
        # 1. 优先读 manifest.yaml
        try:
            import yaml
            manifest_path = Path(__file__).parent / "manifest.yaml"
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                models_block = data.get("models", {}) or {}
                dv = models_block.get("default_variant")
                if dv:
                    logger.info("从 manifest 读取 default_variant: %s", dv)
                    return dv
        except Exception as e:
            logger.warning("读取 manifest default_variant 失败: %s", e)

        # 2. 回退到 config.default_variant
        if self.config.default_variant:
            logger.info("从 config 读取 default_variant: %s", self.config.default_variant)
            return self.config.default_variant

        # 3. None
        logger.info("default_variant 未配置（manifest 和 config 均未声明）")
        return None

    def _refresh_scan(self) -> None:
        """刷新模型扫描结果（同步 fallback，用于 __init__ 阶段）。

        v0.3.0：此方法走 fallback 本地扫描逻辑。
        平台 API 异步扫描请调用 `_refresh_scan_async()`（lifespan startup 阶段调用）。
        """
        logger.info("刷新模型扫描 (fallback): %s", self.config.checkpoint_dir)
        self._available_variants = self.scanner.scan_variants()
        # v0.3.9+ 方案 C2：记录扫描来源模式与时间
        self._last_scan_mode = self.scanner.last_scan_mode or "fallback"
        self._last_scan_time = time.time()

    async def _refresh_scan_async(self) -> None:
        """v0.3.0：异步刷新模型扫描结果（优先平台 API）。

        在 lifespan startup 阶段调用，此时平台 API 已就绪。
        平台不可达时 fallback 到本地扫描。
        """
        logger.info("异步刷新模型扫描（平台 API 优先）: %s", self.config.checkpoint_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()
        # v0.3.9+ 方案 C2：记录扫描来源模式与时间
        self._last_scan_mode = self.scanner.last_scan_mode or "fallback"
        self._last_scan_time = time.time()

    def _maybe_rescan_before_load(self) -> bool:
        """v0.3.9+ 方案 C2：判断 /load 前是否需要触发扫描自愈重扫。

        触发条件（全部满足）：
            1. 最近一次扫描为 fallback 模式（平台 API 未成功）
            2. AIRELIABLE_CORE_URL 环境变量已设置（平台可能只是启动时未就绪）
            3. 进程内自愈重扫尝试次数 < 2（防平台长期离线时
               每次 /load 都卡 SDK 重试链 2+5+10s + 超时）

        v0.3.9+ 方案 A 已让 fallback 命名对齐 manifest（即使重扫失败，
        INI 也能生成正确的 manifest 名），此自愈是防御纵深：
        让 available 字段 / ctx override / planned_ctx 上报等平台语义恢复。

        Returns:
            bool: True = 需要重扫（调用方 await _refresh_scan_async()）
        """
        core_url = os.environ.get("AIRELIABLE_CORE_URL")
        if not core_url:
            return False
        if self._last_scan_mode == "platform":
            return False
        if self._scan_rescue_attempts >= 2:
            logger.info(
                "扫描自愈：已达重试上限（%d 次），保持当前 fallback 扫描结果",
                self._scan_rescue_attempts,
            )
            return False
        return True

    # ============ 7 个 _xxx_impl 实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回可用 GGUF variant 列表。"""
        from .schemas import VariantCtxSchema

        variants: list[LlamaVariantInfo] = []
        for v in self._available_variants:
            # v0.3.3+ 透传 per-variant ctx 配置
            ctx_schema: Optional[VariantCtxSchema] = None
            if v.ctx is not None:
                ctx_schema = VariantCtxSchema(
                    recommended=v.ctx.recommended,
                    min=v.ctx.min,
                    max=v.ctx.max,
                )

            variants.append(LlamaVariantInfo(
                name=v.name,
                base_model=v.base_model,
                precision=v.precision,
                loader=v.loader,
                vram_required=v.vram_required or "未知",
                available=v.available,
                path=v.path.name if v.path else "",
                file_size=v.file_size,
                supports_mmproj=v.supports_mmproj,
                mmproj_path=str(v.mmproj_path) if v.mmproj_path else None,
                alias=v.alias,
                ctx=ctx_schema,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=self._available_loras,
            supported_sizes=[],                      # chat 引擎不用 size
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema（MCP 专属层动态 tool 数据源）
            input_schema=InferInputs.model_json_schema(),
        )

    async def _load_impl(self, variant: Optional[str] = None) -> LoadResponse:
        """POST /load：启动 router mode 子进程。

        v0.3.1 语义：
            - /load 仅启动子进程（router mode），**不预加载任何模型**
            - 模型由 router mode 在第一次 chat 请求时按需自动加载（--models-autoload）
            - 用户在 UI 切换模型时，router mode 自动加载，无需手动调 /models/load

        v0.3.1 变更（vs v0.3.0）：
            - 删除 _preload_variant 调用：启动后不占用显存，等第一次交互再加载
            - 新增 mmproj 文件校验：启动前检查所有 variant 声明的 mmproj 文件存在性

        文件完整性校验（project-rules.md §2.2）：
            - 不下载，只校验
            - 校验失败抛 RuntimeError 指引用户去模型管理界面下载

        v0.3.9+ 方案 C2：入口处扫描自愈。若引擎在 core 重启窗口内启动导致
        平台 API 扫描 502 回退 fallback（如 2026-09-09 qwen3.8-27b 远程调用
        404/400 事故），此处趁平台恢复后重扫一次，恢复 manifest 命名空间 +
        available / ctx override 平台语义。方案 A 已保证 fallback 命名对齐，
        此重扫是防御纵深（重扫失败不阻断 /load）。
        """
        # 0. v0.3.9+ 方案 C2：fallback 模式 + 平台可达 → 自愈重扫
        if self._maybe_rescan_before_load():
            self._scan_rescue_attempts += 1
            logger.info(
                "扫描自愈：当前为 fallback 模式（%s），尝试平台 API 重扫"
                "（第 %d/2 次）",
                self._last_scan_mode, self._scan_rescue_attempts,
            )
            prev_mode = self._last_scan_mode
            try:
                await self._refresh_scan_async()
            except Exception as e:
                # 重扫失败不阻断启动（方案 A 已保证 fallback 命名对齐）
                logger.warning("扫描自愈重扫失败（不阻断启动）: %s", e)
            if self._last_scan_mode == "platform":
                logger.info(
                    "扫描自愈成功：%s → platform（variant 命名空间已恢复 manifest）",
                    prev_mode,
                )
            else:
                logger.warning(
                    "扫描自愈重扫后仍为 fallback 模式（平台可能离线），"
                    "继续用当前扫描结果启动（方案 A 保证命名对齐）"
                )

        # 1. 文件完整性校验（不下载，只校验 + 抛错指引）
        self._check_models_dir_integrity()

        # 2. v0.3.1+ mmproj 文件校验（缺失时警告，不阻断启动）
        self._check_mmproj_files()

        logger.info("开始启动 router mode 子进程（v0.3.1+ 不预加载模型，等第一次交互再加载）")

        # v0.3.3+ 记录 per-variant ctx 配置（日志可见性，便于排查显存问题）
        self._log_variant_ctx_config()

        # 3. 启动 router mode 子进程
        # v0.3.2+ 传入 variants 列表，用于生成 models-preset INI（含 mmproj 关联，多模态必需）
        start_time = time.time()
        try:
            result = await self.process_manager.start_router_mode(
                variants=self._available_variants
            )
        except FileNotFoundError:
            # llama-server 二进制 / models_dir 不存在
            raise
        except Exception as e:
            logger.exception("启动 llama-server router mode 失败")
            # 附带最近的子进程日志
            recent_logs = self.process_manager.read_recent_logs(50)
            if recent_logs:
                logger.error("llama-server 最近日志:\n%s", recent_logs)
            raise RuntimeError(f"启动 llama-server router mode 失败: {e}") from e

        load_time = result.get("load_time", 0.0)
        internal_port = result.get("internal_port")
        pid = result.get("pid")

        logger.info(
            "router mode 启动完成: pid=%s, port=%s, load_time=%.2fs（未加载模型，等待第一次交互）",
            pid, internal_port, load_time,
        )

        # v0.3.6+ 上报 planned_ctx（每个 variant 的生效 ctx）到 core，
        # 供前端运行参数 Tab 显示"引擎推荐（按显存自适应）"默认值。
        # 失败不阻断启动（core 不可达时降级，前端用 manifest 兜底）。
        try:
            await self._report_planned_ctx()
        except Exception as e:
            logger.warning("上报 planned_ctx 失败（不阻断启动）: %s", e)

        # v0.3.1+ 不预加载任何模型，返回 router-mode 标识
        # 模型由 router mode 在第一次 chat 请求时按需自动加载（--models-autoload）
        return LoadResponse(
            variant="router-mode",
            state="ready",
            vram_used="0MB",                          # 未加载模型，显存占用为 0
            load_time=load_time,
        )

    async def _report_planned_ctx(self) -> None:
        """v0.3.6+ 上报各 variant 的生效 ctx（planned_ctx）到平台 core。

        planned_ctx 来源：llama_process._generate_models_preset_ini 生成的
        per-variant ctx-size（override > 显存自适应 > manifest.recommended），
        即引擎实际会写入 models-preset INI 的值。

        前端运行参数 Tab 据此显示"引擎推荐（按显存自适应）"默认值；
        用户手动设置（override）后，此值随引擎重启更新。

        Returns:
            None；失败仅记录 warning（不阻断引擎启动）
        """
        planned = getattr(self.process_manager, "_planned_ctx", None)
        if not planned:
            return
        try:
            from aireliable_engine import get_platform_vram_manager
            vram_mgr = await get_platform_vram_manager()
            ok = await vram_mgr.report_planned_ctx(dict(planned))
            if ok:
                logger.info(
                    "已上报 planned_ctx: %d 个 variant %s",
                    len(planned), {k: v for k, v in list(planned.items())[:3]},
                )
        except Exception as e:
            logger.warning("上报 planned_ctx 异常（不阻断启动）: %s", e)

    def _check_mmproj_files(self) -> None:
        """v0.3.1+ 校验所有 variant 声明的 mmproj 文件存在性。

        缺失时记录警告（不阻断启动），让 router mode 自行处理。
        用户在 UI 切换到该 variant 时，router mode 会报 Model failed，
        但日志中已有警告便于排查。

        检查范围：
            - manifest.yaml 中所有 variant 声明的 mmproj 文件
            - 文件路径：{checkpoint_dir}/{mmproj_filename}
        """
        try:
            models_dir = self.config.checkpoint_dir
            for v in self._available_variants:
                mmproj_path = getattr(v, "mmproj_path", None)
                if not mmproj_path:
                    continue
                # v0.2.15+ 修复：路径解析健壮性
                # mmproj_path 可能是绝对路径、相对项目根目录的路径、或相对 checkpoint_dir 的路径
                # 仅检查 is_absolute() 会把相对项目根目录的路径错误拼接到 models_dir，产生重复前缀
                p = Path(mmproj_path)
                if not p.exists() and not p.is_absolute():
                    candidate = models_dir / p
                    if candidate.exists():
                        p = candidate
                if not p.exists():
                    logger.warning(
                        "variant %s 的 mmproj 文件缺失: %s "
                        "（切换到该模型时 router mode 会报 Model failed，请到模型管理界面下载）",
                        v.name, p,
                    )
                else:
                    # 校验文件大小（mmproj 文件至少 1MB）
                    size_mb = p.stat().st_size / (1024 * 1024)
                    if size_mb < 1:
                        logger.warning(
                            "variant %s 的 mmproj 文件可能损坏（size=%.2fMB < 1MB）: %s",
                            v.name, size_mb, p,
                        )
        except Exception as e:
            logger.warning("mmproj 文件校验异常（不阻断启动）: %s", e)

    def _log_variant_ctx_config(self) -> None:
        """v0.3.3+ 记录 per-variant ctx 配置（启动前日志可见性）。

        输出每个 variant 的：
            - manifest/GGUF 推导的 ctx 配置（recommended/min/max）
            - 用户覆盖值（来自 config.variant_ctx_overrides）
            - 实际生效的 ctx-size

        用于排查显存问题和确认 per-variant ctx 配置正确性。
        """
        overrides = self.config.variant_ctx_overrides or {}
        global_ctx = self.config.ctx

        logger.info("===== per-variant ctx 配置（v0.3.3+）=====")
        logger.info("全局默认 ctx: %d", global_ctx)
        if overrides:
            logger.info("用户覆盖（来自 LLAMA_VARIANT_CTX_OVERRIDES）: %s", overrides)
        else:
            logger.info("用户覆盖: 无")

        for v in self._available_variants:
            v_ctx = getattr(v, "ctx", None)
            override_val = overrides.get(v.name)

            if override_val is not None and v_ctx is not None:
                if v_ctx.min <= override_val <= v_ctx.max:
                    effective = override_val
                    source = "override"
                else:
                    effective = v_ctx.recommended
                    source = "manifest（override 超范围，回退）"
            elif override_val is not None:
                effective = override_val
                source = "override（无 manifest 约束）"
            elif v_ctx is not None:
                effective = v_ctx.recommended
                source = "manifest/gguf"
            else:
                effective = global_ctx
                source = "global-default"

            if v_ctx is not None:
                logger.info(
                    "  - %s: manifest[min=%d, recommended=%d, max=%d] | override=%s | "
                    "生效 ctx-size=%d（%s）",
                    v.name, v_ctx.min, v_ctx.recommended, v_ctx.max,
                    override_val if override_val is not None else "无",
                    effective, source,
                )
            else:
                logger.info(
                    "  - %s: manifest=无 | override=%s | 生效 ctx-size=%d（%s）",
                    v.name,
                    override_val if override_val is not None else "无",
                    effective, source,
                )
        logger.info("================================================")

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：停止子进程（释放所有模型显存）。"""
        if not self.process_manager.is_running:
            return UnloadResponse(state="idle", vram_freed="0GB")

        result = await self.process_manager.stop()
        logger.info("卸载完成: %s", result)
        return UnloadResponse(
            state=result.get("state", "idle"),
            vram_freed=result.get("vram_freed", "0GB"),
        )

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：转发到 llama-server /v1/chat/completions（解析 + 落盘）。

        v0.3.4+ 变更：
            - 解析 OpenAI SSE 字节流，提取 delta.content 累积完整文本
            - 发送 AIReliable 进度帧（stage=generating, delta=text）
            - 对话结果落盘到 output/text/{task_id}.json + {task_id}.txt
            - 完成帧含 text_url / text_path / finish_reason / completion_tokens
            - 集成 SDK 任务上报（report_created / report_running / report_succeeded / report_failed）
            - 修复两个 Bug：bytes 帧类型错误 + make_final_frame 缺 timing 参数

        v0.3.0 保留：
            - 支持 router mode：通过 model 字段路由到已加载模型
            - 若指定 variant 但模型未加载，自动调 /models/load

        错误处理：内部捕获异常，推送 error 帧后 return。
        """
        # ----- 0. 生成 task_id + 初始化 SDK 任务上报（v0.3.4+）-----
        task_id = request.task_id or f"llama_{uuid.uuid4().hex[:16]}"
        start_time = time.time()

        # SDK 任务上报（失败不阻断推理）
        task_reporter = None
        try:
            from aireliable_engine import get_platform_task_reporter
            task_reporter = await get_platform_task_reporter()
            inputs_dict = request.inputs.model_dump() if hasattr(request.inputs, "model_dump") else {}
            request_summary = {
                "model": inputs_dict.get("model", ""),
                "max_tokens": inputs_dict.get("max_tokens", 2048),
                "temperature": inputs_dict.get("temperature", 0.7),
                "message_count": len(inputs_dict.get("messages", [])),
            }
            await task_reporter.report_created(
                task_id=task_id,
                request_summary=request_summary,
                variant=request.variant or "",
            )
        except Exception as e:
            logger.warning("SDK 任务上报初始化失败（不阻断推理）: %s", e)
            task_reporter = None

        # ----- 0.5 上报 running 状态 -----
        if task_reporter:
            try:
                task_reporter.fire_report_running(task_id, stage="infer_start")
            except Exception as e:
                logger.warning("report_running 失败（不阻断推理）: %s", e)

        if not self.process_manager.is_running:
            stream.put_threadsafe(
                make_error_frame("llama-server 子进程未启动，请先调 /load", "not_loaded")
            )
            if task_reporter:
                task_reporter.fire_report_failed(
                    task_id,
                    error={"type": "not_loaded", "message": "llama-server 子进程未启动"},
                )
            return

        inputs = request.inputs
        if not inputs.messages:
            stream.put_threadsafe(
                make_error_frame("messages 不能为空", "invalid_messages")
            )
            if task_reporter:
                task_reporter.fire_report_failed(
                    task_id,
                    error={"type": "invalid_messages", "message": "messages 不能为空"},
                )
            return

        # 解析 router mode 路由用的 model 字段
        # 优先级：inputs.model > request.variant > 默认（不指定，由 llama-server 选）
        model_name = inputs.model or request.variant

        # v0.3.8+ 修复：alias → preset 名翻译（/infer 路径缺失 alias 翻译 Bug）
        #
        # 反代层（server.py 的 catch-all 路由，v0.3.4+）对 OpenAI 端点调用
        # _translate_model_alias 会把 alias（如 qwen3.6-unc-q6）翻译为
        # variant.name（即 llama-server preset 名，如
        # qwen3.6-35b-a3b-uncensored-q6-k-p）。但网关经 /infer 路径传入时，
        # _infer_impl 直接把 inputs.model / request.variant 值原样传给
        # llama-server，不经过反代层的 alias 翻译逻辑——当 extra.inputs.model
        # 显式传入 alias（或上层/前端误把 alias 写入 inputs.model）时，
        # llama-server 找不到该 preset → /models/load 404 +
        # /v1/chat/completions 400 → 网关 500。
        #
        # 修复策略（与反代层保持一致）：
        #   1. 先在 variant 列表里按传入 model_name 做一次精确匹配
        #      （_find_variant 接受 variant.name 或 variant.alias）
        #   2. 如果匹配到 variant，且 model_name != variant.name，则说明是 alias，
        #      翻译为 variant.name（preset 名）
        #   3. 同步更新 inputs.model（若该字段存在），确保后续写入 body
        #      透传 llama-server /v1/chat/completions 的 model 字段也是 preset 名
        if model_name:
            _matched_v = self._find_variant(model_name)
            if _matched_v is not None and model_name != _matched_v.name:
                _translated = _matched_v.name
                logger.info(
                    "/infer alias 翻译: %s -> %s", model_name, _translated,
                )
                model_name = _translated
                # inputs.model 如果存在且原值是 alias，一并写回为 preset 名
                try:
                    if inputs.model and inputs.model != _translated:
                        inputs.model = _translated
                except Exception:
                    # 不可写的 Pydantic 对象直接忽略（靠上游 model_name 兜底）
                    pass

        # v0.3.4+ 修复：variant name 即 preset 名，不再映射为文件名 stem
        # llama-server router mode 的 model 字段需要 preset 名，
        # 而 preset 名 = INI 文件的 section 名 = variant.name（v0.3.2+ _generate_models_preset_ini 用 v.name 作为 section_name）
        # 之前的逻辑把 variant name 映射为文件名 stem（如 Qwen_Qwen3.6-35B-A3B-Q6_K），
        # 但 router mode 注册的 preset 名是 variant name（如 qwen3.6-35b-a3b-q6-k），
        # 导致 router mode 找不到 preset 返回 400 Bad Request。
        # 若调用方在 inputs.model 中显式指定了 preset 名，则原样使用。
        if model_name and not inputs.model and request.variant:
            logger.debug(
                "router mode model 字段: %s（variant name 即 preset name）",
                model_name,
            )

        # 若指定了 variant 但未加载，尝试自动加载
        # 注意：models_autoload=True 时 router mode 会自动按需加载，
        # 此处 load_model 仍保留作为显式预加载（失败不阻断，router 会自动加载）
        if model_name and model_name not in self.process_manager.loaded_models:
            try:
                # v0.3.3+ 注入 per-variant ctx-size
                ctx_val = self._resolve_ctx_for_preset(model_name)
                logger.info(
                    "模型未加载，自动加载: %s (ctx-size=%s)",
                    model_name, ctx_val if ctx_val else "INI默认",
                )
                await self.process_manager.load_model(model_name, ctx_size=ctx_val)

                # v0.3.3+ 加载成功后通知平台 core 显存状态
                try:
                    from aireliable_engine import get_platform_vram_manager
                    vram_mgr = await get_platform_vram_manager()
                    vram_used_bytes = self._estimate_variant_vram_bytes(model_name)
                    variant = self._find_variant_by_preset_name(model_name)
                    variant_name = variant.name if variant else model_name
                    # v0.3.6+ 上报运行时 ctx（= preset INI 写入的 ctx-size，即实际加载值）
                    await vram_mgr.report_loaded(
                        variant=variant_name,
                        vram_used_bytes=vram_used_bytes,
                        ctx_size=ctx_val,
                    )
                    logger.info(
                        "已通知平台显存状态: variant=%s vram_estimated=%.2fGB"
                        "%s",
                        variant_name, vram_used_bytes / (1024 ** 3),
                        f" ctx={ctx_val}" if ctx_val else "",
                    )
                except Exception as report_err:
                    # 上报失败不阻断推理
                    logger.warning("通知平台显存状态失败（不阻断推理）: %s", report_err)

            except Exception as e:
                # autoload=True 时，加载失败不阻断（router mode 会按需自动加载）
                # 只记录警告，继续尝试 forward_chat
                logger.warning(
                    "显式加载模型失败: %s (err=%s)，router mode 将按需自动加载",
                    model_name, e,
                )

        # 构造 OpenAI 消息
        # v0.3.4+ 修复：exclude_none=True 排除 None 字段（如 name=None），
        # 否则 llama-server 解析 messages 时报 "type must be string, but is null"
        messages = [m.model_dump(exclude_none=True) for m in inputs.messages]

        try:
            # 转发到 llama-server（返回字节流 async iterator）
            # v0.3.4+ 透传 chat_template_kwargs（用于禁用思维链等场景，None 不传）
            #
            # v0.3.8+ B2 修复：引擎侧兜底默认注入 {"enable_thinking": False}（双保险）。
            # - Gateway 路径：openai_endpoints.py L720-740 已经默认注入（对齐反代层），
            #   inputs.chat_template_kwargs 通常非 None，会优先使用上游值。
            # - 直连 /infer 路径（前端内部直连引擎端口、脚本测试、debug SDK 等）：
            #   上游未显式传值时，inputs.chat_template_kwargs 为 None（InferInputs 默认），
            #   此时兜底注入与 server.py catch-all 反代层 L927-936 保持一致，避免
            #   Qwen Uncensored 等模型默认开启思维链 → thinking token 占满 max_tokens
            #   → completion_tokens=0 finish=length content=[]。
            forward_kwargs: dict[str, Any] = {}
            if inputs.chat_template_kwargs is not None:
                forward_kwargs["chat_template_kwargs"] = inputs.chat_template_kwargs
            else:
                forward_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
                logger.debug(
                    "/infer 兜底注入 chat_template_kwargs=%s（上游未显式传）",
                    forward_kwargs["chat_template_kwargs"],
                )
            # v0.4.x OpenAI 工具调用透传：tools / tool_choice 经 forward_chat 的
            # **extra 直达 llama-server（request_body.update）。不透传时模型收不到
            # 工具定义，只能输出文本格式的工具调用，客户端解析 message.tool_calls
            # 崩溃（Hermes "object with .c..." 问题根因）。
            if inputs.tools is not None:
                forward_kwargs["tools"] = inputs.tools
                logger.info(
                    "/infer 工具调用透传: tools=%d 个, tool_choice=%s",
                    len(inputs.tools), inputs.tool_choice,
                )
            if inputs.tool_choice is not None:
                forward_kwargs["tool_choice"] = inputs.tool_choice
            # v0.4.x G11：OpenAI 标准采样参数透传（llama-server 原生支持）
            if inputs.frequency_penalty is not None:
                forward_kwargs["frequency_penalty"] = inputs.frequency_penalty
            if inputs.presence_penalty is not None:
                forward_kwargs["presence_penalty"] = inputs.presence_penalty
            # v0.4.x response_format（JSON mode / 结构化输出）透传：
            # llama-server 原生支持 {"type": "json_object"} 等
            if inputs.response_format is not None:
                forward_kwargs["response_format"] = inputs.response_format
                logger.info(
                    "/infer response_format 透传: %s", inputs.response_format,
                )
            # v0.4.x usage 统计：引擎内部始终流式，显式请求 llama-server 在
            # 流式末 chunk 返回 usage（OpenAI stream_options 规范；不支持时
            # 被忽略无害，completion_tokens 仍有逐 chunk 计数兜底）
            forward_kwargs["stream_options"] = {"include_usage": True}
            byte_stream = self.process_manager.forward_chat(
                messages=messages,
                model=model_name,
                max_tokens=inputs.max_tokens,
                temperature=inputs.temperature,
                top_p=inputs.top_p,
                stream=inputs.stream,
                seed=inputs.seed,
                stop=inputs.stop,
                **forward_kwargs,
            )

            # v0.3.4+ 解析 OpenAI SSE 字节流，提取 delta.content
            # 累积完整文本 + 发送 AIReliable 进度帧
            full_text_parts: list[str] = []
            completion_tokens = 0
            finish_reason = "stop"
            # v0.4.x usage / reasoning 完整统计：
            #   prompt_tokens_final / total_tokens_final：llama-server 流式末 chunk
            #   （stream_options.include_usage=True）返回的真实 usage 值，缺失时保持 0
            #   reasoning_parts：思维链增量累积（delta.reasoning_content），
            #   完成帧 result.reasoning_content 输出拼接结果
            prompt_tokens_final = 0
            total_tokens_final = 0
            reasoning_parts: list[str] = []
            # v0.4.x 工具调用聚合器：index → {"id","type","function":{"name","arguments"}}
            # 流式增量按 index 合并（id/name 覆盖式、arguments 拼接式），
            # 完成帧 result.tool_calls 输出完整列表
            tool_calls_acc: dict[int, dict] = {}

            async for chunk in byte_stream:
                # v0.3.4+ Bug 1 修复：chunk 是 bytes，不能直接传给 put_threadsafe
                # 解析 SSE 字节流，提取 delta.content
                if not chunk:
                    continue

                try:
                    chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                except Exception:
                    logger.warning("SSE chunk 解码失败，跳过")
                    continue

                # 本 chunk 内各 data 行提取值的暂存（发一帧）
                delta_text = ""
                reasoning_text = ""
                tool_calls_delta: list[dict] = []

                # 解析 SSE data 行
                # 格式：data: {"id":"...","choices":[{"delta":{"content":"你好"}}]}\n\n
                # 或：data: [DONE]\n\n
                for line in chunk_str.split("\n"):
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()  # 去掉 "data:" 前缀
                    if data_str == "[DONE]":
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # 提取 delta.content 和 delta.reasoning_content
                    choices = data.get("choices", [])
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta", {}) or {}
                        delta_text = delta.get("content", "") or ""
                        reasoning_text = delta.get("reasoning_content", "") or ""
                        if delta_text:
                            full_text_parts.append(delta_text)
                            completion_tokens += 1
                        if reasoning_text:
                            reasoning_parts.append(reasoning_text)

                        # v0.4.x 提取 delta.tool_calls 增量（OpenAI 流式格式）
                        tc_list = delta.get("tool_calls") or []
                        for tc in tc_list:
                            if not isinstance(tc, dict):
                                continue
                            tool_calls_delta.append(tc)
                            idx = tc.get("index", 0)
                            try:
                                idx = int(idx)
                            except (TypeError, ValueError):
                                idx = 0
                            acc = tool_calls_acc.setdefault(idx, {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                            if tc.get("id"):
                                acc["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                # name 一般在首个增量整体出现，覆盖式
                                acc["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                # arguments 分片到达，拼接式
                                acc["function"]["arguments"] += fn["arguments"]

                        # 检查 finish_reason
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr

                    # 提取 usage（如果有）
                    # v0.4.x 完整提取 prompt/completion/total（旧代码只捡
                    # completion_tokens，prompt_tokens 被丢弃 → 客户端上下文
                    # 管理/成本统计全错，见分析文档 §3.1）
                    usage = data.get("usage") or {}
                    if isinstance(usage, dict):
                        try:
                            if usage.get("prompt_tokens") is not None:
                                prompt_tokens_final = int(usage["prompt_tokens"])
                            if usage.get("completion_tokens") is not None:
                                completion_tokens = int(usage["completion_tokens"])
                            if usage.get("total_tokens") is not None:
                                total_tokens_final = int(usage["total_tokens"])
                        except (TypeError, ValueError):
                            logger.warning("usage 字段非法（忽略）: %s", usage)

                # 发送 AIReliable 进度帧（v0.3.4+ Bug 1 修复：dict 帧替代 bytes）
                # v0.3.5+ 修复：delta 字段只放提取的 content 文本，不放原始 SSE 文本
                # 之前发送 chunk_str（原始 "data: {...}" 文本）导致 gateway 把整个
                # SSE 文本作为 content 返回给客户端。
                # 现在改为：只发送 delta_text（提取的 content）；
                # reasoning_content（思维链）通过 extra 字段传递；
                # tool_calls 增量通过 tool_calls 字段传递（gateway 原样转发给客户端）。
                if delta_text or reasoning_text or tool_calls_delta:
                    frame_data: dict = {
                        "progress": 0.5,
                        "stage": "generating",
                        "delta": delta_text,           # content 文本（gateway 期望）
                    }
                    if reasoning_text:
                        frame_data["extra"] = {"reasoning_content": reasoning_text}
                    if tool_calls_delta:
                        frame_data["tool_calls"] = tool_calls_delta
                    stream.put_threadsafe(frame_data)

                # SDK 进度上报
                if task_reporter:
                    try:
                        task_reporter.fire_report_progress(
                            task_id, progress=0.5, stage="generating",
                        )
                    except Exception:
                        pass

            full_text = "".join(full_text_parts)
            reasoning_content = "".join(reasoning_parts)
            elapsed = time.time() - start_time

            # v0.4.x 完整 usage 统计（OpenAI 规范三字段）：
            #   - llama-server 返回了真实 usage（stream_options.include_usage）→ 用真实值
            #   - 未返回（老版本 server 不支持）→ prompt/total 按 0 上报，
            #     completion_tokens 仍有逐 chunk 计数兜底（gateway 侧还会二次兜底）
            usage_result: dict = {
                "prompt_tokens": prompt_tokens_final,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens_final
                or (prompt_tokens_final + completion_tokens),
            }

            # v0.4.x 聚合 tool_calls 完整列表（按 index 排序，无工具调用时为空）
            merged_tool_calls: list[dict] = [
                tool_calls_acc[i] for i in sorted(tool_calls_acc)
            ]

            # v0.3.4+ T13：落盘到 output/text/{task_id}.json + {task_id}.txt
            text_url = ""
            text_path = ""
            text_preview_url = ""
            try:
                json_path = self.get_output_path("text", f"{task_id}.json")
                txt_path = self.get_output_path("text", f"{task_id}.txt")

                # JSON 格式：完整对话记录（请求 + 响应 + 元信息）
                conversation = {
                    "task_id": task_id,
                    "model": model_name or "default",
                    "variant": request.variant or "",
                    "messages": messages,
                    "response": full_text,
                    "tool_calls": merged_tool_calls,
                    "reasoning_content": reasoning_content,
                    "usage": usage_result,
                    "finish_reason": finish_reason,
                    "completion_tokens": completion_tokens,
                    "elapsed_seconds": round(elapsed, 3),
                    "created_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
                    ),
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(conversation, f, ensure_ascii=False, indent=2)

                # TXT 格式：仅响应文本（人读友好）
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

                text_url = f"/output/text/{task_id}.json"
                text_path = str(json_path)
                text_preview_url = f"/output/text/{task_id}.txt"
                logger.info(
                    "对话已落盘: %s + %s (%d tokens, %.2fs, tool_calls=%d)",
                    json_path.name, txt_path.name, completion_tokens, elapsed,
                    len(merged_tool_calls),
                )
            except Exception as save_err:
                logger.warning("对话落盘失败（不阻断推理）: %s", save_err)

            # v0.3.4+ Bug 2 修复：make_final_frame 补 timing 参数
            result = {
                "model": model_name or "default",
                "variant": request.variant or "",
                "text": full_text,
                "text_url": text_url,
                "text_path": text_path,
                "text_preview_url": text_preview_url,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                # v0.4.x 透传给 gateway：真实 usage（OpenAI 三字段）+ 思维链全文
                # gateway 同步模式读 result.usage / result.reasoning_content 构造响应
                "usage": usage_result,
                "reasoning_content": reasoning_content,
            }
            # v0.4.x 工具调用透传：聚合 tool_calls 放入完成帧 result
            # （gateway 同步模式读 result.tool_calls 构造 message.tool_calls）
            if merged_tool_calls:
                result["tool_calls"] = merged_tool_calls
                logger.info(
                    "工具调用完成: %d 个, finish_reason=%s, names=%s",
                    len(merged_tool_calls), finish_reason,
                    [tc["function"].get("name", "") for tc in merged_tool_calls],
                )
            timing = {
                "total_seconds": round(elapsed, 3),
                "tokens": completion_tokens,
            }
            stream.put_threadsafe(make_final_frame(result, timing))

            # SDK 上报成功
            if task_reporter:
                try:
                    task_reporter.fire_report_succeeded(task_id, result=result)
                except Exception as e:
                    logger.warning("report_succeeded 失败（不阻断）: %s", e)

        except Exception as e:
            logger.exception("推理失败")
            stream.put_threadsafe(make_error_frame(f"推理失败: {e}", "infer_failed"))
            # SDK 上报失败
            if task_reporter:
                try:
                    task_reporter.fire_report_failed(
                        task_id,
                        error={"type": "infer_failed", "message": str(e)},
                    )
                except Exception:
                    pass

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态（含 router mode 多模型信息）。"""
        pm = self.process_manager
        # 直接用本地 _loaded_models set（llama-server GET /models 返回所有 preset，不是已加载模型）
        loaded_models = list(pm.loaded_models)

        return LlamaStatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=loaded_models[0] if loaded_models else None,
            vram_used="router-mode multi-model" if pm.is_running else "0GB",
            loaded_loras=[],                     # v0.1 不支持 LoRA
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            internal_port=pm.internal_port,
            llama_pid=pm.pid,
            models_max=self.config.models_max,
            loaded_models=loaded_models,
            ngl=None,                            # router mode 无全局 ngl
            ctx=self.config.ctx,
        )

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：v0.1 不支持，返回 501。"""
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="Llama.cpp 引擎 v0.1 不支持运行时 LoRA 切换（需重启子进程）",
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：v0.1 不支持，返回 501。"""
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="Llama.cpp 引擎 v0.1 不支持运行时 LoRA 切换（需重启子进程）",
        )

    # ============ 工具方法 ============

    def _check_models_dir_integrity(self) -> None:
        """校验模型目录完整性（project-rules.md §2.2）。

        v0.3.0 router mode：只校验 models_dir 存在 + 至少有一个 .gguf 文件。
        不校验具体 variant 文件大小（router mode 允许任意模型组合）。

        Raises:
            RuntimeError: 目录不存在 / 无 .gguf 文件，指引用户去模型管理界面下载
        """
        models_dir = self.config.checkpoint_dir
        if not models_dir.exists():
            raise RuntimeError(
                f"模型目录不存在：\n"
                f"  路径：{models_dir}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，"
                f"选择镜像源后下载。\n"
                f"HuggingFace 仓库：见 manifest.yaml 中 model_files[].source"
            )

        # 检查目录内是否有至少一个 .gguf 文件（不算 mmproj）
        gguf_files = [
            f for f in models_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".gguf"
            and not f.name.lower().startswith("mmproj")
        ]
        if not gguf_files:
            raise RuntimeError(
                f"模型目录为空（无 .gguf 文件）：\n"
                f"  路径：{models_dir}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，"
                f"选择镜像源后下载。\n"
                f"HuggingFace 仓库：见 manifest.yaml 中 model_files[].source"
            )

        logger.info(
            "模型目录校验通过：%s（发现 %d 个 .gguf 文件）",
            models_dir, len(gguf_files),
        )

    async def _preload_variant(self, variant: str) -> str:
        """预加载指定 variant 到 router mode 子进程。

        Args:
            variant: variant_id 或 alias

        Returns:
            str: 实际加载的模型名（用于 LoadResponse.variant 字段）

        Raises:
            FileNotFoundError: variant 不存在
            RuntimeError: 加载失败
        """
        info = self._find_variant(variant)
        if info is None:
            logger.warning(
                "预加载 variant 不存在: %s（可用: %s），跳过预加载",
                variant, [v.name for v in self._available_variants],
            )
            return variant  # 仍返回，让上层认为加载了

        if info.path is None or not info.path.exists():
            logger.warning(
                "variant 文件不存在: %s (path=%s)，跳过预加载",
                variant, info.path,
            )
            return variant

        # 用 GGUF 文件名（去 .gguf 后缀）作为 llama-server 的 model 字段
        # llama-server router mode 的 preset 名 = 文件名去 .gguf
        # （如 "Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf" → "Qwen_Qwen3.5-35B-A3B-Q4_K_M"）
        model_name = info.path.stem
        try:
            await self.process_manager.load_model(model_name)
            logger.info("预加载 variant 完成: %s (model=%s)", variant, model_name)
        except Exception as e:
            logger.warning(
                "预加载 variant 失败: %s (err=%s)，router mode 仍可按需加载",
                variant, e,
            )
        return variant

    def _find_variant(self, variant_id: str) -> Optional[VariantInfo]:
        """按 variant.name 或 alias（支持字符串/列表，与网关 config.py 对齐）查找 variant。

        v0.3.8+ 与网关 `GatewayConfig.get_engine_for_model` 匹配规则保持一致：
          - alias 是 str 时精确相等；
          - alias 是 list 时成员包含即命中。
        用于 /infer 路径 alias → preset 名（variant.name）翻译，
        也作为 Gateway 层统一翻译之外的防御兜底。
        """
        if not variant_id:
            return None
        for v in self._available_variants:
            if v.name == variant_id:
                return v
            alias = getattr(v, "alias", None)
            if alias is None:
                continue
            if isinstance(alias, list):
                if variant_id in alias:
                    return v
            else:
                if alias == variant_id:
                    return v
        return None

    def _find_variant_by_preset_name(self, preset_name: str) -> Optional[VariantInfo]:
        """v0.3.3+ 按 llama-server preset 名查找 variant。

        llama-server router mode 的 preset 名 = GGUF 文件名去 .gguf 后缀
        （如 "Qwen_Qwen3.6-35B-A3B-Q6_K"）。
        反代层拦截 /models/load 时收到的 model 字段是 preset 名，
        需要映射回 variant 信息以查询 ctx 配置和显存估算。

        Args:
            preset_name: llama-server preset 名（GGUF 文件 stem）

        Returns:
            VariantInfo | None
        """
        if not preset_name:
            return None
        for v in self._available_variants:
            if v.path is not None and v.path.stem == preset_name:
                return v
            # 兼容：variant.name 也可能是 preset 名
            if v.name == preset_name:
                return v
        return None

    def _resolve_ctx_for_preset(self, preset_name: str) -> Optional[int]:
        """v0.3.3+ 解析 preset 名对应的 variant 的生效 ctx-size。

        v0.3.6+ 变更：无 override 时优先按显存自适应推荐（与 llama_process 的
        _resolve_variant_ctx 保持一致，保证 auto-load 注入的 ctx 与 preset INI 一致）。

        优先级：
            1. config.variant_ctx_overrides[variant.name]（用户覆盖，需在 [min, max] 内）
            2. 显存自适应推荐（v0.3.6+，依据可用显存 + 模型权重估算）
            3. variant.ctx.recommended（manifest 或 GGUF 推导）
            4. None（使用 INI 文件默认值，不注入）

        Args:
            preset_name: llama-server preset 名

        Returns:
            int | None: ctx-size 值；None 表示用 INI 默认值
        """
        variant = self._find_variant_by_preset_name(preset_name)
        if variant is None:
            return None

        overrides = self.config.variant_ctx_overrides or {}
        override_val = overrides.get(variant.name)

        if override_val is not None and variant.ctx is not None:
            if variant.ctx.min <= override_val <= variant.ctx.max:
                return override_val
            logger.warning(
                "preset %s 的 ctx override=%d 超出 manifest 范围 [%d, %d]，"
                "回退到 recommended=%d",
                preset_name, override_val, variant.ctx.min, variant.ctx.max,
                variant.ctx.recommended,
            )
            return variant.ctx.recommended
        elif override_val is not None:
            return override_val
        elif variant.ctx is not None:
            # v0.3.6+ 显存自适应推荐（与 llama_process._resolve_variant_ctx 一致）
            try:
                free_vram_gb = self.process_manager._get_free_vram_gb()
                if free_vram_gb:
                    suggested = self.process_manager._suggest_ctx_by_vram(
                        variant, free_vram_gb,
                    )
                    if suggested is not None:
                        logger.info(
                            "preset %s 显存自适应 ctx: %d（free=%.1fGB）",
                            preset_name, suggested, free_vram_gb,
                        )
                        return suggested
            except Exception as e:
                logger.debug("preset %s 显存自适应 ctx 计算失败: %s", preset_name, e)
            return variant.ctx.recommended
        return None

    def _estimate_variant_vram_bytes(self, preset_name: str) -> int:
        """v0.3.3+ 估算 variant 加载后的显存占用（字节）。

        用于向平台 core 上报显存状态（report_loaded）。

        估算策略：
            1. 优先用 manifest 的 vram_required 字段（如 "48GB"）
            2. 否则用文件大小 + 2GB overhead（ctx 缓存等）

        Args:
            preset_name: llama-server preset 名

        Returns:
            int: 估算的显存字节数
        """
        variant = self._find_variant_by_preset_name(preset_name)
        if variant is None:
            # 未知 variant，保守估算 4GB
            return 4 * 1024 ** 3

        # 1. 优先用 manifest 的 vram_required
        vram_required = (variant.vram_required or "").strip()
        if vram_required:
            parsed = _parse_vram_str_to_bytes(vram_required)
            if parsed > 0:
                return parsed

        # 2. 文件大小 + 2GB overhead
        file_size = variant.file_size or 0
        if file_size > 0:
            return file_size + 2 * 1024 ** 3

        # 3. 兜底 4GB
        return 4 * 1024 ** 3
