"""llama-server 子进程生命周期管理（v0.3.0 router mode 重构版 + v0.3.1 显存回收增强）。

职责：
    1. 启动 llama-server 子进程（router mode：--models-dir + --models-max）
    2. 字节流透传 chat 请求到 llama-server（直接转发 OpenAI SSE 流，不做格式转换）
    3. router mode 模型管理：load_model / unload_model / list_loaded_models
    4. 停止子进程（terminate → kill → 进程树清理 兜底）
    5. 健康检查（轮询 /health）

v0.3.0 关键变更：
    - 启动命令从单模型 `--model <path>` 改为 router mode `--models-dir <dir>`
    - 新增 `load_model(model_name)` / `unload_model(model_name)` 方法（调 llama-server API）
    - `forward_chat` 改为字节流透传（删除 SSE 格式转换）
    - 删除单 variant 专用逻辑（current_variant / current_params）
    - 新增 `loaded_models` 属性（router mode 多模型列表）

v0.3.1 显存回收增强：
    - Windows: 用 Job Object 绑定子进程生命周期，父进程死时 OS 自动 kill 子进程
    - 停止时调用 taskkill /T /F 杀整个进程树（防止孤儿进程残留显存）
    - Linux/Mac: start_new_session=True + os.killpg 已足够

关键设计：
    - subprocess.Popen(start_new_session=True) 创建新进程组（Linux/Mac）
    - Windows: creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP + Job Object
    - stdout/stderr 重定向到 logs/engines/llama_cpp_server/llama_cpp_server.log（v0.2.15+ 统一目录）
    - 隐藏端口范围 8100-8200，自动扫描空闲端口
"""

from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Optional

from aireliable_engine import decode_subprocess_output
from aireliable_engine.device import probe as _sdk_probe

from .config import EngineConfig

logger = logging.getLogger(__name__)


def _parse_vram_str_to_bytes(vram_str: str) -> int:
    """解析显存字符串为字节数（v0.3.6+ 新增，与 engine.py 保持一致）。

    支持格式："48GB" / "2048MB" / "1024"（字节）

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


class LlamaProcessManager:
    """llama-server 子进程管理器（v0.3.0 router mode 版）。"""

    def __init__(self, config: EngineConfig):
        """
        Args:
            config: 引擎配置
        """
        self.config = config

        # 子进程状态
        self._process: Optional[subprocess.Popen] = None
        self._internal_port: Optional[int] = None
        self._log_file_handle = None
        self._start_time: Optional[float] = None
        # router mode 已加载模型列表（由 llama-server 管理，本地缓存镜像）
        self._loaded_models: set[str] = set()
        # v0.3.1+ Windows Job Object 句柄（绑定子进程生命周期，父进程死时自动 kill 子进程）
        self._job_handle: Optional[Any] = None
        # v0.3.6+ 最近一次生成的 planned_ctx（variant → 生效 ctx，override > 显存自适应 > recommended）
        # 由 _generate_models_preset_ini 填充，引擎启动成功后上报 core 供前端显示
        self._planned_ctx: dict[str, int] = {}

        # ============ llama-server 日志转发（v1.0）============
        # 将 logs/engines/llama_cpp_server/llama_cpp_server.log 内容 tail 到
        # SDK logger → JSON stdout → Core LogManager → SSE → 前端「运行日志」终端
        # 链路完全复用现有架构，Core/前端 0 改动；保持 Popen 原文件写入模式合规（§6.7）
        self._stop_tail = threading.Event()          # tail 线程停止信号
        self._tail_thread: Optional[threading.Thread] = None
        self._tail_crash_count: int = 0              # 崩溃重试计数（最大 3 次，避免死循环）
        self._log_file_tail_path: Optional[Path] = None

    # ============ llama-server 日志转发工具方法（v1.1：修复 timings 误杀 + 白名单 TOKEN/S 高亮）============

    # v1.1 关键字分类（🔴 2026-08-30 用户反馈"看不到思考速度 token/s"的修复）：
    #   TIMING_PATTERNS（白名单）：eval tokens/s / prompt eval tokens/s / timings 行
    #       → 强制 INFO 级 + 加 TOKEN/S 醒目前缀（用户最想看的指标，核心诉求！）
    #   IMPORTANT_PATTERNS（白名单）：HTTP access / model loaded / error / warn / slot 分配
    #       → 保留 INFO 级（前端可见）
    #   NOISY_PATTERNS（黑名单）：ggml_/cuBLAS 等底层实现细节（用户不关心）
    #       → 降级 DEBUG 级（仅文件级有，前端终端不可见）
    # 匹配顺序（精确控制）：先匹配 TIMING → 命中则 INFO+📊 并 return；否则 IMPORTANT
    #   → INFO；否则 NOISY → DEBUG；否则 fallback INFO。避免"先命中黑名单再白名单"漏匹配。
    _TIMING_PATTERNS: tuple[str, ...] = (
        # ===== v1.2 新增：b10034+ 实时速度行（推理进行中每 ~3s 一条，用户"思考过程速度"核心诉求）=====
        #   格式：`I slot print_timing: id  3 | task 0 | n_decoded =  100, tg =  26.25 t/s, tg_3s =  26.25 t/s`
        "print_timing",           # 同时覆盖 print_timing（单数，实时行）与 print_timings（复数，汇总行）
        " tg = ",                 # 实时 tg / tg_3s 速度字段
        " t/s",                   # 任何 xx.x t/s 速度值（新 build 简写，等价旧版 tokens/s）
        # ===== v1.1 原有：汇总 timings 行（推理结束统计）=====
        "timings =",             # llama.cpp 标准 timings 行（所有模型通用）
        "timings:",              # 部分 build 的变体写法
        "eval time",             # eval_time ms 变体
        "prompt eval time",      # prompt_eval_time 变体
        "eval_time",             # 兼容 key=value 写法
        "prompt_eval_time",
        "eval tokens per",       # "eval tokens per second"
        "prompt eval tokens per",
        "tokens/s",              # 任何 tokens/s 指标行
    )

    @staticmethod
    def _flush_pending_lines(
        pending_lines: list[str],
        noisy_patterns: tuple[str, ...],
        engine_logger: logging.Logger,
    ) -> None:
        """G1 v1.1 三级分类 + G3 200ms 攒批刷新。

        行分类流程（从高到低匹配）：
          1) TIMING_PATTERNS（白名单，强制 INFO + TOKEN/S 醒目前缀）
              → eval tokens/s、prompt eval tokens/s、timings 汇总行
              → 用户最想看的推理速度指标（核心诉求）
          2) 其余行：命中 NOISY_PATTERNS → logger.debug（仅文件级，前端不可见）
              未命中 → logger.info（进 JSON stdout → SSE → 前端终端可见）

        批量行若同时含 TIMING + 普通行，则拆分：TIMING 单独一批 TOKEN/S 前缀，
        普通 INFO/DEBUG 按原策略走，避免前缀混淆。

        Args:
            pending_lines: 待输出行列表（本方法清空）
            noisy_patterns: 视为噪音的关键字元组（NOISY_PATTERNS，黑名单）
            engine_logger: 当前引擎的 SDK logger（模块级 logger）
        """
        timing_lines: list[str] = []          # TOKEN/S 白名单
        info_lines: list[str] = []            # 普通 INFO（前端可见）
        debug_lines: list[str] = []           # 黑名单 DEBUG（仅文件级）
        TIMING = LlamaProcessManager._TIMING_PATTERNS

        for ln in pending_lines:
            if any(p in ln for p in TIMING):
                timing_lines.append(ln)
            elif any(p in ln for p in noisy_patterns):
                debug_lines.append(ln)
            else:
                info_lines.append(ln)

        # TOKEN/S 指标行（核心诉求，强制 INFO + 醒目前缀）——用户一眼能找到
        if timing_lines:
            prefix = (
                "[llama-server] [TOKEN/S]"
                if len(timing_lines) == 1
                else f"[llama-server] [TOKEN/S (batch {len(timing_lines)})]"
            )
            engine_logger.info("%s %s", prefix, "\n".join(timing_lines))
        # 普通 INFO
        if info_lines:
            prefix = (
                "[llama-server]"
                if len(info_lines) == 1
                else f"[llama-server (batch {len(info_lines)})]"
            )
            engine_logger.info("%s %s", prefix, "\n".join(info_lines))
        # 黑名单 DEBUG（前端不可见，仅 SDK 文件级日志有）
        if debug_lines:
            engine_logger.debug(
                "[llama-server DEBUG (batch %d)] %s",
                len(debug_lines), "\n".join(debug_lines),
            )
        pending_lines.clear()

    # ============ llama-server 日志转发主方法（v1.0）============

    def _tail_llama_server_log_file(self) -> None:
        """后台守护线程：tail llama_cpp_server.log → 攒批 → SDK logger。

        设计（对齐分析报告 §3.2）：
          • 只读不写文件 → 0 反压风险；llama-server 直接写 OS 文件，消费慢仅延迟不阻塞推理
          • seek(0, 2) 跳到文件末尾 → 从"本次启动后新内容"开始 tail，不转发之前历史
          • errors="replace" → 防止 llama-server 偶尔非 UTF-8 字节导致 decode 崩溃
          • 50ms × EOF 轮询 = 20 次/秒 CPU 消耗可忽略
          • G3 200ms × 最多 50 行攒批刷新 → 减少 logger 调用 & JSON stdout 解析压力
          • G1 NOISY_PATTERNS → 行分流 INFO / DEBUG（噪音行仅文件级 DEBUG，前端不可见）
          • 异常 → 最大 3 次崩溃重试（指数退避 2s/4s/6s），_tail_crash_count 超限则死
          • _stop_tail 置位 or 进程退出 poll() not None → 退出
        """
        log = logger  # 复用模块级 SDK logger（已配置 JSON stdout + RotatingFileHandler）
        # v1.1 黑名单（🔴 已剔除 TIMING 相关！eval/prompt_eval/timings 被提升到白名单 TOKEN/S）
        # 只保留真正"用户不关心"的 CUDA/ggml 底层细节、KV cache 内部状态行
        NOISY_PATTERNS: tuple[str, ...] = (
            "ggml_", "cuBLAS:", "cuBLAS ", "CLBlast:",
            "llama_model_load_internal", "n_ctx_train", "n_vocab",
            "rope_freq", "rope_scale", "mapped.",
            "slot_process", "slot id=", "slot state change:",
            "decode_time",                          # 内部 decode 毫秒非指标（非 eval_time）
            "loading model ", "model_load", "kv_cache",
            "compute: ", "paged_cache:",
            "eval(",                                # "eval(N tokens)" 这种逐批次计数行，非最终指标
        )

        # 1. 打开文件（r，UTF-8 容错）
        try:
            fh = open(
                self._log_file_tail_path, "r",
                encoding="utf-8", errors="replace",
            )
        except Exception as e:
            log.error(
                "[llama-server tail] 无法打开日志文件 %s: %s",
                self._log_file_tail_path, e,
            )
            return
        # 2. 跳到文件末尾（本次启动清空后的 EOF）
        #    _cleanup_resources 会重新 open("w") 清空，我们在 start_router_mode
        #    中于 Popen 之前就已经设置 _log_file_tail_path 并 seek 到末尾，
        #    所以这里天然从"本次启动的新日志"开始读，不会重复转发上一次启动历史
        try:
            fh.seek(0, 2)
        except Exception as e:
            log.warning("[llama-server tail] seek(END) 失败: %s（从文件开头继续读）", e)

        pending_lines: list[str] = []
        last_flush = time.monotonic()
        FLUSH_INTERVAL = 0.2       # G3 200ms
        BATCH_MAX = 50             # G3 最多攒 50 行一次 flush
        POLL_INTERVAL = 0.05       # EOF 时 50ms 轮询

        try:
            while not self._stop_tail.is_set():
                line = fh.readline()
                if not line:
                    # EOF：要么 llama-server 还没写完，要么进程已退出
                    now = time.monotonic()
                    need_flush = bool(pending_lines) and (
                        now - last_flush >= FLUSH_INTERVAL
                        or len(pending_lines) >= BATCH_MAX
                    )
                    if need_flush:
                        LlamaProcessManager._flush_pending_lines(
                            pending_lines, NOISY_PATTERNS, log,
                        )
                        last_flush = now

                    # 进程已退出且文件里最后一点都 flush 完了 → break
                    if (
                        self._process is not None
                        and self._process.poll() is not None
                        and not pending_lines
                    ):
                        break

                    time.sleep(POLL_INTERVAL)
                    continue

                # 读到行
                line_clean = line.rstrip("\r\n")
                if not line_clean:
                    continue
                pending_lines.append(line_clean)

                # 达到批量阈值 → 立即 flush（避免爆发时堆积过多）
                if len(pending_lines) >= BATCH_MAX:
                    LlamaProcessManager._flush_pending_lines(
                        pending_lines, NOISY_PATTERNS, log,
                    )
                    last_flush = time.monotonic()

            # 退出循环前 flush 剩余行
            if pending_lines:
                LlamaProcessManager._flush_pending_lines(
                    pending_lines, NOISY_PATTERNS, log,
                )

        except Exception as e:
            self._tail_crash_count += 1
            log.error(
                "[llama-server tail] 崩溃 (count=%d/3): %s",
                self._tail_crash_count, e,
            )
            if self._tail_crash_count <= 3:
                backoff = self._tail_crash_count * 2
                log.info(
                    "[llama-server tail] %ds 后自动重启（index=%d/3）...",
                    backoff, self._tail_crash_count,
                )
                time.sleep(backoff)
                if not self._stop_tail.is_set():
                    # 递归重启（深度 ≤ 3）
                    self._tail_llama_server_log_file()
        finally:
            try:
                fh.close()
            except Exception:
                pass
            # 用户级 INFO 行：在前端终端能看到 tail 已退出（作为进程停止的可观测性事件）
            log.info("[llama-server tail] 线程已退出（转发停止）")

    # ============ 状态属性 ============

    @property
    def is_running(self) -> bool:
        """子进程是否在运行。"""
        return self._process is not None and self._process.poll() is None

    @property
    def internal_port(self) -> Optional[int]:
        """当前隐藏端口。"""
        return self._internal_port

    @property
    def internal_url(self) -> Optional[str]:
        """llama-server 内部 URL。"""
        if self._internal_port is None:
            return None
        return f"http://127.0.0.1:{self._internal_port}"

    @property
    def pid(self) -> Optional[int]:
        """子进程 PID。"""
        if self._process is None:
            return None
        return self._process.pid

    @property
    def loaded_models(self) -> list[str]:
        """已加载的模型列表（router mode 多模型）。"""
        return sorted(self._loaded_models)

    # ============ 启动（router mode）============

    async def start_router_mode(self, variants: Optional[list] = None) -> dict:
        """启动 llama-server 子进程（router mode）。

        router mode 启动命令（v0.3.2+ 改用 --models-preset INI 文件）：
            llama-server --models-preset <ini_path> --host 127.0.0.1 --port <internal>
                         --models-max N --models-autoload
                         -c <ctx> --cache-type-k <k> --cache-type-v <v>
                         --cors-origins *

        v0.3.2 关键变更：
            - 从 --models-dir 改为 --models-preset INI 文件
            - INI 文件中为多模态 variant 声明 model + mmproj 配对
            - 解决 --models-dir 模式下 llama-server 不关联 mmproj 导致多模态失效的问题
            - 移除 --tools all（触发 UI AgenticStore bug，导致聊天崩溃）

        Args:
            variants: 可用 variant 列表（含 mmproj_path 信息）；None 则用 --models-dir 兜底

        Returns:
            dict: {
                "internal_port": 81xx,
                "pid": int,
                "load_time": float,
                "models_dir": str,
                "models_max": int,
                "preset_ini": str（v0.3.2+ 新增），
            }

        Raises:
            FileNotFoundError: llama-server 二进制不存在 / models_dir 不存在
            RuntimeError: 启动失败 / 超时
        """
        if self.is_running:
            raise RuntimeError(
                f"已有 llama-server 子进程在运行（pid={self.pid}），请先 stop()"
            )

        # 检查 llama-server 二进制
        server_bin = self.config.llama_server_path
        if not server_bin.exists():
            raise FileNotFoundError(
                f"llama-server 二进制不存在: {server_bin} "
                f"（请检查 LLAMA_BINS_DIR={self.config.llama_bins_dir}）"
            )

        # 检查 models_dir
        models_dir = self.config.checkpoint_dir
        if not models_dir.exists():
            raise FileNotFoundError(
                f"模型目录不存在: {models_dir}（请检查 AIRELIABLE_MODELS_ROOT + "
                f"AIRELIABLE_MODEL_LAYOUT_BASE 环境变量）"
            )

        # 分配隐藏端口
        internal_port = allocate_internal_port(
            self.config.internal_port_start,
            self.config.internal_port_end,
        )
        logger.info("分配隐藏端口: %d", internal_port)

        # v0.3.2+ 生成 models-preset INI 文件（含 mmproj 关联）
        preset_ini_path: Optional[Path] = None
        if variants:
            try:
                # v0.3.6+ 查询一次 GPU 可用显存，用于显存自适应推荐 ctx
                free_vram_gb = self._get_free_vram_gb()
                preset_ini_path = self._generate_models_preset_ini(
                    variants, free_vram_gb=free_vram_gb,
                )
                logger.info("已生成 models-preset INI: %s", preset_ini_path)
            except Exception as e:
                logger.warning("生成 models-preset INI 失败，回退到 --models-dir: %s", e)
                preset_ini_path = None

        # 构造 router mode 启动命令
        full_cmd = self._build_router_mode_command(
            internal_port, models_dir, preset_ini_path
        )
        logger.info("启动 llama-server (router mode): %s", " ".join(full_cmd))

        # 打开日志文件（v0.2.15+ 单独分目录到 logs/engines/llama_cpp_server/）
        server_log_dir = self.config.server_log_dir
        server_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = server_log_dir / "llama_cpp_server.log"
        self._log_file_handle = open(log_file, "w", encoding="utf-8")

        # ===== llama-server 日志转发 v1.0：Popen 前启动 tail 守护线程 =====
        # 时机：open("w") 刚清空文件 → 设置路径 + clear stop + seek(0,2)
        # → llama-server 启动后的第一行新日志都能被 tail 到（不重复、不遗漏）
        self._log_file_tail_path = log_file
        self._stop_tail.clear()
        self._tail_crash_count = 0
        self._tail_thread = threading.Thread(
            target=self._tail_llama_server_log_file,
            name=(
                "LlamaServerLogTail-"
                f"{internal_port if internal_port else 'prep'}"
            ),
            daemon=True,  # 兜底：引擎 Python 进程意外挂掉时 OS 自动回收线程
        )
        self._tail_thread.start()
        logger.info(
            "llama-server 日志 tail 线程已启动（转发到 SDK logger → 运行日志终端）: "
            "file=%s, port=%s", log_file, internal_port,
        )

        # 启动子进程
        start_time = time.time()
        try:
            self._process = subprocess.Popen(
                full_cmd,
                stdout=self._log_file_handle,
                stderr=subprocess.STDOUT,                # stderr 合并到 stdout
                stdin=subprocess.DEVNULL,
                # Windows 不弹窗 + 创建新进程组
                creationflags=_get_creation_flags(),
                start_new_session=(sys.platform != "win32"),  # Linux/Mac 新会话
                close_fds=True,
            )
            self._internal_port = internal_port
            self._start_time = start_time
            self._loaded_models.clear()

            # v0.3.1+ Windows: 绑定 Job Object，父进程死时 OS 自动 kill 子进程
            # 防止平台异常退出时 llama-server 变孤儿继续占用显存
            self._assign_to_job_object(self._process)

            logger.info(
                "llama-server 子进程启动 (router mode): pid=%d, port=%d, models_dir=%s",
                self._process.pid, internal_port, models_dir,
            )

        except Exception as e:
            self._cleanup_resources()
            raise RuntimeError(f"启动 llama-server 失败: {e}") from e

        # 等待健康检查通过
        try:
            await wait_for_ready(
                process=self._process,
                internal_port=internal_port,
                timeout=self.config.server_timeout,
            )
        except Exception as e:
            # 健康检查失败，停止子进程
            await self.stop()
            raise RuntimeError(f"llama-server 健康检查失败: {e}") from e

        load_time = time.time() - start_time
        logger.info("llama-server 启动完成 (router mode): 耗时 %.2fs", load_time)

        return {
            "internal_port": internal_port,
            "pid": self._process.pid,
            "load_time": load_time,
            "models_dir": str(models_dir),
            "models_max": self.config.models_max,
            "preset_ini": str(preset_ini_path) if preset_ini_path else None,
        }

    def _build_router_mode_command(
        self,
        internal_port: int,
        models_dir: Path,
        preset_ini_path: Optional[Path] = None,
    ) -> list[str]:
        """构造 router mode 启动命令。

        v0.3.2+ 变更：优先使用 --models-preset INI 文件（支持 mmproj 关联），
        fallback 到 --models-dir（无 mmproj 关联）。

        Args:
            internal_port: 内部端口
            models_dir: 模型目录（fallback 用）
            preset_ini_path: models-preset INI 文件路径（v0.3.2+ 新增）

        Returns:
            list[str]: 完整命令行
        """
        server_bin = str(self.config.llama_server_path)
        cmd: list[str] = [
            server_bin,
        ]

        # v0.3.2+ 优先用 --models-preset（支持 mmproj 关联，多模态必需）
        if preset_ini_path and preset_ini_path.exists():
            cmd.extend(["--models-preset", str(preset_ini_path)])
        else:
            # fallback: --models-dir（不关联 mmproj，多模态不可用）
            cmd.extend(["--models-dir", str(models_dir)])

        cmd.extend([
            "--host", "127.0.0.1",
            "--port", str(internal_port),
            "--models-max", str(self.config.models_max),
            # v1.0+ parallel_slots（全局 router CLI，对齐 bat --parallel 1，
            #   影响 n_slots / KV 预分配槽数；默认 1 避免 n_slots=4 触发 KV ×4 预占显存）
            "--parallel", str(getattr(self.config, "parallel_slots", 1)),
        ])
        # v0.3.6+ 修复：不再传全局 --cache-type-k / --cache-type-v
        # 原因 1：同 ctx-size 的 server-models.cpp preset.merge(base_preset) 语义
        #   —— router 启动命令 CLI 全局 cache-type 会覆盖每个 INI model section
        #   里按 ctx 阈值策略写的 per-variant cache-type（见 T1.3 阈值逻辑），
        #   导致我们之后写 q8_0 也会被 CLI q4_0 强制盖回去（v0.3.6 ctx 的同款 BUG）。
        # 原因 2：cache-type 属于"每模型 KV 精度"（就像 ctx-size），
        #   应该在 INI 每个 section 差异化写入（ctx≤262k 用 q8_0，>262k 用 q4_0，
        #   以及用户显式 env 设置的 LLAMA_CACHE_TYPE_K/V）。
        #   全局 CLI 不应再插手 cache-type 的覆盖行为。
        #（旧代码：--cache-type-k config.cache_type_k；--cache-type-v config.cache_type_v）
        logger.info(
            "router CLI 不再传全局 --cache-type-k/v（避免覆盖 INI per-model 阈值策略；"
            "已移到 _generate_models_preset_ini 按 variant 差异化写入）"
        )
        # v0.3.6+ 修复：不再传全局 -c（ctx-size）
        # 原因：llama-server router mode 的 load_models() 会执行
        #   preset.merge(base_preset) —— 即 router 启动命令的 CLI 参数
        #   覆盖每个 model preset 的同名配置（llama.cpp tools/server/server-models.cpp）。
        # 之前传 -c 8192 导致 preset INI 里 per-variant 的 ctx-size=32768
        # 被全局 8192 覆盖，模型实际只以 8192 ctx 加载，
        # 大请求（如 >8192 tokens）被 llama-server 以 400 拒绝：
        #   "request (N tokens) exceeds the available context size (8192 tokens)"
        # 去掉 -c 后，每个模型使用 preset INI 中自己的 ctx-size（
        # 由 _generate_models_preset_ini 按 override/显存自适应/recommended 写入）。

        # 自动加载开关
        if self.config.models_autoload:
            cmd.append("--models-autoload")
        else:
            cmd.append("--no-models-autoload")

        # v0.3.5+ 设备策略（平台注入 AIRELIABLE_DEVICE_POLICY）：
        #   gpu     → -ngl all --fit off：全部层上 GPU，显存不足直接报错
        #   gpu+cpu → -ngl auto --fit on --fit-target 512 --fit-ctx 4096
        #             （新版 llama.cpp 用户要求的"GPU不够自动移CPU"原生机制）：
        #               --fit on            ：自动按实际显存调 offload 层数（溢出层放 CPU）
        #               --fit-target 512    ：每 GPU 预留 512MiB 安全余量（避免算太紧贴OOM边界）
        #               --fit-ctx 4096      ：fit 自动降 ctx 时的下限（不降到 < 4096 才能正常聊天）
        #             （--fit-target / --fit-ctx 是新版新增参数，help 已实测支持）
        #   cpu     → -ngl 0 --fit off：纯 CPU 推理
        #   auto    → 不干预（llama-server 默认即 ngl=auto + fit=on）
        # 注意：router mode 下 CLI 全局参数会 merge 覆盖每个 model preset 的
        # 同名配置（见上方 v0.3.6 注释），因此此处全局 -ngl 对所有模型生效。
        policy = getattr(self.config, "device_policy", "gpu")
        if policy == "gpu":
            cmd.extend(["-ngl", "all", "--fit", "off"])
            logger.info("设备策略 gpu：-ngl all --fit off（强制全 GPU，显存不足将报错；对齐 bat --n-gpu-layers 999）")
        elif policy == "gpu+cpu":
            cmd.extend([
                "-ngl", "auto",
                "--fit", "on",
                "--fit-target", "512",
                "--fit-ctx", "4096",
            ])
            logger.info(
                "设备策略 gpu+cpu：-ngl auto --fit on --fit-target 512 --fit-ctx 4096"
                "（新版 llama.cpp 原生 GPU 不够自动移 CPU + 留 512MiB 安全余量 + ctx ≥ 4096）"
            )
        elif policy == "cpu":
            cmd.extend(["-ngl", "0", "--fit", "off"])
            logger.info("设备策略 cpu：-ngl 0 --fit off（纯 CPU 推理）")

        # v0.3.2+ 修复：不再启用 --tools all
        # 原因：--tools all 会触发 llama-server UI 的 AgenticStore，但 UI bundle 在
        # 解析 chat completions 响应时存在 bug（Cannot read properties of undefined
        # reading 'length'），导致第一条消息就崩溃。
        # 不启用 --tools 时 /tools 端点返回 403，但这只是 F12 警告，不影响聊天功能。
        # 优先保证聊天功能正常。

        # 显式设置 CORS 允许所有源（独立参数，不依赖 --tools）
        cmd.extend(["--cors-origins", "*"])

        # v1.0+：启用 --cont-batching（对齐 start_llama_server.bat；单请求场景 overhead≈0，
        #   多请求/新任务能减少 slot 切换开销，并可能改变 KV 预留算法避免硬按 n_slots×全ctx 预占）
        # 新版 llama-server：-cb/--cont-batching 默认状态（查 help：-nocb/--no-cont-batching），
        # 显式传防止 LLAMA_ARG_CONT_BATCHING env 覆盖关闭
        cmd.append("--cont-batching")
        logger.info("router CLI：--cont-batching 已启用（对齐 bat）")

        # ===== KV 前缀复用（v1.1.2+，方向 1：根治 agent 多轮长上下文重复计算）=====
        # 🔴 2026-09-10 事故：Hermes agent 60 轮迭代把上下文滚到 190K tokens，
        #   每轮全量重发 → 单次 prompt processing 9-10 分钟（progress 0.66→0.94 实测）。
        # 机制（llama-server 0.3.0-dev --help 已确认）：
        #   --cache-prompt    ：prompt 缓存开关（默认 enabled；显式传防
        #                       LLAMA_ARG_CACHE_PROMPT 环境变量关闭）
        #   --cache-reuse 256 ：KV shifting 最小复用块（默认 0=关闭）。即使新请求与
        #                       缓存前缀不完全一致（agent 迭代可能微调中间轮次），
        #                       只要 ≥256 tokens 前缀命中就从缓存恢复，只算新增部分。
        #   --cache-ram 8192  ：host 侧缓存上限（默认 8192MiB，190K token KV 足够，不传）
        # 注意：cache 由 llama-server 自管（GPU KV + host cache），空闲 slot 默认
        #   保存到 prompt cache（--cache-idle-slots 默认开启）。
        cmd.extend(["--cache-prompt", "--cache-reuse", "256"])
        logger.info(
            "router CLI：--cache-prompt + --cache-reuse 256（KV 前缀复用；"
            "agent 多轮长上下文只算增量，避免 190K prompt 每轮重算）"
        )

        # ===== llama-server 日志转发 v1.2 G2 源头收敛（FIX：v1.1 的 2 是 WARNING 级，掐死 timings）=====
        # 🔴 2026-08-30 用户反馈"推理时看不到 token/s"根因（v1.2 实测 --help 修正）：
        #   llama-server b10034+ 的 --log-verbosity 语义与旧版相反——数值越小越啰嗦：
        #     0: generic | 1: error | 2: warning | 3: info（默认）| 4: trace | 5: debug
        #   v1.1 误设为 2（=WARNING 级），比默认 3 更安静 →
        #   llama-server 原生 eval tokens/s / prompt eval tokens/s（INFO 级）被源头截流，
        #   llama_cpp_server.log 在 ready 后 0 条 timings 行 → 前端终端永远看不到思考速度！
        #   v1.2 修复：改为 3（INFO 级，与默认一致，显式声明防止外部环境变量 LLAMA_ARG_LOG_VERBOSITY 干扰）。
        #   配合 G1 文件层 TIMING_PATTERNS 白名单（强制 INFO 且加 TOKEN/S 前缀）双保险：
        #     ✅ 源头 CLI 保留所有 INFO 行（含 timings）
        #     ✅ 文件级 tail 强制 timings 行 INFO + 醒目前缀
        #     ✅ 其余 ggml_/slot_ 等 23 种噪音模式仍降级 DEBUG（不影响终端干净度）
        # ⚠️ 注意：绝不启用 `--log-prompts-dir`（§6.9 禁止落盘敏感 prompt；S1 --help 已发现该参数）
        cmd.extend(["--log-verbosity", "3"])
        logger.info(
            "llama-server 启动参数 G2(v1.2 修复)：--log-verbosity 3（INFO 级；"
            "v1.1 的 2 实为 WARNING 级导致 timings/token/s 被源头截流，已修正）"
        )

        return cmd

    def _generate_models_preset_ini(
        self, variants: list, free_vram_gb: Optional[float] = None,
    ) -> Path:
        """v0.3.2+ 生成 llama-server models-preset INI 文件。

        v0.3.3+ 变更：使用 per-variant ctx（来自 manifest/GGUF/用户覆盖），不再用全局 ctx。

        v0.3.6+ 变更：新增 free_vram_gb 参数，无用户 override 时按显存自适应推荐 ctx。

        INI 格式（每个 variant 一个 section）：
            version = 1

            [variant-name]
            model = /path/to/main.gguf
            mmproj = /path/to/mmproj.gguf        # 仅多模态 variant 写此行
            ctx-size = 32768                      # v0.3.3+ per-variant
            cache-type-k = q4_0
            cache-type-v = q4_0
            tools = all

        per-variant ctx 优先级（高 → 低）：
            1. config.variant_ctx_overrides[variant_name]（用户前端 UI 滑块覆盖）
            2. 显存自适应推荐（v0.3.6+，依据可用显存 + 模型权重估算，仅当无 override）
            3. variant.ctx.recommended（来自 manifest 或 GGUF 元数据推导）
            4. config.ctx（全局默认）

        当使用 override 时，会校验是否在 [variant.ctx.min, variant.ctx.max] 范围内，
        超出范围时回退到 recommended 并记录警告。

        Args:
            variants: VariantInfo 列表（含 path / mmproj_path / supports_mmproj / ctx）
            free_vram_gb: GPU 当前可用显存（GB）；None 表示无法检测（跳过显存自适应）


        Returns:
            Path: INI 文件路径（写入 data 目录）

        Raises:
            RuntimeError: 生成失败
        """
        import configparser

        config_parser = configparser.ConfigParser(interpolation=None)

        # version 必须为 1（llama-server 要求）
        config_parser["DEFAULT"] = {"version": "1"}

        models_dir = self.config.checkpoint_dir
        global_ctx = self.config.ctx
        # v1.0+ cache_type 默认策略（对齐 bat L110-116）：
        #   - 如果用户显式覆盖（env LLAMA_CACHE_TYPE_K/V 或 CLI --cache-type-k/v），
        #     判定方法：与 DEFAULT_CACHE_TYPE_K/V(="q4_0") 不同 → 视为显式覆盖 → 直接用用户值；
        #   - 否则按 ctx 阈值分支：ctx ≤ 262144 → q8_0；ctx > 262144 → q4_0
        #     （bat 相同逻辑："ctx≤262k 省反量化开销；ctx>262k 怕显存炸强制 q4_0"）
        # 注意：router CLI 全局 cache-type 已经被删掉（见 _build_router_mode_command），
        # 避免 CLI 覆盖 INI 的 per-model 差异化值。
        from .config import (  # 延迟导入避免循环；同级包相对导入（父级..config → engines.config 不存在，会报 No module named 'engines.config'）
            DEFAULT_CACHE_TYPE_K as _DEF_CK,
            DEFAULT_CACHE_TYPE_V as _DEF_CV,
        )
        cache_k_user = self.config.cache_type_k
        cache_v_user = self.config.cache_type_v
        _user_override_k = cache_k_user != _DEF_CK
        _user_override_v = cache_v_user != _DEF_CV
        # 阈值常量：与 bat 对齐（262144 tokens）
        CTX_Q8_THRESHOLD = 262144
        overrides = self.config.variant_ctx_overrides or {}

        multi_modal_count = 0
        text_only_count = 0
        ctx_log_lines: list[str] = []
        # 收集：哪些 variant 用了阈值 q8_0（最后 INFO 汇总打日志）
        _q8_count = 0
        _q4_count = 0
        _explicit_count = 0

        for v in variants:
            # variant 名称（作为 section 名）
            section_name = v.name
            if not section_name:
                continue

            # 主模型路径
            if v.path is None:
                logger.warning("variant %s 无 path，跳过", section_name)
                continue

            # v0.3.5+ 修复：路径解析健壮性
            # v.path 可能是以下三种形式之一：
            #   1. 绝对路径（如 D:\...\xxx.gguf）
            #   2. 相对项目根目录的路径（如 models/llm/checkpoints/llama/xxx.gguf）
            #      — ModelScanner._resolve_gguf_path 尝试 3 返回的路径（models_root / rel_path）
            #      — 当 checkpoint_dir 是相对路径时，此路径相对 cwd 存在
            #   3. 相对 checkpoint_dir 的路径（如 xxx.gguf，仅文件名）
            #      — ModelScanner._resolve_gguf_path 尝试 2 返回的路径（checkpoint_dir / p.name）
            # 之前只检查 is_absolute() 会导致情况 2 被错误拼接到 models_dir，产生重复前缀：
            #   models/llm/checkpoints/llama + models/llm/checkpoints/llama/xxx.gguf
            #   → models/llm/checkpoints/llama/models/llm/checkpoints/llama/xxx.gguf（404）
            # v0.3.5+ 修复：先检查路径是否直接存在，不存在再拼接到 models_dir
            # 并且检查路径是否已经包含 models_dir 前缀，避免重复拼接
            main_path = Path(v.path)
            
            # 情况 1：绝对路径
            if main_path.is_absolute():
                if not main_path.exists():
                    logger.warning(
                        "variant %s 主模型文件不存在: %s",
                        section_name, main_path,
                    )
                    continue
            # 情况 2：相对路径
            else:
                # 尝试 1：直接相对于 cwd
                if main_path.exists():
                    pass  # 使用原路径
                # 尝试 2：检查路径是否已经以 models_dir 开头（避免重复拼接）
                else:
                    main_path_str = str(main_path).replace("\\", "/")
                    models_dir_str = str(models_dir).replace("\\", "/")
                    
                    if main_path_str.startswith(models_dir_str + "/"):
                        # 路径已经包含 models_dir 前缀，直接使用
                        pass
                    # 尝试 3：只取文件名，拼接到 models_dir
                    elif (models_dir / main_path.name).exists():
                        main_path = models_dir / main_path.name
                    # 尝试 4：完整路径拼接到 models_dir
                    elif (models_dir / main_path).exists():
                        main_path = models_dir / main_path
                    else:
                        logger.warning(
                            "variant %s 主模型文件不存在，跳过: %s",
                            section_name, main_path,
                        )
                        continue

            if not main_path.exists():
                logger.warning(
                    "variant %s 主模型文件不存在，跳过: %s",
                    section_name, main_path,
                )
                continue

            # v0.3.3+ 解析 per-variant ctx（v0.3.6+ 传入 free_vram_gb 支持显存自适应）
            chosen_ctx, ctx_source = self._resolve_variant_ctx(
                v, overrides, global_ctx, free_vram_gb=free_vram_gb,
            )
            # v0.3.6+ 记录生效 ctx（供引擎启动后上报 core 作为"引擎推荐"）
            self._planned_ctx[section_name] = chosen_ctx
            ctx_log_lines.append(f"  - {section_name}: ctx-size={chosen_ctx}（来源：{ctx_source}）")

            # v1.0+ cache-type 按 ctx 阈值策略（对齐 bat L110-116）：
            # 用户显式覆盖优先；否则 ctx≤262144→q8_0；ctx>262144→q4_0。
            # ⚠️ KV q8_0 V cache 某些后端需要 Flash-Attention（来自经验库 1217599）：
            #   若启动后 llama_cpp_server.log 出现 "quantized V cache was requested,
            #   but this requires Flash Attention" → 请把 LLAMA_CACHE_TYPE_V 设为 q4_0
            #   （或 ctx 档升到 > 262144 自动 q4_0）
            if _user_override_k or _user_override_v:
                effective_cache_k = cache_k_user
                effective_cache_v = cache_v_user
                _explicit_count += 1
            else:
                if chosen_ctx <= CTX_Q8_THRESHOLD:
                    effective_cache_k = "q8_0"
                    effective_cache_v = "q8_0"
                    _q8_count += 1
                else:
                    effective_cache_k = "q4_0"
                    effective_cache_v = "q4_0"
                    _q4_count += 1
            ctx_log_lines[-1] += f", cache-k={effective_cache_k}, cache-v={effective_cache_v}"

            section: dict[str, str] = {
                "model": str(main_path),
                "ctx-size": str(chosen_ctx),
                "cache-type-k": effective_cache_k,
                "cache-type-v": effective_cache_v,
            }
            # v1.0+：tools=all 仅多模态 variant 写（对齐 bat 不传 --tools；
            #   bat 纯文本场景不启用 experimental tools 标志位）
            #   多模态 tools 写在 supports_mmproj 分支里（见下文）。

            # 多模态：添加 mmproj
            mmproj_path = getattr(v, "mmproj_path", None)
            supports_mmproj = getattr(v, "supports_mmproj", False)
            if supports_mmproj and mmproj_path:
                mmproj_p = Path(mmproj_path)
                # v0.2.15+ 修复：与 main_path 相同的路径解析健壮性修复
                if not mmproj_p.exists() and not mmproj_p.is_absolute():
                    candidate = models_dir / mmproj_p
                    if candidate.exists():
                        mmproj_p = candidate
                if mmproj_p.exists():
                    section["mmproj"] = str(mmproj_p)
                    # v1.0+：tools=all 仅多模态 variant 开启（对齐 bat 不传 --tools，
                    #   纯文本不加载 experimental tools）。多模态必须 tools=all 才能
                    #   在多模态推理路径中让 UI/工作流调用非 builtin 功能。
                    section["tools"] = "all"
                    multi_modal_count += 1
                    logger.info(
                        "variant %s 配置多模态: mmproj=%s, ctx-size=%d",
                        section_name, mmproj_p.name, chosen_ctx,
                    )
                else:
                    logger.warning(
                        "variant %s 声明多模态但 mmproj 文件不存在: %s",
                        section_name, mmproj_p,
                    )
            else:
                text_only_count += 1

            config_parser[section_name] = section

        # 写入 INI 文件（data 目录下，v0.2.15+ 直接用 data_root）
        ini_dir = self.config.data_root
        ini_dir.mkdir(parents=True, exist_ok=True)
        ini_path = ini_dir / f"llama_cpp_models_preset_{self.config.engine_id}.ini"

        with open(ini_path, "w", encoding="utf-8") as f:
            config_parser.write(f)

        logger.info(
            "models-preset INI 生成完成: %s（多模态 %d 个，纯文本 %d 个）\n"
            "cache-type 策略汇总：显式覆盖 %d 个，q8_0(ctx≤%d) %d 个，q4_0(ctx>%d) %d 个\n"
            "per-variant ctx + cache 配置：\n%s",
            ini_path, multi_modal_count, text_only_count,
            _explicit_count, CTX_Q8_THRESHOLD, _q8_count, CTX_Q8_THRESHOLD, _q4_count,
            "\n".join(ctx_log_lines) if ctx_log_lines else "  （无 variant）",
        )

        return ini_path

    def _resolve_variant_ctx(
        self,
        variant: Any,
        overrides: dict[str, int],
        global_ctx: int,
        free_vram_gb: Optional[float] = None,
    ) -> tuple[int, str]:
        """v0.3.3+ 解析单个 variant 的 ctx-size。

        v0.3.6+ 新增 free_vram_gb：无用户 override 时，优先按显存自适应推荐。

        优先级（高 → 低）：
            1. overrides[variant.name]（用户前端 UI 滑块覆盖，需在 [min, max] 范围内）
            2. 显存自适应推荐（v0.3.6+，依据可用显存 + 模型权重估算，仅当 free_vram_gb 已知）
            3. variant.ctx.recommended（manifest 或 GGUF 元数据推导）
            4. global_ctx（全局默认）

        当 override 超出 [variant.ctx.min, variant.ctx.max] 范围时：
            - 回退到 variant.ctx.recommended
            - 记录 warning 日志

        Args:
            variant: VariantInfo 对象（含 name 和可选的 ctx: VariantCtxInfo）
            overrides: 用户覆盖字典 {variant_name: ctx_size}
            global_ctx: 全局默认 ctx（如 8192）
            free_vram_gb: GPU 当前可用显存（GB）；None 表示无法检测（跳过显存自适应）

        Returns:
            tuple: (chosen_ctx, source_description)
            source_description 取值：
                "override" | "vram-adaptive" | "manifest" | "gguf" | "global-default"
        """
        v_name = variant.name
        v_ctx = getattr(variant, "ctx", None)

        # 1. 用户覆盖（最高优先级）
        if v_name in overrides:
            override_val = overrides[v_name]
            # 校验 override 是否在 [min, max] 范围内
            if v_ctx is not None:
                if v_ctx.min <= override_val <= v_ctx.max:
                    return override_val, "override"
                logger.warning(
                    "variant %s 的 ctx override=%d 超出 manifest 范围 [%d, %d]，"
                    "回退到 recommended=%d",
                    v_name, override_val, v_ctx.min, v_ctx.max, v_ctx.recommended,
                )
                return v_ctx.recommended, "manifest"
            # 无 manifest ctx 约束时，直接使用 override
            return override_val, "override"

        # 2. 显存自适应推荐（v0.3.6+，仅当无 override 且显存可检测）
        if free_vram_gb and v_ctx is not None:
            suggested = self._suggest_ctx_by_vram(variant, free_vram_gb)
            if suggested is not None:
                return suggested, "vram-adaptive"

        # 3. variant.ctx.recommended（manifest 或 GGUF 推导）
        if v_ctx is not None:
            # 区分来源：manifest 写入的 vs GGUF 推导的（这里无法精确区分，统一标 "manifest/gguf"）
            return v_ctx.recommended, "manifest/gguf"

        # 4. 全局默认
        return global_ctx, "global-default"

    # ============ 显存自适应 ctx（v0.3.6+ 新增）============

    def _get_free_vram_gb(self) -> Optional[float]:
        """查询 GPU 当前可用显存（GB）。

        优先级：
            1. config.gpu_vram_gb（用户显式指定，视为可用显存）
            2. SDK 设备探测（有 core 走平台权威值，否则本地多源探测）

        Returns:
            float | None: 可用显存 GB；检测失败返回 None
        """
        # 用户显式指定（视为可用显存）
        if self.config.gpu_vram_gb and self.config.gpu_vram_gb > 0:
            return float(self.config.gpu_vram_gb)

        # 任务 71：nvidia-smi 直调改为 SDK 探测（本引擎无 torch，走系统层）
        try:
            raw = _sdk_probe(mode="auto")
            free_mb = raw.get("vram_free_mb")
            if free_mb is not None:
                return float(free_mb) / 1024.0
        except Exception as e:
            logger.debug("SDK 显存探测失败: %s", e)
        return None

    def _suggest_ctx_by_vram(
        self, variant: Any, free_vram_gb: float,
    ) -> Optional[int]:
        """v0.3.6+ 按可用显存估算 variant 的推荐 ctx-size。

        估算逻辑：
            1. 模型权重显存：优先 manifest 的 vram_required（如 "48GB"），
               否则用文件大小 × 1.05（mmap 场景权重接近文件大小）
            2. KV cache 显存预算 = 可用显存 - 权重 - 余量(2GB)
            3. 每 ctx-token 的 KV cache 大小按 cache 量化类型 + 并行槽数估算：
               - 基础每 token KV（q4_0）≈ 54KB（35B-A3B 级：2×48层×8kv头×128dim×0.5625B）
               - 乘 1.3 保守系数覆盖额外开销
               - 乘并行槽数（router 默认 4，与 llama-server n_parallel 对齐）
            4. ctx = KV 预算 / 每 ctx-token KV，clamp 到 [variant.ctx.min, variant.ctx.max]

        Args:
            variant: VariantInfo 对象（含 ctx / vram_required / file_size）
            free_vram_gb: GPU 当前可用显存（GB）

        Returns:
            int | None: 推荐 ctx-size；无法估算返回 None
        """
        v_ctx = getattr(variant, "ctx", None)
        if v_ctx is None or free_vram_gb <= 0:
            return None

        # 1. 模型权重显存估算
        # 优先用文件大小 × 1.05（mmap/offload 时权重显存 ≈ 文件大小）；
        # vram_required 是"模型 + KV cache"总需求（如 "48GB"），不能直接当权重，
        # 仅当文件大小未知时兜底用（按总需求 60% 粗估权重）。
        weights_gb: Optional[float] = None
        file_size = getattr(variant, "file_size", 0) or 0
        if file_size > 0:
            weights_gb = file_size / (1024 ** 3) * 1.05
        if not weights_gb:
            vram_str = getattr(variant, "vram_required", "") or ""
            if vram_str:
                vram_bytes = _parse_vram_str_to_bytes(vram_str)
                if vram_bytes > 0:
                    weights_gb = vram_bytes / (1024 ** 3) * 0.6
        if not weights_gb or weights_gb <= 0:
            return None

        # 2. KV cache 显存预算（余量 2GB）
        kv_budget_gb = free_vram_gb - weights_gb - 2.0
        if kv_budget_gb <= 0.5:
            return v_ctx.min

        # 3. 每 ctx-token 的 KV cache 大小
        # 基础每 token KV（q4_0，35B-A3B 级）：2 × n_layers × n_kv_heads × head_dim × 0.5625B
        base_kv_bytes_per_token = 54 * 1024  # 54KB/token（q4_0）
        # cache 类型修正：f16/f32 等比 q4_0 大
        cache_mult = {"q4_0": 1.0, "q4_1": 1.2, "q5_0": 1.1, "q5_1": 1.3,
                      "q8_0": 1.75, "f16": 4.0, "f32": 8.0}
        mult = cache_mult.get(self.config.cache_type_k, 1.0)
        # 并行槽数：llama-server 默认 n_parallel=4（KV cache 按槽数分配，与
        # models_max 并存模型数无关）。系数 1.8 覆盖计算缓冲 / mmproj / 系统开销。
        # llama-server 的 --fit 不调整显式设置的 ctx-size，估算必须保守以防 OOM。
        slots = 4
        kv_per_ctx_token_bytes = base_kv_bytes_per_token * mult * 1.8 * slots

        # 4. 计算并 clamp
        ctx_est = int(kv_budget_gb * (1024 ** 3) / kv_per_ctx_token_bytes)
        suggested = max(v_ctx.min, min(ctx_est, v_ctx.max))
        logger.debug(
            "显存自适应 ctx: variant=%s, free=%.1fGB, weights=%.1fGB, "
            "kv_budget=%.1fGB, kv/ctx-token=%.0fKB, suggested=%d",
            getattr(variant, "name", "?"), free_vram_gb, weights_gb,
            kv_budget_gb, kv_per_ctx_token_bytes / 1024, suggested,
        )
        return suggested

    # ============ 停止 ============

    async def stop(self) -> dict:
        """停止子进程（v0.3.1+ 增强进程树清理）。

        v0.3.1 改动：
            - Windows: 用 taskkill /T /F 杀整个进程树（防止孤儿孙进程残留显存）
            - Linux/Mac: 用 os.killpg 杀整个进程组
            - 保留 terminate → kill 兜底逻辑

        Returns:
            dict: {"state": "idle", "vram_freed": "..."}
        """
        if self._process is None:
            return {"state": "idle", "vram_freed": "0GB"}

        pid = self._process.pid
        logger.info("停止 llama-server 子进程: pid=%d", pid)

        # 估算释放的显存（router mode 多模型，简化为日志记录）
        vram_freed = "router-mode multi-model"

        # v0.3.1+ 优先用进程树清理（杀整个进程树，防止孤儿进程残留显存）
        try:
            await self._kill_process_tree(pid)
            # 等待进程退出
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                    timeout=3.0,
                )
                logger.info("llama-server 进程树已清理: pid=%d", pid)
            except asyncio.TimeoutError:
                logger.warning("进程树清理后 3s 内未退出，回退到 kill: pid=%d", pid)
                try:
                    self._process.kill()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("进程树清理失败，回退到 terminate/kill: %s", e)
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                        timeout=5.0,
                    )
                    logger.info("llama-server 已优雅退出: pid=%d", pid)
                except asyncio.TimeoutError:
                    logger.warning("llama-server 5s 内未退出，强制 kill: pid=%d", pid)
                    self._process.kill()
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._process.wait, 3.0,
                        )
                    except Exception:
                        pass
            except Exception as e2:
                logger.error("停止子进程异常: %s", e2)
                try:
                    self._process.kill()
                except Exception:
                    pass

        self._cleanup_resources()
        return {
            "state": "idle",
            "vram_freed": vram_freed,
        }

    async def _kill_process_tree(self, pid: int) -> None:
        """杀整个进程树（v0.3.1+ 新增）。

        Windows: taskkill /T /F /PID <pid>（/T 递归杀子进程，/F 强制）
        Linux/Mac: os.killpg(os.getpgid(pid), SIGKILL)（杀整个进程组）

        Args:
            pid: 根进程 PID
        """
        if sys.platform == "win32":
            # Windows: taskkill /T /F 杀进程树
            cmd = ["taskkill", "/T", "/F", "/PID", str(pid)]
            logger.info("Windows 进程树清理: %s", " ".join(cmd))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
                if proc.returncode == 0:
                    logger.info("进程树清理成功: pid=%d", pid)
                else:
                    # taskkill 返回非 0（如进程已退出），不视为错误
                    logger.debug(
                        "taskkill 返回 %d: stdout=%s, stderr=%s",
                        proc.returncode,
                        stdout.decode(errors="replace"),
                        stderr.decode(errors="replace"),
                    )
            except asyncio.TimeoutError:
                logger.warning("taskkill 超时: pid=%d", pid)
            except Exception as e:
                logger.warning("taskkill 异常: %s", e)
        else:
            # Linux/Mac: 杀进程组
            try:
                import os
                import signal
                pgid = os.getpgid(pid)
                logger.info("Linux 进程组清理: pgid=%d", pgid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                # 进程已退出
                pass
            except Exception as e:
                logger.warning("killpg 异常: %s", e)

    def _assign_to_job_object(self, process: subprocess.Popen) -> None:
        """Windows: 把子进程加入 Job Object（v0.3.1+ 新增）。

        Job Object 配置：
            - JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: 父进程死时 OS 自动 kill Job 内所有进程
        作用：
            - 防止平台异常退出（崩溃/任务管理器结束）时 llama-server 变孤儿继续占用显存
            - 仅 Windows 需要，Linux/Mac 已用 start_new_session + 进程组

        Args:
            process: subprocess.Popen 对象
        """
        if sys.platform != "win32":
            return

        try:
            import ctypes

            # Windows API 常量
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
            JobObjectExtendedLimitInformation = 9
            PROCESS_ALL_ACCESS = 0x1F0FFF

            # 声明函数原型（use_last_error=True 确保错误码准确）
            kernel32 = ctypes.windll.kernel32
            kernel32.CreateJobObjectW = ctypes.WINFUNCTYPE(
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p,
                use_last_error=True,
            )(("CreateJobObjectW", kernel32))
            kernel32.SetInformationJobObject = ctypes.WINFUNCTYPE(
                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_uint32,
                use_last_error=True,
            )(("SetInformationJobObject", kernel32))
            kernel32.AssignProcessToJobObject = ctypes.WINFUNCTYPE(
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                use_last_error=True,
            )(("AssignProcessToJobObject", kernel32))
            kernel32.OpenProcess = ctypes.WINFUNCTYPE(
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32,
                use_last_error=True,
            )(("OpenProcess", kernel32))

            # Windows SDK 标准结构体定义（sizeof = 144 bytes）
            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            # 创建 Job Object
            job_handle = kernel32.CreateJobObjectW(None, None)
            if not job_handle:
                raise ctypes.WinError(ctypes.get_last_error())

            # 设置 Job Object 信息：KILL_ON_JOB_CLOSE
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            ret = kernel32.SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ret:
                raise ctypes.WinError(ctypes.get_last_error())

            # 把子进程加入 Job Object（用 OpenProcess 获取句柄，兼容 Python 3.12+）
            process_handle = kernel32.OpenProcess(
                PROCESS_ALL_ACCESS, False, process.pid,
            )
            if not process_handle:
                raise ctypes.WinError(ctypes.get_last_error())

            ret = kernel32.AssignProcessToJobObject(job_handle, process_handle)
            if not ret:
                raise ctypes.WinError(ctypes.get_last_error())

            self._job_handle = job_handle
            logger.info("llama-server 子进程已绑定 Job Object: pid=%d, job=%d",
                        process.pid, job_handle)

        except Exception as e:
            # Job Object 绑定失败不阻断启动，仅记录警告
            # 进程树清理（taskkill /T）仍能兜底
            logger.warning("Job Object 绑定失败（不阻断启动）: %s", e)

    # ============ router mode 模型管理 ============

    async def load_model(self, model_name: str, ctx_size: Optional[int] = None) -> dict:
        """调 llama-server API 加载指定模型到显存。

        router mode 下 llama-server 暴露 POST /models/load 接口。

        v0.3.3+ 变更：支持 ctx_size 参数，允许在加载时覆盖 INI 文件中的 ctx-size。
        用于反代层拦截 /models/load 时注入 per-variant ctx。

        Args:
            model_name: 模型名（GGUF 文件名或 alias / preset 名）
            ctx_size: 可选的 ctx-size 覆盖值（v0.3.3+ 新增）
                - None：使用 INI 文件中声明的 ctx-size（默认行为）
                - int：覆盖 INI 文件的 ctx-size（用于 per-variant ctx 注入）

        Returns:
            dict: llama-server 返回的加载结果

        Raises:
            RuntimeError: 子进程未启动 / API 调用失败
        """
        if not self.is_running or self.internal_url is None:
            raise RuntimeError("llama-server 子进程未启动")

        import httpx
        url = f"{self.internal_url}/models/load"
        payload: dict[str, Any] = {"model": model_name}
        if ctx_size is not None and ctx_size > 0:
            payload["ctx-size"] = int(ctx_size)
            logger.info("router mode 加载模型: %s (ctx-size=%d)", model_name, ctx_size)
        else:
            logger.info("router mode 加载模型: %s (ctx-size=INI 默认)", model_name)

        try:
            async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:  # 平台内部回环，不走系统代理  # 大模型加载可能很慢
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                self._loaded_models.add(model_name)
                result = resp.json() if resp.content else {}
                logger.info("模型加载成功: %s", model_name)
                return result
        except Exception as e:
            logger.error("模型加载失败: %s, err=%s", model_name, e)
            raise RuntimeError(f"模型加载失败: {model_name} - {e}") from e

    async def unload_model(self, model_name: str) -> dict:
        """调 llama-server API 卸载指定模型，释放显存。

        router mode 下 llama-server 暴露 POST /models/unload 接口。

        Args:
            model_name: 模型名

        Returns:
            dict: llama-server 返回的卸载结果
        """
        if not self.is_running or self.internal_url is None:
            raise RuntimeError("llama-server 子进程未启动")

        import httpx
        url = f"{self.internal_url}/models/unload"
        logger.info("router mode 卸载模型: %s", model_name)

        try:
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = await client.post(url, json={"model": model_name})
                resp.raise_for_status()
                self._loaded_models.discard(model_name)
                result = resp.json() if resp.content else {}
                logger.info("模型卸载成功: %s", model_name)
                return result
        except Exception as e:
            logger.error("模型卸载失败: %s, err=%s", model_name, e)
            raise RuntimeError(f"模型卸载失败: {model_name} - {e}") from e

    async def list_loaded_models(self) -> list[dict]:
        """调 llama-server API 列出所有已加载模型。

        Returns:
            list[dict]: 已加载模型列表
        """
        if not self.is_running or self.internal_url is None:
            return []

        import httpx
        url = f"{self.internal_url}/models"
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                # llama-server /models 返回格式：{"data": [{"id": "xxx", ...}, ...]}
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                if isinstance(data, list):
                    return data
                return []
        except Exception as e:
            logger.warning("查询已加载模型失败: %s", e)
            return []

    # ============ 字节流透传 chat ============

    async def forward_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
        **extra: Any,
    ) -> AsyncIterator[bytes]:
        """转发到 llama-server /v1/chat/completions（字节流透传）。

        v0.3.0 变更：
            - 删除 SSE 格式转换（不再用 ChatAdapter 转 AIReliable 进度帧）
            - 直接透传 OpenAI SSE 字节流
            - 新增 model 参数（router mode 路由用）

        Args:
            messages: OpenAI 消息列表
            model: router mode 路由用的模型名（variant_id 或 alias）
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: top-p 采样
            stream: 是否流式
            seed: 随机种子
            stop: 停止序列
            **extra: 其他 OpenAI 兼容参数

        Yields:
            bytes: SSE 帧字节流（直接透传，未做格式转换）

        Raises:
            RuntimeError: 子进程未启动
        """
        if not self.is_running or self.internal_url is None:
            raise RuntimeError("llama-server 子进程未启动")

        import httpx

        url = f"{self.internal_url}/v1/chat/completions"
        request_body: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if model is not None:
            request_body["model"] = model
        if seed is not None:
            request_body["seed"] = seed
        if stop:
            request_body["stop"] = stop
        request_body.update(extra)

        # v0.3.4+ 打印请求体（排除 messages）用于排查 chat_template_kwargs 等扩展参数透传
        extra_keys = {k: v for k, v in request_body.items() if k != "messages"}
        logger.info(
            "转发 chat 请求: url=%s, model=%s, messages=%d, max_tokens=%d, stream=%s, "
            "extra_keys=%s, has_chat_template_kwargs=%s",
            url, model, len(messages), max_tokens, stream,
            extra_keys,
            "chat_template_kwargs" in request_body,
        )

        # v0.4.x 读超时 600→3600：大上下文 prompt 处理期间 SSE 静默 650s+，
        # 600s 必超时（与 server.py 反代层 / gateway EngineClient 同步修复）
        async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0), trust_env=False) as client:  # 平台内部回环，不走系统代理
            async with client.stream(
                "POST",
                url,
                json=request_body,
                headers={"Accept": "text/event-stream" if stream else "application/json"},
            ) as response:
                response.raise_for_status()

                if stream:
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
                else:
                    # 非流式：一次性返回
                    content = await response.aread()
                    yield content

    # ============ 通用请求转发（供反代路由调用）============

    async def forward_request(
        self,
        method: str,
        path: str,
        headers: dict,
        body: bytes,
        query_params: Optional[dict] = None,
    ) -> tuple[int, dict, AsyncIterator[bytes]]:
        """转发任意 HTTP 请求到 llama-server（供反向代理路由使用）。

        Args:
            method: HTTP 方法（GET/POST/PUT/DELETE/...）
            path: 路径（不含 host，如 "/v1/chat/completions"）
            headers: 请求头
            body: 请求体字节流
            query_params: 查询参数

        Returns:
            tuple: (status_code, response_headers, body_stream)
            body_stream 是异步迭代器，按 chunk 透传字节流

        Raises:
            RuntimeError: 子进程未启动
        """
        if not self.is_running or self.internal_url is None:
            raise RuntimeError("llama-server 子进程未启动")

        import httpx

        url = f"{self.internal_url}{path}"
        logger.debug("反代转发: %s %s", method, url)

        # 透传请求头，但移除 host 相关字段（避免 llama-server 端冲突）
        forward_headers = {
            k: v for k, v in headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), trust_env=False)  # 平台内部回环，不走系统代理

        try:
            req = client.build_request(
                method,
                url,
                headers=forward_headers,
                content=body if body else None,
                params=query_params,
            )
            response = await client.send(req, stream=True)
            return response.status_code, dict(response.headers), response.aiter_bytes()
        except Exception as e:
            await client.aclose()
            raise

    # ============ 健康检查 ============

    async def health(self) -> bool:
        """检查 llama-server /health 是否 OK。"""
        if not self.is_running or self.internal_url is None:
            return False

        import httpx
        try:
            async with httpx.AsyncClient(trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = await client.get(
                    f"{self.internal_url}/health",
                    timeout=2.0,
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def is_alive(self) -> bool:
        """检查子进程是否存活。"""
        return self.is_running

    # ============ 工具方法 ============

    def _cleanup_resources(self) -> None:
        """清理资源（v0.3.1+ 增加关闭 Job Object 句柄）。"""
        self._process = None
        self._internal_port = None
        self._start_time = None
        self._loaded_models.clear()

        # ===== llama-server 日志转发 v1.0：先停 tail 线程再关文件 =====
        # 顺序：发信号 → 有界 join(0.5s) → 置空引用 → 再 close 文件句柄
        # 避免 tail 在读文件句柄期间被 close 触发异常（虽然线程内部已经吞异常）
        self._stop_tail.set()
        if self._tail_thread is not None and self._tail_thread.is_alive():
            try:
                self._tail_thread.join(timeout=0.5)
                if self._tail_thread.is_alive():
                    logger.warning(
                        "llama-server tail 线程 0.5s 未退出（daemon=True 兜底回收）"
                    )
            except Exception as e:
                logger.warning("join llama-server tail 线程异常: %s", e)
        self._tail_thread = None
        self._log_file_tail_path = None

        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None

        # v0.3.1+ 关闭 Job Object 句柄
        # 关闭后 Job 内残留进程会被 OS 自动 kill（KILL_ON_JOB_CLOSE 标志）
        if self._job_handle is not None:
            try:
                if sys.platform == "win32":
                    import ctypes
                    ctypes.windll.kernel32.CloseHandle(self._job_handle)
            except Exception:
                pass
            self._job_handle = None

    def read_recent_logs(self, n_lines: int = 100) -> str:
        """读取最近的子进程日志（用于错误诊断）。"""
        # v0.2.15+ llama-server 子进程日志在 logs/engines/llama_cpp_server/
        log_file = self.config.server_log_dir / "llama_cpp_server.log"
        if not log_file.exists():
            return ""
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:])
        except Exception as e:
            logger.warning("读取日志失败: %s", e)
            return ""


# ===== 模块级工具函数 =====

def allocate_internal_port(start: int = 8100, end: int = 8200) -> int:
    """从指定范围内找空闲端口。

    Args:
        start: 起始端口
        end: 结束端口

    Returns:
        int: 可用端口

    Raises:
        RuntimeError: 无空闲端口
    """
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"无空闲端口可用（范围 {start}-{end}）")


def _get_creation_flags() -> int:
    """获取 Windows 子进程创建标志（不弹 cmd 窗口）。"""
    if sys.platform == "win32":
        # CREATE_NO_WINDOW = 0x08000000
        # CREATE_NEW_PROCESS_GROUP = 0x00000200（允许 Ctrl+C 不传递到子进程）
        return 0x08000000 | 0x00000200
    return 0


async def wait_for_ready(
    process: subprocess.Popen,
    internal_port: int,
    timeout: float = 120.0,
    interval: float = 0.5,
) -> None:
    """轮询 llama-server /health 直到 OK 或超时。

    Args:
        process: 子进程对象
        internal_port: 隐藏端口
        timeout: 超时秒数
        interval: 轮询间隔

    Raises:
        RuntimeError: 超时 / 子进程退出
    """
    import httpx

    url = f"http://127.0.0.1:{internal_port}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        # 检查子进程是否已退出
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server 子进程已退出（returncode={process.returncode}）"
            )

        try:
            async with httpx.AsyncClient(trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = await client.get(url, timeout=1.0)
                if resp.status_code == 200:
                    return
        except Exception:
            pass

        await asyncio.sleep(interval)

    raise RuntimeError(
        f"llama-server 启动超时（{timeout}s 内 /health 未就绪，端口 {internal_port}）"
    )
