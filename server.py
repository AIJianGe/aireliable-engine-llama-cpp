"""Llama.cpp 引擎 HTTP 服务（v0.3.0 router mode + 反代 llama-server UI）。

实现 8 个标准 HTTP 端点 + 反向代理 llama-server 自带 SvelteKit Web UI。

启动：
    uv run --no-sync --project envs/llm-llama-cpp \\
        python -m engines.llama_cpp_9cdb76f8.server --port 7002

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用 GGUF variant 列表）
    POST /load           启动 router mode 子进程（可选预加载 variant）
    POST /unload         停止子进程（释放所有模型显存）
    POST /infer          推理（SSE 流式，字节流透传 llama-server OpenAI SSE）
    GET  /status         状态查询（含 router mode 多模型信息）
    POST /lora/load      v0.1 返回 501
    POST /lora/unload    v0.1 返回 501
    GET  /output/*       静态文件（文本产物）
    ANY  /{path}         反向代理 → llama-server 自带 UI（最后挂载）

v0.3.0 关键变更：
    - 删除 Gradio UI 挂载（_mount_gradio / _load_gradio_theme）
    - 删除 file_storage_dir 静态挂载（统一到 output/）
    - 删除 registry_variants 加载（engine.py 改用平台 API 优先扫描）
    - 新增 _mount_llama_server_proxy：反向代理路由（最后挂载）
    - 反代支持 SSE 流式透传 + WebSocket 透传
    - 未 /load 时反代返回友好 HTML 提示页（503）
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

import json as _json

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    ErrorResponse,
    HealthResponse,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import LlamaCppEngine
from .schemas import InferRequest

logger = logging.getLogger(__name__)


# ===== v0.3.4+ 反代层 alias 翻译 =====
# 反代路由 /v1/chat/completions、/v1/completions 等路径收到请求后，
# 会读取 body 中的 `model` 字段，如果是 manifest 中声明的 alias
# （如 qwen3.6-unc-q6），则翻译为 llama-server preset 名（variant.name）
# （如 qwen3.6-35b-a3b-uncensored-q6-k-p）。
# 目的：让工具可以直连 7002 引擎端口使用 alias 调用，无需绕道 8000 gateway。
# 翻译失败（model 不在 alias 表中）保持原 body 透传，让 llama-server 自己处理。

# 需要做 alias 翻译的 OpenAI 兼容端点
# 注意：仅翻译 model 字段，其他字段原样透传
_ALIAS_TRANSLATED_PATHS = frozenset({
    "v1/chat/completions",
    "v1/completions",
    "v1/responses",                  # OpenAI 较新接口，部分工具使用
})


def _translate_model_alias(
    engine: LlamaCppEngine,
    body_bytes: bytes,
) -> tuple[bytes, bool]:
    """反代层 alias 翻译辅助函数。

    读取请求体，解析 `model` 字段；如果该值是 manifest 中声明的 alias
    （即等于某个 variant.alias 但不等于任何 variant.name），则替换为
    对应的 variant.name（llama-server preset 名）。

    Args:
        engine: 引擎实例（用于访问 _available_variants 列表）
        body_bytes: 原始请求体

    Returns:
        (modified_body, translated)：
            modified_body: 翻译后的 body（若无修改则原样返回）
            translated: 是否触发了翻译（用于日志）
    """
    if not body_bytes:
        return body_bytes, False

    try:
        body_json = _json.loads(body_bytes)
    except (_json.JSONDecodeError, UnicodeDecodeError):
        # 非 JSON 或编码异常，原样透传（让 llama-server 报错）
        return body_bytes, False

    if not isinstance(body_json, dict):
        return body_bytes, False

    model_value = body_json.get("model")
    if not isinstance(model_value, str) or not model_value:
        return body_bytes, False

    # 先按 variant.name / alias 做匹配（_find_variant 已支持 alias: str | list[str]）
    matched = engine._find_variant(model_value)
    if matched is None:
        # 不在 alias/name 表中，记录 WARNING 便于定位（v0.3.8+ B1 修复可观测性）
        _aliases_overview = []
        for _v in getattr(engine, "_available_variants", []) or []:
            _n = getattr(_v, "name", "")
            _a = getattr(_v, "alias", None)
            if isinstance(_a, list):
                _aliases_overview.append(f"{_n}(aliases:{','.join(_a)})")
            elif isinstance(_a, str) and _a:
                _aliases_overview.append(f"{_n}(alias:{_a})")
            else:
                _aliases_overview.append(f"{_n}")
        logger.warning(
            "反代 alias 未匹配：model=%s，可用 variant+alias=%s",
            model_value, _aliases_overview,
        )
        return body_bytes, False

    # 已经是 variant.name（全名）→ 不翻译
    if model_value == matched.name:
        return body_bytes, False

    # 判断 model_value 是否真等于某个 alias（兼容 alias: str | list[str]）
    _alias = getattr(matched, "alias", None)
    _is_alias = False
    if isinstance(_alias, list):
        _is_alias = model_value in _alias
    elif isinstance(_alias, str) and _alias:
        _is_alias = (model_value == _alias)
    # _find_variant 命中但既不等于 name 也不等于 alias → 理论不会到这里
    if not _is_alias:
        # 兜底：当 name 也不匹配但 _find_variant 找到了（例如 _find_variant 用模糊匹配），
        # 仍按 name 翻译，但记一条 DEBUG 便于审计
        logger.debug(
            "反代 alias 翻译：_find_variant 命中但 alias 字段未显式包含 model=%s, "
            "variant.name=%s, variant.alias=%s，仍翻译为 name",
            model_value, matched.name, _alias,
        )

    # 触发翻译：alias → variant.name（preset 名）
    body_json["model"] = matched.name
    new_body = _json.dumps(body_json, ensure_ascii=False).encode("utf-8")
    return new_body, True


def _safe_extract_model(body_bytes: bytes) -> str:
    """从请求体中安全提取 model 字段（用于日志，不抛异常）。"""
    try:
        obj = _json.loads(body_bytes)
    except Exception:
        return "<unparseable>"
    if not isinstance(obj, dict):
        return "<not-object>"
    model = obj.get("model")
    if isinstance(model, str):
        return model
    return "<no-model>"


# ===== 全局引擎实例 =====
# FastAPI 启动时初始化，所有端点共享
_engine: Optional[LlamaCppEngine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> LlamaCppEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== v0.3.1+ asyncio 噪声日志过滤 =====

def _setup_asyncio_noise_filter() -> None:
    """设置自定义 asyncio 异常处理器，过滤 Windows ProactorEventLoop 噪声日志。

    过滤目标：
        - ConnectionResetError [WinError 10054]：健康检查轮询时 socket 关闭触发
        - _ProactorBasePipeTransport._call_connection_lost 回调异常

    这些异常是 Windows asyncio 已知缺陷，不影响功能（健康检查最终成功）。
    仅过滤特定组合，其他异常仍走默认 handler。
    """
    import asyncio
    import sys

    if sys.platform != "win32":
        return  # 仅 Windows 需要

    loop = asyncio.get_event_loop()

    def _noise_filter(loop, context):
        # 获取异常和回调信息
        exception = context.get("exception")
        message = context.get("message", "")
        callback = context.get("handle")

        # 过滤 ConnectionResetError [WinError 10054]
        if isinstance(exception, ConnectionResetError):
            # 检查是否来自 _call_connection_lost（ProactorBasePipeTransport）
            cb_repr = repr(callback) if callback else ""
            if "_call_connection_lost" in cb_repr or "WinError 10054" in str(exception):
                logger.debug(
                    "已过滤 asyncio 噪声: ConnectionResetError [WinError 10054] "
                    "（Windows ProactorEventLoop 已知缺陷，不影响功能）"
                )
                return

        # 过滤 _call_connection_lost 相关的异常（即使不是 ConnectionResetError）
        if callback is not None:
            cb_repr = repr(callback)
            if "_call_connection_lost" in cb_repr:
                logger.debug(
                    "已过滤 asyncio 噪声: _call_connection_lost 异常 (%s): %s",
                    type(exception).__name__ if exception else "unknown",
                    exception,
                )
                return

        # 其他异常走默认 handler
        loop.default_exception_handler(context)

    loop.set_exception_handler(_noise_filter)
    logger.debug("已设置 asyncio 噪声过滤器（过滤 WinError 10054）")


# ===== v0.4.x httpx 噪音过滤（SDK internal progress 上报） =====

class _HttpxInternalProgressFilter(logging.Filter):
    """过滤 httpx logger 的 INFO 级 internal progress 请求日志。

    过滤目标：
        - SDK task_reporter POST /v1/tasks/internal/{task_id}/progress|succeeded|failed
          每次上报产生一行 ``HTTP Request: POST http://... 200 OK``，
          长推理期间进度帧高频上报（每 token 批次一次）→ 运行日志终端刷屏。

    不过滤：
        - WARNING 及以上级别（错误必须可见）
        - 其他路径的 INFO 日志（反代请求日志保留，便于排查）
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "/v1/tasks/internal/" not in msg


def _setup_httpx_noise_filter() -> None:
    """给 httpx logger 挂载 internal progress 过滤器（幂等）。"""
    httpx_logger = logging.getLogger("httpx")
    # 幂等保护：避免 lifespan 重入时重复挂 filter（重复挂载会导致同一条日志被多次过滤判断）
    for f in httpx_logger.filters:
        if getattr(f, "_aireliable_internal_progress_filter", False):
            return
    flt = _HttpxInternalProgressFilter()
    flt._aireliable_internal_progress_filter = True
    httpx_logger.addFilter(flt)
    logger.debug("已设置 httpx 噪音过滤器（过滤 /v1/tasks/internal/ 的 INFO 请求日志）")


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理。

    启动时初始化引擎（含异步模型扫描），关闭时清理资源（停止子进程）。
    """
    global _engine, _config

    # 解析配置（如果尚未从命令行注入）
    if _config is None:
        _config = parse_args()

    # 配置日志
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    # v0.3.1+ 设置自定义异常处理器：抑制 Windows asyncio ProactorEventLoop 噪声日志
    # 过滤 _call_connection_lost + ConnectionResetError [WinError 10054] 组合
    # 这是 Windows asyncio 已知缺陷，健康检查轮询时 socket 关闭触发，不影响功能
    _setup_asyncio_noise_filter()

    # v0.4.x httpx 噪音过滤（SDK internal progress 上报的 INFO 请求日志）
    _setup_httpx_noise_filter()

    logger.info("===== Llama.cpp 引擎启动（v0.3.1 router mode + 显存回收增强）=====")
    logger.info("配置: %s", _config.to_dict())

    # 初始化引擎
    _engine = LlamaCppEngine(_config)

    # v0.3.0：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await _engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d",
        _engine.state_machine.state.value,
        len(_engine._available_variants),
    )

    yield

    # 关闭清理：停止子进程
    logger.info("===== Llama.cpp 引擎关闭 =====")
    if _engine is not None and _engine.process_manager.is_running:
        try:
            await _engine._unload_impl()
        except Exception as e:
            logger.warning("关闭时停止子进程失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="Llama.cpp Engine",
        description=(
            "AIReliable 平台的 Llama.cpp 引擎（v0.3.0 router mode + 反代自带 UI）"
        ),
        version="0.3.0",
        lifespan=lifespan,
    )

    # CORS（允许主控跨域调用，iframe 嵌入需要）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],                  # v0.1 简化，生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== 注册 8 个标准端点 + /output 静态挂载 =====
    _register_endpoints(app)
    _mount_output_static(app)

    # ===== v0.3.3+ 反代拦截器：/models/load / /models/unload =====
    # 必须在 catch-all 反代之前注册，拦截 llama-server 的模型加载/卸载请求，
    # 注入 per-variant ctx + 通知平台 core 显存状态变化
    _register_proxy_interceptors(app)

    # ===== 反向代理 llama-server UI（必须最后挂载）=====
    # 注意：反代路由使用 catch-all /{path:path}，会拦截未匹配的所有路径，
    # 因此必须在 8 端点 + /output + 拦截器之后挂载，否则会拦截 /health /load 等关键端点
    _mount_llama_server_proxy(app)

    return app


# ===== 8 端点注册 =====

def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准 HTTP 端点。"""

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（可用 GGUF variant 列表）。

        注意：手动序列化 variants 以保留 LlamaVariantInfo 扩展字段
        （path / file_size / supports_mmproj / mmproj_path / alias），
        避免 Pydantic 按基类 VariantInfo 序列化时丢弃子类字段。
        """
        resp = await get_engine().capabilities()
        data = resp.model_dump()
        data["variants"] = [v.model_dump() for v in resp.variants]
        return data

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """启动 router mode 子进程（可选预加载 variant）。"""
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            # 文件完整性校验失败 / 子进程启动失败
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """停止子进程（释放所有模型显存）。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """推理（SSE 流式返回，字节流透传 llama-server OpenAI SSE）。

        请求体参考 InferRequest schema：
            - variant: 可选，若未加载则自动调 /models/load
            - inputs.messages: OpenAI 兼容消息列表
            - inputs.model: router mode 路由用的模型名（优先级高于 variant）
            - inputs.max_tokens / temperature / top_p / stream / seed / stop
            - task_id: 可选任务 ID
        """
        engine = get_engine()

        # BaseEngine.infer 返回异步迭代器（SSE 帧）
        sse_stream = engine.infer(request)

        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",          # Nginx 不缓冲
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态查询（含 router mode 多模型信息）。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load（v0.1 返回 501）
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA（v0.1 不支持）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload（v0.1 返回 501）
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA（v0.1 不支持）。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 错误处理
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常统一格式。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )

    # v0.2.5+ T6.2：日志级别动态调整端点（GET /log_level + POST /log_level）
    register_log_level_endpoint(app)


# ===== /output 静态挂载 =====

def _mount_output_static(app: FastAPI) -> None:
    """挂载 /output/* 静态文件（文本产物下载）。"""
    if _config is None:
        return
    app.mount(
        "/output",
        StaticFiles(directory=str(_config.output_root)),
        name="output",
    )
    logger.info("输出目录挂载: /output -> %s", _config.output_root)


# ===== 状态机恢复 + auto-load 公共 helper（v0.3.7+）=====

async def auto_load_if_needed(engine) -> None:
    """确保 llama-server 子进程已启动，必要时清理状态机并重新加载。

    v0.3.7+ 关键修复：状态机与子进程状态不一致恢复
    ----------------------------------------------------------------
    历史问题：核心服务重启/异常退出时，Python 引擎父进程崩溃但状态机
    卡在 READY/INFERRING；子进程被 OS 通过 Job Object 自动 kill；之后
    所有 auto-load 都会因为 `ready → loading` 非法转换失败，陷入死循环。

    修复策略（先清理后加载）：
        1. 子进程运行中 → 直接 return
        2. 子进程死了 + 状态机 READY → 调 unload() 清理状态（READY → IDLE）
        3. 子进程死了 + 状态机 INFERRING/ERROR → 强制 reset 到 IDLE
        4. 子进程死了 + 状态机 IDLE/LOADING → 直接 load()

    Args:
        engine: LlamaCppEngine 实例

    Raises:
        RuntimeError: 状态清理后 _load_impl 仍失败（文件缺失、二进制缺失等）
    """
    from aireliable_engine import RuntimeState

    pm_local = engine.process_manager
    sm = engine.state_machine

    # 1. 子进程已运行 → 直接放行
    if pm_local.is_running and pm_local.internal_url is not None:
        return

    # 2. 子进程死了但状态机卡住：先清理状态再 load
    current_state = sm.state
    if current_state == RuntimeState.READY:
        logger.warning(
            "auto_load_if_needed：状态机 READY 但子进程已死，"
            "先调 unload() 清理状态（避免 ready → loading 非法转换）"
        )
        try:
            await engine.unload()
            logger.info("auto_load_if_needed：unload 完成，状态机已转 IDLE")
        except Exception as e:
            # unload 失败（极端情况），强制 reset 状态机
            logger.warning("auto_load_if_needed：unload 失败，强制 reset 状态机: %s", e)
            try:
                sm.reset()
            except Exception:
                # 状态不在 ERROR 时 reset 会失败，强制 transition
                pass
    elif current_state in (RuntimeState.INFERRING, RuntimeState.ERROR):
        logger.warning(
            "auto_load_if_needed：状态机 %s 但子进程已死，强制 reset 到 IDLE",
            current_state.value,
        )
        try:
            if current_state == RuntimeState.ERROR:
                sm.reset()  # ERROR → IDLE
            else:
                # INFERRING 是非终态：强制转 ERROR 再 reset
                sm.transition(RuntimeState.ERROR)
                sm.reset()
        except Exception as e:
            logger.warning("auto_load_if_needed：状态机 reset 失败: %s", e)
    elif current_state == RuntimeState.LOADING:
        logger.warning(
            "auto_load_if_needed：状态机 LOADING 但子进程已死，"
            "可能有并发 load 操作，等待或强制 reset"
        )
        # LOADING 状态下不应该被反代路由打断（_op_lock 会保护）
        # 但子进程死了说明 load 一定失败了，先转 ERROR
        try:
            sm.transition(RuntimeState.ERROR)
            sm.reset()
        except Exception as e:
            logger.warning("auto_load_if_needed：状态机 reset 失败: %s", e)

    # 3. 此时状态机应该是 IDLE，调 load 启动子进程
    logger.info(
        "auto_load_if_needed：触发 load(variant=None) 启动子进程"
    )
    await engine.load(LoadRequest(variant=None))
    logger.info("auto_load_if_needed：成功，子进程已启动（未加载模型）")


# ===== 反向代理 llama-server UI（最后挂载）=====

# 未 /load 时返回的友好提示页（503）
_UNREADY_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Llama.cpp 引擎未启动</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         display: flex; align-items: center; justify-content: center;
         min-height: 100vh; margin: 0; background: #f5f5f5; color: #333; }
  .card { background: white; padding: 40px; border-radius: 8px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 480px; text-align: center; }
  h1 { font-size: 20px; margin: 0 0 16px; color: #d97706; }
  p { font-size: 14px; line-height: 1.6; color: #666; margin: 8px 0; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px;
         font-family: Consolas, Monaco, monospace; color: #c41e3a; }
  .hint { margin-top: 20px; padding-top: 16px; border-top: 1px solid #eee;
          font-size: 12px; color: #999; }
</style>
</head>
<body>
  <div class="card">
    <h1>⚠️ Llama.cpp 引擎未启动</h1>
    <p>llama-server 子进程尚未运行，无法提供 Web UI 和推理服务。</p>
    <p>请先在平台详情页点击 <code>启动</code> 按钮（调 <code>POST /load</code>）。</p>
    <p>启动完成后刷新此页面即可访问 llama-server 自带 SvelteKit UI。</p>
    <div class="hint">
      引擎端点：<code>http://127.0.0.1:7002</code><br>
      健康检查：<code>GET /health</code> &nbsp;|&nbsp; 状态：<code>GET /status</code>
    </div>
  </div>
</body>
</html>
"""


def _register_proxy_interceptors(app: FastAPI) -> None:
    """v0.3.3+ 注册反代拦截器：/models/load / /models/unload。

    这些路由必须在 catch-all 反代之前注册，拦截 llama-server 的模型加载/卸载请求，
    实现两个目标：
        1. 注入 per-variant ctx-size（来自 manifest / GGUF / 用户覆盖）
        2. 通知平台 core 显存状态变化（report_loaded / report_unloaded）

    拦截流程（/models/load）：
        1. 读取请求体，解析 model 字段（preset 名）
        2. 查找 variant 信息，解析生效的 ctx-size
        3. 若请求未指定 ctx-size，注入生效值
        4. 转发到 llama-server
        5. 成功后调 SDK report_loaded 通知平台（含显存估算值）

    拦截流程（/models/unload）：
        1. 转发到 llama-server
        2. 成功后调 SDK report_unloaded 通知平台

    注意：拦截器不阻断反代，仅做"注入 + 上报"，转发逻辑复用反代的流式透传。
    """
    import json as _json

    @app.post("/models/load")
    async def proxy_models_load(request: Request):
        """拦截 /models/load：注入 per-variant ctx + 上报显存。

        llama-server /models/load 请求体格式：
            {"model": "<preset_name>", "ctx-size": 32768（可选）}

        注入策略：
            - 若请求体已含 ctx-size：不注入（尊重调用方显式指定）
            - 若请求体无 ctx-size：
                - 查找 variant 的生效 ctx（override > manifest.recommended > 不注入）
                - 注入到请求体
        """
        engine = get_engine()
        pm = engine.process_manager

        # auto-load：子进程未启动时自动启动（使用统一 helper，含状态机恢复）
        if not pm.is_running or pm.internal_url is None:
            logger.info("/models/load 拦截器：子进程未启动，触发 auto-load")
            try:
                await auto_load_if_needed(engine)
            except Exception as e:
                logger.exception("/models/load 拦截器 auto-load 失败: %s", e)
                return JSONResponse(
                    status_code=503,
                    content={"error": {
                        "type": "AutoLoadFailed",
                        "message": f"llama-server 启动失败: {e}",
                        "code": 503,
                    }},
                )

        # 读取请求体
        raw_body = await request.body()
        try:
            body_json = _json.loads(raw_body) if raw_body else {}
        except _json.JSONDecodeError:
            # 非 JSON 请求体，原样转发
            body_json = None

        model_name = ""
        modified_body: bytes = raw_body

        if isinstance(body_json, dict):
            model_name = body_json.get("model", "") or ""
            # v0.3.3+ 注入 per-variant ctx-size（仅当请求未显式指定时）
            if "ctx-size" not in body_json and model_name:
                ctx_val = engine._resolve_ctx_for_preset(model_name)
                if ctx_val is not None and ctx_val > 0:
                    body_json["ctx-size"] = int(ctx_val)
                    modified_body = _json.dumps(body_json).encode("utf-8")
                    logger.info(
                        "/models/load 拦截器：注入 per-variant ctx-size=%d (model=%s)",
                        ctx_val, model_name,
                    )

        # 转发到 llama-server
        target_url = f"{pm.internal_url}/models/load"
        forward_headers = _strip_hop_by_hop_headers(dict(request.headers))

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0), trust_env=False,  # 大模型加载可能很慢（平台内部回环，不走系统代理）
            follow_redirects=False,
        )
        try:
            req = client.build_request(
                "POST",
                target_url,
                headers=forward_headers,
                content=modified_body if modified_body else None,
            )
            response = await client.send(req, stream=True)
        except Exception as e:
            await client.aclose()
            logger.exception("/models/load 拦截器转发失败: %s", e)
            return JSONResponse(
                status_code=502,
                content={"error": {
                    "type": "ProxyError",
                    "message": f"转发 llama-server 失败: {e}",
                    "code": 502,
                }},
            )

        # 等待响应完成（非流式，一次性读取）
        response_body = await response.aread()
        await client.aclose()

        # 更新本地 loaded_models 缓存
        if response.status_code == 200 and model_name:
            pm._loaded_models.add(model_name)
            logger.info("/models/load 拦截器：模型加载成功 %s", model_name)

            # v0.3.3+ 通知平台 core 显存状态
            try:
                from aireliable_engine import get_platform_vram_manager
                vram_mgr = await get_platform_vram_manager()
                vram_used_bytes = engine._estimate_variant_vram_bytes(model_name)
                # 查找 variant 名（preset → variant.name 映射）
                variant = engine._find_variant_by_preset_name(model_name)
                variant_name = variant.name if variant else model_name
                await vram_mgr.report_loaded(
                    variant=variant_name,
                    vram_used_bytes=vram_used_bytes,
                )
                logger.info(
                    "/models/load 拦截器：已通知平台显存状态 variant=%s vram_estimated=%.2fGB",
                    variant_name, vram_used_bytes / (1024 ** 3),
                )
            except Exception as e:
                # 上报失败不阻断加载
                logger.warning("/models/load 拦截器：通知平台显存状态失败（不阻断）: %s", e)

        # 返回响应给客户端
        response_headers = _strip_hop_by_hop_headers(dict(response.headers))
        response_headers.pop("content-encoding", None)
        response_headers.pop("Content-Encoding", None)
        return JSONResponse(
            status_code=response.status_code,
            content=_json.loads(response_body) if response_body else {},
            headers=response_headers,
        )

    @app.post("/models/unload")
    async def proxy_models_unload(request: Request):
        """拦截 /models/unload：上报显存释放。

        llama-server /models/unload 请求体格式：
            {"model": "<preset_name>"}
        """
        engine = get_engine()
        pm = engine.process_manager

        if not pm.is_running or pm.internal_url is None:
            # 子进程未启动，无需卸载
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "message": "子进程未运行，无需卸载"},
            )

        # 读取请求体
        raw_body = await request.body()
        try:
            body_json = _json.loads(raw_body) if raw_body else {}
        except _json.JSONDecodeError:
            body_json = None

        model_name = ""
        if isinstance(body_json, dict):
            model_name = body_json.get("model", "") or ""

        # 转发到 llama-server
        target_url = f"{pm.internal_url}/models/unload"
        forward_headers = _strip_hop_by_hop_headers(dict(request.headers))

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False,  # 平台内部回环，不走系统代理
            follow_redirects=False,
        )
        try:
            req = client.build_request(
                "POST",
                target_url,
                headers=forward_headers,
                content=raw_body if raw_body else None,
            )
            response = await client.send(req, stream=True)
        except Exception as e:
            await client.aclose()
            logger.exception("/models/unload 拦截器转发失败: %s", e)
            return JSONResponse(
                status_code=502,
                content={"error": {
                    "type": "ProxyError",
                    "message": f"转发 llama-server 失败: {e}",
                    "code": 502,
                }},
            )

        response_body = await response.aread()
        await client.aclose()

        # 更新本地 loaded_models 缓存
        if response.status_code == 200 and model_name:
            pm._loaded_models.discard(model_name)
            logger.info("/models/unload 拦截器：模型卸载成功 %s", model_name)

            # v0.3.3+ 通知平台 core 显存释放
            try:
                from aireliable_engine import get_platform_vram_manager
                vram_mgr = await get_platform_vram_manager()
                await vram_mgr.report_unloaded()
                logger.info("/models/unload 拦截器：已通知平台显存释放")
            except Exception as e:
                logger.warning("/models/unload 拦截器：通知平台显存释放失败（不阻断）: %s", e)

        response_headers = _strip_hop_by_hop_headers(dict(response.headers))
        response_headers.pop("content-encoding", None)
        response_headers.pop("Content-Encoding", None)
        return JSONResponse(
            status_code=response.status_code,
            content=_json.loads(response_body) if response_body else {},
            headers=response_headers,
        )


def _mount_llama_server_proxy(app: FastAPI) -> None:
    """挂载反向代理路由到 llama-server 自带 UI。

    v0.3.0 关键设计：
        - catch-all 路由 /{path:path}，必须最后挂载（在 8 端点 + /output 之后）
        - 支持 HTTP 方法：GET / POST / PUT / DELETE / OPTIONS / PATCH
        - 支持 SSE 流式透传（llama-server /completion 是 SSE）
        - 支持 WebSocket 透传（llama-server MCP 客户端用 WS）
        - 未 /load 时返回友好 HTML 提示页（503）
        - 排除已注册的 8 端点 + /output + /openapi.json + /docs

    透传策略：
        - 请求：透传 method / path / query / headers / body
        - 响应：透传 status_code / headers / body（流式）
        - 跳过 hop-by-hop headers（host / content-length / transfer-encoding 等）

    注意：本函数仅声明闭包路由处理器，不立即访问 _engine / process_manager。
    路由处理器在请求到达时才调用 get_engine()，此时 lifespan 已初始化引擎。
    """
    # ===== HTTP 反向代理 =====
    async def _ensure_subprocess_running() -> Optional[HTMLResponse]:
        """确保 llama-server 子进程已启动（反代路由 auto-load 辅助函数）。

        v0.3.1 语义：
            - 用户访问 UI 时自动启动子进程（router mode）
            - **不预加载任何模型**：variant=None，等第一次 chat 请求时由 router mode 自动加载
            - 行为对齐 foleycrafter/voxcpm 的 Gradio UI auto-/load 逻辑

        v0.3.7+：委托给统一的 auto_load_if_needed helper（含状态机恢复），
        避免子进程死了但状态机卡在 READY 时直接 load 触发非法转换。

        若子进程已运行，返回 None（继续反代）。
        若子进程未运行，自动调 _load_impl(variant=None) 启动；
          - 启动成功返回 None（继续反代）
          - 启动失败返回 503 HTMLResponse（显示错误提示页）
        """
        engine = get_engine()
        try:
            await auto_load_if_needed(engine)
            return None
        except Exception as e:
            logger.exception("反代 auto-load 失败：%s", e)
            # 返回错误提示页（复用 _UNREADY_HTML，但加错误信息）
            err_html = _UNREADY_HTML.replace(
                "请先在平台详情页点击 启动 按钮（调 POST /load）。",
                f"自动启动失败：{e}<br><br>请检查模型文件是否完整，或手动调 POST /load。",
            )
            return HTMLResponse(content=err_html, status_code=503)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    )
    async def reverse_proxy(path: str, request: Request):
        """反向代理到 llama-server 内部端口。

        v0.3.0：子进程未启动时自动调 _load_impl（auto-load），
        而非返回 503 提示页。对齐 foleycrafter/voxcpm 的 UI auto-/load 规范。

        Args:
            path: 路径（不含前导 /，如 "v1/chat/completions"）
            request: FastAPI Request 对象
        """
        # 排除 OpenAPI / Swagger UI 路径（FastAPI 自带，但 catch-all 会拦截）
        # 注意：favicon.ico 不在此列，应反代到 llama-server（上游 UI 自带 favicon）
        if path in ("openapi.json", "docs", "redoc"):
            raise HTTPException(status_code=404, detail=f"未找到路径: /{path}")

        # v0.4.x sw.js 短路：浏览器打开 web UI 后 service worker 周期请求 /sw.js，
        # llama-server 没有 sw.js 恒 404。直接短路返回（不转发），
        # 避免 httpx INFO 请求日志刷屏 + 无谓转发开销。
        if path == "sw.js":
            return JSONResponse(status_code=404, content={"error": "not found"})

        # auto-load：子进程未启动时自动启动
        err = await _ensure_subprocess_running()
        if err is not None:
            return err

        engine = get_engine()
        pm_local = engine.process_manager

        # 构造转发请求
        target_url = f"{pm_local.internal_url}/{path}"
        if request.url.query:
            target_url = f"{target_url}?{request.url.query}"

        # 透传请求头（移除 hop-by-hop headers）
        forward_headers = _strip_hop_by_hop_headers(dict(request.headers))

        # 读取请求体
        body = await request.body()

        # v0.3.4+ 反代层 alias 翻译：仅对 OpenAI 兼容端点（chat/completions 等）
        # 翻译后 body 用于转发给 llama-server，原 body 不修改
        # 翻译失败（model 不是 alias）原样透传，让 llama-server 自己处理
        if path in _ALIAS_TRANSLATED_PATHS and body:
            translated_body, translated = _translate_model_alias(engine, body)
            if translated:
                logger.info(
                    "反代 alias 翻译: %s -> %s (path=%s)",
                    _safe_extract_model(body),
                    _safe_extract_model(translated_body),
                    path,
                )
                body = translated_body
                # 同步更新 Content-Length，避免下游 httpx 重新计算导致请求异常
                forward_headers["content-length"] = str(len(body))

            # v0.3.5+ 反代层默认注入 chat_template_kwargs（方案 D）
            # 解决 qwen3.6-uncensored 等模型多轮对话时思维链累积导致 400 的问题
            # 如果工具未显式传 chat_template_kwargs，平台自动注入 {"enable_thinking": false}
            # 工具显式传则使用工具的值（覆盖默认）
            body_json = _json.loads(body)
            if isinstance(body_json, dict) and "chat_template_kwargs" not in body_json:
                body_json["chat_template_kwargs"] = {"enable_thinking": False}
                body = _json.dumps(body_json, ensure_ascii=False).encode("utf-8")
                forward_headers["content-length"] = str(len(body))
                logger.debug("反代默认注入 chat_template_kwargs: %s", body_json["chat_template_kwargs"])

        logger.debug("反代转发: %s %s -> %s", request.method, path, target_url)

        # 流式转发（支持 SSE）
        # v0.4.x 读超时 600→3600：220K+ token 大上下文 prompt 处理期间 llama-server
        # 不发送任何 SSE 字节（实测静默 650s+），600s 读超时必崩（ReadTimeout，
        # 见 GATEWAY_LLAMA_TIMEOUT_COMPAT_ANALYSIS.md §2.1）。3600s 覆盖 262K
        # 满上下文 prompt eval（~770s）+ 排队 + 生成期 chunk 间隔。
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(3600.0, connect=10.0),
            follow_redirects=False,
            trust_env=False,  # 平台内部回环，不走系统代理
        )
        try:
            req = client.build_request(
                request.method,
                target_url,
                headers=forward_headers,
                content=body if body else None,
            )
            response = await client.send(req, stream=True)
        except Exception as e:
            await client.aclose()
            logger.exception("反代转发失败: %s %s", request.method, path)
            return JSONResponse(
                status_code=502,
                content={"error": {
                    "type": "ProxyError",
                    "message": f"反向代理 llama-server 失败: {e}",
                    "code": 502,
                }},
            )

        # 透传响应头（移除 hop-by-hop + content-encoding）
        # 注意：httpx 的 aiter_bytes() 默认返回已解压的字节流，
        # 若保留 Content-Encoding 头，浏览器会尝试二次解压 → ERR_CONTENT_DECODING_FAILED
        # 因此必须移除 content-encoding，让浏览器按原始字节接收
        response_headers = _strip_hop_by_hop_headers(dict(response.headers))
        response_headers.pop("content-encoding", None)
        response_headers.pop("Content-Encoding", None)

        # 流式响应（SSE / chunked / 普通 JSON 都走这个分支）
        async def stream_body():
            try:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=response.status_code,
            headers=response_headers,
        )

    # ===== WebSocket 反向代理（llama-server MCP 客户端用 WS）=====
    @app.websocket("/{path:path}")
    async def websocket_proxy(websocket: WebSocket, path: str):
        """WebSocket 反向代理到 llama-server。

        用于 llama-server MCP 客户端 / 实时流式场景。
        v0.3.0：子进程未启动时自动调 _load_impl（auto-load，同 HTTP 反代）。
        v0.3.7+：使用统一的 auto_load_if_needed helper，含状态机恢复逻辑。
        """
        engine = get_engine()
        pm_local = engine.process_manager

        # v0.3.7+ auto-load：使用统一 helper（含状态机恢复，避免 ready→loading 非法转换）
        if not pm_local.is_running or pm_local.internal_url is None:
            logger.info(
                "WS 反代 auto-load 触发：子进程未启动，使用统一 helper 启动"
                "（v0.3.1+ 不预加载，等第一次交互再加载）"
            )
            try:
                await auto_load_if_needed(engine)
                logger.info("WS 反代 auto-load 成功：子进程已启动（未加载模型）")
            except Exception as e:
                logger.exception("WS 反代 auto-load 失败：%s", e)
                await websocket.close(code=1011, reason=f"llama-server 启动失败: {e}")
                return

        # 接受客户端 WS 连接
        await websocket.accept()

        # 构造到 llama-server 的 WS URL
        target_ws_url = f"{pm_local.internal_url.replace('http', 'ws')}/{path}"
        if websocket.query_params:
            query_str = "&".join(f"{k}={v}" for k, v in websocket.query_params.items())
            target_ws_url = f"{target_ws_url}?{query_str}"

        logger.debug("WS 反代: /%s -> %s", path, target_ws_url)

        # 用 httpx.AsyncClient 升级到 WebSocket
        # httpx 本身不支持 WS，用 websockets 库（FastAPI 已依赖）
        try:
            import websockets
        except ImportError:
            logger.error("websockets 库未安装，无法反代 WS")
            await websocket.close(code=1011, reason="WS 代理依赖缺失")
            return

        # 双向桥接：client <-> llama-server
        try:
            async with websockets.connect(
                target_ws_url,
                additional_headers=dict(websocket.headers),
                max_size=None,
            ) as upstream_ws:
                import asyncio

                async def client_to_upstream():
                    """客户端 → llama-server。"""
                    try:
                        while True:
                            msg = await websocket.receive()
                            if msg.get("type") == "websocket.disconnect":
                                break
                            if "bytes" in msg and msg["bytes"] is not None:
                                await upstream_ws.send(msg["bytes"])
                            elif "text" in msg and msg["text"] is not None:
                                await upstream_ws.send(msg["text"])
                    except WebSocketDisconnect:
                        pass

                async def upstream_to_client():
                    """llama-server → 客户端。"""
                    try:
                        async for msg in upstream_ws:
                            if isinstance(msg, bytes):
                                await websocket.send_bytes(msg)
                            else:
                                await websocket.send_text(msg)
                    except Exception:
                        pass

                # 并发执行两个方向的转发
                await asyncio.gather(
                    client_to_upstream(),
                    upstream_to_client(),
                    return_exceptions=True,
                )
        except Exception as e:
            logger.warning("WS 反代异常: %s", e)
        finally:
            try:
                await websocket.close()
            except Exception:
                pass


def _strip_hop_by_hop_headers(headers: dict) -> dict:
    """移除 hop-by-hop headers（不能被代理转发的头）。

    参考 RFC 7230 §6.1：以下 headers 必须由代理移除：
        Connection / Keep-Alive / Proxy-Authenticate / Proxy-Authorization /
        TE / Trailer / Transfer-Encoding / Upgrade

    另外移除 Host / Content-Length（由下游 client 重新计算）。
    """
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
    return {
        k: v for k, v in headers.items()
        if k.lower() not in hop_by_hop
    }


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    # 解析命令行参数
    _config = parse_args()

    # 配置日志（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info(
        "启动 Llama.cpp 引擎 (v0.3.0 router mode): host=%s, port=%d",
        _config.host, _config.port,
    )

    # 创建 app
    app = create_app(_config)

    # 用 uvicorn 启动
    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
