"""Llama.cpp 引擎专用数据模型（v0.3.0 router mode 重构版）。

复用 SDK 的标准 schema，仅扩展 chat 引擎特有字段：
    - ChatMessage / InferInputs：OpenAI 兼容的 chat 输入
    - InferRequest：扩展 SDK InferRequest，加 messages / max_tokens / temperature 等
    - LlamaVariantInfo：扩展 SDK VariantInfo，加 GGUF 特有字段
    - LlamaStatusResponse：扩展 SDK StatusResponse，加子进程端口/PID 等
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from aireliable_engine import (
    StatusResponse,
    VariantInfo,
)


# ===== Chat Inputs（OpenAI 兼容）=====

class ChatMessage(BaseModel):
    """OpenAI 兼容的 chat 消息。

    v0.4.x 工具调用透传：补齐 tool_calls / tool_call_id 字段，content 改 Optional——
    多轮工具对话历史（assistant 消息的 tool_calls、role=tool 消息）之前会被
    Pydantic 静默丢弃（多轮 function calling 会话断裂）。
    """

    role: str                                       # user / assistant / system / tool
    # 文本或多模态内容列表；assistant 工具调用消息的 content 可为 None
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None    # assistant 消息的工具调用列表
    tool_call_id: Optional[str] = None                   # role=tool 消息对应的调用 ID


class InferInputs(BaseModel):
    """Llama.cpp 推理输入参数（OpenAI 兼容）。

    router mode 下 model 字段必填（指定要路由到哪个已加载模型）。
    """

    messages: list[ChatMessage]
    model: Optional[str] = None                     # router mode 路由用的模型名（variant_id 或 alias）
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True                             # llama-server 内部流式
    seed: Optional[int] = None
    stop: Optional[list[str]] = None                # 停止序列
    # v0.4.x G11：OpenAI 标准采样参数透传（gateway 经 inputs 下发，
    # forward_chat **extra 直达 llama-server 请求体）
    frequency_penalty: Optional[float] = None       # -2.0 ~ 2.0
    presence_penalty: Optional[float] = None        # -2.0 ~ 2.0
    # v0.4.x OpenAI 工具调用透传（function calling，agent 客户端如 Hermes 依赖）：
    # llama-server 收到 tools 后注入工具调用模板并把模型输出解析为结构化 tool_calls
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    # v0.4.x response_format 透传（JSON mode / 结构化输出）：
    # OpenAI 规范 {"type": "json_object"} 等，llama-server 原生支持
    response_format: Optional[dict[str, Any]] = None
    # llama-server / vLLM 扩展：透传到 chat 模板的额外参数
    # 典型用途：{"enable_thinking": false} 禁用 Qwen3 / uncensored 模型的思维链
    chat_template_kwargs: Optional[dict[str, Any]] = None


# ===== Infer Request =====

class InferRequest(BaseModel):
    """Llama.cpp 推理请求。

    v0.3.0 router mode 重构：
        - variant 字段保留兼容（用于自动加载模型），可选
        - inputs.model 字段优先级高于 variant（router mode 路由）
    """

    variant: Optional[str] = None                   # 可选：若未加载则自动调 /models/load
    inputs: InferInputs
    loras: list[Any] = Field(default_factory=list)  # v0.1 忽略
    task_id: Optional[str] = None


# ===== Variant Info（Llama.cpp 专用）=====

class VariantCtxSchema(BaseModel):
    """per-variant ctx 配置（v0.3.3+ 新增，与 manifest_schema.VariantCtxSchema 对齐）。

    用于前端 UI 滑块 + 引擎启动 INI 文件 ctx-size 参数。

    Attributes:
        recommended: 推荐的 ctx-size（默认值，平衡显存和性能）。如 32768。
        min: 最小允许的 ctx-size（防止过小导致功能异常）。如 4096。
        max: 最大允许的 ctx-size（受限于显存和 n_ctx_train）。如 262144。
    """

    recommended: int = Field(..., ge=1)
    min: int = Field(..., ge=1)
    max: int = Field(..., ge=1)


class LlamaVariantInfo(VariantInfo):
    """Llama.cpp variant 信息（继承 SDK + 扩展 GGUF 特有字段）。"""

    path: str = ""                                              # GGUF 文件名（相对 checkpoint_dir）
    file_size: Optional[int] = None                             # 文件字节数
    supports_mmproj: bool = False                               # 是否支持多模态
    mmproj_path: Optional[str] = None                           # 多模态投影文件路径
    # v0.4.x 修复 /capabilities 500：fallback 对齐 manifest 后 alias 为
    # list[str]（[文件名stem, *manifest aliases]），旧声明 Optional[str]
    # 在 pydantic 校验时报 string_type 错误
    alias: Optional[Union[str, list[str]]] = None                # 短别名（str 或列表）
    # v0.3.3+ 新增：per-variant ctx 配置（前端 UI 滑块 + 引擎启动 INI 用）
    ctx: Optional[VariantCtxSchema] = None


# ===== Status（Llama.cpp 专用）=====

class LlamaStatusResponse(StatusResponse):
    """Llama.cpp 状态响应（继承 SDK + 扩展子进程信息）。"""

    internal_port: Optional[int] = None                         # llama-server 隐藏端口
    llama_pid: Optional[int] = None                             # llama-server 子进程 PID
    models_max: Optional[int] = None                            # router mode 最大并存数
    loaded_models: list[str] = Field(default_factory=list)      # 已加载的模型列表（router mode 多模型）
    ngl: Optional[int] = None                                   # GPU 层数（router mode 全局默认）
    ctx: Optional[int] = None                                   # 上下文长度（router mode 全局默认）


# ===== OpenAI 兼容出参（/infer 透传时无需使用，保留兼容）=====

class ChatCompletionChoice(BaseModel):
    """OpenAI chat completion 单个 choice。"""

    index: int = 0
    message: dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI 兼容的 chat completion 响应。"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[dict[str, int]] = None
