"""GGUF 元数据读取器（v0.3.3+ 新增）。

读取 GGUF 文件头部的元数据，提取关键信息：
    - general.context_length（n_ctx_train）：模型训练时的最大上下文长度
    - general.name：模型名称
    - general.architecture：模型架构（如 llama / qwen2moe）

用于：
    1. 生成 manifest.yaml 的 variant.ctx 块（推荐 ctx-size）
    2. 启动时校验 --ctx-size 不超过 n_ctx_train
    3. 前端 UI 滑块的 max 值

GGUF 文件格式：
    - 魔数：0x46554747（"GGUF" 小端序）
    - 版本：uint32
    - 张量数：uint64
    - 元数据 KV 数：uint64
    - 元数据 KV 列表：[key_type, key, value_type, value] 重复

详见：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# GGUF 元数据类型枚举
# 详见 https://github.com/ggerganov/ggml/blob/master/include/ggml.h
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGUF 魔数（小端序 "GGUF"）
GGUF_MAGIC = 0x46554747


class GGUFMetadataError(Exception):
    """GGUF 元数据读取错误。"""


def _read_gguf_string(f) -> str:
    """读取 GGUF 格式的字符串（uint64 长度 + UTF-8 字节）。"""
    raw_len = f.read(8)
    if len(raw_len) < 8:
        raise GGUFMetadataError("读取字符串长度失败：文件结束")
    str_len = struct.unpack("<Q", raw_len)[0]
    if str_len > 10 * 1024 * 1024:  # 10MB 安全阈值
        raise GGUFMetadataError(f"字符串长度异常: {str_len}")
    raw_bytes = f.read(str_len)
    if len(raw_bytes) < str_len:
        raise GGUFMetadataError("读取字符串内容失败：文件结束")
    return raw_bytes.decode("utf-8", errors="replace")


def _read_gguf_value(f, value_type: int) -> Any:
    """根据类型 ID 读取 GGUF 元数据值。"""
    if value_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif value_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif value_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif value_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif value_type == GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    elif value_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif value_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif value_type == GGUF_TYPE_ARRAY:
        # 数组：uint32 array_type + uint64 len + len 个值
        raw_arr_type = f.read(4)
        if len(raw_arr_type) < 4:
            raise GGUFMetadataError("读取数组类型失败：文件结束")
        array_type = struct.unpack("<I", raw_arr_type)[0]
        raw_arr_len = f.read(8)
        if len(raw_arr_len) < 8:
            raise GGUFMetadataError("读取数组长度失败：文件结束")
        arr_len = struct.unpack("<Q", raw_arr_len)[0]
        # tokenizer.ggml.tokens 等数组可能 15-20 万元素，放宽到 500 万
        if arr_len > 5_000_000:
            raise GGUFMetadataError(f"数组长度异常: {arr_len}")
        return [_read_gguf_value(f, array_type) for _ in range(arr_len)]
    else:
        raise GGUFMetadataError(f"未知的 GGUF 元数据类型: {value_type}")


def read_gguf_metadata(gguf_path: Path) -> dict[str, Any]:
    """读取 GGUF 文件的所有元数据 KV。

    Args:
        gguf_path: GGUF 文件路径

    Returns:
        dict: 元数据键值对（如 {"general.context_length": 32768, ...}）

    Raises:
        GGUFMetadataError: 文件格式错误或读取失败
        FileNotFoundError: 文件不存在
    """
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF 文件不存在: {gguf_path}")

    metadata: dict[str, Any] = {}

    with open(gguf_path, "rb") as f:
        # 1. 读取魔数
        raw_magic = f.read(4)
        if len(raw_magic) < 4:
            raise GGUFMetadataError("文件过小，无法读取魔数")
        magic = struct.unpack("<I", raw_magic)[0]
        if magic != GGUF_MAGIC:
            raise GGUFMetadataError(
                f"非 GGUF 文件（魔数 0x{magic:08X}，期望 0x{GGUF_MAGIC:08X}）"
            )

        # 2. 读取版本
        raw_version = f.read(4)
        if len(raw_version) < 4:
            raise GGUFMetadataError("读取版本失败：文件结束")
        version = struct.unpack("<I", raw_version)[0]
        if version < 1 or version > 3:
            raise GGUFMetadataError(f"不支持的 GGUF 版本: {version}")
        metadata["_gguf_version"] = version

        # 3. 读取张量数（uint64）
        raw_tensor_count = f.read(8)
        if len(raw_tensor_count) < 8:
            raise GGUFMetadataError("读取张量数失败：文件结束")
        tensor_count = struct.unpack("<Q", raw_tensor_count)[0]
        metadata["_tensor_count"] = tensor_count

        # 4. 读取元数据 KV 数（uint64）
        raw_kv_count = f.read(8)
        if len(raw_kv_count) < 8:
            raise GGUFMetadataError("读取元数据 KV 数失败：文件结束")
        kv_count = struct.unpack("<Q", raw_kv_count)[0]
        if kv_count > 10000:  # 安全阈值
            raise GGUFMetadataError(f"元数据 KV 数异常: {kv_count}")

        # 5. 逐个读取 KV
        # GGUF KV 格式（参考 https://github.com/ggerganov/ggml/blob/master/docs/gguf.md）：
        #   - key: gguf_string（uint64 长度 + UTF-8 字节，key 永远是字符串）
        #   - value_type: uint32（枚举值，见 GGUF_TYPE_* 常量）
        #   - value: 根据 value_type 读取
        # 注意：没有 key_type 字段（key 永远是 string）
        for i in range(kv_count):
            try:
                # key（gguf_string，无 type 前缀）
                key = _read_gguf_string(f)

                # value_type（uint32）
                raw_value_type = f.read(4)
                if len(raw_value_type) < 4:
                    raise GGUFMetadataError(f"读取 KV[{i}] value_type 失败：文件结束")
                value_type = struct.unpack("<I", raw_value_type)[0]
                value = _read_gguf_value(f, value_type)

                metadata[key] = value
            except GGUFMetadataError as e:
                logger.warning("读取 KV[%d] 失败（跳过剩余）: %s", i, e)
                break
            except Exception as e:
                logger.warning("读取 KV[%d] 异常（跳过剩余）: %s", i, e)
                break

    return metadata


def get_n_ctx_train(gguf_path: Path) -> Optional[int]:
    """获取 GGUF 模型的训练上下文长度（n_ctx_train）。

    GGUF 元数据规范：context_length 字段按模型架构命名，不是 `general.context_length`。
    实际字段名是 `<architecture>.context_length`，如：
        - llama.context_length
        - qwen35moe.context_length
        - qwen2moe.context_length
        - gpt2.context_length

    因此需要先读 `general.architecture`，再查 `<arch>.context_length`。

    优化：找到两个字段后立即停止读取，避免解析 tokenizer 数组。

    Args:
        gguf_path: GGUF 文件路径

    Returns:
        int | None: n_ctx_train（如 32768）；读取失败返回 None
    """
    try:
        # 读取 general.architecture 和 <arch>.context_length，命中后立即停止
        metadata = _read_gguf_metadata_until(
            gguf_path,
            # stop_keys 在循环中动态判断（见 _read_gguf_metadata_until 实现），
            # 这里传一个空集合，靠 _read_gguf_metadata_until_with_arch 的逻辑判断
            stop_keys=set(),
            stop_predicate=_ctx_stop_predicate,
        )
        arch = metadata.get("general.architecture")
        if not arch:
            logger.debug("GGUF 元数据中无 general.architecture 字段: %s", gguf_path)
            return None

        ctx_key = f"{arch}.context_length"
        ctx_len = metadata.get(ctx_key)
        if isinstance(ctx_len, int) and ctx_len > 0:
            return ctx_len
        # 兜底：扫所有 key 找 *.context_length
        for k, v in metadata.items():
            if k.endswith(".context_length") and isinstance(v, int) and v > 0:
                return v
        logger.debug(
            "GGUF 元数据中无 %s.context_length 字段: %s",
            arch, gguf_path,
        )
        return None
    except Exception as e:
        logger.warning("读取 GGUF n_ctx_train 失败 %s: %s", gguf_path, e)
        return None


def _ctx_stop_predicate(metadata: dict[str, Any]) -> bool:
    """get_n_ctx_train 的停止谓词：已读到 architecture + 对应 context_length 时停止。

    判断逻辑：
        1. 已读到 general.architecture 且对应的 <arch>.context_length 已读到 → 停止
        2. 否则继续读
    """
    arch = metadata.get("general.architecture")
    if not arch:
        return False
    ctx_key = f"{arch}.context_length"
    return ctx_key in metadata


def _read_gguf_metadata_until(
    gguf_path: Path,
    stop_keys: set[str],
    stop_predicate: Optional[Any] = None,
) -> dict[str, Any]:
    """读取 GGUF 元数据，遇到停止条件后立即停止。

    优化版：避免解析 tokenizer 数组等大字段（只需 general.context_length 等小字段时）。

    Args:
        gguf_path: GGUF 文件路径
        stop_keys: 遇到这些 key 中的任一个并读完值后立即停止
        stop_predicate: 可选的谓词函数 `f(metadata) -> bool`，返回 True 时停止。
            适合需要基于已读字段做联合判断的场景（如先读 architecture 再判断
            context_length 是否已读到）。

    Returns:
        dict: 已读取的元数据键值对（可能不含 stop_keys 之外的键）
    """
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF 文件不存在: {gguf_path}")

    metadata: dict[str, Any] = {}

    with open(gguf_path, "rb") as f:
        # 1. 读取魔数
        raw_magic = f.read(4)
        if len(raw_magic) < 4:
            raise GGUFMetadataError("文件过小，无法读取魔数")
        magic = struct.unpack("<I", raw_magic)[0]
        if magic != GGUF_MAGIC:
            raise GGUFMetadataError(
                f"非 GGUF 文件（魔数 0x{magic:08X}，期望 0x{GGUF_MAGIC:08X}）"
            )

        # 2. 读取版本
        raw_version = f.read(4)
        if len(raw_version) < 4:
            raise GGUFMetadataError("读取版本失败：文件结束")
        version = struct.unpack("<I", raw_version)[0]
        if version < 1 or version > 3:
            raise GGUFMetadataError(f"不支持的 GGUF 版本: {version}")
        metadata["_gguf_version"] = version

        # 3. 读取张量数（uint64）
        raw_tensor_count = f.read(8)
        if len(raw_tensor_count) < 8:
            raise GGUFMetadataError("读取张量数失败：文件结束")
        metadata["_tensor_count"] = struct.unpack("<Q", raw_tensor_count)[0]

        # 4. 读取元数据 KV 数（uint64）
        raw_kv_count = f.read(8)
        if len(raw_kv_count) < 8:
            raise GGUFMetadataError("读取元数据 KV 数失败：文件结束")
        kv_count = struct.unpack("<Q", raw_kv_count)[0]
        if kv_count > 10000:
            raise GGUFMetadataError(f"元数据 KV 数异常: {kv_count}")

        # 5. 逐个读取 KV，遇到停止条件后立即停止
        for i in range(kv_count):
            try:
                key = _read_gguf_string(f)
                raw_value_type = f.read(4)
                if len(raw_value_type) < 4:
                    raise GGUFMetadataError(f"读取 KV[{i}] value_type 失败：文件结束")
                value_type = struct.unpack("<I", raw_value_type)[0]
                value = _read_gguf_value(f, value_type)
                metadata[key] = value

                # 命中 stop_keys 则立即停止
                if key in stop_keys:
                    break
                # 命中谓词则立即停止
                if stop_predicate is not None and stop_predicate(metadata):
                    break
            except GGUFMetadataError as e:
                logger.warning("读取 KV[%d] 失败（跳过剩余）: %s", i, e)
                break
            except Exception as e:
                logger.warning("读取 KV[%d] 异常（跳过剩余）: %s", i, e)
                break

    return metadata


def get_gguf_model_info(gguf_path: Path) -> dict[str, Any]:
    """获取 GGUF 模型的关键信息（用于 manifest 生成）。

    Args:
        gguf_path: GGUF 文件路径

    Returns:
        dict: 含以下字段：
            - name: 模型名称（general.name）
            - architecture: 模型架构（general.architecture）
            - context_length: 训练上下文长度（<arch>.context_length）
            - parameter_count: 参数量（general.parameter_count）
            - quantization_version: 量化版本（general.quantization_version）
    """
    info: dict[str, Any] = {
        "name": None,
        "architecture": None,
        "context_length": None,
        "parameter_count": None,
        "quantization_version": None,
    }
    try:
        metadata = read_gguf_metadata(gguf_path)
        info["name"] = metadata.get("general.name")
        arch = metadata.get("general.architecture")
        info["architecture"] = arch
        # context_length 字段名按架构命名（如 qwen35moe.context_length）
        if arch:
            ctx_len = metadata.get(f"{arch}.context_length")
            if isinstance(ctx_len, int) and ctx_len > 0:
                info["context_length"] = ctx_len
        # 兜底：扫所有 key 找 *.context_length
        if info["context_length"] is None:
            for k, v in metadata.items():
                if k.endswith(".context_length") and isinstance(v, int) and v > 0:
                    info["context_length"] = v
                    break
        param_count = metadata.get("general.parameter_count")
        if isinstance(param_count, (int, float)) and param_count > 0:
            info["parameter_count"] = param_count
        quant_ver = metadata.get("general.quantization_version")
        if isinstance(quant_ver, int):
            info["quantization_version"] = quant_ver
    except Exception as e:
        logger.warning("读取 GGUF 模型信息失败 %s: %s", gguf_path, e)
    return info


def suggest_ctx_config(n_ctx_train: int) -> dict[str, int]:
    """根据 n_ctx_train 生成推荐的 ctx 配置（用于 manifest.yaml）。

    策略：
        - recommended: min(n_ctx_train, 16384) — 平衡显存和性能的默认推荐
          （v0.3.6 用户决策：默认推荐值由 32768 调整为 16384，
           显存自适应推荐仍可在引擎启动时按显卡动态给出更高值）
        - min: 4096 — 最小可用上下文
        - max: n_ctx_train — 不超过模型训练上限

    Args:
        n_ctx_train: GGUF 元数据的 general.context_length

    Returns:
        dict: {"recommended": int, "min": int, "max": int}
    """
    if n_ctx_train <= 0:
        return {"recommended": 4096, "min": 512, "max": 4096}

    recommended = min(n_ctx_train, 16384)
    min_ctx = 4096 if n_ctx_train >= 4096 else max(512, n_ctx_train // 2)
    max_ctx = n_ctx_train

    return {
        "recommended": recommended,
        "min": min_ctx,
        "max": max_ctx,
    }
