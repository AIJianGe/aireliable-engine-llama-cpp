"""manifest.yaml variant.ctx 块更新工具（v0.3.3+ 新增）。

扫描 manifest.yaml 中声明的 variant 对应的 GGUF 文件，读取其元数据中的
`general.context_length`（n_ctx_train），调用 `suggest_ctx_config` 生成
推荐配置，并原子写入 manifest.yaml 的 `variant.ctx` 块。

用途：
    1. 新增 GGUF 模型后批量更新 manifest 的 ctx 块
    2. 平台 API 不可达时手动执行（CLI 工具）
    3. 平台 core 层 manifest 编辑器调用（per-variant ctx 持久化）

使用方式：
    # 命令行（手动执行）
    uv run --no-sync --project envs/llm-llama-cpp \\
        python -m engines.llama_cpp_9cdb76f8.update_manifest_ctx

    # Python API（平台 core 调用）
    from engines.llama_cpp_9cdb76f8.update_manifest_ctx import update_manifest_ctx
    update_manifest_ctx(manifest_path, models_root)

规范参考：
    - VRAM_MANAGEMENT_V033_ANALYSIS.md §四 T9
    - PR/architecture.md §3.35 per-variant ctx 配置（v0.3.3+）
    - project-rules.md §2.1 manifest.yaml model_files 必须覆盖 load() 依赖的所有文件
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

from .gguf_metadata_reader import get_n_ctx_train, suggest_ctx_config

logger = logging.getLogger(__name__)


def update_manifest_ctx(
    manifest_path: Path,
    models_root: Path,
    force: bool = False,
) -> dict:
    """扫描 GGUF 文件并更新 manifest.yaml 的 variant.ctx 块。

    Args:
        manifest_path: manifest.yaml 文件路径
        models_root: 模型根目录（用于解析 variant.path）
        force: True 时强制覆盖已有 ctx 块；False 时仅填充缺失的 ctx 块

    Returns:
        dict: 更新统计 {
            "total_variants": int,
            "updated": int,
            "skipped_has_ctx": int,
            "skipped_not_found": int,
            "skipped_no_nctx": int,
            "errors": list[str],
        }
    """
    manifest_path = Path(manifest_path)
    models_root = Path(models_root)

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 文件不存在: {manifest_path}")

    # 读取 manifest
    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models_block = data.get("models", {}) or {}
    variants = models_block.get("variants", []) or []

    stats = {
        "total_variants": len(variants),
        "updated": 0,
        "skipped_has_ctx": 0,
        "skipped_not_found": 0,
        "skipped_no_nctx": 0,
        "errors": [],
    }

    modified = False

    for v in variants:
        v_name = v.get("name", "<unknown>")
        v_path = v.get("path", "")

        # force=False 时跳过已有 ctx 块的 variant
        if not force and v.get("ctx") is not None:
            stats["skipped_has_ctx"] += 1
            logger.debug("variant %s 已有 ctx 块，跳过（force=False）", v_name)
            continue

        # 解析 GGUF 文件路径
        if not v_path:
            stats["errors"].append(f"variant {v_name} 缺少 path 字段")
            continue

        gguf_path = models_root / v_path
        if not gguf_path.exists():
            stats["skipped_not_found"] += 1
            logger.warning(
                "variant %s 的 GGUF 文件不存在，跳过: %s",
                v_name, gguf_path,
            )
            continue

        # 读取 GGUF n_ctx_train
        try:
            n_ctx_train = get_n_ctx_train(gguf_path)
        except Exception as e:
            stats["errors"].append(
                f"variant {v_name} 读取 GGUF 元数据失败: {e}"
            )
            continue

        if not n_ctx_train or n_ctx_train <= 0:
            stats["skipped_no_nctx"] += 1
            logger.warning(
                "variant %s 的 GGUF 元数据无 general.context_length，跳过: %s",
                v_name, gguf_path,
            )
            continue

        # 生成推荐 ctx 配置
        cfg = suggest_ctx_config(n_ctx_train)
        v["ctx"] = cfg
        stats["updated"] += 1
        modified = True

        logger.info(
            "variant %s 更新 ctx: n_ctx_train=%d → recommended=%d min=%d max=%d",
            v_name, n_ctx_train,
            cfg["recommended"], cfg["min"], cfg["max"],
        )

    # 原子写入 manifest
    if modified:
        _atomic_write_yaml(manifest_path, data)
        logger.info("manifest.yaml 已更新: %s", manifest_path)
    else:
        logger.info("manifest.yaml 无需更新（无 variant 需要填充 ctx）")

    return stats


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """原子写入 YAML 文件（避免写入失败导致文件损坏）。

    策略：先写入临时文件，再 os.replace 覆盖原文件（原子操作）。
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # 用 NamedTemporaryFile 在同目录创建临时文件（确保同分区，os.replace 才原子）
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        os.replace(tmp_path, path)
    except Exception:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main() -> None:
    """CLI 入口：扫描 manifest 的 variant GGUF 文件并更新 ctx 块。"""
    parser = argparse.ArgumentParser(
        description="更新 manifest.yaml 的 variant.ctx 块（v0.3.3+）",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="manifest.yaml 路径（默认: 引擎目录下 manifest.yaml）",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="模型根目录（默认: 环境变量 AIRELIABLE_MODELS_ROOT 或 ./models）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已有 ctx 块（默认仅填充缺失的）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="日志级别（DEBUG/INFO/WARNING/ERROR）",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 默认 manifest 路径：本脚本所在目录的 manifest.yaml
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = Path(__file__).parent / "manifest.yaml"

    # 默认 models_root
    if args.models_root:
        models_root = Path(args.models_root)
    else:
        models_root = Path(
            os.environ.get("AIRELIABLE_MODELS_ROOT", "models")
        )

    logger.info("==== 开始更新 manifest variant.ctx ====")
    logger.info("manifest: %s", manifest_path)
    logger.info("models_root: %s", models_root)
    logger.info("force: %s", args.force)

    try:
        stats = update_manifest_ctx(manifest_path, models_root, force=args.force)
    except Exception as e:
        logger.exception("更新失败: %s", e)
        sys.exit(1)

    logger.info("==== 更新完成 ====")
    logger.info("总 variant 数: %d", stats["total_variants"])
    logger.info("已更新: %d", stats["updated"])
    logger.info("跳过（已有 ctx）: %d", stats["skipped_has_ctx"])
    logger.info("跳过（GGUF 不存在）: %d", stats["skipped_not_found"])
    logger.info("跳过（无 n_ctx_train）: %d", stats["skipped_no_nctx"])
    if stats["errors"]:
        logger.warning("错误 (%d 项):", len(stats["errors"]))
        for err in stats["errors"]:
            logger.warning("  - %s", err)
        sys.exit(2)


if __name__ == "__main__":
    main()
