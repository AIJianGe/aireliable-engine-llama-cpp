"""Llama.cpp 模型目录扫描器（v0.3.0 router mode 重构版 + v0.3.3 per-variant ctx）。

v0.3.0 改造：
    优先调用平台 API（`GET /engines/{id}/variants`）获取 variant 可用性，
    平台不可达时 fallback 到本地文件扫描。

    声明式检测（方案 B）：
        1. 调用 BaseEngine.get_variants() 获取平台 variant 列表（含 available 字段）
        2. 从 manifest 平台返回的 model_file_ids 关联 mmproj 文件
        3. 合并：平台判定 available + 引擎补充 mmproj 路径

    Fallback 检测（保留兼容性，测试环境无平台时使用）：
        1. 扫描 checkpoint_dir 下所有 .gguf 文件（跳过 mmproj-*.gguf）
        2. 按文件名解析 base_model / precision
        3. 模糊匹配 mmproj 文件
        4. 仅检查文件存在性（不检查大小，不比平台 API 严格）

v0.3.3 新增：
    - VariantInfo 新增 `ctx` 字段（per-variant ctx 配置）
    - 扫描时读取 GGUF 元数据的 `general.context_length`（n_ctx_train）
    - 调用 `gguf_metadata_reader.suggest_ctx_config` 生成推荐配置
    - 用于前端 UI 滑块 + 引擎启动 INI 文件 ctx-size 参数

规范参考：
    - PR/architecture.md §3.29 模型检测架构统一（方案 B）
    - PR/architecture.md §3.35 跨引擎显存管理 + per-variant ctx（v0.3.3+）
    - project-rules.md §3.3 本地 fallback 与平台 API 行为必须一致
    - project-rules.md §5.1 本地 fallback 只检查标志文件存在性
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .gguf_metadata_reader import get_n_ctx_train, suggest_ctx_config

logger = logging.getLogger(__name__)


# mmproj 文件名识别正则
MMPROJ_PATTERN = re.compile(r"^mmproj[_-].*\.gguf$", re.IGNORECASE)


@dataclass
class VariantCtxInfo:
    """per-variant ctx 配置（v0.3.3+ 新增）。

    与 manifest_schema.VariantCtxSchema 字段对齐，用于前端 UI 滑块 +
    引擎启动 INI 文件 ctx-size 参数。

    Attributes:
        recommended: 推荐的 ctx-size（默认值，平衡显存和性能）
        min: 最小允许的 ctx-size（防止过小导致功能异常）
        max: 最大允许的 ctx-size（受限于显存和 n_ctx_train）
    """

    recommended: int = 8192
    min: int = 4096
    max: int = 32768

    def to_dict(self) -> dict:
        return {
            "recommended": self.recommended,
            "min": self.min,
            "max": self.max,
        }


@dataclass
class VariantInfo:
    """扫描得到的 variant 信息（v0.3.0 简化版 + v0.3.3 ctx 扩展）。

    与 SDK VariantInfo 字段对齐，便于直接构造 CapabilitiesResponse。
    """

    name: str                                       # variant_id（如 qwen3.6-35b-a3b-q6-k）
    # v0.3.9+ 注解放宽：alias 支持 str 或 list[str]
    # （fallback 对齐 manifest 后为 list：[文件名stem, *manifest aliases]，双名可用）
    alias: "str | list[str]" = ""                   # 短别名（如 qwen3.6-q6）
    base_model: str = ""                            # 基础模型（如 qwen3.6-35b-a3b）
    precision: str = ""                             # 量化精度（如 q6_k / q8_0）
    loader: str = "gguf"
    vram_required: str = ""                         # 显存需求（字符串，如 "48GB"）
    path: Optional[Path] = None                     # GGUF 文件绝对路径
    file_size: int = 0                              # 字节数
    supports_mmproj: bool = False
    mmproj_path: Optional[Path] = None              # 多模态投影文件路径
    available: bool = True                          # 模型文件是否存在
    # v0.3.3+ 新增：per-variant ctx 配置（来自 manifest 或 GGUF 元数据推导）
    ctx: Optional[VariantCtxInfo] = None


class ModelScanner:
    """模型目录扫描器（v0.3.0 平台 API 优先 + 本地 fallback）。

    使用方式：
        # 异步（推荐，lifespan startup 阶段）
        scanner = ModelScanner(checkpoint_dir, engine=self)
        variants = await scanner.scan_variants_async()

        # 同步（__init__ 阶段 fallback）
        variants = scanner.scan_variants()
    """

    def __init__(self, checkpoint_dir: Path, engine=None):
        """初始化扫描器。

        Args:
            checkpoint_dir: 模型 checkpoint 目录（如 models/llm/checkpoints/llama/）
            engine: 可选的 BaseEngine 实例（v0.2.11+ 方案 B 注入）。
                传入时优先调用平台 API 检测 variant；
                为 None 时走 fallback 本地扫描逻辑（兼容旧测试）。
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.engine = engine
        # v0.3.9+（方案 C2）：最近一次扫描的来源模式
        #   "platform" = 平台 API（manifest 命名空间）
        #   "fallback" = 本地文件扫描（v0.3.9+ 已对齐 manifest 命名，见 _load_manifest_mapping）
        # 引擎 /load 前检查此值，fallback 模式且平台可达时触发自愈重扫
        self.last_scan_mode: str = ""
        # v0.3.9+（方案 A）：manifest filename → variant 声明映射（惰性加载缓存）
        self._manifest_map: Optional[dict] = None

    # ============ 异步扫描（平台 API 优先）============

    async def scan_variants_async(self) -> list[VariantInfo]:
        """v0.2.11+ 方案 B：异步扫描，优先调用平台 API。

        调用链：
            1. engine.get_variants() → 平台 API（声明式检测）
            2. 平台返回 None → fallback 到 _scan_locally()
            3. 合并平台 available 字段 + 本地 mmproj 路径解析

        日志规范（project-rules.md §3.4）：
            记录 variant 扫描路径（平台 API / 本地 fallback）及原因
        """
        if self.engine is not None:
            core_url = self.engine._get_platform_core_url()
            if core_url:
                logger.info("variant 扫描路径：平台 API（AIRELIABLE_CORE_URL=%s）", core_url)
                try:
                    platform_variants = await self.engine.get_variants()
                except Exception as e:
                    logger.warning(
                        "variant 扫描路径：本地 fallback（原因：平台 API 调用异常 %s）", e
                    )
                    return self._scan_locally()

                if platform_variants is not None:
                    self.last_scan_mode = "platform"
                    return self._merge_platform_with_local(platform_variants)
                # platform_variants is None：平台不可达，fallback
                logger.warning(
                    "variant 扫描路径：本地 fallback（原因：平台 API 返回 None，可能 AIRELIABLE_CORE_URL 未设置）"
                )
            else:
                logger.warning(
                    "variant 扫描路径：本地 fallback（原因：AIRELIABLE_CORE_URL 环境变量未设置）"
                )
        else:
            logger.warning(
                "variant 扫描路径：本地 fallback（原因：未注入 engine 实例）"
            )

        return self._scan_locally()

    # ============ 同步扫描（fallback）============

    def scan_variants(self) -> list[VariantInfo]:
        """同步扫描接口（向后兼容）。

        v0.3.0 注：若需使用方案 B 平台 API，请调用 scan_variants_async()。
        此同步方法走 fallback 本地扫描逻辑。
        """
        logger.info("variant 扫描路径：本地 fallback（同步模式，未调平台 API）")
        return self._scan_locally()

    def _scan_locally(self) -> list[VariantInfo]:
        # v0.3.9+：统一在此设置 fallback 模式标记（供引擎 /load 自愈重扫判断）
        self.last_scan_mode = "fallback"
        return self._scan_locally_impl()

    # ============ 平台 API 结果合并 ============

    def _merge_platform_with_local(self, platform_variants: list[dict]) -> list[VariantInfo]:
        """合并平台 variant 数据 + 本地 mmproj 路径解析。

        Args:
            platform_variants: 平台 API 返回的 variant 列表（含 available 字段）

        Returns:
            VariantInfo 列表（available 来自平台，mmproj 路径来自本地扫描）

        平台返回字段（参考 BaseEngine.get_variants）：
            name / alias / base_model / precision / loader / vram_required /
            available / model_file_ids / missing_file_ids

        v0.3.0 修复：平台 API 不返回 manifest 的 path 字段，
        需要 fallback 到本地 checkpoint_dir 按 precision + base_model 匹配 GGUF 文件。
        """
        result: list[VariantInfo] = []

        # 预扫描本地 checkpoint_dir 中所有 .gguf 文件（含 mmproj），
        # 供 _match_local_gguf 和 find_mmproj 使用
        local_gguf_files: list[Path] = []
        if self.checkpoint_dir.exists():
            local_gguf_files = [
                f for f in self.checkpoint_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".gguf"
            ]

        for pv in platform_variants:
            v_name = pv.get("name", "")
            available = bool(pv.get("available", False))

            # 1. 优先用平台返回的 path 字段
            gguf_path = self._resolve_gguf_path(pv)

            # 2. 平台不返回 path 时，fallback 到本地按 precision + base_model 匹配
            if gguf_path is None and available and local_gguf_files:
                gguf_path = self._match_local_gguf(pv, local_gguf_files)
                if gguf_path:
                    logger.debug(
                        "本地匹配 GGUF: variant=%s → %s",
                        v_name, gguf_path.name,
                    )

            # 解析 mmproj 路径（本地扫描）
            mmproj_path = None
            supports_mmproj = bool(pv.get("supports_mmproj", False))
            # 若 manifest 声明了 mmproj_filename（通过 model_file_ids 关联）
            mmproj_filename = pv.get("mmproj_filename") or pv.get("mmproj_path")

            # v0.3.2+ 修复：平台 API 可能不返回 supports_mmproj 字段（旧版本兼容）
            # 改为：只要 gguf_path 存在，就尝试 find_mmproj 模糊匹配
            # 如果匹配到 mmproj 文件，自动设 supports_mmproj=True
            if gguf_path is not None:
                base_model = pv.get("base_model", "")
                mmproj_path = self.find_mmproj(gguf_path, base_model, mmproj_filename)
                if mmproj_path:
                    supports_mmproj = True

            # 文件大小
            file_size = 0
            if gguf_path is not None and gguf_path.exists():
                try:
                    file_size = gguf_path.stat().st_size
                except Exception:
                    pass

            # v0.3.3+ 解析 per-variant ctx 配置
            # 优先级：平台 manifest 的 ctx 块 > GGUF 元数据推导 > None
            ctx_info = self._resolve_ctx_info(pv, gguf_path)

            info = VariantInfo(
                name=v_name,
                alias=pv.get("alias", ""),
                base_model=pv.get("base_model", ""),
                precision=pv.get("precision", ""),
                loader=pv.get("loader", "gguf"),
                vram_required=pv.get("vram_required", ""),
                path=gguf_path if available else None,
                file_size=file_size,
                supports_mmproj=supports_mmproj,
                mmproj_path=mmproj_path if available else None,
                available=available,
                ctx=ctx_info,
            )
            result.append(info)

            logger.info(
                "  - %s (alias=%s, base=%s, precision=%s, available=%s, size=%.2fGB, mmproj=%s, ctx=%s)",
                info.name, info.alias, info.base_model, info.precision,
                info.available, file_size / (1024 ** 3) if file_size else 0,
                info.mmproj_path.name if info.mmproj_path else "无",
                ctx_info.to_dict() if ctx_info else "无",
            )

        logger.info(
            "平台 API 扫描完成：声明 %d 个 variant，可用 %d 个",
            len(result), sum(1 for v in result if v.available),
        )
        return result

    def _resolve_gguf_path(self, platform_variant: dict) -> Optional[Path]:
        """从平台 variant 数据解析 GGUF 文件路径。

        平台返回的 path 字段是相对 models_root 的路径（如 llm/checkpoints/llama/xxx.gguf）。
        本地 checkpoint_dir 是 models_root / model_layout.base。
        """
        # 优先用 platform_variant["path"]（manifest 中的 path 字段）
        rel_path = platform_variant.get("path", "")
        if rel_path:
            # path 可能是相对 models_root 的完整路径，也可能是相对 checkpoint_dir 的文件名
            p = Path(rel_path)
            # 尝试 1：作为绝对路径
            if p.is_absolute() and p.exists():
                return p
            # 尝试 2：相对 checkpoint_dir
            local_path = self.checkpoint_dir / p.name
            if local_path.exists():
                return local_path
            # 尝试 3：相对 models_root（platform path 形如 "llm/checkpoints/llama/xxx.gguf"）
            # checkpoint_dir 父目录的父目录 = models_root
            try:
                models_root = self.checkpoint_dir.parents[2]
                full_path = models_root / rel_path
                if full_path.exists():
                    return full_path
            except (IndexError, OSError):
                pass

        return None

    def _resolve_ctx_info(
        self,
        platform_variant: dict,
        gguf_path: Optional[Path],
    ) -> Optional[VariantCtxInfo]:
        """v0.3.3+ 解析 per-variant ctx 配置。

        优先级：
            1. 平台 manifest 的 ctx 块（用户在前端调整过的值）
            2. GGUF 元数据推导（general.context_length → suggest_ctx_config）
            3. None（无法获取，使用全局默认）

        Args:
            platform_variant: 平台 API 返回的 variant dict（可能含 ctx 字段）
            gguf_path: 本地 GGUF 文件路径（用于读取元数据）

        Returns:
            VariantCtxInfo | None
        """
        # 1. 优先用平台返回的 ctx 块
        pv_ctx = platform_variant.get("ctx")
        if isinstance(pv_ctx, dict):
            try:
                recommended = int(pv_ctx.get("recommended", 0))
                min_ctx = int(pv_ctx.get("min", 0))
                max_ctx = int(pv_ctx.get("max", 0))
                if recommended > 0 and min_ctx > 0 and max_ctx > 0:
                    if min_ctx <= recommended <= max_ctx:
                        logger.debug(
                            "variant %s 使用平台 manifest ctx: %s",
                            platform_variant.get("name", ""), pv_ctx,
                        )
                        return VariantCtxInfo(
                            recommended=recommended,
                            min=min_ctx,
                            max=max_ctx,
                        )
                    logger.warning(
                        "variant %s 平台 ctx 配置非法（min/recommended/max 范围不一致）: %s",
                        platform_variant.get("name", ""), pv_ctx,
                    )
            except (TypeError, ValueError) as e:
                logger.warning(
                    "variant %s 平台 ctx 解析失败: %s, raw=%s",
                    platform_variant.get("name", ""), e, pv_ctx,
                )

        # 2. 从 GGUF 元数据推导
        if gguf_path is not None and gguf_path.exists():
            try:
                n_ctx_train = get_n_ctx_train(gguf_path)
                if n_ctx_train and n_ctx_train > 0:
                    cfg = suggest_ctx_config(n_ctx_train)
                    logger.debug(
                        "variant %s 从 GGUF 推导 ctx: n_ctx_train=%d → %s",
                        platform_variant.get("name", ""), n_ctx_train, cfg,
                    )
                    return VariantCtxInfo(
                        recommended=cfg["recommended"],
                        min=cfg["min"],
                        max=cfg["max"],
                    )
            except Exception as e:
                logger.warning(
                    "variant %s 读取 GGUF ctx 失败: %s",
                    platform_variant.get("name", ""), e,
                )

        # 3. None（使用全局默认 LLAMA_CTX）
        return None

    def _match_local_gguf(
        self, platform_variant: dict, local_files: list[Path]
    ) -> Optional[Path]:
        """平台不返回 path 时，按 precision + base_model 匹配本地 GGUF 文件。

        匹配规则：
            1. precision 转大写（如 q4_k_m → Q4_K_M）
            2. 文件名必须以 `-{precision}.gguf` 结尾（精确匹配，避免 Q6_K 误匹配 Q6_K_P）
            3. base_model 含 "uncensored" → 文件名必须含 "uncensored"（不区分大小写）
               base_model 不含 "uncensored" → 文件名不含 "uncensored"

        示例：
            platform: precision=q4_k_m, base_model=qwen3.5-35b-a3b
            → 匹配 Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf

            platform: precision=q6_k_p, base_model=qwen3.6-35b-a3b-uncensored
            → 匹配 Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf
        """
        precision = platform_variant.get("precision", "").upper().replace("-", "_")
        base_model = platform_variant.get("base_model", "").lower()
        is_uncensored = "uncensored" in base_model

        if not precision:
            return None

        # 精确后缀匹配：-{precision}.gguf（不区分大小写）
        suffix = f"-{precision}.gguf".lower()

        candidates: list[Path] = []
        for f in local_files:
            # 跳过 mmproj 文件
            if MMPROJ_PATTERN.match(f.name):
                continue
            # precision 后缀匹配
            if not f.name.lower().endswith(suffix):
                continue
            # base_model uncensored 区分
            name_lower = f.name.lower()
            if is_uncensored and "uncensored" not in name_lower:
                continue
            if not is_uncensored and "uncensored" in name_lower:
                continue
            candidates.append(f)

        if not candidates:
            logger.warning(
                "本地未找到匹配的 GGUF: precision=%s, base_model=%s, suffix=%s",
                precision, base_model, suffix,
            )
            return None
        if len(candidates) > 1:
            logger.warning(
                "多个 GGUF 文件匹配 precision=%s base=%s: %s，取第一个",
                precision, base_model, [c.name for c in candidates],
            )
        return candidates[0]

    # ============ 本地 fallback 扫描 ============

    def _scan_locally_impl(self) -> list[VariantInfo]:
        """fallback 本地扫描实现（无平台 API 时使用）。

        v0.3.0 策略（project-rules.md §5.1）：
            - 只检查 .gguf 文件存在性（标志文件）
            - 不检查文件大小（避免断点续传残文件被排除）
            - 不比平台 API 严格（保证 variant 列表可见，可触发下载入口）

        v0.3.9+（方案 A）：扫描结果按 manifest.yaml 对齐命名空间。
            文件名命中 manifest model_files 声明时，variant.name 使用 manifest
            variant 名（连字符，如 qwen3.8-27b-q6-k-l），文件名 stem 降级为
            alias 之一（双名可用）。详见 _load_manifest_mapping。
        """
        variants: list[VariantInfo] = []

        if not self.checkpoint_dir.exists():
            logger.warning("checkpoint 目录不存在: %s", self.checkpoint_dir)
            return variants

        # v0.3.9+ 方案 A：预加载 manifest 映射（空映射 = 未命中，保持原命名）
        manifest_map = self._load_manifest_mapping()

        # 扫描所有 .gguf 文件（跳过 mmproj-*.gguf）
        for entry in sorted(self.checkpoint_dir.iterdir()):
            if not (entry.is_file() and entry.suffix.lower() == ".gguf"):
                continue
            if MMPROJ_PATTERN.match(entry.name):
                continue  # 跳过 mmproj 文件

            try:
                info = self._parse_gguf_file(entry)
                if info:
                    # v0.3.9+ 方案 A：命中 manifest 则对齐命名空间
                    matched = manifest_map.get(entry.name.lower())
                    if matched is not None:
                        info = self._apply_manifest_naming(info, matched)
                    variants.append(info)
            except Exception as e:
                logger.warning("扫描条目失败: %s, err=%s", entry, e)

        logger.info(
            "本地 fallback 扫描完成：%s 发现 %d 个 variant（manifest 命中 %d 个）",
            self.checkpoint_dir, len(variants),
            sum(1 for v in variants if v.name in
                {m.get("name", "") for m in manifest_map.values()}),
        )
        return variants

    def _load_manifest_mapping(self) -> dict:
        """v0.3.9+ 方案 A：加载 manifest 的 filename → variant 声明映射。

        解析引擎自身 manifest.yaml（与 model_scanner.py 同目录）：
            1. models.model_files[]：id → filename（主权重条目）
            2. models.variants[]：name / alias / base_model / precision /
               vram_required / ctx / model_file_ids

        构建 filename(lower) → variant 声明 dict 映射，用于 fallback 扫描
        时对齐 manifest 命名空间（根治"平台 API 名 vs 文件名派生名"断裂）。

        Returns:
            dict: {filename_lower: {name, alias, base_model, precision,
                  vram_required, ctx}}；manifest 缺失/解析失败/pyyaml 不可用
                  时返回空 dict（fallback 保持原文件名派生命名，降级不报错）
        """
        if self._manifest_map is not None:
            return self._manifest_map

        mapping: dict = {}
        try:
            import yaml
            manifest_path = Path(__file__).resolve().parent / "manifest.yaml"
            if not manifest_path.exists():
                logger.warning(
                    "fallback 命名对齐：manifest 不存在（保持文件名派生命名）: %s",
                    manifest_path,
                )
                self._manifest_map = mapping
                return mapping

            with open(manifest_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            models = data.get("models") or {}
            model_files = models.get("model_files") or []
            variants_decl = models.get("variants") or []

            # id → filename 索引（跳过 mmproj 条目，主权重才会被扫描循环处理）
            id_to_filename: dict[str, str] = {}
            for mf in model_files:
                mf_id = str(mf.get("id", "")).strip()
                filename = str(mf.get("filename", "")).strip()
                if not mf_id or not filename:
                    continue
                if MMPROJ_PATTERN.match(filename):
                    continue  # mmproj 文件由 find_mmproj 单独关联
                id_to_filename[mf_id] = filename

            # variant 声明 → 主权重 filename 映射
            for v in variants_decl:
                v_name = str(v.get("name", "")).strip()
                if not v_name:
                    continue
                # 取该 variant 引用的第一个主权重 model_file
                for mf_id in v.get("model_file_ids") or []:
                    filename = id_to_filename.get(str(mf_id).strip())
                    if not filename:
                        continue
                    key = filename.lower()
                    if key in mapping:
                        logger.warning(
                            "fallback 命名对齐：文件 %s 被多个 variant 声明"
                            "（%s / %s），取第一个",
                            filename, mapping[key].get("name"), v_name,
                        )
                        continue
                    # ctx 块透传（VariantCtxInfo 在 _apply_manifest_naming 中构造）
                    mapping[key] = {
                        "name": v_name,
                        "alias": v.get("alias") or [],
                        "base_model": v.get("base_model", ""),
                        "precision": v.get("precision", ""),
                        "vram_required": v.get("vram_required", ""),
                        "ctx": v.get("ctx"),
                    }
                    break

            logger.info(
                "fallback 命名对齐：manifest 映射加载完成，%d 个 variant 声明"
                "（%s）",
                len(mapping), sorted(m.get("name", "") for m in mapping.values()),
            )
        except ImportError:
            logger.warning(
                "fallback 命名对齐：pyyaml 不可用，保持文件名派生命名"
            )
        except Exception as e:
            logger.warning(
                "fallback 命名对齐：manifest 解析失败（保持文件名派生命名）: %s", e
            )

        self._manifest_map = mapping
        return mapping

    def _apply_manifest_naming(
        self, info: VariantInfo, matched: dict
    ) -> VariantInfo:
        """v0.3.9+ 方案 A：用 manifest 声明覆盖 fallback 解析的 VariantInfo 字段。

        覆盖规则：
            - name：manifest variant 名（连字符，llama-server preset 名）
            - alias：[文件名stem.lower(), *manifest aliases]（双名可用，
              文件名 stem 保留为 alias 保证向后兼容）
            - base_model / precision / vram_required：manifest 值优先
              （fallback 的文件名正则解析可能拆错，manifest 是权威声明）
            - ctx：manifest ctx 块优先；无 manifest ctx 时保留 GGUF 元数据推导值

        Args:
            info: fallback 解析出的 VariantInfo（文件名派生命名）
            matched: manifest 映射条目（见 _load_manifest_mapping）

        Returns:
            VariantInfo: 对齐 manifest 命名后的新 VariantInfo（原对象不修改）
        """
        m_name = matched.get("name", "")
        if not m_name:
            return info

        # alias 列表：文件名 stem（向后兼容）+ manifest aliases（去重）
        stem_alias = info.name  # fallback 解析的 name 即文件名 stem.lower()
        m_aliases = matched.get("alias") or []
        if isinstance(m_aliases, str):
            m_aliases = [m_aliases]
        aliases = [stem_alias]
        for a in m_aliases:
            if a and a not in aliases:
                aliases.append(a)

        # ctx：manifest 优先，无则保留 GGUF 推导（info.ctx）
        m_ctx = matched.get("ctx")
        ctx_info = info.ctx
        if isinstance(m_ctx, dict):
            try:
                recommended = int(m_ctx.get("recommended", 0))
                min_ctx = int(m_ctx.get("min", 0))
                max_ctx = int(m_ctx.get("max", 0))
                if recommended > 0 and min_ctx > 0 and max_ctx > 0:
                    ctx_info = VariantCtxInfo(
                        recommended=recommended, min=min_ctx, max=max_ctx,
                    )
            except (TypeError, ValueError):
                logger.warning(
                    "fallback 命名对齐：variant %s manifest ctx 非法: %s",
                    m_name, m_ctx,
                )

        logger.info(
            "fallback 命名对齐：%s → %s（alias=%s，来源 manifest）",
            info.path.name if info.path else stem_alias, m_name, aliases,
        )

        return VariantInfo(
            name=m_name,
            alias=aliases,
            base_model=matched.get("base_model", "") or info.base_model,
            precision=matched.get("precision", "") or info.precision,
            loader=info.loader,
            vram_required=matched.get("vram_required", "") or info.vram_required,
            path=info.path,
            file_size=info.file_size,
            supports_mmproj=info.supports_mmproj,
            mmproj_path=info.mmproj_path,
            available=info.available,
            ctx=ctx_info,
        )

    def _parse_gguf_file(self, path: Path) -> Optional[VariantInfo]:
        """从文件名解析 GGUF variant 信息（fallback 模式）。

        示例：
            Qwen_Qwen3.6-35B-A3B-Q6_K.gguf
            → name=qwen_qwen3.6-35b-a3b-q6_k
            → base=Qwen_Qwen3.6-35B-A3B
            → precision=q6_k

        v0.3.3+：读取 GGUF 元数据填充 ctx 字段（无平台 API 时使用）
        """
        # 从文件名提取 base 和 quant
        quant_match = re.search(
            r"[._-](Q[0-9]+(?:_[A-Z0-9_]+)?)\.gguf$",
            path.name,
            re.IGNORECASE,
        )
        if not quant_match:
            logger.warning("GGUF 文件名无法识别量化类型: %s", path.name)
            return None

        quant_upper = quant_match.group(1).upper()
        quant_lower = quant_upper.lower()
        base = path.name[:quant_match.start()].rstrip("-_.")
        # 去掉前缀 Qwen_ 等
        if base.startswith("Qwen_"):
            base = base[5:]

        # variant_id 用文件名 stem 的小写
        variant_id = path.stem.lower()

        # 尝试匹配 mmproj
        mmproj_path = self.find_mmproj(path, base, None)
        supports_mmproj = mmproj_path is not None

        # 估算显存（基于文件大小 + 2GB 缓存）
        file_size = path.stat().st_size
        vram_gb = file_size / (1024 ** 3) + 2
        vram_required = f"{vram_gb:.0f}GB"

        # v0.3.3+ 从 GGUF 元数据推导 ctx 配置
        ctx_info: Optional[VariantCtxInfo] = None
        try:
            n_ctx_train = get_n_ctx_train(path)
            if n_ctx_train and n_ctx_train > 0:
                cfg = suggest_ctx_config(n_ctx_train)
                ctx_info = VariantCtxInfo(
                    recommended=cfg["recommended"],
                    min=cfg["min"],
                    max=cfg["max"],
                )
                logger.debug(
                    "fallback 扫描：variant %s 从 GGUF 推导 ctx: n_ctx_train=%d → %s",
                    variant_id, n_ctx_train, cfg,
                )
        except Exception as e:
            logger.warning("fallback 扫描：variant %s 读取 GGUF ctx 失败: %s", variant_id, e)

        return VariantInfo(
            name=variant_id,
            alias=variant_id,
            base_model=base,
            precision=quant_lower,
            loader="gguf",
            vram_required=vram_required,
            path=path,
            file_size=file_size,
            supports_mmproj=supports_mmproj,
            mmproj_path=mmproj_path,
            available=True,
            ctx=ctx_info,
        )

    # ============ mmproj 匹配 ============

    def find_mmproj(
        self,
        model_path: Path,
        base_model: str,
        explicit_filename: Optional[str] = None,
    ) -> Optional[Path]:
        """按规则匹配 mmproj 文件。

        v0.3.2+ 修复：区分 Uncensored 和非 Uncensored 模型，避免误匹配。
        - base_model 含 "uncensored" → 只匹配含 "uncensored" 的 mmproj 文件
        - base_model 不含 "uncensored" → 只匹配不含 "uncensored" 的 mmproj 文件

        Args:
            model_path: 模型文件路径（用于同目录查找）
            base_model: 模型族关键字（如 Qwen3.6-35B-A3B）
            explicit_filename: manifest 显式声明的文件名

        Returns:
            Path | None: 找到的 mmproj 路径
        """
        if not self.checkpoint_dir.exists():
            return None

        # 1. 优先用显式声明的文件名
        if explicit_filename:
            # explicit_filename 可能是路径，也可能是纯文件名
            explicit_path = self.checkpoint_dir / Path(explicit_filename).name
            if explicit_path.exists() and explicit_path.is_file():
                return explicit_path
            logger.warning("显式声明的 mmproj 文件不存在: %s", explicit_path)

        # 2. 按模型族关键字模糊匹配 mmproj-* 文件
        keyword = self._extract_mmproj_keyword(base_model)
        if not keyword:
            return None

        keyword_lower = keyword.lower()
        # 处理含下划线的情况：Qwen3.6 → qwen_qwen3.6 / qwen3.6
        keyword_variants = [
            keyword_lower,
            keyword_lower.replace("qwen", "qwen_qwen"),
        ]

        # v0.3.2+ 修复：区分 uncensored，避免非 uncensored 模型误匹配 uncensored mmproj
        is_uncensored = "uncensored" in base_model.lower()

        # 收集所有候选，按精确度排序后返回最佳匹配
        candidates: list[Path] = []
        for entry in sorted(self.checkpoint_dir.iterdir()):
            if not (entry.is_file() and MMPROJ_PATTERN.match(entry.name)):
                continue
            entry_lower = entry.name.lower()
            # uncensored 区分：base_model 和 mmproj 文件必须一致
            entry_is_uncensored = "uncensored" in entry_lower
            if is_uncensored != entry_is_uncensored:
                continue
            for kw in keyword_variants:
                if kw in entry_lower:
                    candidates.append(entry)
                    break

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # 多个候选时，优先选关键字最长的（最精确的匹配）
        candidates.sort(key=lambda p: len(p.name), reverse=True)
        logger.warning(
            "多个 mmproj 匹配 base_model=%s: %s，取 %s",
            base_model, [c.name for c in candidates], candidates[0].name,
        )
        return candidates[0]

    @staticmethod
    def _extract_mmproj_keyword(base_model: str) -> str:
        """从 base_model 提取用于匹配 mmproj 的关键字。

        v0.3.2+ 修复：不再统一移除 uncensored 后缀。
        - 含 uncensored → 保留完整 base_model（用于精确匹配 uncensored mmproj）
        - 不含 uncensored → 用原始 base_model（匹配时通过 find_mmproj 排除 uncensored 文件）

        示例：
            Qwen3.6-35B-A3B → Qwen3.6-35B-A3B
            Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive → Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
        """
        # 首字母大写
        parts = base_model.split("-")
        capitalized = "-".join(p.capitalize() for p in parts if p)
        return capitalized

    # ============ 工具方法 ============

    def find_variant(self, variant_id: str) -> Optional[VariantInfo]:
        """按 variant_id 查找（同步 fallback 模式）。"""
        for info in self.scan_variants():
            if info.name == variant_id or info.alias == variant_id:
                return info
        return None

    async def find_variant_async(self, variant_id: str) -> Optional[VariantInfo]:
        """按 variant_id 查找（异步，优先平台 API）。"""
        for info in await self.scan_variants_async():
            if info.name == variant_id or info.alias == variant_id:
                return info
        return None

    def list_available_variants(self) -> list[str]:
        """返回所有可用 variant_id 列表（同步 fallback 模式）。"""
        return [v.name for v in self.scan_variants() if v.available]
