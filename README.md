# Llama.cpp 引擎使用指南

> 引擎 ID：`llama_cpp_9cdb76f8`
> 引擎类型：llm / chat
> 基于：[llama.cpp](https://github.com/ggml-org/llama.cpp) 官方预编译二进制 `llama-server.exe`

---

## 一、模型放哪里？

### 1.1 放置目录

所有 llama_cpp 引擎的模型文件统一放到项目根目录下：

```
models/llm/checkpoints/llama/
```

**Windows 完整路径示例**：
```
D:\myProInstallDir\AIReliable\models\llm\checkpoints\llama\Qwen_Qwen3.6-35B-A3B-Q6_K.gguf
```

**Linux 完整路径示例**：
```
/opt/aireliable/models/llm/checkpoints/llama/Qwen_Qwen3.6-35B-A3B-Q6_K.gguf
```

> ⚠️ 不要放到其他目录（如 `D:\myProInstallDir\VLLM\llama\models\`）。引擎只扫描 `models/llm/checkpoints/llama/`，放错位置会导致启动按钮变灰。

### 1.2 多模态 mmproj 投影文件

视觉多模态模型（如 Qwen-VL）需要额外配 `mmproj-*.gguf` 投影文件：

- **放置位置**：与主模型文件**同目录**（`models/llm/checkpoints/llama/`）
- **命名规则**：`mmproj-<模型族>.gguf`
- **来源**：与主模型同仓库的 HuggingFace repo

---

## 二、支持哪些模型？

### 2.1 支持的文件格式

| 格式 | 扩展名 | 支持 | 说明 |
|---|---|---|---|
| **GGUF** | `.gguf` | ✅ 原生支持 | llama.cpp 官方格式，直接使用 |
| Safetensors | `.safetensors` | ❌ 不支持 | 需先用 `convert_hf_to_gguf.py` 转换 |
| PyTorch | `.bin` / `.pt` | ❌ 不支持 | 需先转换 |
| Diffusers | 目录形式 | ❌ 不支持 | 需先用 `convert_hf_to_gguf.py` 转换 |

### 2.2 支持的模型族

llama.cpp 引擎支持所有 GGUF 格式的 LLM，包括但不限于：

| 模型族 | 示例 | 多模态支持 |
|---|---|---|
| Qwen | Qwen3.5 / Qwen3.6 / Qwen2.5 / Qwen-VL | ✅（配 mmproj） |
| DeepSeek | DeepSeek-V3 / DeepSeek-R1 | 纯文本 |
| Llama | Llama 3 / 3.1 / 3.2 / 3.3 | 纯文本 |
| Mistral | Mistral / Mixtral / Codestral | 纯文本 |
| Yi | Yi-1.5 / Yi-VL | ✅（配 mmproj） |
| Gemma | Gemma 2 / Gemma 3 | ✅（配 mmproj） |
| Phi | Phi-3 / Phi-3-Vision | ✅（配 mmproj） |
| InternLM | InternLM2 / InternVL | ✅（配 mmproj） |
| Baichuan | Baichuan 2 | 纯文本 |
| Command-R | Command-R / Command-R+ | 纯文本 |

### 2.3 量化级别推荐

| 量化 | 比特数 | 文件大小（35B 模型） | 质量 | 速度 | 推荐显存 |
|---|---|---|---|---|---|
| Q4_0 | 4 | ~18GB | 一般 | 最快 | 24GB |
| Q4_K_M | 4 | ~22GB | 好 | 快 | 24GB |
| Q5_K_M | 5 | ~26GB | 较好 | 中 | 32GB |
| Q6_K | 6 | ~30GB | 很好 | 中慢 | 48GB |
| Q8_0 | 8 | ~37GB | 极好 | 慢 | 48GB+ |
| F16 | 16 | ~70GB | 完整 | 最慢 | 80GB+ |

**推荐**：
- 日常使用：Q4_K_M / Q5_K_M（速度和质量平衡）
- 高质量需求：Q6_K / Q8_0
- 显存吃紧：Q4_0

---

## 三、新增 variant（模型档）步骤

### 3.1 放模型文件

把 `.gguf` 文件（和配套的 `mmproj-*.gguf`）放到 `models/llm/checkpoints/llama/`。

### 3.2 在 registry.yaml 声明

打开 `config/registry.yaml`，找到 `llama_cpp_9cdb76f8` 引擎块，在 `variants` 和 `model_files` 列表下各加一条：

```yaml
engines:
  - engine_id: llama_cpp_9cdb76f8
    # ... 其他配置 ...
    variants:
      - name: my-new-model-q6                # 内部名称
        alias: my-model-q6                    # 短别名（≤16 字符）
        params:
          ctx: 32768                           # 上下文长度（可选，默认自动计算）
          ngl: 999                             # GPU 层数（可选）

    model_files:
      - variant: my-new-model-q6
        filename: MyModel-35B-Q6_K.gguf
        target_dir: llm/checkpoints/llama/    # 固定这个目录
        size_bytes: 30000000000               # 文件大小（字节）
        source:
          type: huggingface
          repo_id: MyOrg/MyModel-GGUF
          filename: MyModel-35B-Q6_K.gguf
        mirror:
          type: huggingface_mirror
          url: https://hf-mirror.com/MyOrg/MyModel-GGUF/resolve/main/MyModel-35B-Q6_K.gguf
```

### 3.3 重启程序

重启 AIReliable，引擎详情页会自动显示新 variant 和文件就绪状态。

### 3.4 命名建议

文件名建议包含以下信息以便识别：

```
<模型族>_<参数量>-<版本>-<量化>.gguf
```

例：`Qwen_Qwen3.6-35B-A3B-Q6_K.gguf`

- `Qwen_Qwen3.6`：模型族
- `35B`：总参数量
- `A3B`：MoE 激活参数量（A=Active，仅 MoE 模型有）
- `Q6_K`：量化级别

---

## 四、当前已声明的 variant

| alias | 模型 | 量化 | 文件大小 | 多模态 |
|---|---|---|---|---|
| `qwen3.5-q4` | Qwen3.5-35B-A3B | Q4_K_M | ~22GB | 否 |
| `qwen3.6-q6` | Qwen3.6-35B-A3B | Q6_K | ~30GB | 是 |
| `qwen3.6-q8` | Qwen3.6-35B-A3B | Q8_0 | ~37GB | 是 |
| `qwen3.6-unc-q6` | Qwen3.6-35B-A3B Uncensored | Q6_K_P | ~30GB | 是 |
| `qwen3.6-unc-q8` | Qwen3.6-35B-A3B Uncensored | Q8_K_P | ~40GB | 是 |

---

## 五、使用方式

### 5.1 启动引擎

1. 打开 AIReliable 桌面程序
2. 进入"引擎管理"页
3. 找到 llama_cpp 卡片，点击"启动"按钮（按钮灰色表示模型文件未就绪，点卡片进详情页看下载指引）
4. 等待状态变为 Ready

### 5.2 使用 Gradio 聊天界面

1. 引擎 Ready 后，点 llama_cpp_9cdb76f8 卡片进入详情页
2. 点"打开网页"按钮打开 Gradio UI
3. 选择 variant（如 `qwen3.6-q6`）
4. 直接输入消息发送 —— 系统会自动加载模型（首次约 60-90 秒）
5. 切换 variant 后发送消息，系统会自动卸载旧模型、加载新模型

### 5.3 调用 OpenAI 兼容 API

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-q6",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": "sse"
  }'
```

- `model` 字段填 variant 的 alias
- `stream` 参数：`"sse"`（流式，默认）/ `"json"`（同步）/ `"async"`（异步任务）

---

## 六、常见问题

### Q1：启动按钮是灰色的

**A**：模型文件未就绪。点卡片进详情页，查看"模型文件"区块，每个未就绪文件会显示：
- 下载链接（官方 + 镜像）
- 目标路径（"放到"那一行）
- "打开目录"按钮

按提示把文件放到 `models/llm/checkpoints/llama/` 即可。

### Q2：加载模型失败 / 500 错误

**A**：可能原因：
1. **加载超时**：28GB+ 的大模型加载需要 60-120 秒，已设置 120 秒超时
2. **显存不足**：检查 nvidia-smi，确认显存够大（Q6_K 需要 48GB+）
3. **文件损坏**：检查文件大小是否与 registry.yaml 声明一致（1% 容差）

### Q3：模型文件已放好，但程序没识别

**A**：
1. 确认路径是 `models/llm/checkpoints/llama/`，不是 `models/llm/` 或 `models/`
2. 在引擎详情页点"刷新"按钮
3. 如果还不行，检查 `config/registry.yaml` 中 `filename` 是否与实际文件名**完全一致**（区分大小写）

### Q4：如何转换非 GGUF 格式

**A**：使用 llama.cpp 提供的转换工具：

```bash
# HuggingFace 模型转 GGUF
python convert_hf_to_gguf.py <hf_model_dir> --outfile <output.gguf> --outtype q6_k

# 重新量化
llama-quantize.exe <input.gguf> <output.gguf> Q6_K
```

详细见 [llama.cpp 文档](https://github.com/ggml-org/llama.cpp/tree/master/examples)。

---

## 七、技术架构

详见 [PR/engine/llama-cpp/llama-cpp-architecture.md](../../PR/engine/llama-cpp/llama-cpp-architecture.md)。

- **引擎 Python 层**：监听 `127.0.0.1:7002`，对外暴露 AIReliable 8 端点契约
- **llama-server 二进制**：监听 `127.0.0.1:<internal_port>`（隐藏，从 8100-8200 选取空闲端口）
- **Gradio UI**：挂载在 Python 层根路径 `/`
- **8 端点契约**：/health /capabilities /load /unload /infer /status /lora/load /lora/unload
