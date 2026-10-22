"""Llama.cpp 引擎模块入口。

支持 `python -m engines.llama_cpp_9cdb76f8` 直接启动。

启动命令：
    uv run --project envs/llm-llama-cpp \\
        python -m engines.llama_cpp_9cdb76f8.server --port 7002
"""

from .server import main

if __name__ == "__main__":
    main()
