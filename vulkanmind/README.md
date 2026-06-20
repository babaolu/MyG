# VulkanMind

VulkanMind is a local multi-agent system for modern Vulkan/C++ graphics engineering. It uses Python orchestration, LangGraph, Claude API agents, Qdrant vector search, FastAPI, SQLite persistence, and structured platform-aware routing.

## Configuration

Edit `config.yaml` before running the service:

- `llm.api_key_env` points to the environment variable containing the Anthropic API key.
- `qdrant.host` and `qdrant.port` configure the vector database connection.
- `storage.database_path` controls the SQLite database used for sessions, bug history, build queue, and self-update diffs.

Export the API key before starting:

```bash
export ANTHROPIC_API_KEY="..."
```

## Local services

Start Qdrant:

```bash
docker compose up -d
```

Install dependencies with `uv`:

```bash
uv sync --extra dev
```

Run the FastAPI service:

```bash
uv run uvicorn main:app --reload
```

## Native validation requirements

The validation tools are subprocess wrappers with a 60-second hard timeout. Install these on the host or target toolchain path:

- CMake 3.25+
- `clang-tidy` with a CppCoreGuidelines profile
- `spirv-val`
- `glslangValidator`
- Optional Android target queries: `adb`

The hardware governor freezes compiler threads when thermal mode is `THERMAL_THROTTLE`.
