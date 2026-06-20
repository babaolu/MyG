# VulkanMind

VulkanMind is a local multi-agent AI system for modern Vulkan graphics programming and high-performance C++ systems engineering. It orchestrates specialized agents for platform detection, knowledge retrieval, Vulkan-Hpp/VMA code generation, validation, debugging, and self-update monitoring.

The implementation lives in `vulkanmind/` and is configured for Python 3.11+, `uv`, FastAPI, LangGraph, Claude API, Qdrant, SQLite, Pydantic, structlog, and optional native Vulkan/C++ validation tools.

## Repository layout

```text
vulkanmind/
├── main.py
├── pyproject.toml
├── config.yaml
├── docker-compose.yml
├── orchestrator/
│   ├── graph.py
│   ├── router.py
│   └── state.py
├── agents/
│   ├── platform_intelligence/
│   ├── code_generation/
│   ├── debugger/
│   ├── knowledge_retrieval/
│   └── self_update/
├── tools/
├── db/
└── tests/
```

## Core architecture

VulkanMind uses a deterministic LangGraph state graph:

1. `platform_intelligence_node` always runs first.
2. `router_node` classifies the user request.
3. `code_generation`, `debug`, and `knowledge_query` tasks first pass through `knowledge_retrieval_node`.
4. `code_generation_node` emits Vulkan-Hpp/VMA-oriented C++ and CMake snippets.
5. If generated code validation fails, the graph automatically routes to `debugger_node`.
6. SQLite checkpointing persists session state and history.

Shared state is defined in `orchestrator/state.py` as `VulkanMindState`. Every agent receives the active `PlatformContext` in its Claude system prompt through `build_agent_system_prompt()`.

## Agents

### Platform Intelligence Agent

Location: `agents/platform_intelligence/`

Responsibilities:

- Detect host OS, architecture, compiler, and compiler version.
- Resolve target platform from ADB first, then user declaration.
- Refuse to guess a target if neither ADB nor a user declaration is available.
- Determine cross-compilation strategy for Linux-to-Android, Linux-to-Windows, Windows-to-Linux, and macOS-to-Android.
- Read thermal state and derive hardware governor mode.

Key functions:

- `detect_host_platform() -> HostContext`
- `detect_target_platform(user_declared, adb_connected) -> TargetContext`
- `determine_cross_compile(host, target) -> ToolchainContext`
- `read_thermal_state(target) -> ThermalContext`
- `build_platform_context(...) -> PlatformContext`

### Code Generation Agent

Location: `agents/code_generation/`

Responsibilities:

- Generate Vulkan-Hpp-oriented C++20/C++23-style code.
- Require VMA for device allocations.
- Generate platform-specific CMake snippets.
- Validate generated code through native tools when available.

Generated code follows these mandates:

- Vulkan-Hpp type-safe bindings.
- VMA wrappers instead of raw `vkAllocateMemory`.
- RAII ownership.
- `std::expected` or `std::optional` for error states.
- No owning raw pointers or raw `new`/`delete`.

### Debugger Agent

Location: `agents/debugger/`

Responsibilities:

- Classify validation/build failures as `ISOLATABLE` or `CROSS_SYSTEM`.
- Match Vulkan validation logs against seeded bug patterns.
- Produce ranked hypotheses, fix templates, and validation steps.
- Isolate source fragments by suspected subsystem.
- Queue speculative build candidates.
- Freeze compilation when the hardware governor reports `THERMAL_THROTTLE`.

Seeded pattern groups include:

- `BLACK_SCREEN`
- `GPU_HANG`
- `VALIDATION_SYNC`
- `VALIDATION_LAYOUT`

### Knowledge Retrieval Agent

Location: `agents/knowledge_retrieval/`

Responsibilities:

- Ingest PDFs through LlamaParse.
- Scrape Khronos, ARM, Qualcomm, JCGT, and Advances in Real-Time Rendering sources.
- Chunk documents into metadata-aware segments.
- Build Qdrant queries filtered by GPU vendor, target OS, Vulkan version, and topic.

Qdrant query construction is in `agents/knowledge_retrieval/retrieval/query_builder.py`.

### Self-Update Agent

Location: `agents/self_update/`

Responsibilities:

- Poll KhronosGroup/Vulkan-Docs releases.
- Store pending update diffs in SQLite.
- Require explicit human confirmation before applying changes to knowledge or profile data.

No self-update is applied automatically.

## Configuration

Edit `config.yaml` before running:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY

embeddings:
  provider: anthropic
  model: voyage-3

qdrant:
  host: localhost
  port: 6333
  collection: vulkanmind_chunks

storage:
  database_path: data/vulkanmind.sqlite3

hardware_governor:
  default_mode: NORMAL
  max_compiler_timeout_seconds: 60

self_update:
  require_human_confirmation: true
```

Export the required API key before starting:

```bash
export ANTHROPIC_API_KEY="..."
```

If OpenAI embeddings are selected in code, export:

```bash
export OPENAI_API_KEY="..."
```

## Local services

Start Qdrant:

```bash
cd vulkanmind
docker compose up -d
```

Install dependencies with `uv`:

```bash
cd vulkanmind
uv sync --extra dev
```

Run the FastAPI service:

```bash
cd vulkanmind
uv run uvicorn main:app --reload
```

The API listens on `http://127.0.0.1:8000` by default.

## API reference

### Start a session

```http
POST /session/start
Content-Type: application/json
```

Request:

```json
{
  "user_request": "Generate a Vulkan-Hpp triangle with VMA allocation",
  "target_platform_declared": {
    "os": "Linux",
    "arch": "x86_64",
    "gpu_vendor": "AMD",
    "gpu_model": "Radeon",
    "vulkan_version": "1.3",
    "supported_extensions": ["VK_KHR_swapchain"]
  },
  "attached_files": []
}
```

Response:

```json
{
  "session_id": "..."
}
```

### Send a message

```http
POST /session/{session_id}/message
Content-Type: application/json
```

Request:

```json
{
  "message": "Generate a minimal Vulkan-Hpp compute pipeline",
  "validation_output": null,
  "build_log": null,
  "attached_files": []
}
```

Response includes:

- `agent_response`
- `generated_code`
- `debug_report`
- `knowledge_citations`

### Inspect platform context

```http
GET /session/{session_id}/platform_context
```

### Inspect build queue

```http
GET /session/{session_id}/build_queue
```

### Confirm or discard self-update

```http
POST /session/{session_id}/confirm_update
Content-Type: application/json
```

Request:

```json
{
  "update_id": "khronos-1.3.296",
  "confirmed": true
}
```

### Health check

```http
GET /health
```

Response includes:

- `status`
- `qdrant_connected`
- `hardware_governor_mode`

## Native validation requirements

The validation wrappers enforce a 60-second hard timeout and treat missing tools, non-zero exits, warnings, and timeouts as validation faults.

Install these tools on the host or target toolchain path:

- CMake 3.25+
- `clang-tidy` with a CppCoreGuidelines profile
- `spirv-val`
- `glslangValidator`
- Optional Android target queries: `adb`

The hardware governor blocks compilation when thermal mode is `THERMAL_THROTTLE`.

## Testing

Run the test suite:

```bash
cd vulkanmind
uv run pytest -q
```

Run linting:

```bash
cd vulkanmind
uv run ruff check .
```

The current implementation includes unit tests for platform detection, bug classification, hardware governor behavior, and quirk profiles, plus integration tests for routing and knowledge query construction.

## Environment notes

- Python: 3.11+
- Package manager: `uv`
- Vector DB: Qdrant via Docker Compose
- Storage: SQLite
- API service: FastAPI
- LLM: Anthropic Claude `claude-sonnet-4-6`
- Embeddings: configurable; local fallback exists for tests
- Native C++ target: C++20/C++23 tracking
- Vulkan bindings: Vulkan-Hpp
- Allocation abstraction: VMA

## Operational cautions

- The system refuses target detection when no ADB device or user-declared target is available.
- Generated C++ validation fails closed when native tools are missing.
- Self-update diffs are stored but never applied without explicit confirmation.
- Claude API calls require `ANTHROPIC_API_KEY`.
- Qdrant must be running for knowledge retrieval with real vector search.
