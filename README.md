# VulkanMind

VulkanMind is a local multi-agent AI system for modern Vulkan graphics programming and high-performance C++ systems engineering. It orchestrates specialized agents for platform detection, knowledge retrieval, Vulkan-Hpp/VMA code generation, validation, debugging, and self-update monitoring.

The implementation lives at the repository root and is configured for Python 3.11+, `uv`, FastAPI, LangGraph, Claude API, Qdrant, SQLite, Pydantic, structlog, and optional native Vulkan/C++ validation tools.

## Repository layout

```text
main.py
pyproject.toml
config.yaml
docker-compose.yml
orchestrator/
├── graph.py
├── router.py
└── state.py
agents/
├── platform_intelligence/
├── code_generation/
├── debugger/
├── knowledge_retrieval/
├── self_update/
└── self_improvement/
tools/
db/
tests/
README.md
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

## Self-Improvement Loop

VulkanMind improves through structured agent-layer memory, not by changing the Claude model. Each completed code generation, validation, and debug cycle can produce an `ExecutionTrace`. Successful non-trivial debug cycles may be converted into reusable `VulkanSkill` objects, which are platform-tagged and injected into future sessions.

The loop has four parts:

- **Execution trace storage:** `db/execution_traces.py` records every complete cycle with platform context, validation result, matched bug pattern, active fix, and outcome.
- **Skill extraction:** `agents/self_improvement/skill_extractor.py` extracts reusable Vulkan-Hpp/VMA fix procedures from successful multi-iteration debug sessions and writes them to `db/skill_writebacks.py`.
- **Session memory injection:** `agents/self_improvement/memory_injector.py` injects trusted skills, recent same-platform fixes, and platform pattern hit rates into agent system prompts before RAG retrieval.
- **Curation and prompt refinement:** `agents/self_improvement/pattern_curator.py` audits skills and patterns weekly, while `agents/self_improvement/prompt_refiner.py` proposes guarded prompt changes from recurring failures.

Important human gates:

- Prompt refinement proposals require explicit approval through `/skills/proposals/{proposal_id}/approve`.
- Contradictory or low-performing skills are moved to review and require human confirmation.
- Low-hit-rate skill-writeback patterns are marked for review and are never auto-retired.

Useful endpoints:

- `GET /skills` lists active skills, optionally filtered by `gpu_vendor`, `domain`, or `confidence`.
- `GET /skills/trusted` lists high-confidence accumulated knowledge.
- `GET /skills/review` shows skills under review, pending prompt proposals, and human review items.
- `GET /skills/stats` shows total skills, trusted skills, total traces, success rate by platform, top resolved symptoms, and average iterations to resolve.

## Graphify Knowledge Graph

VulkanMind's `knowledge_retrieval_node` runs two retrieval paths in parallel:

1. **Qdrant** — vector search filtered by GPU vendor / OS / Vulkan version.
2. **Graphify** — a local knowledge graph built from the VulkanMind codebase
   (and any ingested docs), enabling structural queries that vector search
   cannot answer.

Graphify is read-only from the agent's perspective. The CLI owns all writes:

```bash
# Re-build the graph from scratch (AST-only, no LLM).
graphify update . --force

# Re-extract with semantic edges (LLM-backed).
graphify extract . --mode deep

# Run a BFS question over the graph.
graphify query "what does the LangGraph dispatcher do"
```

The Python adapter lives in `tools/graphify_reader.py`:

```python
from tools.graphify_reader import build_graphify_reader

reader = build_graphify_reader({"graph_path": "graphify-out/graph.json"})
if reader is not None:
    excerpt = reader.format_for_prompt(
        "How does self_improvement_node feed memory into the prompt?"
    )
```

The reader shells out to the `graphify` CLI for `query` and `path` operations
and reads `graph.json` directly for relationship/edge lookups. If the CLI is
absent or the graph file does not exist, `build_graphify_reader` returns
`None` and the knowledge retrieval node falls through to the Qdrant path
without raising.

The live graph for this repository is committed under `graphify-out/` (457
nodes, 27 communities across the `agents/`, `db/`, `orchestrator/`, `tools/`
packages plus the design docs in `vulkan_mind.md`).

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

self_improvement:
  enabled: true
  skill_extraction:
    min_iterations_to_extract: 2
    min_confidence_to_inject: medium
    max_skills_in_prompt: 5
    max_tokens_for_memory: 2000
  curation:
    audit_schedule_days: 7
    stale_threshold_days: 90
    min_hit_rate_to_keep: 0.30
    auto_promote_threshold: 3
    require_human_review_below: 0.40
  prompt_refinement:
    enabled: true
    min_failures_to_analyse: 5
    min_confidence_to_propose: 0.70
    lookback_days: 30
    require_human_approval: true
  memory_injection:
    inject_trusted_skills: true
    inject_recent_fixes: true
    recent_fixes_limit: 10
    inject_hit_rates: true
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

The API listens on `http://127.0.0.1:8000` by default.

## Using the Shell

VulkanMind has an interactive shell — no curl required.

Start the backend:

```bash
uv run uvicorn main:app --reload
```

In a second terminal, start the shell:

```bash
uv run python shell.py
```

Or via the `Makefile` from the repo root:

```bash
make shell    # terminal 2 — interactive shell
make backend  # terminal 1 — FastAPI service (do this first)
make dev      # starts both in one process (backend in background)
```

Shell commands:

```text
/help          Show all commands
/platform      Show detected platform context
/skills        Show accumulated skill statistics
/trusted       List trusted skills
/queue         Show speculative build queue
/review        Show items needing human review
/new           Start a new session
/project PATH  Switch to a different project's Graphify graph
/validation X  Attach validation layer output to next message
/buildlog X    Attach build log to next message
/clear         Clear the screen
/exit          Exit
```

Everything else is sent as a natural language message to VulkanMind.

Point at a remote backend:

```bash
uv run python shell.py --host http://192.168.1.100:8000
```

The shell is a thin HTTP client. All routing, retrieval, generation, and
state lives in the backend — restart `uvicorn` to pick up backend changes,
the shell will resume the active session automatically via the local
`.vulkanmind_session` file.

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
uv run pytest -q
```

Run linting:

```bash
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
