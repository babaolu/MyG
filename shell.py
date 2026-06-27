#!/usr/bin/env python3
"""
VulkanMind Interactive Shell

Usage:
    # Start VulkanMind backend first (in a separate terminal):
    uv run uvicorn main:app --reload

    # Then launch the shell:
    uv run python shell.py

    # Or point at a remote VulkanMind instance:
    uv run python shell.py --host http://192.168.1.100:8000
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

BASE_URL = "http://localhost:8000"
HISTORY_FILE = Path.home() / ".vulkanmind_history"
SESSION_FILE = Path(".vulkanmind_session")

console = Console()


# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------


class VulkanMindClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=120)

    def health(self) -> dict[str, Any]:
        try:
            r = self._http.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ConnectionError(
                f"Cannot reach VulkanMind at {self.base_url}\n"
                f"Is the backend running? Start it with:\n"
                f"  uv run uvicorn main:app --reload"
            ) from e

    def start_session(
        self,
        initial_message: str,
        target_platform: dict | None = None,
    ) -> str:
        r = self._http.post(
            f"{self.base_url}/session/start",
            json={
                "user_request": initial_message,
                "target_platform_declared": target_platform,
                "attached_files": [],
            },
        )
        r.raise_for_status()
        return r.json()["session_id"]

    def send_message(
        self,
        session_id: str,
        message: str,
        validation_output: str | None = None,
        build_log: str | None = None,
    ) -> dict[str, Any]:
        r = self._http.post(
            f"{self.base_url}/session/{session_id}/message",
            json={
                "message": message,
                "validation_output": validation_output,
                "build_log": build_log,
            },
        )
        r.raise_for_status()
        return r.json()

    def get_platform_context(self, session_id: str) -> dict[str, Any]:
        r = self._http.get(f"{self.base_url}/session/{session_id}/platform_context")
        r.raise_for_status()
        return r.json()

    def get_skills_stats(self) -> dict[str, Any]:
        r = self._http.get(f"{self.base_url}/skills/stats")
        r.raise_for_status()
        return r.json()

    def get_trusted_skills(self) -> dict[str, Any]:
        r = self._http.get(f"{self.base_url}/skills/trusted")
        r.raise_for_status()
        return r.json()

    def get_build_queue(self, session_id: str) -> dict[str, Any]:
        r = self._http.get(f"{self.base_url}/session/{session_id}/build_queue")
        r.raise_for_status()
        return r.json()

    def get_skills_review(self) -> dict[str, Any]:
        r = self._http.get(f"{self.base_url}/skills/review")
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _render_response(response: dict[str, Any]) -> None:
    """Render a VulkanMind response in a readable format."""

    # Generated code — show with syntax highlighting
    if response.get("generated_code"):
        console.print()
        console.print(Rule("[bold cyan]Generated Code[/bold cyan]"))
        console.print(
            Syntax(
                response["generated_code"],
                "cpp",
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )
        )

    # Debug report
    debug = response.get("debug_report")
    if debug and debug.get("classification"):
        console.print()
        console.print(Rule("[bold yellow]Debug Report[/bold yellow]"))

        classification = debug.get("classification", {})
        if isinstance(classification, dict):
            bug_type = classification.get("classification", "unknown")
        else:
            bug_type = str(classification)

        console.print(
            Panel(
                f"[bold]Classification:[/bold] {bug_type}\n"
                f"[bold]Active Fix:[/bold] {debug.get('active_fix') or 'none yet'}",
                title="Bug Classification",
                border_style="yellow",
            )
        )

        hypotheses = debug.get("hypotheses", [])
        if hypotheses:
            table = Table(title="Hypotheses (ranked by probability)")
            table.add_column("Probability", style="cyan", width=12)
            table.add_column("Description", style="white")
            table.add_column("Fix", style="green")
            for h in hypotheses:
                if isinstance(h, dict):
                    table.add_row(
                        f"{h.get('probability', 0):.0%}",
                        h.get("description", ""),
                        textwrap.shorten(h.get("fix_template", ""), width=60),
                    )
            console.print(table)

    # Knowledge citations
    citations = response.get("knowledge_citations", [])
    if citations:
        console.print()
        console.print(Rule("[dim]Knowledge Sources[/dim]"))
        for chunk in citations[:3]:
            if isinstance(chunk, dict):
                source = chunk.get("source_title", chunk.get("source", "unknown"))
                confidence = chunk.get("confidence", "")
                console.print(
                    f"  [dim]• {source}[/dim]"
                    + (f" [{confidence}]" if confidence else "")
                )

    # General agent response text
    agent_response = response.get("agent_response", {})
    if isinstance(agent_response, dict):
        # Look for a human-readable message field
        for field in ("message", "response", "summary", "error"):
            value = agent_response.get(field)
            if value and isinstance(value, str):
                console.print()
                console.print(Markdown(value))
                break
        # If validation warning present
        if agent_response.get("validation_passed") is False:
            console.print(
                "[yellow]⚠ Validation not yet passed — "
                "debug cycle continuing.[/yellow]"
            )
        if agent_response.get("skill_extracted"):
            console.print(
                f"[green]✓ New skill extracted and saved "
                f"(id: {agent_response.get('skill_id', 'unknown')})[/green]"
            )


def _render_platform_context(ctx: dict[str, Any]) -> None:
    target = ctx.get("target", {})
    thermal = ctx.get("thermal", {})
    toolchain = ctx.get("toolchain", {})

    table = Table(title="Platform Context", show_header=False)
    table.add_column("Key", style="cyan", width=24)
    table.add_column("Value", style="white")

    table.add_row("Target OS", target.get("os", "unknown"))
    table.add_row("GPU Vendor", target.get("gpu_vendor", "unknown"))
    table.add_row("GPU Model", target.get("gpu_model", "unknown"))
    table.add_row("Vulkan Version", target.get("vulkan_version", "unknown"))
    table.add_row("Architecture", target.get("arch", "unknown"))
    table.add_row("Cross Compile", str(toolchain.get("is_cross_compile", False)))
    table.add_row("Thermal Mode", thermal.get("mode", "unknown"))
    table.add_row(
        "Parallel Workers",
        str(thermal.get("max_parallel_workers", "unknown")),
    )
    table.add_row(
        "Runtime Validation",
        str(ctx.get("runtime_validation_available", False)),
    )
    console.print(table)

    quirks = target.get("quirk_profile", {})
    if quirks:
        console.print("\n[bold]Active Quirk Profile:[/bold]")
        for k, v in quirks.items():
            console.print(f"  [cyan]{k}[/cyan]: {v}")


def _render_skills_stats(stats: dict[str, Any]) -> None:
    table = Table(title="VulkanMind Skill Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Skills", str(stats.get("total_skills", 0)))
    table.add_row("Trusted Skills", str(stats.get("trusted_skills", 0)))
    table.add_row("Total Traces", str(stats.get("total_traces", 0)))
    table.add_row(
        "Avg Iterations to Resolve",
        f"{stats.get('avg_iterations_to_resolve', 0):.1f}",
    )
    console.print(table)

    success_rates = stats.get("success_rate_by_platform", {})
    if success_rates:
        console.print("\n[bold]Success Rate by Platform:[/bold]")
        for platform, rate in success_rates.items():
            bar = "█" * int(rate * 20)
            console.print(f"  {platform:<20} {bar:<20} {rate:.0%}")


# ---------------------------------------------------------------------------
# Shell commands
# ---------------------------------------------------------------------------


COMMANDS = {
    "/help": "Show this help message",
    "/platform": "Show current platform context",
    "/skills": "Show skill statistics",
    "/trusted": "List trusted skills",
    "/queue": "Show current build queue",
    "/review": "Show items needing human review",
    "/validation <text>": "Send message with validation layer output attached",
    "/buildlog <text>": "Send message with build log attached",
    "/project <path>": "Switch Graphify graph to a different project",
    "/new": "Start a new session",
    "/session": "Show current session ID",
    "/clear": "Clear the screen",
    "/exit": "Exit VulkanMind shell",
}


def _show_help() -> None:
    table = Table(title="VulkanMind Shell Commands", show_header=True)
    table.add_column("Command", style="cyan", width=30)
    table.add_column("Description", style="white")
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)
    console.print(
        "\n[dim]For everything else, just type naturally. "
        "VulkanMind will classify your intent automatically.[/dim]"
    )


def _save_session(session_id: str) -> None:
    SESSION_FILE.write_text(session_id)


def _load_session() -> str | None:
    if SESSION_FILE.exists():
        sid = SESSION_FILE.read_text().strip()
        return sid if sid else None
    return None


def _append_history(line: str) -> None:
    try:
        with open(HISTORY_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Platform setup wizard
# ---------------------------------------------------------------------------


def _platform_setup_wizard() -> dict | None:
    """
    Run on first session start if no platform is declared.
    Walks the user through declaring a target platform interactively.
    Returns a platform dict or None if the user skips.
    """
    console.print()
    console.print(
        Panel(
            "No target platform declared.\n"
            "VulkanMind will try to auto-detect via ADB if an Android "
            "device is connected.\n"
            "Or you can declare it now for accurate platform-specific output.",
            title="Platform Setup",
            border_style="cyan",
        )
    )

    choice = Prompt.ask(
        "Declare platform now?",
        choices=["yes", "no", "adb"],
        default="adb",
    )

    if choice == "adb":
        console.print(
            "[cyan]VulkanMind will query your connected Android device via ADB.[/cyan]"
        )
        return None  # backend handles ADB detection

    if choice == "no":
        console.print("[dim]Skipping — platform will be detected or can be set later.[/dim]")
        return None

    # Manual declaration
    os_choice = Prompt.ask(
        "Target OS",
        choices=["Android", "Linux", "Windows", "Embedded"],
        default="Android",
    )
    gpu_vendor = Prompt.ask(
        "GPU Vendor",
        choices=["Qualcomm", "ARM", "PowerVR", "NVIDIA", "AMD", "Intel"],
        default="Qualcomm",
    )
    gpu_model = Prompt.ask("GPU Model", default="Adreno 740")
    vulkan_version = Prompt.ask(
        "Vulkan Version",
        choices=["1.0", "1.1", "1.2", "1.3"],
        default="1.3",
    )
    arch = Prompt.ask(
        "Architecture",
        choices=["arm64-v8a", "x86_64", "armv7", "x86"],
        default="arm64-v8a",
    )

    return {
        "os": os_choice,
        "gpu_vendor": gpu_vendor,
        "gpu_model": gpu_model,
        "vulkan_version": vulkan_version,
        "arch": arch,
    }


# ---------------------------------------------------------------------------
# Main shell loop
# ---------------------------------------------------------------------------


def run_shell(base_url: str) -> None:
    client = VulkanMindClient(base_url)

    # Check backend is up
    console.print(Rule("[bold cyan]VulkanMind[/bold cyan]"))
    try:
        health = client.health()
        qdrant = "✓" if health.get("qdrant_connected") else "✗ (Qdrant offline)"
        governor = health.get("hardware_governor_mode", "unknown")
        console.print(
            f"[green]Backend connected[/green] | "
            f"Qdrant: {qdrant} | "
            f"Governor: {governor}"
        )
    except ConnectionError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(
        "[dim]Type naturally to talk to VulkanMind. "
        "Type /help for shell commands.[/dim]\n"
    )

    # Resume or start session
    session_id = _load_session()
    target_platform = None

    if session_id:
        console.print(f"[dim]Resuming session: {session_id}[/dim]")
    else:
        target_platform = _platform_setup_wizard()
        initial = Prompt.ask("\n[cyan]What are you working on?[/cyan]")
        _append_history(initial)
        try:
            session_id = client.start_session(initial, target_platform)
            _save_session(session_id)
            console.print(f"[dim]Session started: {session_id}[/dim]\n")

            with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                response = client.send_message(session_id, initial)
            _render_response(response)
        except httpx.HTTPError as e:
            console.print(f"[red]Error starting session: {e}[/red]")
            sys.exit(1)

    # Pending context for /validation and /buildlog commands
    _pending_validation: str | None = None
    _pending_buildlog: str | None = None

    # Main input loop
    while True:
        try:
            console.print()
            user_input = Prompt.ask("[bold green]you[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        _append_history(user_input)

        # --- Shell commands ---

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() == "/clear":
            console.clear()
            continue

        if user_input.lower() == "/help":
            _show_help()
            continue

        if user_input.lower() == "/session":
            console.print(f"[dim]Session ID: {session_id}[/dim]")
            continue

        if user_input.lower() == "/platform":
            try:
                ctx = client.get_platform_context(session_id)
                _render_platform_context(ctx)
            except httpx.HTTPError as e:
                console.print(f"[red]Could not fetch platform context: {e}[/red]")
            continue

        if user_input.lower() == "/skills":
            try:
                stats = client.get_skills_stats()
                _render_skills_stats(stats)
            except httpx.HTTPError as e:
                console.print(f"[red]Could not fetch skills: {e}[/red]")
            continue

        if user_input.lower() == "/trusted":
            try:
                trusted = client.get_trusted_skills()
                skills = trusted.get("skills", [])
                if not skills:
                    console.print("[dim]No trusted skills yet.[/dim]")
                else:
                    table = Table(title=f"Trusted Skills ({len(skills)})")
                    table.add_column("Name", style="cyan")
                    table.add_column("Domain", style="white")
                    table.add_column("Symptom", style="yellow")
                    table.add_column("Times Used", style="green")
                    for s in skills:
                        table.add_row(
                            s.get("name", ""),
                            s.get("domain", ""),
                            textwrap.shorten(s.get("symptom", ""), 40),
                            str(s.get("times_successful", 0)),
                        )
                    console.print(table)
            except httpx.HTTPError as e:
                console.print(f"[red]Could not fetch trusted skills: {e}[/red]")
            continue

        if user_input.lower() == "/queue":
            try:
                queue = client.get_build_queue(session_id)
                items = queue.get("build_queue", [])
                if not items:
                    console.print("[dim]Build queue is empty.[/dim]")
                else:
                    table = Table(title="Build Queue")
                    table.add_column("ID", style="dim", width=10)
                    table.add_column("Status", style="cyan")
                    table.add_column("Probability", style="white")
                    table.add_column("Description", style="yellow")
                    for item in items:
                        table.add_row(
                            str(item.get("candidate_id", ""))[:8],
                            item.get("status", ""),
                            f"{item.get('probability', 0):.0%}",
                            textwrap.shorten(item.get("fix_description", ""), 50),
                        )
                    console.print(table)
            except httpx.HTTPError as e:
                console.print(f"[red]Could not fetch build queue: {e}[/red]")
            continue

        if user_input.lower() == "/review":
            try:
                review = client.get_skills_review()
                under_review = review.get("skills_under_review", [])
                proposals = review.get("pending_proposals", [])
                if not under_review and not proposals:
                    console.print("[green]Nothing needs review.[/green]")
                else:
                    if under_review:
                        console.print(
                            f"[yellow]{len(under_review)} skill(s) under review[/yellow]"
                        )
                        for s in under_review:
                            console.print(
                                f"  • {s.get('name')} "
                                f"(id: {s.get('skill_id')})"
                            )
                    if proposals:
                        console.print(
                            f"[yellow]{len(proposals)} prompt refinement "
                            f"proposal(s) pending[/yellow]"
                        )
                        for p in proposals:
                            console.print(
                                f"  • {p.get('agent_target')} agent: "
                                f"{textwrap.shorten(p.get('rationale', ''), 60)}"
                            )
            except httpx.HTTPError as e:
                console.print(f"[red]Could not fetch review queue: {e}[/red]")
            continue

        if user_input.lower() == "/new":
            confirm = Prompt.ask(
                "Start a new session? Current session will be ended",
                choices=["yes", "no"],
                default="no",
            )
            if confirm == "yes":
                SESSION_FILE.unlink(missing_ok=True)
                target_platform = _platform_setup_wizard()
                initial = Prompt.ask("What are you working on?")
                try:
                    session_id = client.start_session(initial, target_platform)
                    _save_session(session_id)
                    console.print(f"[dim]New session: {session_id}[/dim]\n")
                    with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                        response = client.send_message(session_id, initial)
                    _render_response(response)
                except httpx.HTTPError as e:
                    console.print(f"[red]Error: {e}[/red]")
            continue

        if user_input.lower().startswith("/project "):
            project_path = user_input[9:].strip()
            project_graph = Path(project_path) / "graphify-out" / "graph.json"
            if not project_graph.exists():
                console.print(
                    f"[yellow]No graphify-out/graph.json found in {project_path}[/yellow]\n"
                    f"Build it with:\n"
                    f"  cd {project_path}\n"
                    f"  graphify extract . --mode deep\n"
                    f"  graphify hook install"
                )
            else:
                os.environ["GRAPHIFY_GRAPH"] = str(project_graph)
                console.print(
                    f"[green]Switched Graphify graph to {project_graph}[/green]\n"
                    f"[dim]Note: Restart VulkanMind backend to apply this change.[/dim]"
                )
            continue

        if user_input.lower().startswith("/validation "):
            _pending_validation = user_input[12:].strip()
            console.print(
                "[dim]Validation output captured. "
                "Type your message now and it will be attached.[/dim]"
            )
            continue

        if user_input.lower().startswith("/buildlog "):
            _pending_buildlog = user_input[10:].strip()
            console.print(
                "[dim]Build log captured. "
                "Type your message now and it will be attached.[/dim]"
            )
            continue

        # --- Normal message — send to backend ---
        try:
            with console.status(
                "[cyan]VulkanMind is thinking...[/cyan]", spinner="dots"
            ):
                response = client.send_message(
                    session_id,
                    user_input,
                    validation_output=_pending_validation,
                    build_log=_pending_buildlog,
                )
            # Clear pending context after use
            _pending_validation = None
            _pending_buildlog = None

            console.print(Rule("[dim]vulkanmind[/dim]"))
            _render_response(response)

        except httpx.HTTPError as e:
            console.print(f"[red]Request failed: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VulkanMind Interactive Shell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Start the VulkanMind backend first:
                uv run uvicorn main:app --reload

            Then launch the shell:
                uv run python shell.py

            Point at a remote instance:
                uv run python shell.py --host http://192.168.1.100:8000
        """),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("VULKANMIND_HOST", BASE_URL),
        help="VulkanMind backend URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    run_shell(args.host)


if __name__ == "__main__":
    main()
