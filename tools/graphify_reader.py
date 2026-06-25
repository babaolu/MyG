"""Read-only adapter over the local Graphify knowledge graph.

The agent ecosystem never writes to ``graph.json`` directly — that is owned by
the ``graphify`` CLI. This module exposes the read paths the
``knowledge_retrieval_node`` needs:
  * concept lookup (label / id resolution)
  * relationship queries (incoming + outgoing edges)
  * shortest path between two concepts
  * prompt-ready formatting for retrieval snippets, parallel to the Qdrant
    retrieval path

All read paths shell out to ``graphify`` so the agent stack does not import
the graphify internals. Callers should not block on the CLI; they should accept
``GraphifyReader`` failures as "no extra context" and fall through to Qdrant.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_GRAPH = Path("graphify-out/graph.json")


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    file_type: str
    source_file: str | None
    source_location: str | None
    community: int | None
    norm_label: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    confidence: str
    confidence_score: float
    source_file: str | None = None
    source_location: str | None = None
    context: str | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class GraphifyGraph:
    """In-memory snapshot of a graph.json file.

    Cheap to construct because we just keep the raw JSON; queries do targeted
    node-by-id lookups so the cost scales with the answer, not the graph.
    """

    path: Path
    raw: dict

    @classmethod
    def load(cls, path: Path | str | None = None) -> GraphifyGraph:
        graph_path = Path(path) if path else _DEFAULT_GRAPH
        if not graph_path.exists():
            raise FileNotFoundError(f"graph.json not found at {graph_path}")
        return cls(path=graph_path, raw=json.loads(graph_path.read_text(encoding="utf-8")))

    # -- canonical lookups ------------------------------------------------

    def get_node(self, node_id: str) -> GraphNode | None:
        for node in self._nodes():
            if node.get("id") == node_id:
                return _to_node(node)
        return None

    def find_node_by_label(self, label: str) -> GraphNode | None:
        normalised = label.strip().lower()
        for node in self._nodes():
            if (node.get("norm_label") or "").lower() == normalised or (
                node.get("label", "").strip().lower() == normalised
            ):
                return _to_node(node)
        return None

    def nodes(self) -> list[GraphNode]:
        return [_to_node(node) for node in self._nodes()]

    def edges(self) -> list[GraphEdge]:
        return [_to_edge(link) for link in self._links()]

    def outgoing(self, node_id: str) -> list[GraphEdge]:
        return [edge for edge in self.edges() if edge.source == node_id]

    def incoming(self, node_id: str) -> list[GraphEdge]:
        return [edge for edge in self.edges() if edge.target == node_id]

    def nodes_in_community(self, community: int) -> list[GraphNode]:
        return [node for node in self.nodes() if node.community == community]

    # -- private helpers --------------------------------------------------

    def _nodes(self) -> list[dict]:
        return list(self.raw.get("nodes", []) or [])

    def _links(self) -> list[dict]:
        return list(self.raw.get("links", []) or [])


def _to_node(node: dict) -> GraphNode:
    return GraphNode(
        node_id=str(node.get("id", "")),
        label=str(node.get("label", "")),
        file_type=str(node.get("file_type", "")),
        source_file=node.get("source_file"),
        source_location=node.get("source_location"),
        community=node.get("community"),
        norm_label=node.get("norm_label"),
        raw=node,
    )


def _to_edge(link: dict) -> GraphEdge:
    return GraphEdge(
        source=str(link.get("source", "")),
        target=str(link.get("target", "")),
        relation=str(link.get("relation", "")),
        confidence=str(link.get("confidence", "")),
        confidence_score=float(link.get("confidence_score", 0.0) or 0.0),
        source_file=link.get("source_file"),
        source_location=link.get("source_location"),
        context=link.get("context"),
        weight=float(link.get("weight", 1.0) or 1.0),
    )


# ---------------------------------------------------------------------------
# CLI-backed query API
# ---------------------------------------------------------------------------


class GraphifyReader:
    """Read-only Graphify facade.

    Args:
        graph_path: path to ``graph.json`` (defaults to ``graphify-out/graph.json``).
        cli_path: explicit path to the ``graphify`` CLI; autodetected via ``shutil.which``.
        budget_tokens: token cap passed to ``graphify query`` (default 2000).
    """

    def __init__(
        self,
        graph_path: Path | str | None = None,
        cli_path: str | None = None,
        budget_tokens: int = 2000,
    ) -> None:
        self.graph = GraphifyGraph.load(graph_path)
        # Caller may pass an explicit binary path; otherwise we probe PATH. If
        # we can't find it, the reader still works for in-graph lookups
        # (format_for_prompt degrades to "").
        if cli_path is None:
            cli_path = shutil.which("graphify")
        self.cli_path = cli_path
        self.budget_tokens = budget_tokens

    @property
    def available(self) -> bool:
        """True when the graphify CLI is on PATH or explicitly provided.

        Lookups like `query_concept` only fire when this is True; relationship
        queries and `format_for_prompt` degrade gracefully otherwise.
        """
        if self.cli_path is None:
            return False
        return shutil.which(self.cli_path) is not None or Path(self.cli_path).exists()

    # -- concept BFS -----------------------------------------------------

    def query_concept(self, question: str) -> str:
        """Return a prompt-ready string covering the BFS neighbourhood of ``question``."""
        if not self.available:
            raise RuntimeError("graphify CLI is not installed; install with `pip install graphify`")
        completed = subprocess.run(
            [
                self.cli_path or "graphify",
                "query",
                question,
                "--budget",
                str(self.budget_tokens),
                "--graph",
                str(self.graph.path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"graphify query failed (exit={completed.returncode}): {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    # -- relationship queries --------------------------------------------

    def query_relationships(self, node_label: str) -> list[GraphEdge]:
        node = self.graph.find_node_by_label(node_label)
        if node is None:
            return []
        return self.graph.outgoing(node.node_id) + self.graph.incoming(node.node_id)

    # -- shortest path between two concepts ------------------------------

    def query_path(self, source_label: str, target_label: str) -> list[GraphEdge]:
        source = self.graph.find_node_by_label(source_label)
        target = self.graph.find_node_by_label(target_label)
        if source is None or target is None or not self.available:
            return []
        completed = subprocess.run(
            [
                self.cli_path or "graphify",
                "path",
                source.node_id,
                target.node_id,
                "--graph",
                str(self.graph.path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            return []
        # The path command prints a sequence of node ids; reconstruct edges
        # using the in-memory graph so we return GraphEdge objects.
        node_ids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        edges_by_endpoints = {
            (edge.source, edge.target): edge for edge in self.graph.edges()
        }
        ordered_edges: list[GraphEdge] = []
        for start, end in zip(node_ids, node_ids[1:], strict=False):
            edge = edges_by_endpoints.get((start, end))
            if edge is not None:
                ordered_edges.append(edge)
        return ordered_edges

    # -- prompt assembly --------------------------------------------------

    def format_for_prompt(
        self,
        question: str,
        *,
        max_chars: int = 4000,
        max_edges: int = 24,
    ) -> str:
        """Render a condensed BFS answer as a prompt section."""
        try:
            answer = self.query_concept(question)
        except Exception:
            return ""
        if not answer:
            return ""
        # Pull supporting edges out of the in-memory graph for inline citations.
        supporting = self._edges_for_text(question, max_edges=max_edges)
        supporting_block = (
            "\n".join(
                f"- {edge.relation} {edge.source} -> {edge.target} "
                f"(confidence={edge.confidence}, score={edge.confidence_score:.2f})"
                for edge in supporting
            )
            if supporting
            else "- (no supporting edges resolved)"
        )
        block = (
            "=== Graphify Knowledge Graph context ===\n"
            f"Q: {question}\n\n"
            f"{answer[:max_chars]}\n\n"
            f"Supporting edges:\n{supporting_block}\n"
        )
        return block

    def _edges_for_text(self, text: str, *, max_edges: int) -> list[GraphEdge]:
        nodes = self._candidate_nodes(text)
        if not nodes:
            return []
        node_ids = {node.node_id for node in nodes}
        edges: list[GraphEdge] = []
        for edge in self.graph.edges():
            if edge.source in node_ids or edge.target in node_ids:
                edges.append(edge)
            if len(edges) >= max_edges:
                break
        return edges

    def _candidate_nodes(self, text: str) -> list[GraphNode]:
        tokens = [token.lower() for token in text.split() if len(token) >= 3]
        if not tokens:
            return []
        by_label: dict[str, GraphNode] = {}
        for node in self.graph.nodes():
            label = (node.norm_label or node.label).lower()
            if any(token in label for token in tokens):
                by_label[node.node_id] = node
        return list(by_label.values())


def build_graphify_reader(graphify_config: dict | None = None) -> GraphifyReader | None:
    """Construct a GraphifyReader from `config["graphify"]`.

    Returns ``None`` if the graph file does not exist; callers should treat
    Graphify as an opt-in retrieval path and fall through to Qdrant otherwise.
    """
    config = dict(graphify_config or {})
    raw_path = config.get("graph_path") or os.environ.get("GRAPHIFY_GRAPH")
    graph_path = Path(raw_path) if raw_path else _DEFAULT_GRAPH
    if not graph_path.exists():
        return None
    return GraphifyReader(
        graph_path=graph_path,
        cli_path=config.get("cli_path"),
        budget_tokens=int(config.get("budget_tokens", 2000)),
    )
