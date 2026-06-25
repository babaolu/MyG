"""Unit tests for ``tools.graphify_reader``.

The reader shells out to the ``graphify`` CLI for BFS queries and reads
graph.json directly for relationship lookups. We patch the CLI out so the
tests don't depend on packaging or PATH.
"""
from __future__ import annotations

import pytest

from tools.graphify_reader import (
    GraphEdge,
    GraphifyGraph,
    GraphifyReader,
    build_graphify_reader,
)


@pytest.fixture
def sample_graph(tmp_path):
    """Produce a minimal graph.json layout for tests."""
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        """
{
  "directed": false,
  "multigraph": false,
  "graph": {},
  "nodes": [
    {"id": "a", "label": "alpha", "file_type": "code", "source_file": "x.py"},
    {"id": "b", "label": "beta", "file_type": "concept", "community": 1, "source_file": "y.md"},
    {"id": "c", "label": "gamma", "file_type": "code", "source_file": "z.py", "community": 1}
  ],
  "links": [
    {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED", "confidence_score": 1.0},
    {"source": "b", "target": "c", "relation": "references", "confidence": "INFERRED", "confidence_score": 0.85}
  ],
  "hyperedges": []
}
""".strip()
    )
    return graph_path


def test_load_returns_graph_with_nodes_and_edges(sample_graph) -> None:
    graph = GraphifyGraph.load(sample_graph)
    assert len(graph.nodes()) == 3
    assert len(graph.edges()) == 2
    assert all(isinstance(edge, GraphEdge) for edge in graph.edges())


def test_find_node_by_label_matches_norm_label(sample_graph) -> None:
    graph = GraphifyGraph.load(sample_graph)
    node = graph.find_node_by_label("alpha")
    assert node is not None
    assert node.node_id == "a"


def test_outgoing_and_incoming_partition_edges(sample_graph) -> None:
    graph = GraphifyGraph.load(sample_graph)
    b = graph.find_node_by_label("beta")
    assert b is not None
    incoming = graph.incoming(b.node_id)
    outgoing = graph.outgoing(b.node_id)
    assert [edge.source for edge in incoming] == ["a"]
    assert [edge.target for edge in outgoing] == ["c"]


def test_nodes_in_community_filters_by_community(sample_graph) -> None:
    graph = GraphifyGraph.load(sample_graph)
    community_nodes = graph.nodes_in_community(1)
    assert {node.node_id for node in community_nodes} == {"b", "c"}


def test_query_concepts_raises_when_cli_unavailable(sample_graph, monkeypatch) -> None:
    """Force a non-existent CLI path so the assumption is portable."""
    fake_path = "/nonexistent/graphify-cli-binary-for-tests"
    reader = GraphifyReader(graph_path=sample_graph, cli_path=fake_path)
    assert reader.available is False  # explicit fake path is not on PATH
    with pytest.raises(RuntimeError):
        reader.query_concept("alpha")


def test_format_for_prompt_returns_empty_when_unavailable(sample_graph) -> None:
    """With no CLI available, format_for_prompt degrades to an empty string.

    The knowledge retrieval node relies on this for graceful degradation when
    Graphify isn't installed alongside VulkanMind.
    """
    fake_path = "/nonexistent/graphify-cli-binary-for-tests"
    reader = GraphifyReader(graph_path=sample_graph, cli_path=fake_path)
    assert reader.available is False
    assert reader.format_for_prompt("alpha") == ""


def test_build_graphify_reader_returns_none_for_missing_graph(tmp_path) -> None:
    assert build_graphify_reader({"graph_path": str(tmp_path / "missing.json")}) is None


def test_build_graphify_reader_returns_reader_for_existing_graph(sample_graph) -> None:
    reader = build_graphify_reader({"graph_path": str(sample_graph)})
    assert isinstance(reader, GraphifyReader)
    assert reader.graph.path == sample_graph
