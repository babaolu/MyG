from orchestrator.router import classify_task_type, router_node


def test_router_classifies_code_generation() -> None:
    assert classify_task_type("generate Vulkan-Hpp code with VMA") == "code_generation"


def test_router_node_returns_structured_task_type() -> None:
    result = router_node({"user_request": "debug validation sync error", "agent_trace": []})
    assert result["task_type"] == "debug"
    assert result["agent_trace"]
