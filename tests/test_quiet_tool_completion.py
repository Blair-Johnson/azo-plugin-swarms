"""Agent-initiated messaging needs no success interrupt; failures remain visible."""
import asyncio
from unittest.mock import AsyncMock

import pytest
import swarm
from test_recovery import runtime
from test_swarm_tools import tool_runtime


@pytest.mark.parametrize("action", ["bcast", "post", "interrupt", "continue", "cancel"])
@pytest.mark.parametrize("outcome", ["info", "warning", "exception"])
def test_background_notification_policy(tool_runtime, action, outcome):
    t = tool_runtime
    c, state = t.controller, t.runtime.state
    state.pending_interrupts = ["Unrelated notice"]
    handler = AsyncMock(return_value=dict(summary="Operation result", level=outcome))
    if outcome == "exception":
        handler.side_effect = ValueError("Delivery failed")
    setattr(c, "post" if action == "post" else "control", handler)
    if action == "post":
        receipt = c.post_message("sw0", 0, "general", "hello", "stable", state)
    elif action == "bcast":
        receipt = c.broadcast("sw0", "hello", state=state)
    else:
        method = "continue_swarm" if action == "continue" else action
        receipt = getattr(c, method)("sw0", state)
    assert receipt and not t.futures[0].done()
    result = asyncio.run(t.factories[0]())
    result["access"] = {"sw0": "", "sw1": ""}
    t.futures[0].set_result(result)
    check = swarm.SwarmCompletionCheck(c)
    check(state)
    check(state)
    assert state.swarm_access["sw1"] == ""  # Quiet completion still applies state.
    assert not c.pending
    expected = ["Unrelated notice"]
    if action not in {"bcast", "post"} or outcome != "info":
        expected.append(result["summary"])
    assert state.pending_interrupts == expected


def test_real_board_post_and_duplicate_are_quiet(tool_runtime):
    t = tool_runtime
    c, state = t.controller, t.runtime.state
    for index in range(2):
        c.post_message("sw0", 0, "general", "durable", "stable", state)
        result = asyncio.run(t.factories[index]())
        assert "committed" in result["summary"]
        t.futures[index].set_result(result)
        swarm.SwarmCompletionCheck(c)(state)
    assert len(t.runtime.store.messages("pod-1", "general")) == 1
    assert not state.pending_interrupts and not c.pending


@pytest.mark.parametrize("action", ["bcast", "post"])
def test_future_failure_still_notifies_parent(tool_runtime, action):
    t = tool_runtime
    c, state = t.controller, t.runtime.state
    if action == "bcast":
        c.broadcast("sw0", "hello", state=state)
    else:
        c.post_message("sw0", 0, "general", "hello", "stable", state)
    t.futures[0].set_exception(RuntimeError("IO loop failed"))
    swarm.SwarmCompletionCheck(c)(state)
    assert state.pending_interrupts == ["Swarm operation failed: IO loop failed"]
    assert not c.pending
