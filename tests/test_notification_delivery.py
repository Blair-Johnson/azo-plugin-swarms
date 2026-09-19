"""Parent notices reach model input; cancelled pools stay silent on restore."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from agent_utils import Session
from agent_utils.components import MessageRenderer
from agent_zoo.runtime.projection import entry_to_items
from agents.llm import _extract_system, _messages_to_responses_input
import swarm


@pytest.mark.parametrize("action", ["create", "bcast", "interrupt", "continue", "cancel"])
def test_parent_completion_reaches_model_as_system_generated_user(action):
    session = Session(system_prompt="Initial instructions")
    session.pipeline = [MessageRenderer()]
    session.state._session_kind = "default"
    session.inject("Previous assistant response", role="assistant")
    text = '  Review this code.\n"Keep this quotation"\n  '
    result = dict(summary=f"sw1: {action} complete.")
    if action == "create":
        result["model_message"] = "The user launched swarm sw1. Read the Swarms skill."
    if action == "bcast":
        result.update(pool={"id": "sw1"}, operation=dict(
            action="bcast", message=text, outcomes=[{"label": "sw1p0a0"}]))
    ctx = SimpleNamespace(session=session, state=session.state)
    notice = swarm.SwarmController({}).complete(ctx, result)
    entry = session.state.entries[-1]
    assert entry.role == "user" and entry.system_generated
    assert entry_to_items(entry)[0].role == "interrupt"
    content = entry.messages[0]["content"]
    instructions, start = _extract_system(session.state.rendered_messages)
    assert instructions == "Initial instructions"
    items = _messages_to_responses_input(session.state.rendered_messages[start:])
    assert any(item.get("role") == "user" and any(
        block.get("text") == content for block in item.get("content", [])
    ) for item in items)
    if action == "bcast":
        assert content.endswith(text) and "sw1p0a0" in content
    assert notice == swarm.CommandResult.notice(result["summary"], level="info")


@pytest.mark.parametrize("other", [None, "active", "error"])
def test_repeated_restore_skips_cancelled_pool_but_preserves_other_notices(other):
    cancelled = SimpleNamespace(pool=Mock(return_value={"desired_state": "cancelled"}))
    active = SimpleNamespace(pool=Mock(return_value={"desired_state": "running"}))
    stores = [("sw0", cancelled, None)]
    if other == "active":
        stores.append(("sw1", active, None))
    elif other == "error":
        stores.append(("sw1", None, ValueError("missing journal")))
    for instance in ("before-reload", "after-reload"):
        state = SimpleNamespace(_session_kind="default", _session_id="parent",
                                _instance_id=instance, swarm_access={}, grants={},
                                pending_interrupts=[])
        controller = swarm.SwarmController({})
        controller.recover = AsyncMock(return_value=dict(summary="sw1: 2 agents restored (paused)."))
        result = asyncio.run(controller.recover_owned(state, stores=stores))
        future = Future()
        future.set_result(result)
        controller.pending.append((future, controller.identity(state)))
        swarm.SwarmCompletionCheck(controller)(state)
        assert not any("sw0" in notice for notice in state.pending_interrupts)
        assert state.swarm_access["sw0"] == ""  # Retain inspection access.
        if other == "active":
            controller.recover.assert_awaited_once_with(state, active, {})
            assert state.pending_interrupts == ["sw1: 2 agents restored (paused)."]
        elif other == "error":
            controller.recover.assert_not_awaited()
            assert "sw1: recovery blocked: missing journal" in state.pending_interrupts[0]
        else:
            controller.recover.assert_not_awaited()
            assert result["summary"] == "" and state.pending_interrupts == []
