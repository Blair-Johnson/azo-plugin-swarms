"""Parent notices reach model input; cancelled pools stay silent on restore."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from agent_utils import Entry, PipelineBuilder, Session, core_feature, interrupts_feature
from agent_utils.core.events import IoCompleted
from agent_utils.core.io import IoTable
from agent_zoo.events import RunPluginTask
from agent_zoo.features.projection.transcript import entry_to_items
from agent_zoo.llm.providers.openai_responses import _extract_system, _messages_to_responses_input
import swarm
from conftest import run_swarm_workflow


@pytest.mark.parametrize("action", ["create", "bcast", "interrupt", "continue", "cancel"])
def test_parent_completion_reaches_model_as_system_generated_user(action):
    session = Session(system_prompt="Initial instructions")
    controller = swarm.SwarmController({}, session)
    handler = swarm.SwarmCompletionHandler(controller)
    session.pipeline = (
        PipelineBuilder()
        .add(core_feature())
        .add(interrupts_feature())
        .add(
            swarm.Feature(
                "swarm_test",
                components=[handler],
                order=[swarm.IoFence, swarm.SwarmCompletionHandler, swarm.InterruptDelivery],
            )
        )
        .build()
    )
    session.ensure_initialized()
    session.state._session_kind = "default"
    session.state._session_id = "parent"
    session.state._instance_id = "inst-1"
    session.state.entries.append(
        Entry(
            messages=[{"role": "assistant", "content": "Previous assistant response"}],
            index=len(session.state.entries),
            step=session.state.step,
        )
    )
    text = '  Review this code.\n"Keep this quotation"\n  '
    result = dict(summary=f"sw1: {action} complete.")
    if action == "create":
        result["model_message"] = "The user launched swarm sw1. Read the Swarms skill."
    if action == "bcast":
        result.update(pool={"id": "sw1"}, operation=dict(
            action="bcast", message=text, outcomes=[{"label": "sw1p0a0"}]))
    op_id = session.state.io.submit(RunPluginTask(fn=lambda: None), tag="swarm")
    controller.in_flight[op_id] = (controller.identity(session.state), "command", {"action": action})
    session.step(IoCompleted(op_id=op_id, value=result))
    session.render()
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
    notice = controller.complete(SimpleNamespace(pending_interrupts=[]), result)
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
                                pending_interrupts=[], io=IoTable())
        controller = swarm.SwarmController({})
        recover_calls = []

        def fake_recover(st, store, req):
            recover_calls.append((st, store, req))
            return dict(summary="sw1: 2 agents restored (paused).")
            yield  # pragma: no cover

        controller._wf_recover = fake_recover
        result = run_swarm_workflow(controller, state, controller._wf_recover_owned(state, stores=stores))
        controller.in_flight["op_restore"] = (controller.identity(state), "startup", {})
        swarm.SwarmCompletionHandler(controller).handle(
            IoCompleted(op_id="op_restore", value=result), state, lambda _ev: None
        )
        assert not any("sw0" in notice for notice in state.pending_interrupts)
        assert state.swarm_access["sw0"] == ""  # Retain inspection access.
        if other == "active":
            assert recover_calls == [(state, active, {})]
            assert state.pending_interrupts == ["sw1: 2 agents restored (paused)."]
        elif other == "error":
            assert not recover_calls
            assert "sw1: recovery blocked: missing journal" in state.pending_interrupts[0]
        else:
            assert not recover_calls
            assert result["summary"] == "" and state.pending_interrupts == []
