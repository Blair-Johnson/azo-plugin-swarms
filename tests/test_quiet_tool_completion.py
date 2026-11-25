"""Agent-initiated messaging needs no success interrupt; failures remain visible."""
import pytest
from agent_utils.core.events import IoCompleted, IoFailed
import swarm
from conftest import drain_first_request
from test_recovery import runtime
from test_swarm_tools import tool_runtime


@pytest.mark.parametrize("action", ["bcast", "post", "interrupt", "continue", "cancel"])
@pytest.mark.parametrize("outcome", ["info", "warning", "exception"])
def test_background_notification_policy(tool_runtime, action, outcome):
    t = tool_runtime
    c, state = t.controller, t.runtime.state
    state.pending_interrupts = ["Unrelated notice"]
    if action == "post":
        def fake_post(snapshot, request):
            if outcome == "exception":
                raise ValueError("Delivery failed")
            return dict(summary="Operation result", level=outcome)
            yield  # pragma: no cover
        c._wf_post = fake_post
        receipt = c.post_message("sw0", 0, "general", "hello", "stable", state)
    else:
        def fake_control(snapshot, request, *, store=None):
            if outcome == "exception":
                raise ValueError("Delivery failed")
            return dict(summary="Operation result", level=outcome)
            yield  # pragma: no cover
        c._wf_control = fake_control
        if action == "bcast":
            receipt = c.broadcast("sw0", "hello", state=state)
        else:
            method = "continue_swarm" if action == "continue" else action
            receipt = getattr(c, method)("sw0", state)
    effects = t.take_effects()
    if effects:
        effect = effects[0]
        result = drain_first_request(c, state, effect.request)
    else:
        # Generator completed immediately when fake_control didn't yield.
        assert receipt
        return
    result["access"] = {"sw0": "", "sw1": ""}
    c.in_flight[effect.op_id] = (c.identity(state), "tool", {"action": action})
    handler_comp = swarm.SwarmCompletionHandler(c)
    emitted = []
    handler_comp.handle(IoCompleted(op_id=effect.op_id, value=result), state, emitted.append)
    handler_comp.handle(IoCompleted(op_id=effect.op_id, value=result), state, emitted.append)
    assert state.swarm_access["sw1"] == ""  # Quiet completion still applies state.
    assert not c.in_flight
    expected = ["Unrelated notice"]
    if action not in {"bcast", "post"} or outcome != "info":
        expected.append(result["summary"])
    assert state.pending_interrupts == expected


def test_real_board_post_and_duplicate_are_quiet(tool_runtime):
    t = tool_runtime
    c, state = t.controller, t.runtime.state
    for _ in range(2):
        c.post_message("sw0", 0, "general", "durable", "stable", state)
        effect, = t.take_effects()
        result = drain_first_request(c, state, effect.request)
        assert "committed" in result["summary"]
        c.in_flight[effect.op_id] = (c.identity(state), "tool", {"action": "post"})
        swarm.SwarmCompletionHandler(c).handle(
            IoCompleted(op_id=effect.op_id, value=result), state, lambda _ev: None
        )
    assert len(t.runtime.store.messages("pod-1", "general")) == 1
    assert not state.pending_interrupts and not c.in_flight


@pytest.mark.parametrize("action", ["bcast", "post"])
def test_future_failure_still_notifies_parent(tool_runtime, action):
    t = tool_runtime
    c, state = t.controller, t.runtime.state
    if action == "bcast":
        c.broadcast("sw0", "hello", state=state)
    else:
        c.post_message("sw0", 0, "general", "hello", "stable", state)
    effect, = t.take_effects()
    # Direct top-level failure (not wrapped inside _wf) delivers failure notice to parent.
    c._workflows.clear()
    c.in_flight[effect.op_id] = (c.identity(state), "tool", {"action": action})
    swarm.SwarmCompletionHandler(c).handle(
        IoFailed(op_id=effect.op_id, error=RuntimeError("IO loop failed")), state, lambda _ev: None
    )
    assert state.pending_interrupts == ["Swarm operation failed: IO loop failed"]
    assert not c.in_flight
