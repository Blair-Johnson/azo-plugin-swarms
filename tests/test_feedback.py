"""Human-readable feedback still reaches both parent and status bar."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import swarm

OP_ID = "a" * 32


@pytest.mark.parametrize("action, expected", [
    ("bcast", "message sent to 4 agents"),
    ("interrupt", "pause requested for 4 agents"),
    ("continue", "continue requested for 4 agents"),
    ("cancel", "shutdown requested for 4 agents"),
    ("create", "4 agents ready (paused)"),
    ("recover", "4 agents restored (paused)"),
])
def test_completion_is_plain_language_without_internal_ids(action, expected):
    ready = action in {"create", "recover"}
    operation = dict(id=OP_ID, action=action, outcomes=[
        dict(state="ready_paused" if ready else "sent") for _ in range(4)])
    text = swarm.operation_summary(dict(id="sw0", name="research"), operation)
    assert text == f"research (sw0): {expected}."
    assert OP_ID not in text and len(text) < 90
    ctx = SimpleNamespace(state=SimpleNamespace(), session=SimpleNamespace(inject=Mock()))
    result = swarm.SwarmController({}).complete(ctx, dict(summary=text))
    ctx.session.inject.assert_called_once_with(text, role="user", system_generated=True)
    assert result == swarm.CommandResult.notice(text, level="info")


@pytest.mark.parametrize("action", ["bcast", "interrupt", "continue", "cancel", "recover"])
def test_control_receipt_has_no_diagnostic_boilerplate(action):
    text = swarm.SwarmController.receipt(dict(action=action, target="sw0", pods=None, operation_id=OP_ID))
    assert "sw0" in text and OP_ID not in text and len(text) < 60
    assert "pending" not in text and "polling" not in text


def test_scope_singular_and_actual_failure_are_readable():
    operation = dict(id=OP_ID, action="bcast", pods=[2], outcomes=[dict(state="sent")])
    assert swarm.operation_summary({"id": "sw0"}, operation) == "sw0: message sent to 1 agent — pods 2."
    operation["outcomes"] += [dict(state="unknown", label="sw0p2a1", error="Send timed out")]
    text = swarm.operation_summary({"id": "sw0"}, operation)
    assert "1 of 2 agents" in text and "1 delivery unknown" in text
    assert "sw0p2a1: Send timed out" in text
    assert "proof" not in text and "retry" not in text


def test_long_runtime_ids_and_errors_stay_out_of_feedback():
    text = swarm.brief_error("Runtime " + OP_ID + " failed\n" + "x" * 500)
    assert OP_ID not in text and "\n" not in text and len(text) <= 160


def test_record_save_failure_is_not_hidden():
    text = swarm.operation_summary({"id": "sw0"}, dict(id=OP_ID, action="interrupt",
        outcomes=[dict(state="sent")]), audit_error="Disk full")
    assert "pause requested for 1 agent" in text
    assert "Could not save the operation record: Disk full" in text
