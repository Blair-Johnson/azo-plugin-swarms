"""Controller regression coverage under v2 ownership, typed requests, and AgentZooSessionDriver."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from agent_zoo.features.commands import Command, CommandError, CommandRegistry
import swarm
from conftest import execute_swarm_request
from test_adoption import client_for, lineage
from test_recovery import runtime


def test_registration_exposes_one_raw_user_command_without_work(monkeypatch):
    features = []
    forbidden = Mock(side_effect=AssertionError("registration must not do work"))
    for attribute in ("allocate_pool", "resolve_pool", "reserve_member_launch", "store_for"):
        monkeypatch.setattr(swarm, attribute, forbidden)
    swarm.register_features(SimpleNamespace(add=features.append), session=SimpleNamespace(), config={})
    commands = [c for f in features for c in f.components if isinstance(c, Command)]
    command, = commands
    assert command.path == "/swarm" and command.raw and not command.args
    assert not command.cancels_lifecycle
    raw = '  bcast sw0 -p 0 "  first\n\t\u96ea last  "  '
    assert CommandRegistry(commands).parse("/swarm" + raw).arguments == {"raw_args": raw}
    forbidden.assert_not_called()


@pytest.mark.parametrize("raw", ["", "-n0", "-n3 -p2", 'bcast "unterminated', "cancel sw0 sw1"])
def test_invalid_command_has_no_background_or_storage_work(runtime, monkeypatch, raw):
    ctx = SimpleNamespace(state=runtime.state, emit=Mock(), submit_io=Mock(return_value="op_1"))
    forbidden = Mock(side_effect=AssertionError("invalid command side effect"))
    monkeypatch.setattr(swarm, "store_for", forbidden)
    with pytest.raises(CommandError):
        swarm.SwarmController({}).command(ctx, raw_args=raw)
    ctx.submit_io.assert_not_called(); ctx.emit.assert_not_called(); forbidden.assert_not_called()


def test_valid_slash_control_uses_detached_shared_request(runtime):
    r = runtime
    c = swarm.SwarmController({})
    c.grants = r.state.grants.copy()
    ctx = SimpleNamespace(state=r.state, emit=Mock(), submit_io=Mock(return_value="op_1"))
    c.command(ctx, raw_args='bcast sw0 -p 0 "  first\nlast  "')
    assert ctx.submit_io.call_count == 1
    task = ctx.submit_io.call_args.args[0]
    assert isinstance(task, swarm.SwarmStoreRequest) and task.op == "control_begin"
    _, _, meta = c.in_flight["op_1"]
    assert meta["message"] == "  first\nlast  " and meta["pods"] == (0,)
    assert meta["operation_id"]
    # Mutating live state after command() does not affect the detached snapshot captured in the workflow.
    with r.store.mutation() as (records, _):
        for member in r.members:
            records.put("members", member["session_id"], dict(recovery_status="ready_paused"))
    r.state._agent_zoo_context["project_name"] = "changed"
    store, pool, members, op = swarm.SwarmStoreDriver().execute(task)
    assert pool["id"] == "sw0" and len(members) == 1 and op["pods"] == (0,)


def test_allocation_unique_names_stable_ids_and_balanced_topology(runtime):
    r = runtime
    store, pool = swarm.allocate_pool(r.state, swarm.parse_command("-n4 -p2 --name explore --channels comms,breakthroughs"))
    assert pool["id"] == "sw1" and pool["channels"] == ["comms", "breakthroughs"]
    assert [m["label"] for p in pool["pods"] for m in p["members"]] == ["sw1p0a0", "sw1p0a1", "sw1p1a0", "sw1p1a1"]
    assert len({m["session_id"] for p in pool["pods"] for m in p["members"]}) == 4
    assert swarm.resolve_pool(r.state, "explore")[1]["id"] == "sw1"
    with pytest.raises(ValueError, match="already exists"):
        swarm.allocate_pool(r.state, swarm.parse_command("-n1 --name explore"))
    assert r.pool["resource_uid"] != pool["resource_uid"]


@pytest.mark.parametrize("actor", ["other-owner", "peer"])
def test_forged_saved_parent_binding_does_not_authorize_controls(runtime, actor):
    r = runtime
    state = SimpleNamespace(**vars(r.state))
    state._session_id = r.members[0]["session_id"] if actor == "peer" else actor
    state.swarm_access = {"sw0": ""}
    result = execute_swarm_request(swarm.SwarmController({}), state, swarm.ControlRequest("interrupt", "sw0"))
    assert result["summary"].startswith("sw0:") and result["level"] == "warning"
    assert r.store.records().list("operations") == []


def test_same_session_new_instance_cannot_reuse_grant(runtime):
    r = runtime
    state = SimpleNamespace(**vars(r.state)); state._instance_id = "foreign-copy"
    state.grants = {"sw0": dict(r.store.grant, instance_id=state._instance_id)}
    result = execute_swarm_request(swarm.SwarmController({}), state, swarm.ControlRequest("interrupt", "sw0"))
    assert "Stale owner instance" in result["summary"]
    assert not r.store.records().list("operations")


@pytest.mark.parametrize("action", ["bcast", "interrupt", "continue", "cancel"])
def test_all_controls_report_partial_outcomes_and_durable_opid(lineage, monkeypatch, action):
    l = lineage; r = l.runtime
    with r.store.mutation() as (records, _):
        for member in r.members:
            records.put("members", member["session_id"], dict(recovery_status="ready_paused"))
    client, _ = client_for(monkeypatch, l.a)
    request = swarm.ControlRequest(action, "sw0", message="text")
    result = execute_swarm_request(swarm.SwarmController({}), r.state, request)
    assert "1 of 2 agents" in result["summary"] and "1 unavailable" in result["summary"]
    operation = r.store.records().get("operations", request.operation_id)
    assert operation["epoch"] == 1 and len(operation["outcomes"]) == 2
    client.send_json.assert_awaited_once()
    frame = client.send_json.call_args.args[0]
    if action == "bcast":
        assert frame["channel"] == "user_text" and frame["payload"] == {"text": "text"}
    elif action == "cancel":
        assert frame["channel"] == "control" and frame["payload"]["type"] == "shutdown"
        assert r.store.pool()["desired_state"] == "cancelled" and "shutdown requested" in result["summary"]
    else:
        assert frame["channel"] == "slash_command" and frame["payload"] == {"raw": "/" + action}


def test_cancelled_pool_stays_terminal_and_not_default(runtime):
    r = runtime
    with r.store.mutation() as (records, pool):
        pool.update(desired_state="cancelled", phase="cancelled"); records.put("swarm", "pool", pool)
    with pytest.raises(ValueError):
        swarm.resolve_pool(r.state, None)
    result = execute_swarm_request(swarm.SwarmController({}), r.state, swarm.ControlRequest("continue", "sw0"))
    assert "cancelled" in result["summary"] and not r.store.records().list("operations")


def test_control_preflight_refuses_incomplete_recovery(runtime):
    r = runtime
    result = execute_swarm_request(swarm.SwarmController({}), r.state, swarm.ControlRequest("continue", "sw0"))
    assert "incomplete" in result["summary"] and "no work sent" in result["summary"]
    assert not r.store.records().list("operations")


def test_swarm_lock_fence_registered_on_session_driver():
    driver = swarm.make_swarm_session_driver()
    assert swarm.SWARM_LOCK_FENCE_NAME in driver._lock_fences
    assert isinstance(driver._lock_fences[swarm.SWARM_LOCK_FENCE_NAME], swarm.SwarmLockFenceAdapter)
