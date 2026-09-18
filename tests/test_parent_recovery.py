"""Parent restoration must not run saved work or invalidate proven live children."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from agent_utils.files.buffer_manager import BufferManager
import swarm
from test_recovery import runtime, reserve


def notify(state, controller):
    event = SimpleNamespace(payload=dict(session_id=state._session_id,
        instance_id=state._instance_id, restored=True, revision_id="saved",
        startup_mode="normal", paused=False))
    swarm.SwarmRuntimeReady(controller).handle_harness_event(event, state)
    return event


@pytest.mark.parametrize("saved_access", [True, False])
def test_parent_pauses_before_first_restored_work_pass(runtime, saved_access):
    state = runtime.state
    state._instance_id = "restored-parent"
    state.grants = {}
    if not saved_access:
        state.swarm_access = {}
    state._runtime_startup_paused = False
    state.buffer_manager = BufferManager()
    controller = swarm.SwarmController({})
    notify(state, controller)
    assert state._runtime_startup_paused is True
    assert state.buffer_manager.special_buffer_namespaces() == ["swarm:"]
    assert len(controller.ready_events) == 1
    assert controller.ready_events[0][0]["paused"] is True
    state.session_launcher.start.assert_not_called()


def test_unrelated_restored_session_hold_can_be_released(runtime):
    state = runtime.state
    state._session_id = "unrelated"
    state._instance_id = "other"
    state.swarm_access = {}
    state.grants = {}
    state._runtime_startup_paused = False
    state.buffer_manager = BufferManager()
    controller = swarm.SwarmController({})
    notify(state, controller)
    assert state._runtime_startup_paused is True
    controller.apply(state, {"release_startup_hold": True})
    assert state._runtime_startup_paused is False


def test_proven_parent_successor_reattaches_children_without_epoch_rotation(runtime, monkeypatch):
    r = runtime
    member = r.members[0]
    attempt = reserve(r, member, identity=r.identity.to_dict())
    before = r.store.pool()
    old_parent = swarm.SwarmStore(r.store.durable, r.store.cache, grant=r.store.grant)
    child_grant = dict(r.store.grant, role="member", session_id=member["session_id"],
                       instance_id=attempt["instance_id"], attempt_id=attempt["attempt_id"])
    child_store = swarm.SwarmStore(r.store.durable, r.store.cache, grant=child_grant)
    r.state._instance_id = "proven-successor"
    r.state.grants = {}
    monkeypatch.setattr(swarm, "owner_replaced", lambda state, pool: True)
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "alive")
    controller = swarm.SwarmController({})
    calls = []

    async def no_launch(*args, **kwargs):
        pytest.fail("Verified parent replacement must not relaunch live children")

    async def interrupt(state, request):
        calls.append(deepcopy(request))
        assert request["action"] == "interrupt" and request["target"] == before["id"]
        return dict(summary="sent=2; pause requested", level="info")

    monkeypatch.setattr(controller, "launch_members", no_launch)
    monkeypatch.setattr(controller, "control", interrupt)
    result = asyncio.run(controller.recover(r.state, r.store, {}))
    after = r.store.pool()
    assert calls and "pause requested" in result["summary"]
    assert after["owner"]["epoch"] == before["owner"]["epoch"]
    assert after["owner"]["token"] == before["owner"]["token"]
    assert after["owner"]["parent_instance_id"] == "proven-successor"
    assert r.store.records().get("attempts", member["session_id"]) == attempt
    with pytest.raises(ValueError, match="[Ss]tale"):
        old_parent.post("pod-1", "general", "late parent write", "parent")
    child_store.post("pod-1", "general", "child grant survived", member["session_id"])
    r.state.session_launcher.start.assert_not_called()
