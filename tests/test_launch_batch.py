"""Spawn the whole batch without waiting for individual worker readiness."""
import asyncio
from types import SimpleNamespace

import pytest
import swarm
from test_recovery import runtime


@pytest.mark.parametrize("recovering", [False, True])
@pytest.mark.parametrize("failed_worker", [None, 0])
def test_all_workers_spawn_before_any_becomes_ready(runtime, monkeypatch, recovering, failed_worker):
    r = runtime
    store, pool = swarm.allocate_pool(r.state, swarm.parse_command("-n16 -p4"))
    members = [m for pod in pool["pods"] for m in pod["members"]]
    expected_ids = [m["session_id"] for m in members]
    checkpoints = {sid: object() for sid in expected_ids}
    monkeypatch.setattr(swarm, "pin_checkpoint", lambda state, store, member: checkpoints[member["session_id"]])

    async def scenario():
        ready_gate, all_waiting = asyncio.Event(), asyncio.Event()
        waiting = set()
        controller = swarm.SwarmController({})

        def launch(state, swarm_id, member_id, settings, *, handles, checkpoint):
            assert checkpoint is (checkpoints[member_id] if recovering else None)
            return SimpleNamespace(session_id=member_id)

        async def ready(state, store, member, handle):
            assert handle.session_id == member["session_id"]
            waiting.add(member["session_id"])
            if len(waiting) == len(members):
                all_waiting.set()
            await ready_gate.wait()
            if failed_worker is not None and member["session_id"] == expected_ids[failed_worker]:
                raise ValueError("readiness failed")
            return dict(label=member["label"], session_id=member["session_id"], state="ready_paused")

        monkeypatch.setattr(swarm, "launch_member", launch)
        monkeypatch.setattr(swarm, "await_member_ready", ready)
        task = asyncio.create_task(controller.launch_members(
            r.state, store, pool, recovering=recovering, settings=r.state.settings))
        try:
            await asyncio.wait_for(all_waiting.wait(), timeout=3)
            assert waiting == set(expected_ids)
            assert not task.done()
        finally:
            ready_gate.set()
            outcomes = await task
        assert [row["session_id"] for row in outcomes] == expected_ids
        for index, row in enumerate(outcomes):
            assert row["state"] == ("blocked" if index == failed_worker else "ready_paused")
        if failed_worker is not None:
            record = store.records().get("members", expected_ids[failed_worker])
            assert record["recovery_status"] == "blocked"
            assert record["error"] == "readiness failed"

    asyncio.run(scenario())
