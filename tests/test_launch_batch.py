"""Spawn the whole batch without waiting for individual worker readiness via SpawnSession."""
import threading
from types import SimpleNamespace

import pytest
from agent_zoo.events import SessionSpawnResult
import swarm
from conftest import run_swarm_workflow
from test_recovery import runtime


@pytest.mark.parametrize("recovering", [False, True])
@pytest.mark.parametrize("failed_worker", [None, 0])
def test_all_workers_spawn_before_any_becomes_ready(runtime, monkeypatch, recovering, failed_worker):
    r = runtime
    store, pool = swarm.allocate_pool(r.state, swarm.parse_command("-n16 -p4"))
    members = [m for pod in pool["pods"] for m in pod["members"]]
    expected_ids = [m["session_id"] for m in members]
    checkpoints = {sid: swarm.CheckpointRef(sid, f"rev-{i}") for i, sid in enumerate(expected_ids)}
    monkeypatch.setattr(swarm, "pin_checkpoint", lambda state, store, member: checkpoints[member["session_id"]])

    ready_gate = threading.Event()
    all_spawned = threading.Event()
    spawned_ids: set[str] = set()
    lock = threading.Lock()

    class _BatchLauncher:
        def prepare(self, launch, *, cwd=None, env=None):
            if recovering:
                assert launch.source.checkpoint == checkpoints[launch.target.session_id]
            return SimpleNamespace(request=launch, ready_file="/offline/ready")

        def start(self, plan):
            sid = plan.request.target.session_id
            iid = plan.request.target.instance_id
            with lock:
                spawned_ids.add(sid)
                if len(spawned_ids) == len(members):
                    all_spawned.set()

            def _poll():
                if not ready_gate.is_set():
                    return SimpleNamespace(status="starting", ready=None)
                if failed_worker is not None and sid == expected_ids[failed_worker]:
                    return SimpleNamespace(status="unknown", error="readiness failed", ready=None)
                return SimpleNamespace(
                    status="ready",
                    ready=dict(
                        format_version=1, session_id=sid, instance_id=iid, pid=100,
                        process_identity=r.identity.to_dict(), websocket_url="ws://127.0.0.1:12000/session",
                        startup_mode="paused", paused=True, revision_id=f"rev-{expected_ids.index(sid)}",
                    ),
                )

            return SimpleNamespace(
                process=SimpleNamespace(pid=100, poll=lambda: None, terminate=lambda: None, kill=lambda: None, wait=lambda timeout=None: 0),
                ready_file="/offline/ready", stdout_path="out", stderr_path="err", poll=_poll,
            )

    monkeypatch.setattr(
        swarm, "finalize_member_ready",
        lambda state, store, member, attempt_id, res, **kw: dict(
            label=member["label"], session_id=member["session_id"], state="ready_paused",
        ),
    )
    import agent_zoo.drivers.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "capture_process_identity", lambda pid: r.identity)

    driver = swarm.make_swarm_session_driver(launcher=_BatchLauncher(), store_root=r.state.settings.store_root)
    controller = swarm.SwarmController({})
    holder: dict = {}

    def _run():
        holder["outcomes"] = run_swarm_workflow(
            controller, r.state,
            controller._wf_launch_members(r.state, store, pool, recovering=recovering, settings=r.state.settings),
            session_driver=driver,
        )

    t = threading.Thread(target=_run)
    t.start()
    try:
        assert all_spawned.wait(timeout=5.0), f"only spawned {len(spawned_ids)}/{len(members)}"
        assert spawned_ids == set(expected_ids)
        assert t.is_alive()
    finally:
        ready_gate.set()
        t.join(timeout=10.0)
        driver.close()
    assert not t.is_alive()
    outcomes = holder["outcomes"]
    assert [row["session_id"] for row in outcomes] == expected_ids
    for index, row in enumerate(outcomes):
        assert row["state"] == ("blocked" if index == failed_worker else "ready_paused")
    if failed_worker is not None:
        record = store.records().get("members", expected_ids[failed_worker])
        assert record["recovery_status"] == "blocked"
        assert "readiness failed" in record["error"]
