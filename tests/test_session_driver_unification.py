"""End-to-end deterministic SimulationRuntime + SimSessionDriver unification tests for Swarm."""
from __future__ import annotations

from agent_utils import Feature, PipelineBuilder, Session, core_feature, interrupts_feature
from agent_utils.runtime.simulation import SimOutcome, SimulationRuntime
from agent_zoo.drivers.sessions import SimSessionDriver
from agent_zoo.events import RunPluginTask, SessionArtifact, SessionDataReceived
from agent_zoo.features.special_buffers import special_buffers_feature
from tmux_pilot.process_identity import HostIdentity, ProcessIdentity
import swarm


def _make_sim_swarm(tmp_path, monkeypatch):
    (tmp_path / "registry").mkdir(parents=True, exist_ok=True)
    host = HostIdentity("sim-host", "sim-boot", "sim-ns", "sim")
    parent_ident = ProcessIdentity("sim-host", "sim-boot", "sim-ns", 9001, "start-9001")
    monkeypatch.setattr(swarm, "local_host_identity", lambda: host)
    monkeypatch.setattr(swarm, "native_self", lambda: parent_ident.to_dict())
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: parent_ident if pid == 9001 else None)

    config = {
        "llm": {"default": "offline", "models": {"offline": {"enabled": True}}},
    }
    session = Session(system_prompt="sim parent", cwd=str(tmp_path))
    controller = swarm.SwarmController(config, session)
    pipeline = (
        PipelineBuilder()
        .add(core_feature())
        .add(interrupts_feature())
        .add(special_buffers_feature())
        .add(
            Feature(
                "swarm_sim",
                components=[
                    swarm.SwarmBuffers(),
                    swarm.SwarmRuntimeReady(controller),
                    swarm.SwarmCompletionHandler(controller),
                ],
                order=[
                    swarm.IoFence,
                    swarm.RegisterSpecialBuffers,
                    swarm.SwarmBuffers,
                    swarm.SwarmRuntimeReady,
                    swarm.SwarmCompletionHandler,
                    swarm.InterruptDelivery,
                ],
            )
        )
        .build()
    )
    session.pipeline = pipeline
    session.ensure_initialized()
    st = session.state
    st._session_id = "parent"
    st._instance_id = "parent-sim"
    st._project_name = "test"
    st._store_root = str(tmp_path / "registry")
    st._agent_zoo_context = {"project_name": "test", "project_store_root": str(tmp_path / "registry")}
    st._persistence_local_root = str(tmp_path / "local")
    st.swarm_access = {}
    st.swarm_resource_uids = {}

    sim_sessions = SimSessionDriver().register(sim := SimulationRuntime(session, seed=42))
    sim.register_handler(
        RunPluginTask,
        lambda op_id, req, s: SimOutcome(value=req.fn(*req.args, **req.kwargs), delay=0.001),
    )
    return session, controller, sim, sim_sessions, config


def test_sim_full_swarm_lifecycle_spawn_bcast_race_retry_unknown_data_cancel(tmp_path, monkeypatch):
    session, controller, sim, sim_sessions, _ = _make_sim_swarm(tmp_path, monkeypatch)
    st = session.state

    # Simplify child bootstrap check in pure simulation (child subprocess isn't running to write bootstrap).
    def _sim_finalize(state, store, member, attempt_id, spawn_res, *, timeout=30.0):
        store.update_attempt(
            member["session_id"], expected_attempt=attempt_id, state="ready_paused",
            pid=spawn_res.pid, process_identity=spawn_res.process_identity, ready=spawn_res.ready,
        )
        with store.mutation() as (records, _):
            row = records.get("members", member["session_id"], {})
            row.update(session_id=member["session_id"], recovery_status="ready_paused", work_blocked=False)
            records.put("members", member["session_id"], row)
        return dict(label=member["label"], session_id=member["session_id"], state="ready_paused")

    monkeypatch.setattr(swarm, "finalize_member_ready", _sim_finalize)

    def _drain():
        sim._dispatch_effects(st.io.take_outbox())
        sim.run_until_idle()

    # 1. Create swarm with 2 agents across 2 pods via pure SpawnSession requests
    ctx = type("Ctx", (), {"state": st, "submit_io": st.io.submit})()
    receipt = controller.command(ctx, raw_args="-n 2 -p 2 --name simpool --profile offline")
    assert "simpool" in str(receipt)
    _drain()
    assert not controller.in_flight
    assert "sw0" in st.swarm_access
    assert len(sim_sessions.sessions) == 2

    store = swarm.store_for(st, "sw0")
    members = [m for p in store.pool()["pods"] for m in p["members"]]
    m0, m1 = members[0]["session_id"], members[1]["session_id"]

    # 2. Reload m0 twice in SimSessionDriver and broadcast:
    #    m0 gets 2 pre_send_race retries then succeeds; m1 gets delivery_unknown (NEVER retried!)
    inst0 = store.records().get("attempts", m0)["instance_id"]
    b0 = sim_sessions.add_lineage_successor(inst0, kind="reload", now=sim.now)
    c0 = sim_sessions.add_lineage_successor(b0.instance_id, kind="reload", now=sim.now)
    sim_sessions.inject_delivery_outcome(m0, "pre_send_race", error="CAS lost #1")
    sim_sessions.inject_delivery_outcome(m0, "pre_send_race", error="CAS lost #2")
    sim_sessions.inject_delivery_outcome(m1, "delivery_unknown", error="socket dropped after write")

    controller.broadcast("sw0", "hello unified driver", state=st)
    _drain()
    assert not controller.in_flight
    assert any(
        d["session_id"] == m0 and d["instance_id"] == c0.instance_id and d["payload"] == {"text": "hello unified driver"}
        for d in sim_sessions.delivered_inputs
    )
    assert not any(d["session_id"] == m1 for d in sim_sessions.delivered_inputs)
    # Warning summary surfaced because m1 had delivery_unknown (buffered by InterruptDelivery while last entry is user)
    delivered_texts = (
        list(st.pending_interrupts)
        + list(getattr(st, "_buffered_interrupts", []))
        + [str(m.get("content", "")) for e in st.entries for m in e.messages]
    )
    assert any("delivery unknown" in msg for msg in delivered_texts)
    bcast_ops = [op for op in store.records().list("operations") if op.get("action") == "bcast"]
    assert len(bcast_ops) == 1
    assert {row["session_id"]: row["state"] for row in bcast_ops[0]["outcomes"]} == {m0: "sent", m1: "unknown"}

    # 3. Structured swarm artifact transfer via SendSessionData
    art = SessionArtifact(name="sw0:pod-1:general", kind="json", data={"finding": 42}, content_type="application/json")
    controller.share_data("sw0", [art], channel="swarm:sw0:pod-1:general", message="board update", pods=[0], state=st)
    _drain()
    assert not controller.in_flight
    assert len(sim_sessions.delivered_data) == 1
    assert sim_sessions.delivered_data[0].artifacts[0].data == {"finding": 42}

    # Verify incoming SessionDataReceived on swarm: channel is captured by SwarmCompletionHandler
    sim_sessions.deliver_incoming_data(
        sim, sender_session_id=m0, channel="swarm:sw0:pod-1:general", artifacts=(art,), message="peer reply",
    )
    _drain()
    assert len(st.swarm_received_data) == 1
    assert isinstance(st.swarm_received_data[0], SessionDataReceived)
    assert st.swarm_received_data[0].artifacts[0].data == {"finding": 42}

    # 4. Cancel via StopSession(mode='graceful_ws', save=True, reason='swarm-cancel')
    controller.cancel("sw0", state=st)
    _drain()
    assert not controller.in_flight
    assert store.pool()["desired_state"] == "cancelled"
