"""Core/AU/plugin startup boundaries; isolated journals, no providers or processes."""
from contextlib import contextmanager
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent_utils import Entry, Feature, PipelineBuilder, Runtime, Session, ToolCall, core_feature, interrupts_feature
from agent_utils.core.events import BeforeLLM, RunToolCall
from agent_utils.features.buffers import BufferManager
from agent_utils.storage.repository import SessionRepository
from agent_utils.storage.serialization import capture_session_document
from agent_zoo.events import SessionReady
from agent_zoo.features.special_buffers import special_buffers_feature
from agent_zoo.features.turn import turn_orchestration_feature
from agent_zoo.storage.load import LoadedSession
from agent_zoo.storage.serialization import (
    SESSION_PERSISTENCE_EXCLUDED_ATTRS,
    restore_session_state,
)
import swarm
from test_recovery import runtime, reserve


class WorkCounter:
    handles = (BeforeLLM, RunToolCall)

    def __init__(self, work):
        self.work = work

    def handle(self, event, state, emit):
        self.work.append(type(event).__name__)
        if any(e.tool_calls for e in state.entries):
            self.work.append("historical-tools")


class SessionReadyObserver:
    handles = (SessionReady,)

    def __init__(self, seen):
        self.seen = seen

    def handle(self, event, state, emit):
        self.seen.append(
            dict(
                session_id=event.session_id,
                instance_id=event.instance_id,
                restored=event.restored,
                revision_id=event.revision_id,
                startup_mode=event.startup_mode,
                paused=event.paused,
            )
        )


@contextmanager
def host_runtime(r, monkeypatch, *, sid, iid, checkpoint=None):
    """Real Sans-I/O Session + Runtime + SwarmDriver, zero provider or subprocess launches."""
    monkeypatch.chdir(r.tmp)
    work, seen = [], []
    config = {
        "llm": {
            "default": "fake",
            "models": {"fake": {"enabled": True, "openai": {"request": {"model": "fake"}}}},
        },
        "default": {"system_prompt": "SYS"},
    }
    session = Session(system_prompt="SYS", cwd=str(r.tmp))
    session._session_id = sid
    controller = swarm.SwarmController(config, session)
    pipeline = (
        PipelineBuilder()
        .add(core_feature())
        .add(interrupts_feature())
        .add(turn_orchestration_feature(config))
        .add(special_buffers_feature())
        .add(
            Feature(
                "swarm_integration",
                components=[
                    WorkCounter(work),
                    SessionReadyObserver(seen),
                    swarm.SwarmBuffers(),
                    swarm.SwarmRuntimeReady(controller),
                    swarm.SwarmCompletionHandler(controller),
                ],
                order=[
                    swarm.IoFence,
                    swarm.RegisterSpecialBuffers,
                    swarm.SwarmBuffers,
                    SessionReadyObserver,
                    swarm.SwarmRuntimeReady,
                    swarm.SwarmCompletionHandler,
                    WorkCounter,
                    swarm.InterruptDelivery,
                ],
                drivers=[swarm.SwarmDriver],
            )
        )
        .build()
    )
    session.pipeline = pipeline
    session.ensure_initialized()

    state = session.state
    state._session_id = sid
    state._instance_id = iid
    state._project_name = "test"
    state._store_root = str(r.tmp / "registry")
    state._agent_zoo_context = {
        "project_name": "test",
        "project_store_root": str(r.tmp / "registry"),
    }
    state._persistence_local_root = str(r.tmp / "local")
    state.session_launcher = r.state.session_launcher
    startup_mode = "normal" if checkpoint else "paused"
    state._runtime_startup_mode = startup_mode
    state._runtime_startup_paused = startup_mode == "paused"
    state.startup_paused = startup_mode == "paused"

    if checkpoint is not None:
        restore_session_state(session, checkpoint, take_ownership=True)
        state._session_id = sid
        state._instance_id = iid
        state._project_name = "test"
        state._store_root = str(r.tmp / "registry")
        state._agent_zoo_context = {
            "project_name": "test",
            "project_store_root": str(r.tmp / "registry"),
        }
        state.session_launcher = r.state.session_launcher
        state._runtime_startup_paused = False
        state.startup_paused = False
        revision_id = checkpoint.state.commit_id
    else:
        # Publish the exact initial shared seed commit before SessionReady.
        repo = SessionRepository(swarm.project_directory(state) / "sessions" / sid)
        writer = repo.create_instance(session_id=sid, instance_id=iid, origin_host_id="host-a")
        commit = writer.commit(
            capture_session_document(
                state,
                session.pipeline,
                session_id=sid,
                exclude_attrs=SESSION_PERSISTENCE_EXCLUDED_ATTRS,
            )
        )
        revision_id = commit.commit_id

    driver = swarm.SwarmDriver()
    rt = Runtime(session, drivers=[driver])
    rt.start()
    rt.post(
        SessionReady(
            session_id=sid,
            instance_id=iid,
            restored=checkpoint is not None,
            revision_id=revision_id,
            startup_mode=startup_mode,
            paused=(startup_mode == "paused"),
        )
    )
    # Step SessionReady synchronously on entry so startup hold and submission are active.
    assert rt.step_once(timeout=1.0)

    def drain_until(predicate, *, timeout_s=5.0):
        deadline = time.monotonic() + timeout_s
        while not predicate():
            rt.step_once(timeout=0.02)
            assert time.monotonic() < deadline, "runtime did not reach expected state"

    try:
        yield SimpleNamespace(
            session=session,
            runtime=rt,
            work=work,
            seen=seen,
            controller=controller,
            drain_until=drain_until,
        )
    finally:
        rt.close()


def test_real_fresh_paused_member_bootstrap_publishes_exact_shared_seed(runtime, monkeypatch):
    r = runtime
    member = r.members[0]
    attempt = reserve(r, member, state="reserved", identity=r.identity.to_dict())
    monkeypatch.setenv("AZO_SWARM_ID", "sw0")
    monkeypatch.setenv("AZO_SWARM_POD_ID", r.store.member_pod(member["session_id"]))
    with host_runtime(r, monkeypatch, sid=member["session_id"], iid=attempt["instance_id"]) as b:
        state = b.session.state
        assert b.work == []
        assert len(b.seen) == 1 and b.seen[0]["paused"] is True
        revision = b.seen[0]["revision_id"]
        assert revision
        assert swarm.member_resource(b.controller.snapshot(state)) is not None, state._agent_zoo_context
        b.drain_until(lambda: not b.controller.in_flight)
        assert b.work == []
        row = r.store.records().get("attempts", member["session_id"])
        assert row.get("bootstrap", {}).get("revision_id") == revision, state.pending_interrupts
        assert row["bootstrap"]["paused"] is True
        shared = swarm.shared_checkpoint(b.controller.snapshot(state), member["session_id"], revision)
        assert shared.commit_id == revision
        assert b.controller.grants["sw0"]["instance_id"] == attempt["instance_id"]
        assert state._runtime_startup_paused and state.startup_paused
        assert state.pending_interrupts  # completion retained, not injected/delivered while paused
        assert len(state.entries) == 1


def parent_checkpoint(r):
    session = Session(system_prompt="saved parent")
    session.pipeline = PipelineBuilder().add(core_feature()).build()
    session.ensure_initialized()
    session.state._session_id = "parent"
    session.state._instance_id = "old-parent"
    session.state.entries.append(
        Entry(
            messages=[{"role": "user", "content": "historical turn"}],
            index=len(session.state.entries),
            step=0,
        )
    )
    call = ToolCall("old-call", "never_execute", "{}", {})
    session.state.entries.append(
        Entry(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "old-call",
                            "type": "function",
                            "function": {"name": "never_execute", "arguments": "{}"},
                        }
                    ],
                }
            ],
            index=len(session.state.entries),
            step=0,
            tool_calls=[call],
        )
    )
    session.state.pending_tool_calls = [call]
    session.state.last_tool_calls = [call]
    session.state._await_requested = True
    repo = SessionRepository(r.tmp / "parent-source")
    writer = repo.create_instance(session_id="parent", instance_id="old-parent", origin_host_id="host-a")
    saved = writer.commit(capture_session_document(session.state, session.pipeline, session_id="parent"))
    return LoadedSession(repo.load(session_id="parent", commit_id=saved.commit_id), "local", "exact")


def test_real_restored_parent_holds_saved_work_through_recovery_poll(runtime, monkeypatch):
    r = runtime
    checkpoint = parent_checkpoint(r)
    # A provably live prior owner blocks takeover. Discovery and recovery still run for real.
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "alive")
    with host_runtime(r, monkeypatch, sid="parent", iid="restored-parent", checkpoint=checkpoint) as b:
        assert b.seen[0]["restored"] is True
        assert b.seen[0]["revision_id"] == checkpoint.state.commit_id
        assert b.session.state._runtime_startup_paused and b.session.state.startup_paused
        assert b.work == []
        b.drain_until(lambda: not b.controller.in_flight)
        assert b.work == []
        assert b.session.state._runtime_startup_paused and b.session.state.startup_paused
        assert b.session.state.entries[-1].tool_calls[0].id == "old-call"
        assert any("blocked" in str(x) for x in b.session.state.pending_interrupts)
        r.state.session_launcher.start.assert_not_called()
