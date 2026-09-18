"""Startup submission across the real websocket thread boundary; no providers."""
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent_utils import Session
from agent_zoo.session_io import SessionIO
from agent_zoo.session_websocket import SessionWebSocketServer

import swarm


@pytest.fixture
def startup_runtime(tmp_path, monkeypatch):
    session = Session(system_prompt="offline startup submission")
    controller = swarm.SwarmController({}, session)
    work = []

    def model_work(state):
        work.append("model/tool pipeline")
        raise AssertionError("paused startup must not run model/tool work")

    session.pipeline = [model_work, swarm.SwarmCompletionCheck(controller)]
    session.ensure_initialized()
    state = session.state
    state._session_id = "startup-session"
    state._instance_id = "startup-instance"
    state._runtime_startup_paused = True
    state.session_cwd = str(tmp_path)
    state._agent_zoo_context = {}
    state.swarm_access = {}
    state.swarm_resource_uids = {}
    server = SessionWebSocketServer(SessionIO(session_id=state._session_id),
                                    instance_id=state._instance_id)
    state._session_websocket_server = server
    threads = []
    result = dict(summary="startup completed once", access={"sw0": ""}, grants={})

    async def startup(snapshot, event):
        threads.append(threading.get_ident())
        return result

    factory = Mock(side_effect=startup)
    monkeypatch.setattr(controller, "startup", factory)
    apply = Mock(wraps=controller.apply)
    monkeypatch.setattr(controller, "apply", apply)
    event = SimpleNamespace(kind="runtime.session_ready", payload=dict(
        session_id=state._session_id, instance_id=state._instance_id,
        restored=False, paused=True, startup_mode="paused", revision_id="seed"))
    swarm.SwarmRuntimeReady(controller).handle_harness_event(event, state)
    entries = list(state.entries)

    def poll():
        assert session.poll_interrupts(deliver=False) is False
        assert state._runtime_startup_paused is True
        assert state.entries == entries
        assert work == []

    runtime = SimpleNamespace(session=session, state=state, controller=controller,
                              server=server, factory=factory, apply=apply, poll=poll,
                              threads=threads, result=result, event=event)
    try:
        yield runtime
    finally:
        assert server.stop_background(timeout_s=5)


def poll_until(runtime, predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        runtime.poll()
        assert time.monotonic() < deadline, "background startup did not finish"
        time.sleep(0.005)


def test_real_server_retains_ready_event_until_started(startup_runtime, monkeypatch, caplog):
    r = startup_runtime
    caplog.set_level(logging.WARNING, logger="swarm")
    submit = Mock(wraps=r.server.submit_background)
    monkeypatch.setattr(r.server, "submit_background", submit)
    queued = list(r.controller.ready_events)
    assert len(queued) == 1
    for _ in range(3):
        r.poll()
        assert r.controller.ready_events == queued
        assert not r.controller.pending
        assert not r.state.pending_interrupts
        r.factory.assert_not_called()
        r.apply.assert_not_called()
    assert submit.call_count == 3  # Actual server rejection, never a mocked error.
    assert not caplog.records

    assert r.server.start_background(timeout_s=5).startswith("ws://127.0.0.1:")
    r.poll()
    poll_until(r, lambda: r.apply.call_count == 1)
    r.factory.assert_called_once()
    snapshot, event = r.factory.call_args.args
    assert snapshot is not r.state
    assert (snapshot._session_id, snapshot._instance_id) == ("startup-session", "startup-instance")
    assert event == queued[0][0]
    assert r.threads and r.threads[0] != threading.get_ident()
    r.apply.assert_called_once_with(r.state, r.result)
    assert r.state.swarm_access == {"sw0": ""}
    assert r.state.pending_interrupts == ["startup completed once"]
    assert not r.controller.ready_events and not r.controller.pending
    for _ in range(3):
        r.poll()
    assert submit.call_count == 4
    r.factory.assert_called_once()
    r.apply.assert_called_once()
    assert r.state.pending_interrupts == ["startup completed once"]


@pytest.mark.parametrize("message", [
    "unrelated runtime failure",
    "session websocket background server is not running or is stopping: other failure",
])
def test_non_readiness_runtime_error_is_logged_not_retried(startup_runtime, monkeypatch, caplog, message):
    r = startup_runtime
    submit = Mock(side_effect=RuntimeError(message))
    monkeypatch.setattr(r.server, "submit_background", submit)
    for _ in range(3):
        r.poll()
    submit.assert_called_once()
    r.factory.assert_not_called()
    r.apply.assert_not_called()
    assert not r.controller.ready_events and not r.controller.pending
    assert r.state.pending_interrupts == [f"Swarm startup failed: {message}"]
    records = [record for record in caplog.records if "Swarm startup submission failed" in record.message]
    assert len(records) == 1 and records[0].exc_info


def test_snapshot_failure_is_logged_not_retried(startup_runtime, monkeypatch, caplog):
    r = startup_runtime
    snapshot = Mock(side_effect=ValueError("snapshot unavailable"))
    submit = Mock(wraps=r.server.submit_background)
    monkeypatch.setattr(r.controller, "snapshot", snapshot)
    monkeypatch.setattr(r.server, "submit_background", submit)
    for _ in range(3):
        r.poll()
    snapshot.assert_called_once_with(r.state)
    submit.assert_not_called()
    r.factory.assert_not_called()
    assert not r.controller.ready_events and not r.controller.pending
    assert r.state.pending_interrupts == ["Swarm startup failed: snapshot unavailable"]
    records = [record for record in caplog.records if "Swarm startup submission failed" in record.message]
    assert len(records) == 1 and records[0].exc_info


def test_real_background_future_failure_is_logged_not_retried(startup_runtime, caplog):
    r = startup_runtime

    async def fail(snapshot, event):
        raise RuntimeError("startup future failed")

    r.factory.side_effect = fail
    r.server.start_background(timeout_s=5)
    r.poll()
    poll_until(r, lambda: bool(r.state.pending_interrupts))
    for _ in range(3):
        r.poll()
    r.factory.assert_called_once()
    r.apply.assert_not_called()
    assert not r.controller.ready_events and not r.controller.pending
    assert r.state.pending_interrupts == [
        "Swarm operation failed: startup future failed"]
    records = [record for record in caplog.records if "Swarm background operation failed" in record.message]
    assert len(records) == 1 and records[0].exc_info


def test_stale_ready_event_is_logged_without_submission(startup_runtime, monkeypatch, caplog):
    r = startup_runtime
    submit = Mock(wraps=r.server.submit_background)
    monkeypatch.setattr(r.server, "submit_background", submit)
    r.state._instance_id = "replacement-instance"
    for _ in range(3):
        r.poll()
    submit.assert_not_called()
    r.factory.assert_not_called()
    r.apply.assert_not_called()
    assert not r.controller.ready_events and not r.controller.pending
    assert not r.state.pending_interrupts
    records = [record for record in caplog.records if "Discarding stale swarm startup event" in record.message]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
