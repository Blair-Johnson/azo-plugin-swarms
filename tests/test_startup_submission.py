"""Startup submission across the real AgentZooSessionDriver + SwarmStoreDriver boundary; no providers."""
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent_utils import Feature, PipelineBuilder, Runtime, Session, core_feature, interrupts_feature
from agent_utils.core.events import BeforeLLM, IoCompleted
from agent_utils.runtime.drivers.timer import TimerDriver
from agent_zoo.events import SessionReady

import swarm


class _NoModelWork:
    handles = (BeforeLLM,)

    def __init__(self, work):
        self.work = work

    def handle(self, event, state, emit):
        self.work.append("model/tool pipeline")
        raise AssertionError("paused startup must not run model/tool work")


@pytest.fixture
def startup_runtime(tmp_path, monkeypatch):
    session = Session(system_prompt="offline startup submission", cwd=str(tmp_path))
    controller = swarm.SwarmController({}, session)
    work = []
    pipeline = (
        PipelineBuilder()
        .add(core_feature())
        .add(interrupts_feature())
        .add(
            Feature(
                "swarm_startup_test",
                components=[
                    _NoModelWork(work),
                    swarm.SwarmRuntimeReady(controller),
                    swarm.SwarmCompletionHandler(controller),
                ],
                order=[
                    swarm.IoFence,
                    _NoModelWork,
                    swarm.SwarmRuntimeReady,
                    swarm.SwarmCompletionHandler,
                    swarm.InterruptDelivery,
                ],
                drivers=[swarm.make_swarm_session_driver, swarm.SwarmStoreDriver, TimerDriver],
            )
        )
        .build()
    )
    session.pipeline = pipeline
    session.ensure_initialized()
    state = session.state
    state._session_id = "startup-session"
    state._instance_id = "startup-instance"
    state.startup_paused = True
    state._agent_zoo_context = {}
    state.swarm_access = {}
    state.swarm_resource_uids = {}

    driver = swarm.make_swarm_session_driver()
    runtime = Runtime(session, drivers=[driver, swarm.SwarmStoreDriver(), TimerDriver()])
    threads = []
    result = dict(summary="startup completed once", access={"sw0": ""}, grants={})

    def startup(snapshot, event):
        threads.append(threading.get_ident())
        return result

    factory = Mock(side_effect=startup)
    monkeypatch.setattr(controller, "startup", factory, raising=False)
    apply = Mock(wraps=controller.apply)
    monkeypatch.setattr(controller, "apply", apply)
    event = SessionReady(
        session_id=state._session_id,
        instance_id=state._instance_id,
        restored=False,
        paused=True,
        startup_mode="paused",
        revision_id="seed",
    )
    entries = list(state.entries)

    def drain_until(predicate, *, timeout_s=5.0):
        deadline = time.monotonic() + timeout_s
        while not predicate():
            runtime.step_once(timeout=0.02)
            assert state.startup_paused is True
            assert state.entries == entries
            assert work == []
            assert time.monotonic() < deadline, "background startup did not finish"

    ctx = SimpleNamespace(
        session=session,
        state=state,
        controller=controller,
        driver=driver,
        runtime=runtime,
        factory=factory,
        apply=apply,
        drain_until=drain_until,
        threads=threads,
        result=result,
        event=event,
    )
    try:
        yield ctx
    finally:
        runtime.close()


def test_real_driver_runs_startup_off_thread_and_applies_once(startup_runtime):
    r = startup_runtime
    r.runtime.start()
    r.runtime.post(r.event)
    r.drain_until(lambda: r.apply.call_count == 1)
    r.factory.assert_called_once()
    snapshot, payload = r.factory.call_args.args
    assert snapshot is not r.state
    assert (snapshot._session_id, snapshot._instance_id) == ("startup-session", "startup-instance")
    assert payload["revision_id"] == "seed" and payload["paused"] is True
    assert r.threads and r.threads[0] != threading.get_ident()
    r.apply.assert_called_once_with(r.state, r.result)
    assert r.state.swarm_access == {"sw0": ""}
    # Completion is retained in pending_interrupts while startup-paused, not delivered into entries.
    assert r.state.pending_interrupts == ["startup completed once"]
    assert not r.controller.in_flight
    # Duplicate SessionReady is ignored.
    r.runtime.post(r.event)
    r.runtime.step_once(timeout=0.05)
    r.factory.assert_called_once()
    r.apply.assert_called_once()


def test_snapshot_failure_is_logged_not_retried(startup_runtime, monkeypatch, caplog):
    r = startup_runtime
    caplog.set_level(logging.ERROR, logger="swarm")
    snapshot = Mock(side_effect=ValueError("snapshot unavailable"))
    monkeypatch.setattr(r.controller, "snapshot", snapshot)
    r.runtime.start()
    r.runtime.post(r.event)
    r.runtime.step_once(timeout=1.0)
    snapshot.assert_called_once_with(r.state)
    r.factory.assert_not_called()
    assert not r.controller.in_flight
    assert r.state.pending_interrupts == ["Swarm startup failed: snapshot unavailable"]
    records = [record for record in caplog.records if "Swarm startup submission failed" in record.message]
    assert len(records) == 1 and records[0].exc_info


def test_real_background_future_failure_is_logged_not_retried(startup_runtime, caplog):
    r = startup_runtime
    caplog.set_level(logging.WARNING, logger="swarm")

    def fail(snapshot, event):
        raise RuntimeError("startup future failed")

    r.factory.side_effect = fail
    r.runtime.start()
    r.runtime.post(r.event)
    r.drain_until(lambda: bool(r.state.pending_interrupts))
    r.factory.assert_called_once()
    r.apply.assert_not_called()
    assert not r.controller.in_flight
    assert r.state.pending_interrupts == ["Swarm operation failed: startup future failed"]
    records = [record for record in caplog.records if "Swarm background operation failed" in record.message]
    assert len(records) == 1


def test_stale_completion_is_logged_and_discarded(startup_runtime, caplog):
    r = startup_runtime
    caplog.set_level(logging.WARNING, logger="swarm")
    r.runtime.start()
    r.runtime.post(r.event)
    # Process SessionReady, then mutate _instance_id before IoCompleted arrives.
    assert r.runtime.step_once(timeout=1.0)
    assert len(r.controller.in_flight) == 1
    r.state._instance_id = "replacement-instance"
    r.drain_until(lambda: not r.controller.in_flight)
    r.apply.assert_not_called()
    assert not r.state.pending_interrupts
    records = [record for record in caplog.records if "Discarding stale swarm completion" in record.message]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
