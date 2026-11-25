"""Use this checkout's plugin with the selected integrated host interpreter."""
from __future__ import annotations

from pathlib import Path
import queue
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from agent_utils.core.events import IoCompleted, IoFailed, TimerFired
from agent_utils.core.io import IoRequest, OpId, StartTimer
from agent_utils.runtime.drivers.timer import TimerDriver
from agent_zoo.drivers.sessions import AgentZooSessionDriver
import swarm


@pytest.fixture(autouse=True)
def isolated_storage_environment(tmp_path, monkeypatch):
    # HOME overrides STATE_ROOT; never let a live harness redirect test fixtures.
    monkeypatch.delenv("AGENT_ZOO_HOME", raising=False)
    monkeypatch.setenv("AGENT_ZOO_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))


def run_driver_op(driver: AgentZooSessionDriver, req: IoRequest, *, timeout_s: float = 10.0) -> IoCompleted | IoFailed:
    """Submit one ``SessionRequest`` to an ``AgentZooSessionDriver`` and wait for its completion."""
    events: queue.Queue = queue.Queue()
    started_here = driver._emit is None
    if started_here:
        driver.start(events.put)
    else:
        orig_emit = driver._emit

        def _tee(ev):
            events.put(ev)
            if orig_emit is not None:
                orig_emit(ev)

        driver._emit = _tee
    op_id: OpId = f"test-op-{time.monotonic_ns()}"
    try:
        driver.submit(op_id, req)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"driver op {type(req).__name__} timed out after {timeout_s}s")
            ev = events.get(timeout=remaining)
            if isinstance(ev, (IoCompleted, IoFailed)) and ev.op_id == op_id:
                return ev
    finally:
        if started_here:
            driver.close()


def run_swarm_workflow(
    controller: swarm.SwarmController,
    state: Any,
    gen: Any,
    *,
    session_driver: AgentZooSessionDriver | None = None,
    timeout_s: float = 10.0,
) -> Any:
    """Drive a pure ``SwarmController`` workflow generator through real drivers and return its result."""
    owns_driver = session_driver is None
    if session_driver is None:
        session_driver = swarm.make_swarm_session_driver(
            launcher=getattr(state, "session_launcher", None),
            store_root=swarm._state_store_root(state),
        )
    store_driver = swarm.SwarmStoreDriver()
    timer_driver = TimerDriver()
    events: queue.Queue = queue.Queue()
    session_driver.start(events.put)
    store_driver.start(events.put)
    timer_driver.start(events.put)
    next_op = 0

    def _dispatch(req: IoRequest) -> OpId:
        nonlocal next_op
        next_op += 1
        op_id = f"wf-op-{next_op}"
        if session_driver.handles(req):
            session_driver.submit(op_id, req)
        elif store_driver.handles(req):
            store_driver.submit(op_id, req)
        elif timer_driver.handles(req):
            timer_driver.submit(op_id, req)
        else:
            raise TypeError(f"No test driver for {type(req).__name__}")
        return op_id

    try:
        send_val: Any = None
        while True:
            try:
                yielded = gen.send(send_val)
            except StopIteration as done:
                return done.value
            if isinstance(yielded, IoRequest):
                op_id = _dispatch(yielded)
                deadline = time.monotonic() + timeout_s
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"workflow op {type(yielded).__name__} timed out")
                    ev = events.get(timeout=remaining)
                    if isinstance(ev, (IoCompleted, IoFailed, TimerFired)) and ev.op_id == op_id:
                        send_val = ev
                        break
            else:
                reqs = list(yielded)
                if not reqs:
                    send_val = []
                    continue
                op_ids = [_dispatch(r) for r in reqs]
                id_to_idx = {oid: idx for idx, oid in enumerate(op_ids)}
                collected: dict[int, IoCompleted | IoFailed | TimerFired] = {}
                deadline = time.monotonic() + timeout_s
                while len(collected) < len(op_ids):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("workflow batch timed out")
                    ev = events.get(timeout=remaining)
                    if isinstance(ev, (IoCompleted, IoFailed, TimerFired)) and ev.op_id in id_to_idx:
                        collected[id_to_idx[ev.op_id]] = ev
                send_val = [collected[i] for i in range(len(op_ids))]
    finally:
        timer_driver.close()
        store_driver.close()
        if owns_driver:
            session_driver.close()


def execute_swarm_request(
    controller: swarm.SwarmController,
    state: Any,
    request: Any,
    *,
    session_driver: AgentZooSessionDriver | None = None,
    timeout_s: float = 10.0,
) -> dict:
    """Execute a Swarm request dict or ``ControlRequest`` via real ``AgentZooSessionDriver`` + ``PluginTaskDriver``."""
    return run_swarm_workflow(
        controller,
        state,
        controller._wf_execute(state, request),
        session_driver=session_driver,
        timeout_s=timeout_s,
    )


def drain_first_request(
    controller: swarm.SwarmController,
    state: Any,
    first_req: IoRequest,
    *,
    session_driver: AgentZooSessionDriver | None = None,
    timeout_s: float = 10.0,
) -> dict:
    """Complete a workflow whose first ``IoRequest`` was already yielded by ``command()`` / ``submit()``."""
    assert len(controller._workflows) == 1, f"expected 1 queued workflow, got {len(controller._workflows)}"
    wf_id, runner = next(iter(controller._workflows.items()))
    for oid in [k for k, (_, _, m) in controller.in_flight.items() if isinstance(m, dict) and m.get("_wf") == wf_id]:
        controller.in_flight.pop(oid, None)
    controller._workflows.pop(wf_id, None)

    def _resume():
        first_ev = yield first_req
        try:
            next_val = runner.gen.send(first_ev)
        except StopIteration as done:
            return done.value
        while True:
            ev = yield next_val
            try:
                next_val = runner.gen.send(ev)
            except StopIteration as done:
                return done.value

    return run_swarm_workflow(controller, state, _resume(), session_driver=session_driver, timeout_s=timeout_s)
