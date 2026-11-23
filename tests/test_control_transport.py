"""Real websocket/input-channel coverage via AgentZooSessionDriver, without launching an agent or model."""
import threading
from types import SimpleNamespace

import pytest

from agent_utils.core.events import IoCompleted
from agent_zoo.drivers.sessions import decode_session_data_control
from agent_zoo.events import AdoptSession, SessionArtifact, SessionDeliveryResult, SessionStopResult
from agent_zoo.transport.session_io import SessionIO
from agent_zoo.transport.websocket import SessionWebSocketServer
from tmux_pilot.process_identity import capture_process_identity
import swarm
from conftest import run_driver_op


@pytest.mark.parametrize("action", ["bcast", "interrupt", "continue", "cancel", "release", "capture", "afk"])
def test_native_control_channels(tmp_path, monkeypatch, action):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path / "shared"))
    io = SessionIO(session_id="owned-peer")
    received = threading.Event()
    io.set_inbox_observer(lambda _: received.set())
    server = SessionWebSocketServer(io, instance_id="owned-instance", local_state_root=tmp_path / "local")
    driver = swarm.make_swarm_session_driver()
    try:
        url = server.start_background(timeout_s=5)
        proposal = dict(
            session_id="owned-peer", instance_id="owned-instance", label="sw0p0a0",
            websocket_url=url, process_identity=capture_process_identity().to_dict(),
        )
        text = "  first line\n" + ("雪 multiline\n" * 10000) + "last line  "
        if action == "cancel":
            adopt_ev = run_driver_op(driver, AdoptSession(
                session_id="owned-peer", instance_id="owned-instance",
                process_identity=proposal["process_identity"], websocket_url=url, watch=False,
            ))
            assert isinstance(adopt_ev, IoCompleted)
            ev = run_driver_op(driver, swarm.build_member_stop_request(proposal, operation_id="op-cancel"))
            assert isinstance(ev, IoCompleted) and isinstance(ev.value, SessionStopResult)
            assert ev.value.status in {"stopped", "stopping"}
        else:
            request = dict(action=action, message=text, operation_id="op-1")
            ev = run_driver_op(driver, swarm.build_member_input_request(proposal, request))
            assert isinstance(ev, IoCompleted) and isinstance(ev.value, SessionDeliveryResult)
            assert ev.value.outcome == "sent"
        assert received.wait(5)
        frames = io.drain()
        assert len(frames) == 1
        frame = frames[0]
        if action == "bcast":
            assert frame.channel == "user_text"
            assert frame.payload == {"text": text}
        elif action == "cancel":
            assert frame.channel == "control"
            assert frame.payload == {
                "type": "shutdown", "save": True, "reason": "swarm-cancel", "spawned_policy": "leave",
            }
        else:
            assert frame.channel == "slash_command"
            assert frame.payload == {"raw": "/afk " + text if action == "afk" else "/" + action}
    finally:
        driver.close()
        assert server.stop_background(timeout_s=5)


def test_structured_board_artifact_transfer_over_send_session_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path / "shared"))
    io = SessionIO(session_id="owned-peer")
    received = threading.Event()
    io.set_inbox_observer(lambda _: received.set())
    server = SessionWebSocketServer(io, instance_id="owned-instance", local_state_root=tmp_path / "local")
    driver = swarm.make_swarm_session_driver()
    try:
        url = server.start_background(timeout_s=5)
        proposal = dict(
            session_id="owned-peer", instance_id="owned-instance", label="sw0p0a0",
            websocket_url=url, process_identity=capture_process_identity().to_dict(),
        )
        art = SessionArtifact(
            name="sw0:pod-1:general", kind="json",
            data={"board": "general", "entries": [{"id": "m1", "text": "finding"}]},
            content_type="application/json",
        )
        req = swarm.build_member_data_request(
            proposal, artifacts=(art,), channel="swarm:sw0:pod-1:general",
            message="board snapshot", sender_session_id="parent", sender_instance_id="parent-a",
        )
        ev = run_driver_op(driver, req)
        assert isinstance(ev, IoCompleted) and ev.value.outcome == "sent"
        assert received.wait(5)
        frames = io.drain()
        assert len(frames) == 1 and frames[0].channel == "control"
        decoded = decode_session_data_control(frames[0].payload)
        assert decoded.channel == "swarm:sw0:pod-1:general"
        assert decoded.sender_session_id == "parent"
        assert len(decoded.artifacts) == 1 and decoded.artifacts[0].data["board"] == "general"
    finally:
        driver.close()
        assert server.stop_background(timeout_s=5)


def test_agent_tool_requires_background_service_not_slash_ingress():
    io = SessionIO(session_id="parent")
    state = SimpleNamespace(_session_id="parent", _instance_id="parent-instance", swarm_access={"sw0": ""},
                            _session_websocket_server=SimpleNamespace(session_io=io),
                            session_cwd=str(__import__("pathlib").Path.cwd()))
    controller = swarm.SwarmController({})
    with pytest.raises(ValueError, match="Background execution unavailable"):
        controller.broadcast("sw0", "exact payload", state=state)
    assert io.drain() == ()
    state.swarm_access = {"sw0": "pod-1"}
    with pytest.raises(ValueError, match="peers"):
        controller.cancel("sw0", state)
    assert io.drain() == ()
