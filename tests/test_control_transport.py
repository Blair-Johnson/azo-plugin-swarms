"""Real websocket/input-channel coverage, without launching an agent or model."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from agent_zoo.session_io import SessionIO
from agent_zoo.session_websocket import SessionWebSocketServer
from tmux_pilot.process_identity import capture_process_identity
import swarm


@pytest.mark.parametrize("action", ["bcast", "interrupt", "continue", "cancel"])
def test_native_control_channels(tmp_path, monkeypatch, action):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path / "shared"))
    io = SessionIO(session_id="owned-peer")
    received = threading.Event()
    io.set_inbox_observer(lambda _: received.set())
    server = SessionWebSocketServer(io, instance_id="owned-instance", local_state_root=tmp_path / "local")
    try:
        url = server.start_background(timeout_s=5)
        target = dict(session_id="owned-peer", instance_id="owned-instance", label="sw0p0a0",
                      ready={"websocket_url": url}, process_identity=capture_process_identity().to_dict())
        text = "  first line\n" + ("雪 multiline\n" * 10000) + "last line  "
        request = dict(action=action, message=text)
        asyncio.run(swarm.send_to_member(target, request))
        assert received.wait(5)
        frames = io.drain()
        assert len(frames) == 1
        frame = frames[0]
        if action == "bcast":
            assert frame.channel == "user_text"
            assert frame.payload == {"text": text}
        elif action == "cancel":
            assert frame.channel == "control"
            assert frame.payload == {"type": "shutdown", "save": True, "reason": "swarm-cancel"}
        else:
            assert frame.channel == "slash_command"
            assert frame.payload == {"raw": "/" + action}
    finally:
        assert server.stop_background(timeout_s=5)


def test_agent_control_tool_uses_the_same_command_ingress():
    io = SessionIO(session_id="parent")
    state = SimpleNamespace(_session_id="parent", swarm_access={"sw0": ""},
                            _session_websocket_server=SimpleNamespace(session_io=io))
    tool = swarm.SwarmControl()
    assert tool.available(state)
    command = 'bcast sw0 -p 0 "  first\nsecond  "'
    assert swarm.control_swarm(command, state).startswith("Queued")
    frame, = io.drain()
    assert frame.channel == "slash_command"
    assert frame.payload == {"raw": "/swarm " + command}
    with pytest.raises(ValueError, match="user"):
        swarm.control_swarm("-n 16 -p 4", state)
    assert io.drain() == ()
    state.swarm_access = {"sw0": "pod-1"}
    assert not tool.available(state)
    with pytest.raises(ValueError, match="peers"):
        swarm.control_swarm("cancel sw0", state)
    assert io.drain() == ()
