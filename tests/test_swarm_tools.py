"""Named model APIs and owner-thread completion; no command-string round trip."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
from agent_utils import Tool, Session
from agent_zoo.runtime.commands import CommandError
import swarm
from test_recovery import runtime


@pytest.fixture
def tool_runtime(runtime):
    r = runtime
    session = SimpleNamespace(state=r.state, pipeline=object())
    c = swarm.SwarmController({"llm": {"default": "offline"}}, session)
    c.grants = r.state.grants.copy()
    futures, factories = [], []
    def submit(factory):
        future = Future(); futures.append(future); factories.append(factory)
        return future
    r.state._session_websocket_server = SimpleNamespace(submit_background=submit,
        session_io=SimpleNamespace(send=Mock(side_effect=AssertionError("slash ingress forbidden"))))
    return SimpleNamespace(runtime=r, controller=c, session=session, futures=futures, factories=factories)


def test_exact_five_tools_have_named_zero_based_arguments():
    features = []
    swarm.register_features(SimpleNamespace(add=features.append), session=SimpleNamespace(), config={})
    tools = {t.name: t for t in features[0].components if isinstance(t, Tool)}
    assert set(tools) == {"swarm_broadcast", "swarm_interrupt", "swarm_continue", "swarm_cancel", "swarm_post"}
    import inspect
    assert list(inspect.signature(tools["swarm_broadcast"].fn).parameters) == ["swarm_id", "message", "pods", "state"]
    assert list(inspect.signature(tools["swarm_post"].fn).parameters) == ["swarm_id", "pod", "channel", "message", "message_id", "state"]
    assert all("command" not in inspect.signature(t.fn).parameters for t in tools.values())


def test_tool_nonblocking_payload_preserved_and_completion_once(tool_runtime):
    t = tool_runtime; c = t.controller; state = t.runtime.state
    text = '  quoted "\\\\\n\u96ea\t  '
    receipt = c.broadcast("sw0", text, [1, 0, 1], state)
    assert receipt == "Sending message to sw0 (pods 1, 0)..."
    assert len(t.futures) == 1 and not t.futures[0].done()
    execute = AsyncMock(return_value=dict(summary="OK sent=2/2", access={"sw0": ""}, grants={}))
    c.execute = execute
    result = asyncio.run(t.factories[0]())
    request = execute.call_args.args[1]
    assert request["message"] == text and request["pods"] == (1, 0)
    assert request["operation_id"] not in receipt
    t.futures[0].set_result(result)
    check = swarm.SwarmCompletionCheck(c)
    check(state); check(state)
    assert state.pending_interrupts == ["OK sent=2/2"] and not c.pending
    state._session_websocket_server.session_io.send.assert_not_called()


@pytest.mark.parametrize("change", ["state", "session", "instance", "pipeline"])
def test_stale_completion_is_observed_but_not_applied(tool_runtime, change):
    t = tool_runtime; state = t.runtime.state
    t.controller.interrupt("sw0", state)
    t.futures[0].set_result(dict(summary="must not deliver", access={"forged": ""}, grants={"forged": {}}))
    if change == "state":
        t.session.state = SimpleNamespace()
    elif change == "session":
        state._session_id = "different"
    elif change == "instance":
        state._instance_id = "different"
    else:
        t.session.pipeline = object()
    swarm.SwarmCompletionCheck(t.controller)(state)
    assert not state.pending_interrupts and "forged" not in state.swarm_access
    assert "forged" not in t.controller.grants


@pytest.mark.parametrize("pods", [[], [True], [-1], [32], ["0"]])
def test_invalid_tool_selectors_never_submit(tool_runtime, pods):
    with pytest.raises(ValueError):
        tool_runtime.controller.broadcast("sw0", "payload", pods, tool_runtime.runtime.state)
    assert tool_runtime.futures == []


def test_missing_background_service_is_immediate_no_submission(tool_runtime):
    state = tool_runtime.runtime.state
    state._session_websocket_server = None
    with pytest.raises(ValueError, match="Background execution unavailable"):
        tool_runtime.controller.interrupt("sw0", state)
    assert not tool_runtime.futures


def test_empty_message_and_frame_limit_before_submission(tool_runtime):
    for text in (" ", "\u96ea" * 600000):
        with pytest.raises(ValueError):
            tool_runtime.controller.broadcast("sw0", text, state=tool_runtime.runtime.state)
    assert not tool_runtime.futures


def test_malformed_slash_only_status_no_model_injection(tool_runtime):
    t = tool_runtime
    ctx = SimpleNamespace(state=t.runtime.state, session=SimpleNamespace(inject=Mock()), defer=Mock())
    with pytest.raises(CommandError):
        t.controller.command(ctx, raw_args='bcast sw0 "unterminated')
    ctx.defer.assert_not_called(); ctx.session.inject.assert_not_called()


def test_user_takeover_parser_is_exact_and_not_model_request():
    parsed = swarm.parse_command("recover sw0 --takeover --expected-epoch 12 --confirmed-stopped")
    assert parsed == dict(action="recover", target="sw0", expected_epoch=12, confirmed_stopped=True)
    for text in ("recover sw0", "recover sw0 --takeover", "recover sw0 --takeover --expected-epoch 12"):
        with pytest.raises(ValueError):
            swarm.parse_command(text)
    with pytest.raises(ValueError, match="model accessible"):
        swarm.ControlRequest("recover", "sw0").as_dict()


def test_post_id_exposed_before_write_and_idempotent_completion(tool_runtime):
    t = tool_runtime; state = t.runtime.state
    receipt = t.controller.post_message("sw0", 0, "general", "durable", "stable-id", state)
    assert "stable-id" in receipt
    first = asyncio.run(t.factories[0]())
    assert "committed" in first["summary"]
    t.controller.post_message("sw0", 0, "general", "durable", "stable-id", state)
    second = asyncio.run(t.factories[1]())
    assert "already committed" in second["summary"]
    assert len(t.runtime.store.messages("pod-1", "general")) == 1
    t.controller.post_message("sw0", 0, "general", "changed", "stable-id", state)
    assert "different content" in asyncio.run(t.factories[2]())["summary"]


def test_event_before_server_schedules_only_from_interrupt_check(tool_runtime):
    t = tool_runtime; state = t.runtime.state
    server = state._session_websocket_server
    state._session_websocket_server = None
    event = SimpleNamespace(payload=dict(session_id=state._session_id, instance_id=state._instance_id,
                                        restored=True, startup_mode="paused", paused=True, revision_id="revision"))
    handler = swarm.SwarmRuntimeReady(t.controller)
    handler.handle_harness_event(event, state); handler.handle_harness_event(event, state)
    assert len(t.controller.ready_events) == 1 and not t.futures
    check = swarm.SwarmCompletionCheck(t.controller)
    check(state)
    assert not t.futures
    state._session_websocket_server = server
    check(state); check(state)
    assert len(t.futures) == 1 and not t.controller.ready_events


def test_partial_summary_keeps_counts_and_bounds_failure_detail():
    outcomes = [dict(label=f"p{i}", state="unavailable", error=f"distinct reason {i}") for i in range(9)]
    outcomes.extend([dict(label="sent-peer", state="sent"), dict(label="unknown-peer", state="unknown", error="send timeout")])
    summary = swarm.operation_summary({"id": "sw0"}, dict(id="op", action="bcast", outcomes=outcomes, pods=[0]))
    assert "sent to 1 of 11 agents" in summary and "9 unavailable" in summary and "1 delivery unknown" in summary
    assert all(f"distinct reason {i}" in summary for i in range(3))
    assert "distinct reason 3" not in summary and "swarm:index" in summary
    assert "acceptance" not in summary and "duplicate" not in summary and len(summary) < 300


@pytest.mark.parametrize("pipeline_name", ["build_default_pipeline", "build_readonly_rlm_pipeline"])
@pytest.mark.parametrize("session_kind,is_member", [
    ("default", False), ("rlm", False), ("fork", False),
    ("sw0p0a0", True), ("sw12p31a63", True),
    ("sw0p0a0-extra", False), ("review_sw0p0a0", False),
])
def test_new_plugin_builds_real_pipelines(tmp_path, monkeypatch, pipeline_name, session_kind, is_member):
    from pathlib import Path
    import agent_zoo.pipelines as pipelines
    import agent_zoo.plugins as plugins
    from agent_zoo.modes import default_modes, rlm_modes
    root = tmp_path / "plugins"; (root / "common").mkdir(parents=True)
    (root / "common" / "swarm.py").write_text(Path(swarm.__file__).read_text())
    monkeypatch.setattr(plugins, "_base_dir", lambda: root)
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda **kwargs: [])
    # A restored member needs no launch environment. Conversely, inherited
    # environment, a matching title, or parent resource access is not membership.
    if is_member:
        monkeypatch.delenv("AZO_SWARM_ID", raising=False)
        monkeypatch.delenv("AZO_SWARM_POD_ID", raising=False)
    else:
        monkeypatch.setenv("AZO_SWARM_ID", "sw0")
        monkeypatch.setenv("AZO_SWARM_POD_ID", "pod-1")
    session = Session(system_prompt="offline", token_budget=120000)
    session.state._session_kind = session_kind
    session.state._session_name = "sw0p0a0"
    session.state.swarm_access = {"sw0": "pod-1" if is_member else ""}
    pipeline = getattr(pipelines, pipeline_name)(session, config={}, skill_paths=[], max_idle=0,
        terminal_backend="headless", mode_defs=default_modes() if pipeline_name == "build_default_pipeline" else rlm_modes(),
        on_interrupt=lambda _: None)
    tools = {c.name for c in pipeline if isinstance(c, Tool)}
    assert {t for t in tools if t.startswith("swarm_")} == {
        "swarm_broadcast", "swarm_interrupt", "swarm_continue", "swarm_cancel", "swarm_post"}
    assert "view" in tools
    assert any(type(c).__name__ == "SwarmCompletionCheck" for c in pipeline)
    has_rlm = pipeline_name == "build_default_pipeline" and not is_member
    assert tools & {"submit_rlm", "rlm_status", "cancel_rlm"} == (
        {"submit_rlm", "rlm_status", "cancel_rlm"} if has_rlm else set())
    has_compaction = pipeline_name == "build_default_pipeline"
    for component in ("RLMProcessCheck", "ContextCompactionTrigger", "ContextCompactionRenderer"):
        assert any(type(c).__name__ == component for c in pipeline) == has_compaction
    if has_compaction:
        poller = next(c for c in pipeline if type(c).__name__ == "RLMProcessCheck")
        trigger = next(c for c in pipeline if type(c).__name__ == "ContextCompactionTrigger")
        assert poller.rlm_queue is trigger.rlm_queue
        pump = Mock(return_value=session.state)
        monkeypatch.setattr(poller.rlm_queue, "poll_due", lambda _: True)
        monkeypatch.setattr(poller.rlm_queue, "pump_queue", pump)
        assert poller(session.state) is session.state
        pump.assert_called_once_with(session.state)
    if pipeline_name == "build_readonly_rlm_pipeline":
        assert "finish_rlm" in tools


def test_old_completion_cannot_roll_back_new_owner_grant(tool_runtime):
    t = tool_runtime; state = t.runtime.state
    old = t.controller.grants["sw0"].copy()
    new = dict(old, epoch=old["epoch"]+1, token="b"*32)
    t.controller.grants["sw0"] = new
    t.controller.apply(state, dict(grants={"sw0": old}))
    assert t.controller.grants["sw0"] == new


def test_snapshot_rejects_grant_from_replaced_runtime(tool_runtime):
    t = tool_runtime
    t.runtime.state._instance_id = "replaced-runtime"
    snapshot = t.controller.snapshot(t.runtime.state)
    with pytest.raises(ValueError, match="different actor instance"):
        swarm.store_for(snapshot, "sw0")


def test_restored_ready_gates_historical_work_until_discovery(tool_runtime):
    t = tool_runtime; state = t.runtime.state
    state._runtime_startup_paused = False
    event = SimpleNamespace(payload=dict(session_id=state._session_id, instance_id=state._instance_id,
        restored=True, paused=False, startup_mode="normal", revision_id="saved"))
    swarm.SwarmRuntimeReady(t.controller).handle_harness_event(event, state)
    assert state._runtime_startup_paused is True
    # A completion with ordinary recovery information cannot release the hold.
    t.controller.apply(state, dict(summary="restored paused", grants={}))
    assert state._runtime_startup_paused is True
    t.controller.apply(state, dict(release_startup_hold=True))
    assert state._runtime_startup_paused is False


def test_explicit_paused_start_is_not_released_by_empty_discovery(tool_runtime):
    t = tool_runtime; state = t.runtime.state
    state._runtime_startup_paused = True
    event = SimpleNamespace(payload=dict(session_id=state._session_id, instance_id=state._instance_id,
        restored=True, paused=True, startup_mode="paused", revision_id="saved"))
    swarm.SwarmRuntimeReady(t.controller).handle_harness_event(event, state)
    t.controller.apply(state, dict(release_startup_hold=True))
    assert state._runtime_startup_paused is True
