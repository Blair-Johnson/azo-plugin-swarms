"""Named model APIs and owner-thread completion; no command-string round trip."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from agent_utils import Tool, Session
from agent_utils.core.events import IoCompleted, IoFailed, IoLost, IoReattached, RenderRequested, SessionRestored, SessionStarted
from agent_utils.core.io import IoTable, SubmitOp
from agent_utils.core.state import ReadinessGate
from agent_zoo.events import EmitNotice, SessionReady
from agent_zoo.features.commands import CommandError
import swarm
from conftest import drain_first_request
from test_recovery import runtime


@pytest.fixture
def tool_runtime(runtime):
    r = runtime
    r.state.io = IoTable()
    session = SimpleNamespace(state=r.state, pipeline=object())
    c = swarm.SwarmController({"llm": {"default": "offline"}})
    c.grants = r.state.grants.copy()

    def take_effects():
        return [eff for eff in r.state.io.take_outbox() if isinstance(eff, SubmitOp)]

    return SimpleNamespace(runtime=r, controller=c, session=session, take_effects=take_effects)


def test_exact_five_tools_have_named_zero_based_arguments():
    features = []
    swarm.register_features(SimpleNamespace(add=features.append), config={})
    swarm_feature = next(f for f in features if f.name == "swarm")
    tools = {t.name: t for t in swarm_feature.components if isinstance(t, Tool)}
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
    effects = t.take_effects()
    assert len(effects) == 1 and isinstance(effects[0].request, swarm.SwarmStoreRequest)
    _, _, meta = c.in_flight[effects[0].op_id]
    assert meta["message"] == text and meta["pods"] == (1, 0)
    assert meta["operation_id"] not in receipt
    # Replace queued runner with direct completion check to verify idempotent delivery.
    wf_id = meta["_wf"]
    c._workflows.pop(wf_id, None)
    c.in_flight[effects[0].op_id] = (c.identity(state), "tool", {"action": "bcast"})
    result = dict(action="bcast", summary="OK sent=2/2", access={"sw0": ""}, grants={})
    handler = swarm.SwarmCompletionHandler(c)
    handler.handle(IoCompleted(op_id=effects[0].op_id, value=result), state, lambda _ev: None)
    handler.handle(IoCompleted(op_id=effects[0].op_id, value=result), state, lambda _ev: None)
    assert not state.pending_interrupts and not c.in_flight


@pytest.mark.parametrize("change", ["state", "session", "instance", "pipeline"])
def test_stale_completion_is_observed_but_not_applied(tool_runtime, change):
    t = tool_runtime; state = t.runtime.state
    t.controller.interrupt("sw0", state)
    effect, = t.take_effects()
    result = dict(summary="must not deliver", access={"forged": ""}, grants={"forged": {}})
    target_state = state
    if change == "state":
        target_state = SimpleNamespace(
            _session_id=state._session_id,
            _instance_id=state._instance_id,
            pending_interrupts=[],
            swarm_access={},
        )
    elif change == "session":
        state._session_id = "different"
    elif change == "instance":
        state._instance_id = "different"
    else:
        state.io_generation = int(getattr(state, "io_generation", 0) or 0) + 1
    swarm.SwarmCompletionHandler(t.controller).handle(
        IoCompleted(op_id=effect.op_id, value=result), target_state, lambda _ev: None
    )
    assert not target_state.pending_interrupts and "forged" not in target_state.swarm_access
    assert not state.pending_interrupts and "forged" not in state.swarm_access
    assert "forged" not in t.controller.grants


@pytest.mark.parametrize("pods", [[], [True], [-1], [32], ["0"]])
def test_invalid_tool_selectors_never_submit(tool_runtime, pods):
    with pytest.raises(ValueError):
        tool_runtime.controller.broadcast("sw0", "payload", pods, tool_runtime.runtime.state)
    assert tool_runtime.take_effects() == []


def test_missing_background_service_is_immediate_no_submission(tool_runtime):
    state = tool_runtime.runtime.state
    state.io = None
    with pytest.raises(ValueError, match="Background execution unavailable"):
        tool_runtime.controller.interrupt("sw0", state)


def test_empty_message_and_frame_limit_before_submission(tool_runtime):
    for text in (" ", "\u96ea" * 600000):
        with pytest.raises(ValueError):
            tool_runtime.controller.broadcast("sw0", text, state=tool_runtime.runtime.state)
    assert not tool_runtime.take_effects()


def test_malformed_slash_only_status_no_model_injection(tool_runtime):
    t = tool_runtime
    ctx = SimpleNamespace(state=t.runtime.state, emit=Mock(), submit_io=Mock(return_value="op_1"))
    with pytest.raises(CommandError):
        t.controller.command(ctx, raw_args='bcast sw0 "unterminated')
    ctx.submit_io.assert_not_called(); ctx.emit.assert_not_called()


@pytest.mark.parametrize("raw", [
    "afk This is an afk message.", "afk missing", "release missing", "capture missing",
    "interrupt missing", "continue missing", "cancel missing", 'bcast missing "hello"',
])
def test_deferred_slash_failure_is_status_only(tool_runtime, monkeypatch, raw):
    t = tool_runtime
    state = t.runtime.state
    ctx = SimpleNamespace(state=state, emit=Mock(), submit_io=lambda req, **kw: state.io.submit(req, **kw))
    t.controller.command(ctx, raw_args=raw)
    effect, = t.take_effects()
    result = drain_first_request(t.controller, state, effect.request)
    assert result["level"] == "warning"
    assert swarm.parse_command(raw)["target"] in result["summary"]
    emitted = []
    t.controller.in_flight[effect.op_id] = (t.controller.identity(state), "command", {"action": "afk"})
    swarm.SwarmCompletionHandler(t.controller).handle(
        IoCompleted(op_id=effect.op_id, value=result), state, emitted.append
    )
    assert any(isinstance(ev, EmitNotice) and ev.text == result["summary"] and ev.level == "warning" for ev in emitted)
    assert not any(type(ev).__name__ == "InterruptQueued" for ev in emitted)
    assert not state.pending_interrupts
    assert not t.runtime.store.records().list("operations")


def test_deferred_tool_failure_still_notifies_model(tool_runtime):
    t = tool_runtime
    state = t.runtime.state
    t.controller.interrupt("missing", state)
    effect, = t.take_effects()
    result = drain_first_request(t.controller, state, effect.request)
    assert result["level"] == "warning"
    t.controller.in_flight[effect.op_id] = (t.controller.identity(state), "tool", {"action": "interrupt"})
    handler = swarm.SwarmCompletionHandler(t.controller)
    handler.handle(IoCompleted(op_id=effect.op_id, value=result), state, lambda _ev: None)
    handler.handle(IoCompleted(op_id=effect.op_id, value=result), state, lambda _ev: None)
    assert state.pending_interrupts == [result["summary"]]
    assert not t.controller.in_flight


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
    first = drain_first_request(t.controller, state, t.take_effects()[0].request)
    assert "committed" in first["summary"]
    t.controller.post_message("sw0", 0, "general", "durable", "stable-id", state)
    second = drain_first_request(t.controller, state, t.take_effects()[0].request)
    assert "already committed" in second["summary"]
    assert len(t.runtime.store.messages("pod-1", "general")) == 1
    t.controller.post_message("sw0", 0, "general", "changed", "stable-id", state)
    assert "different content" in drain_first_request(t.controller, state, t.take_effects()[0].request)["summary"]


def test_session_ready_submits_once_and_deduplicates(tool_runtime):
    t = tool_runtime; state = t.runtime.state
    event = SessionReady(
        session_id=state._session_id,
        instance_id=state._instance_id,
        restored=True,
        startup_mode="paused",
        paused=True,
        revision_id="revision",
    )
    handler = swarm.SwarmRuntimeReady(t.controller)
    handler.handle(event, state, lambda _ev: None)
    handler.handle(event, state, lambda _ev: None)
    effects = t.take_effects()
    assert len(effects) == 1 and len(t.controller.in_flight) == 1


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
    import agent_zoo.plugins.loader as plugin_loader
    from agent_zoo.features.modes import default_modes, rlm_modes
    root = tmp_path / "plugins"; (root / "common").mkdir(parents=True)
    skill_path = swarm.PluginStorage(swarm.PLUGIN_NAME).config_dir / "skills/swarm/SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text((Path(swarm.__file__).resolve().parent.parent / "skills/swarm/SKILL.md").read_text())
    (root / "common" / "swarm.py").write_text(Path(swarm.__file__).read_text())
    monkeypatch.setattr(plugin_loader, "_base_dir", lambda: root)
    monkeypatch.setattr(plugin_loader.metadata, "entry_points", lambda **kwargs: [])
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
    pipeline = getattr(pipelines, pipeline_name)(
        session,
        config={},
        skill_paths=[],
        terminal_backend="headless",
        mode_defs=default_modes() if pipeline_name == "build_default_pipeline" else rlm_modes(),
        include_llm_drivers=False,
    )
    session.pipeline = pipeline
    session.ensure_initialized()
    session.step(SessionStarted())
    session.render()
    assert bool(session.state.skill_registry.get("Swarms")) == (not is_member)
    prompt = next(c for c in pipeline if type(c).__name__ == "SwarmSystemPrompt")
    rendered = prompt.transform([{"role": "system", "content": "Base"}], session.state)[0]["content"]
    assert ("# Swarm worker" in rendered) == is_member
    components = list(pipeline.components)
    reg_idx = next(i for i, c in enumerate(components) if type(c).__name__ == "SwarmSkillRegistration")
    renderer_idx = next(i for i, c in enumerate(components) if isinstance(c, swarm.MessageRenderer))
    assert reg_idx < renderer_idx
    tools = {c.name for c in pipeline if isinstance(c, Tool)}
    assert {t for t in tools if t.startswith("swarm_")} == {
        "swarm_broadcast", "swarm_interrupt", "swarm_continue", "swarm_cancel", "swarm_post"}
    assert "view" in tools
    assert any(type(c).__name__ == "SwarmCompletionHandler" for c in pipeline)
    has_rlm = pipeline_name == "build_default_pipeline" and not is_member
    available_tools = {name for name, entry in session.state.tool_schemas.items() if entry.get("available")}
    assert available_tools & {"submit_rlm", "rlm_status", "cancel_rlm"} == (
        {"submit_rlm", "rlm_status", "cancel_rlm"} if has_rlm else set())
    if pipeline_name == "build_default_pipeline" and is_member:
        from agent_utils.core.events import RunToolCall
        from agent_utils.core.state import ToolCall

        tc = ToolCall(id="rlm-block-1", name="submit_rlm", raw_args="{}", parsed_args={"prompt": "nope"})
        session.state.pending_tool_calls.append(tc)
        session.step(RunToolCall(tc))
        assert tc.error and "not available to swarm workers" in str(tc.result)
    has_compaction = pipeline_name == "build_default_pipeline"
    for component in ("RLMCompletionHandler", "ContextCompactionGate", "ContextCompactionRenderer"):
        assert any(type(c).__name__ == component for c in pipeline) == has_compaction
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
    state.startup_paused = False
    state.readiness = ReadinessGate()
    event = SessionReady(
        session_id=state._session_id,
        instance_id=state._instance_id,
        restored=True,
        paused=False,
        startup_mode="normal",
        revision_id="saved",
    )
    swarm.SwarmRuntimeReady(t.controller).handle(event, state, lambda _ev: None)
    assert not state.readiness.is_ready
    assert "swarm" in state.readiness.holds
    # A completion with ordinary recovery information cannot release the hold.
    t.controller.apply(state, dict(summary="restored paused", grants={}))
    assert not state.readiness.is_ready
    t.controller.apply(state, dict(release_startup_hold=True))
    assert state.readiness.is_ready


def test_explicit_paused_start_is_not_released_by_empty_discovery(tool_runtime):
    t = tool_runtime; state = t.runtime.state
    state.startup_paused = True
    state.readiness = ReadinessGate()
    event = SessionReady(
        session_id=state._session_id,
        instance_id=state._instance_id,
        restored=True,
        paused=True,
        startup_mode="paused",
        revision_id="saved",
    )
    swarm.SwarmRuntimeReady(t.controller).handle(event, state, lambda _ev: None)
    t.controller.apply(state, dict(release_startup_hold=True))
    assert state.startup_paused is True


def test_swarm_module_and_pipeline_are_strictly_sans_io(tmp_path, monkeypatch):
    from agent_utils import PipelineBuilder, core_feature, interrupts_feature
    from agent_zoo.features.special_buffers import special_buffers_feature
    from agent_zoo.testing.sans_io_guard import assert_pipeline_sans_io, audit_module_sans_io

    assert audit_module_sans_io(swarm.__file__) == []

    monkeypatch.setenv("AGENT_ZOO_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    session = Session(system_prompt="sans-io test", cwd=str(tmp_path))
    builder = (
        PipelineBuilder()
        .add(core_feature())
        .add(interrupts_feature())
        .add(special_buffers_feature())
    )
    swarm.register_features(builder, config={})
    pipeline = builder.build()
    assert_pipeline_sans_io(pipeline)


def test_swarm_store_driver_lifecycle_and_recovery_events(tool_runtime):
    t = tool_runtime; state = t.runtime.state
    state.readiness = ReadinessGate()
    driver = swarm.SwarmStoreDriver()
    assert driver.durability == "commit"
    assert swarm.SwarmStoreRequest in driver.request_types

    emitted = []
    driver.start(emitted.append)
    try:
        driver.submit("op-snap", swarm.SwarmStoreRequest(op="refresh_snapshot", params={"snapshot": state}))
        driver.drain(timeout_s=5.0)
        assert any(isinstance(ev, IoCompleted) and ev.op_id == "op-snap" for ev in emitted)
        driver.quiesce(policy="terminate")
    finally:
        driver.close()

    # SessionRestored holds readiness and submits startup; IoReattached records owned instance;
    # IoLost releases startup readiness hold and cleans up in-flight state.
    ready_handler = swarm.SwarmRuntimeReady(t.controller)
    comp_handler = swarm.SwarmCompletionHandler(t.controller)
    ready_handler.handle(SessionRestored(), state, lambda _ev: None)
    assert not state.readiness.is_ready
    effects = t.take_effects()
    assert len(effects) == 1
    startup_op = effects[0].op_id

    comp_handler.handle(IoReattached(op_id="op-re", handle={"instance_id": "child-inst-1"}), state, lambda _ev: None)
    assert "child-inst-1" in t.controller._owned_instances

    comp_handler.handle(IoLost(op_id=startup_op, reason="driver_restarted"), state, lambda _ev: None)
    assert state.readiness.is_ready
    assert not t.controller.in_flight
    assert any("lost" in str(msg).lower() for msg in state.pending_interrupts)
