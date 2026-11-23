"""Role guidance and skill visibility without live workers or provider calls."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from agent_utils import Entry, State
from agent_utils.core.events import IoCompleted, IoFailed, RenderRequested
from agent_utils.core.io import IoTable
from agent_utils.features.skills import SkillRefreshHandler, SkillRefreshRequested, SkillRegistry
from agent_zoo.events import SessionReady
import swarm


def state_for(kind="sw0p2a3", content="Base"):
    state = State(token_budget=10000)
    state._session_kind = kind
    state.entries = [Entry(messages=[dict(role="system", content=content)], index=0, step=0)]
    state.entries[0].initialize_pending_render_channels()
    return state


def render(component, state):
    component.handle(RenderRequested(), state, lambda _ev: None)
    component.handle(RenderRequested(), state, lambda _ev: None)
    state.entries[0].finalize_render_channels()
    return state.entries[0].render(state)[0]["content"]


def test_worker_guidance_survives_replaced_system_entry_and_is_idempotent(monkeypatch):
    component = swarm.SwarmSystemPrompt("Custom guidance: use {literal braces}.")
    state = state_for()
    monkeypatch.setattr(Path, "read_text", Mock(side_effect=AssertionError("render IO")))
    for _ in range(3):
        text = render(component, state)
        assert text.count("# Swarm worker") == 1
        assert "sw0p2a3, worker 3 in swarm sw0, pod 2" in text
        assert "Custom guidance: use {literal braces}." in text
        assert state.entries[0].messages[0]["content"] == "Base"
        state.entries = state_for().entries  # compaction/reload builds a new system entry


@pytest.mark.parametrize("kind", ["default", "fork", "rlm", "sw0p2a3-extra"])
def test_nonworker_role_not_inferred_from_environment_or_name(kind, monkeypatch):
    monkeypatch.setenv("AZO_SWARM_ID", "sw0")
    state = state_for(kind)
    state._session_name = "sw0p2a3"
    assert render(swarm.SwarmSystemPrompt("WORKER ONLY"), state) == "Base"
    state.entries = state_for().entries
    state.swarm_access = {"sw0": ""}
    text = render(swarm.SwarmSystemPrompt("WORKER ONLY"), state)
    assert 'skill(name="Swarms")' in text and "WORKER ONLY" not in text


def test_block_content_is_preserved():
    state = state_for(content=[{"type": "input_text", "text": "Base"}])
    content = render(swarm.SwarmSystemPrompt("Custom"), state)
    assert content[0] == {"type": "input_text", "text": "Base"}
    assert content[1]["type"] == "input_text" and "Custom" in content[1]["text"]


def test_installed_prompt_is_literal_and_only_reloaded_on_pipeline_rebuild(monkeypatch):
    path = Path(swarm.resolve_state_root()) / "plugin-configs/azo-plugin-swarms/config/worker_prompt.md"
    path.parent.mkdir(parents=True)
    path.write_text("Local policy {unchanged}")
    component = swarm.SwarmSystemPrompt(swarm.load_worker_prompt())
    path.write_text("New policy")
    assert "Local policy {unchanged}" in render(component, state_for())
    assert swarm.load_worker_prompt() == "New policy"


def test_default_worker_guidance_has_index_and_posting_tool():
    text = swarm.load_worker_prompt()
    assert 'view(buffer="swarm:index")' in text
    assert "swarm_post" in text


@pytest.mark.parametrize("kind", ["default", "rlm", "fork", "sw0p0a0"])
def test_skill_catalog_is_role_scoped_and_does_no_repeat_io(kind, monkeypatch):
    state = state_for(kind)
    state.skill_registry = SkillRegistry([])
    component = swarm.SwarmSkillRegistration()
    component.handle(RenderRequested(), state, lambda _ev: None)
    assert bool(state.skill_registry.get("Swarms")) == (kind != "sw0p0a0")
    forbidden = Mock(side_effect=AssertionError("repeated discovery"))
    monkeypatch.setattr(Path, "is_file", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    for _ in range(4):
        component.handle(RenderRequested(), state, lambda _ev: None)
        state.skill_registry.list_skills()
    forbidden.assert_not_called()


def test_restored_worker_catalog_excludes_parent_skill_after_refresh():
    state = state_for("default")
    state.skill_registry = SkillRegistry([])
    swarm.SwarmSkillRegistration().handle(RenderRequested(), state, lambda _ev: None)
    assert state.skill_registry.get("Swarms")
    state._session_kind = "sw0p0a0"
    swarm.SwarmSkillRegistration().handle(RenderRequested(), state, lambda _ev: None)
    assert state.skill_registry.get("Swarms") is None
    assert not any(src.kind == "azo-plugin-swarms" for src in state.skill_registry.sources)


@pytest.mark.parametrize("failure", [False, True])
def test_worker_completion_is_silent_but_applies_results(failure, caplog):
    state = state_for()
    state._session_id, state._instance_id = "worker", "instance"
    state.pending_interrupts = ["Unrelated user notification"]
    controller = swarm.SwarmController({})
    controller.in_flight["op_1"] = (controller.identity(state), "startup", {})
    event = (
        IoFailed(op_id="op_1", error=RuntimeError("Worker startup failed"))
        if failure
        else IoCompleted(op_id="op_1", value=dict(summary="agent 9 paused", access={"sw0": "pod-3"}))
    )
    swarm.SwarmCompletionHandler(controller).handle(event, state, lambda _ev: None)
    assert state.pending_interrupts == ["Unrelated user notification"]
    assert not controller.in_flight
    if failure:
        assert "Worker startup failed" in caplog.text
    else:
        assert state.swarm_access == {"sw0": "pod-3"}


def test_worker_startup_submission_failure_is_logged_without_interrupt(caplog):
    state = state_for()
    state._session_id, state._instance_id = "worker", "instance"
    state.pending_interrupts = []
    state.io = IoTable()
    state.io.submit = Mock(side_effect=ValueError("no loop"))
    controller = swarm.SwarmController({})
    event = SessionReady(session_id="worker", instance_id="instance", restored=False, paused=True)
    swarm.SwarmRuntimeReady(controller).handle(event, state, lambda _ev: None)
    assert not state.pending_interrupts and not controller.in_flight
    assert "no loop" in caplog.text


def test_parent_notice_has_exact_user_broadcast_and_receivers():
    message = '  First line\n"quoted" \\ final\n  '
    state = state_for("default")
    state.pending_interrupts = []
    emitted = []
    result = dict(summary="sw0: message sent to 2 agents — pods 0, 2.", pool={"id": "sw0"},
                  operation=dict(action="bcast", message=message,
                                 outcomes=[{"label": "sw0p0a0"}, {"label": "sw0p2a0"}]))
    swarm.SwarmController({}).complete(state, result, emitted.append)
    notice = state.pending_interrupts[0]
    assert "The user broadcast" in notice and "sw0p0a0, sw0p2a0" in notice
    assert "pods 0, 2" in notice and notice.endswith(message)
    assert "do not resend" in notice
    assert any(type(ev).__name__ == "InterruptQueued" for ev in emitted)


def test_parent_launch_message_separate_from_short_status():
    state = state_for("default")
    state.pending_interrupts = []
    emitted = []
    result = dict(summary="sw0: 16 agents ready (paused).", model_message="The user launched swarm sw0. Read the Swarms skill.")
    swarm.SwarmController({}).complete(state, result, emitted.append)
    assert state.pending_interrupts == [result["model_message"]]
    assert any(type(ev).__name__ == "InterruptQueued" for ev in emitted)
    state._session_kind = "sw0p0a0"
    state.pending_interrupts = []
    emitted.clear()
    swarm.SwarmController({}).complete(state, result, emitted.append)
    assert state.pending_interrupts == []
    assert not any(type(ev).__name__ == "InterruptQueued" for ev in emitted)


def test_installed_skill_copy_path_and_explicit_refresh(tmp_path, monkeypatch):
    path = Path(swarm.resolve_state_root()) / "plugin-configs/azo-plugin-swarms/skills/swarm/SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text('---\nname: Swarms\ndescription: Initial description\n---\nUser assigns work too.\n')
    monkeypatch.setattr(swarm, "__file__", str(tmp_path / "plugins/common/swarm.py"))
    state = state_for("default")
    state.skill_registry = SkillRegistry([])
    component = swarm.SwarmSkillRegistration()
    component.handle(RenderRequested(), state, lambda _ev: None)
    assert state.skill_registry.get("Swarms").description == "Initial description"
    path.write_text('---\nname: Swarms\ndescription: Changed description\n---\nNew body.\n')
    component.handle(RenderRequested(), state, lambda _ev: None)
    assert state.skill_registry.get("Swarms").description == "Initial description"
    # Explicit SkillRefreshRequested refreshes from disk via IoCompleted.
    refresher = SkillRefreshHandler(state.skill_registry)
    refresher.handle(SkillRefreshRequested(), state, lambda _ev: None)
    glob_effect, = state.io.take_outbox()
    refresher.handle(IoCompleted(op_id=glob_effect.op_id, value=[str(path)]), state, lambda _ev: None)
    read_effect, = state.io.take_outbox()
    refresher.handle(IoCompleted(op_id=read_effect.op_id, value=path.read_text()), state, lambda _ev: None)
    assert state.skill_registry.get("Swarms").description == "Changed description"
    state.skill_registry = SkillRegistry([])
    component.handle(RenderRequested(), state, lambda _ev: None)
    assert state.skill_registry.get("Swarms").description == "Changed description"


def test_manifest_packages_config_and_skill():
    import tomllib
    root = Path(swarm.__file__).resolve().parent.parent
    manifest = tomllib.loads((root / "pixi.toml").read_text())
    files = manifest["tool"]["agent-zoo"]["plugin"]["config"]["files"]
    assert set(files) == {"config/swarm.json", "config/worker_prompt.md", "skills/swarm/SKILL.md"}
    assert all((root / path).is_file() for path in files)
    skill = (root / "skills/swarm/SKILL.md").read_text()
    assert "The user can assign work directly" in skill
    assert "do not duplicate or override" in skill


def test_model_broadcast_completion_is_not_a_user_assignment_notification():
    state = state_for("default")
    state._session_id, state._instance_id = "parent", "instance"
    state.pending_interrupts = []
    controller = swarm.SwarmController({})
    result = dict(action="bcast", summary="sw0: message sent to 1 agent.", pool={"id": "sw0"},
                  operation=dict(action="bcast", message="agent-generated work",
                                 outcomes=[{"label": "sw0p0a0"}]))
    controller.in_flight["op_1"] = (controller.identity(state), "tool", {"action": "bcast"})
    swarm.SwarmCompletionHandler(controller).handle(
        IoCompleted(op_id="op_1", value=result), state, lambda _ev: None
    )
    assert state.pending_interrupts == []
