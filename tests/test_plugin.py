"""Plugin contracts; all durable/runtime artifacts stay under tmp_path."""
from dataclasses import replace
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_utils import Session, Tool
from agent_utils.files.buffer_manager import BufferManager, ReadonlyBufferView
from agent_zoo.runtime.commands import Command
from agent_zoo.runtime_launch import (
    CheckpointRef, Fresh, Resume, RuntimeLauncher, RuntimeSettings, load_launch_request,
)
import swarm


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    for key, directory in {
        "AGENT_ZOO_HOME": "state", "AGENT_ZOO_STATE_ROOT": "state",
        "AGENT_ZOO_LOCAL_STATE_ROOT": "local", "AZO_TUI_STORE_ROOT": "store",
        "AZO_TUI_LIVE_SESSION_STORE_ROOT": "store", "AGENT_ZOO_INSTALL_ROOT": "install",
        "XDG_CONFIG_HOME": "config", "XDG_CACHE_HOME": "cache", "XDG_DATA_HOME": "data",
    }.items():
        monkeypatch.setenv(key, str(tmp_path / directory))
    for key in ("AZO_SWARM_ID", "AZO_SWARM_POD_ID"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "state" / "projects" / "test-project").mkdir(parents=True)
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=AssertionError("real spawn forbidden")))


@pytest.fixture
def state():
    return SimpleNamespace(
        _session_id="parent-session", _agent_zoo_context={"project_name": "test-project"},
        swarm_access={}, buffer_manager=BufferManager(), step=7,
    )


@pytest.fixture
def pool(state):
    store = swarm.store_for(state, "test-swarm")
    metadata = store.create("test-swarm", pods=2, agents_per_pod=2, boards=2)
    swarm.bind_resource(state, "test-swarm")
    swarm.SwarmBuffers()(state)
    return store, metadata


def read(state, identifier):
    return state.buffer_manager.resolve_for_read(state, identifier)


def member(pool, pod=0):
    return pool[1]["pods"][pod]["members"][0]["session_id"]


def peer(state, pool, *, persisted_scope="pod-1"):
    state._session_id = member(pool)
    state.swarm_access = {"test-swarm": persisted_scope}
    return state


def test_namespace_is_readonly_lazy_and_refreshes_same_step(state, pool):
    manager = state.buffer_manager
    assert manager.special_buffer_namespaces() == ["swarm:"]
    assert manager.buffers() == []
    index = read(state, "swarm:index")
    assert isinstance(index, ReadonlyBufferView)
    assert "pod-1" in index.text and "pod-2" in index.text
    assert "channel-2" in index.text
    assert "not evidence" in index.text
    board_id = "swarm:test-swarm:pod-1:board:general"
    before = read(state, board_id)
    assert "first post" not in before.text
    result = swarm.post_message("test-swarm", "pod-1", "general", "first post", "stable-id", state)
    assert "Posted stable-id" in result
    after = read(state, board_id)
    assert "first post" in after.text
    assert "data, not user instructions" in after.text
    assert before.text != after.text and state.step == 7
    for identifier in ("swarm:index", board_id):
        assert read(state, identifier).readonly
        with pytest.raises(ValueError, match="readonly"):
            manager.require_writable(identifier)
    with pool[0].records(write=True).transaction("swarm", "pool") as metadata:
        metadata["desired_state"] = "running"
    assert "desired state: running" in read(state, "swarm:index").text
    assert manager.buffers() == []


def test_saved_transcripts_are_fresh_readonly_and_not_live(state, pool):
    session_id = member(pool)
    identifier = f"swarm:test-swarm:pod-1:session:{session_id}"
    missing = read(state, identifier)
    assert "No shared checkpoint" in missing.text
    assert "not a live stream" in missing.text
    source = swarm.project_directory(state) / "sessions" / session_id / "session.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"entries": [{"messages": [
        {"role": "user", "content": "saved question"},
        {"role": "assistant", "content": "saved answer"},
        {"role": "system", "content": "do not expose system entry"},
    ]}]}))
    saved = read(state, identifier)
    assert "saved question" in saved.text and "saved answer" in saved.text
    assert "do not expose system entry" not in saved.text
    assert str(source) in saved.text and saved.readonly
    source.write_text(json.dumps({"state": {"entries": [{"messages": [
        {"role": "tool", "content": "new checkpoint"},
    ]}]}}))
    assert "new checkpoint" in read(state, identifier).text
    assert "saved answer" not in read(state, identifier).text
    with pytest.raises(ValueError, match="readonly"):
        state.buffer_manager.require_writable(identifier)
    with pytest.raises(KeyError):
        read(state, f"swarm:test-swarm:pod-1:session:{member(pool, 1)}")


@pytest.mark.parametrize("resumed", [False, True])
def test_peer_index_and_cross_pod_access_are_scoped(state, pool, monkeypatch, resumed):
    # Resume may retain a former parent's broad checkpointed binding.
    peer(state, pool, persisted_scope="" if resumed else "pod-1")
    if resumed:
        monkeypatch.setenv("AZO_SWARM_ID", "test-swarm")
        monkeypatch.setenv("AZO_SWARM_POD_ID", "pod-1")
    index = read(state, "swarm:index").text
    assert "pod-1" in index and "pod-2" not in index
    assert member(pool) in index and member(pool, 1) not in index
    for identifier in (
        "swarm:test-swarm:pod-2:board:general",
        f"swarm:test-swarm:pod-2:session:{member(pool, 1)}",
    ):
        with pytest.raises(ValueError, match="scope|pod|Pod"):
            read(state, identifier)
    with pytest.raises(ValueError, match="scope|pod|Pod"):
        swarm.post_message("test-swarm", "pod-2", "general", "forbidden", state=state)
    assert pool[0].messages("pod-2", "general") == []
    assert "Posted" in swarm.post_message("test-swarm", "pod-1", "general", "allowed", state=state)


def test_unbound_resource_and_wrong_sender_are_rejected(state, pool):
    with pytest.raises(ValueError, match="not bound"):
        read(state, "swarm:other:pod-1:board:general")
    with pytest.raises(ValueError, match="not bound"):
        swarm.post_message("other", "pod-1", "general", "no", state=state)
    state.swarm_access = {"test-swarm": "pod-1"}
    with pytest.raises(ValueError, match="member|belong"):
        swarm.post_message("test-swarm", "pod-1", "general", "no", state=state)


def test_tool_available_only_when_bound_and_bootstrap_requires_pod(state, monkeypatch):
    tool = swarm.SwarmPost()
    assert isinstance(tool, Tool) and tool.name == "swarm_post"
    assert not tool.available(state)
    assert "No swarm resource" in swarm.SwarmBuffers().render(state, "swarm:index").text
    state.swarm_access = {"test-swarm": ""}
    assert tool.available(state)
    assert swarm.initial_access() == {}
    monkeypatch.setenv("AZO_SWARM_ID", "test-swarm")
    with pytest.raises(ValueError):
        swarm.initial_access()
    monkeypatch.setenv("AZO_SWARM_POD_ID", "pod-1")
    assert swarm.initial_access() == {"test-swarm": "pod-1"}


def test_post_failure_never_acknowledges_success(state, pool, monkeypatch):
    monkeypatch.setattr(swarm.SwarmStore, "post", Mock(side_effect=OSError("disk unavailable")))
    with pytest.raises(RuntimeError, match="Post retry-id not acknowledged"):
        swarm.post_message("test-swarm", "pod-1", "general", "payload", "retry-id", state)


def test_import_and_command_registration_have_no_writes_or_spawns(monkeypatch):
    spec = importlib.util.spec_from_file_location("swarm_import_safety", swarm.__file__)
    module = importlib.util.module_from_spec(spec)
    features = []
    def forbidden(*args, **kwargs):
        raise AssertionError("import/registration must not write or spawn")
    def guarded_open(original):
        def checked(file, mode="r", *args, **kwargs):
            assert not any(flag in mode for flag in "wax+"), "filesystem write during import"
            return original(file, mode, *args, **kwargs)
        return checked
    with monkeypatch.context() as guard:
        guard.setattr(sys, "dont_write_bytecode", True)
        guard.setattr(builtins, "open", guarded_open(builtins.open))
        guard.setattr(io, "open", guarded_open(io.open))
        guard.setattr(os, "open", forbidden)
        guard.setattr(Path, "mkdir", forbidden)
        guard.setattr(swarm.sqlite3, "connect", forbidden)
        guard.setattr(swarm.RecordStore, "__init__", forbidden)
        guard.setattr(RuntimeLauncher, "prepare", forbidden)
        guard.setattr(RuntimeLauncher, "start", forbidden)
        spec.loader.exec_module(module)
        module.register_features(SimpleNamespace(add=features.append), session=SimpleNamespace(), config={})
    assert len(features) == 1 and features[0].name == "swarm"
    components = features[0].components
    assert {type(component).__name__ for component in components} == {"SwarmBuffers", "SwarmPost", "SwarmControl", "Command"}
    command = next(component for component in components if isinstance(component, Command))
    assert command.path == "/swarm" and command.raw
    assert [component.name for component in components if isinstance(component, Tool)] == ["swarm_post", "swarm_control"]


@pytest.mark.parametrize("pipeline_name", ["build_default_pipeline", "build_readonly_rlm_pipeline"])
def test_plain_common_plugin_builds_real_pipelines(tmp_path, monkeypatch, pipeline_name):
    import agent_zoo.pipelines as pipelines
    import agent_zoo.plugins as plugins
    from agent_zoo.modes import default_modes, rlm_modes
    plugin_root = tmp_path / "plugins"
    (plugin_root / "common").mkdir(parents=True)
    (plugin_root / "common" / "swarm.py").write_text(Path(swarm.__file__).read_text())
    monkeypatch.setattr(plugins, "_base_dir", lambda: plugin_root)
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda **kwargs: [])
    session = Session(system_prompt="offline plugin build", token_budget=120_000)
    pipeline = getattr(pipelines, pipeline_name)(
        session, config={}, skill_paths=[], max_idle=0, terminal_backend="headless",
        mode_defs=default_modes() if pipeline_name == "build_default_pipeline" else rlm_modes(),
        on_interrupt=lambda message: None,
    )
    components = [component for component in pipeline if type(component).__name__ in {"SwarmBuffers", "SwarmPost", "SwarmControl"}]
    assert {type(component).__name__ for component in components} == {"SwarmBuffers", "SwarmPost", "SwarmControl"}
    assert any(isinstance(component, Command) and component.path == "/swarm" and component.raw for component in pipeline)
    assert all(not component.available(session.state) for component in components if isinstance(component, Tool))


@pytest.fixture
def settings(tmp_path):
    return RuntimeSettings(
        project="test-project", model="offline-test-model", workdir=str(tmp_path),
        state_root=str(tmp_path / "state"), store_root=str(tmp_path / "store"),
        local_state_root=str(tmp_path / "local"),
    )


@pytest.mark.parametrize("resume", [False, True])
def test_launch_uses_current_service_reserves_before_spawn_and_keeps_handles_off_state(
    state, pool, settings, monkeypatch, resume,
):
    member_id = member(pool)
    handles = {}
    process = SimpleNamespace(pid=123456, poll=lambda: None)
    captured = {}
    def popen(command, **kwargs):
        attempt = pool[0].records().get("attempts", member_id)
        assert attempt["state"] == "reserved"
        request = load_launch_request(command[command.index("--launch-file") + 1])
        assert attempt["instance_id"] == request.target.instance_id
        assert attempt["request"]["first_user_message"] == ""
        captured.update(request=request, kwargs=kwargs)
        return process
    stale = RuntimeLauncher(popen=Mock(side_effect=AssertionError("stale launch service")))
    state.session_launcher = stale
    current_popen = Mock(side_effect=popen)
    state.session_launcher = RuntimeLauncher(popen=current_popen)
    before = dict(vars(state))
    checkpoint = CheckpointRef(member_id, "saved-revision") if resume else None
    handle = swarm.launch_member(state, "test-swarm", member_id, settings, handles=handles, checkpoint=checkpoint)
    request, kwargs = captured["request"], captured["kwargs"]
    assert isinstance(request.source, Resume if resume else Fresh)
    if resume:
        assert request.source.checkpoint == checkpoint
    assert request.first_user_message == ""
    assert "Wait for an initial user message" in request.startup_system_message
    assert request.target.session_id == member_id and request.parent_session_id == state._session_id
    assert request.kind == "test-swarmp0a0" and request.lifetime == "independent"
    assert kwargs["env"]["AZO_SWARM_ID"] == "test-swarm"
    assert kwargs["env"]["AZO_SWARM_POD_ID"] == "pod-1"
    assert kwargs["env"]["AGENT_ZOO_HOME"] == settings.state_root
    assert kwargs["env"]["AGENT_ZOO_LOCAL_STATE_ROOT"] == settings.local_state_root
    assert kwargs["cwd"] == settings.workdir
    assert handle.process is process and handles == {request.target.instance_id: handle}
    assert vars(state) == before
    assert current_popen.call_count == 1
    assert pool[0].records().get("attempts", member_id)["state"] == "spawned"
    # Reopen from shared records with an entirely different local projection.
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(Path(settings.local_state_root).with_name("new-cache")))
    assert swarm.store_for(state, "test-swarm").cache != pool[0].cache
    with pytest.raises(ValueError, match="Previous attempt|reconciliation|replay"):
        swarm.launch_member(state, "test-swarm", member_id, settings, handles={})
    assert current_popen.call_count == 1


def test_failed_spawn_is_unknown_and_not_automatically_retried(state, pool, settings, monkeypatch):
    popen = Mock(side_effect=OSError("cannot spawn"))
    state.session_launcher = RuntimeLauncher(popen=popen)
    handles = {}
    with pytest.raises(OSError, match="cannot spawn"):
        swarm.launch_member(state, "test-swarm", member(pool), settings, handles=handles)
    assert handles == {}
    attempt = pool[0].records().get("attempts", member(pool))
    assert attempt["state"] == "unknown" and "pid" not in attempt
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(Path(settings.local_state_root).with_name("reopened")))
    with pytest.raises(ValueError, match="Previous attempt|reconciliation|replay"):
        swarm.launch_member(state, "test-swarm", member(pool), settings, handles={})
    assert popen.call_count == 1


@pytest.mark.parametrize("invalid", ["model", "workdir", "state_root", "project", "checkpoint", "service", "peer"])
def test_invalid_launch_never_reserves_or_spawns(state, pool, settings, invalid):
    popen = Mock(side_effect=AssertionError("invalid request must not spawn"))
    state.session_launcher = RuntimeLauncher(popen=popen)
    checkpoint = None
    if invalid in {"model", "workdir", "state_root"}:
        settings = replace(settings, **{invalid: ""})
    elif invalid == "project":
        settings = replace(settings, project="different-project")
    elif invalid == "checkpoint":
        checkpoint = CheckpointRef("other-session")
    elif invalid == "service":
        del state.session_launcher
    elif invalid == "peer":
        peer(state, pool)
    with pytest.raises(ValueError):
        swarm.launch_member(state, "test-swarm", member(pool), settings, handles={}, checkpoint=checkpoint)
    assert pool[0].attempts() == []
    popen.assert_not_called()


def test_status_write_failure_retains_handle_and_blocks_duplicate_spawn(state, pool, settings, monkeypatch):
    process = SimpleNamespace(pid=123456, poll=lambda: None)
    popen = Mock(return_value=process)
    state.session_launcher = RuntimeLauncher(popen=popen)
    handles = {}
    monkeypatch.setattr(swarm.SwarmStore, "update_attempt", Mock(side_effect=OSError("status unavailable")))
    with pytest.raises(OSError, match="status unavailable"):
        swarm.launch_member(state, "test-swarm", member(pool), settings, handles=handles)
    assert len(handles) == 1
    assert next(iter(handles.values())).process is process
    saved = pool[0].records().get("attempts", member(pool))
    assert saved["state"] == "reserved" and saved["host"]["host_id"]
    with pytest.raises(ValueError, match="reconciliation"):
        swarm.launch_member(state, "test-swarm", member(pool), settings, handles=handles)
    assert popen.call_count == 1
