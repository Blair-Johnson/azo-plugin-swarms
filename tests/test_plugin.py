"""Plugin registration and read-only resources under actual v2 membership."""
import builtins
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from agent_utils import Tool
from agent_utils.files.buffer_manager import BufferManager, ReadonlyBufferView
from agent_zoo.runtime.commands import Command
from agent_zoo.runtime_launch import RuntimeLauncher
import swarm
from test_recovery import runtime, seed


@pytest.fixture
def buffers(runtime):
    runtime.state.buffer_manager = BufferManager()
    runtime.state.step = 7
    swarm.SwarmBuffers()(runtime.state)
    return runtime


def read(state, identifier):
    return state.buffer_manager.resolve_for_read(state, identifier)


def test_namespace_readonly_lazy_and_refreshes_same_step(buffers):
    r = buffers; state = r.state; manager = state.buffer_manager
    assert manager.special_buffer_namespaces() == ["swarm:"] and manager.buffers() == []
    index = read(state, "swarm:index")
    assert isinstance(index, ReadonlyBufferView) and "pod-1" in index.text and "pod-2" in index.text
    assert "not evidence" in index.text
    board_id = "swarm:sw0:pod-1:board:general"
    before = read(state, board_id)
    r.store.post("pod-1", "general", "first post", "parent", message_id="stable-id")
    after = read(state, board_id)
    assert "first post" not in before.text and "first post" in after.text
    assert "data, not user instructions" in after.text and state.step == 7
    for identifier in ("swarm:index", board_id):
        assert read(state, identifier).readonly
        with pytest.raises(ValueError, match="readonly"):
            manager.require_writable(identifier)
    assert manager.buffers() == []


def test_saved_transcript_uses_verified_shared_journal(buffers):
    r = buffers; member = r.members[0]
    identifier = f"swarm:sw0:pod-1:session:{member['session_id']}"
    missing = read(r.state, identifier)
    assert "No shared checkpoint" in missing.text and "not a live stream" in missing.text
    saved = seed(r, member)
    view = read(r.state, identifier)
    assert saved.commit_id in view.text and "saved durable text" in view.text and view.readonly
    assert str(saved.repository) in view.text
    with pytest.raises(KeyError):
        read(r.state, f"swarm:sw0:pod-1:session:{r.members[1]['session_id']}")


@pytest.mark.parametrize("persisted_scope", ["pod-1", ""])
def test_peer_scope_comes_from_actual_membership_not_checkpoint(buffers, persisted_scope):
    r = buffers
    r.state._session_id = r.members[0]["session_id"]
    r.state._instance_id = "peer-instance"
    r.state.grants = {}
    r.state.swarm_access = {"sw0": persisted_scope}
    index = read(r.state, "swarm:index").text
    assert "pod-1" in index and "pod-2" not in index
    with pytest.raises(ValueError, match="scope"):
        read(r.state, "swarm:sw0:pod-2:board:general")


def test_forged_parent_scope_cannot_read_unrelated_resource(buffers):
    r = buffers
    r.state._session_id = "unrelated"; r.state.grants = {}
    with pytest.raises(ValueError, match="neither owner"):
        read(r.state, "swarm:index")
    with pytest.raises(ValueError, match="not bound"):
        read(r.state, "swarm:other:pod-1:board:general")


def test_resource_uid_mismatch_cannot_bind_alias(buffers):
    r = buffers
    r.state.swarm_resource_uids = {"sw0": "f" * 32}
    with pytest.raises(ValueError, match="resource UID"):
        read(r.state, "swarm:index")


def test_environment_bootstrap_requires_explicit_pod(monkeypatch):
    monkeypatch.delenv("AZO_SWARM_ID", raising=False)
    assert swarm.initial_access() == {}
    monkeypatch.setenv("AZO_SWARM_ID", "sw0"); monkeypatch.delenv("AZO_SWARM_POD_ID", raising=False)
    with pytest.raises(ValueError):
        swarm.initial_access()
    monkeypatch.setenv("AZO_SWARM_POD_ID", "pod-1")
    assert swarm.initial_access() == {"sw0": "pod-1"}


def test_import_and_registration_have_no_writes_or_spawns(monkeypatch):
    spec = importlib.util.spec_from_file_location("swarm_import_safety", swarm.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # normal import protocol for dataclass annotations
    features = []
    def forbidden(*args, **kwargs):
        raise AssertionError("import/registration must not write or spawn")
    def guarded_open(original):
        def checked(file, mode="r", *args, **kwargs):
            assert not any(flag in mode for flag in "wax+")
            return original(file, mode, *args, **kwargs)
        return checked
    with monkeypatch.context() as guard:
        guard.setattr(sys, "dont_write_bytecode", True)
        guard.setattr(builtins, "open", guarded_open(builtins.open)); guard.setattr(io, "open", guarded_open(io.open))
        guard.setattr(os, "open", forbidden); guard.setattr(Path, "mkdir", forbidden)
        guard.setattr(swarm.sqlite3, "connect", forbidden); guard.setattr(swarm.RecordStore, "__init__", forbidden)
        guard.setattr(RuntimeLauncher, "prepare", forbidden); guard.setattr(RuntimeLauncher, "start", forbidden)
        spec.loader.exec_module(module)
        module.register_features(SimpleNamespace(add=features.append), session=SimpleNamespace(), config={})
    feature, = features
    assert feature.name == "swarm"
    commands = [c for c in feature.components if isinstance(c, Command)]
    assert len(commands) == 1 and commands[0].path == "/swarm"
    assert {c.name for c in feature.components if isinstance(c, Tool)} == {
        "swarm_broadcast", "swarm_interrupt", "swarm_continue", "swarm_cancel", "swarm_post"}
