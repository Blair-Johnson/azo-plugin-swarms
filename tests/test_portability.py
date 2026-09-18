"""Logical bindings survive relocation; local caches never become shared state."""
import shutil
from types import SimpleNamespace

import pytest

import swarm


def test_relocated_state_home_preserves_resource_without_old_absolute_paths(tmp_path, monkeypatch):
    old, new = tmp_path / "old-home", tmp_path / "new-home"
    monkeypatch.setenv("AGENT_ZOO_HOME", str(old))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local-a"))
    state = SimpleNamespace(_session_id="parent", swarm_access={"demo": ""},
                            _agent_zoo_context={"project_name": "research.v2",
                                               "project_dir": str(old / "obsolete")})
    first = swarm.store_for(state, "demo")
    metadata = first.create("demo", pods=2, agents_per_pod=2)
    first.post("pod-1", "general", "survive relocation", "parent", message_id="one")
    expected = first.messages("pod-1", "general")
    shutil.copytree(old, new)
    shutil.rmtree(old)
    shutil.rmtree(tmp_path / "local-a")
    monkeypatch.setenv("AGENT_ZOO_HOME", str(new))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local-b"))
    restored = swarm.allowed_store(state, "demo", "pod-1")
    assert restored.pool() == metadata
    assert restored.messages("pod-1", "general") == expected
    assert restored.attempts() == []
    assert restored.durable.is_relative_to(new)
    assert not old.exists()


def test_cache_under_shared_home_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path / "shared"))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "shared" / "cache"))
    with pytest.raises(ValueError, match="outside shared"):
        swarm.store_for(SimpleNamespace(), "demo")
    assert not (tmp_path / "shared").exists()
