"""Use this checkout's plugin with the selected integrated host interpreter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import pytest


@pytest.fixture(autouse=True)
def isolated_storage_environment(tmp_path, monkeypatch):
    # HOME overrides STATE_ROOT; never let a live harness redirect test fixtures.
    monkeypatch.delenv("AGENT_ZOO_HOME", raising=False)
    monkeypatch.setenv("AGENT_ZOO_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
