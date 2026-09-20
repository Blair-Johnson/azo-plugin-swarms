"""Plugin-local launch defaults; no live workers or model calls."""
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent_zoo.runtime.commands import CommandError
import swarm
from test_profiles import PROFILE, config, launches
from test_recovery import runtime


@pytest.fixture
def settings_file():
    path = Path(swarm.resolve_state_root()) / "plugin-configs/azo-plugin-swarms/config/swarm.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("settings, explicit, expected", [
    ({"default_profile": PROFILE}, None, PROFILE),
    ({"default_profile": PROFILE}, "other", "other"),
    ({"default_profile": None}, None, "offline"),
    ({}, None, "offline"),
])
def test_default_precedence_and_persistence(runtime, config, launches, settings_file, settings, explicit, expected):
    settings_file.write_text(json.dumps(settings))
    original = deepcopy(config)
    ctx = SimpleNamespace(state=runtime.state, defer=Mock())
    controller = swarm.SwarmController(config)
    controller.command(ctx, raw_args="-n2" + (f" --profile {explicit}" if explicit else ""))
    settings_file.write_text('{"default_profile":"other"}')
    result = asyncio.run(ctx.defer.call_args.args[0]())
    assert result["level"] == "info", result["summary"]
    assert result["pool"]["model_profile"] == expected
    assert swarm.store_for(runtime.state, result["pool"]["id"]).pool()["model_profile"] == expected
    assert all(call.args[0].settings.model == expected for call in launches.prepare.call_args_list)
    assert config == original


@pytest.mark.parametrize("text", ["{", "[]", "null", '{"default_profile":false}',
    '{"default_profile":123}', '{"default_profile":""}', '{"default_profile":"  "}',
    '{"default_model":"typo"}'])
def test_invalid_json_config_is_status_only(runtime, config, settings_file, text):
    settings_file.write_text(text)
    ctx = SimpleNamespace(state=runtime.state, defer=Mock(), session=SimpleNamespace(inject=Mock()))
    with pytest.raises(CommandError, match="swarm config|default_profile|Swarm config"):
        swarm.SwarmController(config).command(ctx, raw_args="-n2")
    ctx.defer.assert_not_called()
    ctx.session.inject.assert_not_called()
    runtime.state.session_launcher.prepare.assert_not_called()


@pytest.mark.parametrize("disabled", [False, True])
def test_unknown_or_disabled_default_rejected_before_defer(runtime, config, settings_file, disabled):
    settings_file.write_text(json.dumps({"default_profile": PROFILE if disabled else "missing"}))
    if disabled:
        config["llm"]["models"][PROFILE]["enabled"] = False
    ctx = SimpleNamespace(state=runtime.state, defer=Mock())
    with pytest.raises(CommandError, match="Unknown swarm model profile|disabled"):
        swarm.SwarmController(config).command(ctx, raw_args="-n1")
    ctx.defer.assert_not_called()


@pytest.mark.parametrize("raw", [f"-n1 --profile {PROFILE}", "cancel", "interrupt sw0", "continue sw0",
    "release", "capture", "afk sw0 keep working", 'bcast sw0 "hello"', "recover sw0 --takeover --expected-epoch 1 --confirmed-stopped"])
def test_explicit_profile_and_lifecycle_skip_config(runtime, config, monkeypatch, raw):
    reader = Mock(side_effect=AssertionError("Unexpected default-config read"))
    monkeypatch.setattr(swarm, "load_default_profile", reader)
    ctx = SimpleNamespace(state=runtime.state, defer=Mock())
    swarm.SwarmController(config).command(ctx, raw_args=raw)
    ctx.defer.assert_called_once()
    reader.assert_not_called()


def test_config_is_declared_and_bundled_default_is_neutral():
    import tomllib
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / "pixi.toml").read_text())
    assert "config/swarm.json" in manifest["tool"]["agent-zoo"]["plugin"]["config"]["files"]
    assert json.loads((root / "config/swarm.json").read_text()) == {"default_profile": None}


def test_installed_config_overrides_bundle_and_rereads(settings_file, tmp_path, monkeypatch):
    source = tmp_path / "source"
    bundled = source / "config/swarm.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text(json.dumps({"default_profile": PROFILE}))
    monkeypatch.setattr(swarm, "__file__", str(source / "src/swarm.py"))
    assert swarm.load_default_profile() == PROFILE
    settings_file.write_text('{"default_profile":"other"}')
    assert swarm.load_default_profile() == "other"
    settings_file.write_text('{"default_profile":null}')
    assert swarm.load_default_profile() is None
    settings_file.unlink()
    bundled.unlink()
    assert swarm.load_default_profile() is None  # Loose single-file installations.


@pytest.mark.parametrize("failure", ["encoding", "directory"])
def test_unreadable_installed_config_does_not_silently_fall_back(settings_file, failure):
    if failure == "encoding":
        settings_file.write_bytes(b"\xff")
    else:
        settings_file.mkdir()
    with pytest.raises(ValueError, match="Cannot read swarm config"):
        swarm.load_default_profile()
