"""Plugin-local launch defaults; no live workers or model calls."""
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent_zoo.features.commands import CommandError
import swarm
from conftest import drain_first_request
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
    ctx = SimpleNamespace(state=runtime.state, submit_io=Mock(return_value="op_1"))
    controller = swarm.SwarmController(config)
    controller.command(ctx, raw_args="-n2" + (f" --profile {explicit}" if explicit else ""))
    settings_file.write_text('{"default_profile":"other"}')
    result = drain_first_request(controller, runtime.state, ctx.submit_io.call_args.args[0])
    assert result["level"] == "info", result["summary"]
    assert result["pool"]["model_profile"] == expected
    assert swarm.store_for(runtime.state, result["pool"]["id"]).pool()["model_profile"] == expected
    assert all(call.args[0].settings.model == expected for call in launches.prepare.call_args_list)
    assert config == original


@pytest.mark.parametrize("text", ["{", "[]", "null", '{"default_profile":false}',
    '{"default_profile":123}', '{"default_profile":""}', '{"default_profile":"  "}',
    '{"default_profile":[]}'])
def test_invalid_json_config_is_status_only(runtime, config, settings_file, text):
    settings_file.write_text(text)
    ctx = SimpleNamespace(state=runtime.state, submit_io=Mock(return_value="op_1"), emit=Mock())
    with pytest.raises(CommandError, match="swarm config|default_profile|Swarm config"):
        swarm.SwarmController(config).command(ctx, raw_args="-n2")
    ctx.submit_io.assert_not_called()
    ctx.emit.assert_not_called()
    runtime.state.session_launcher.prepare.assert_not_called()


@pytest.mark.parametrize("disabled", [False, True])
def test_unknown_or_disabled_default_rejected_before_defer(runtime, config, settings_file, disabled):
    settings_file.write_text(json.dumps({"default_profile": PROFILE if disabled else "missing"}))
    if disabled:
        config["llm"]["models"][PROFILE]["enabled"] = False
    ctx = SimpleNamespace(state=runtime.state, submit_io=Mock(return_value="op_1"))
    with pytest.raises(CommandError, match="enabled saved model profile|disabled"):
        swarm.SwarmController(config).command(ctx, raw_args="-n1")
    ctx.submit_io.assert_not_called()


@pytest.mark.parametrize("raw", [f"-n1 --profile {PROFILE}", "cancel", "interrupt sw0", "continue sw0",
    "release", "capture", "afk sw0 keep working", 'bcast sw0 "hello"', "recover sw0 --takeover --expected-epoch 1 --confirmed-stopped"])
def test_explicit_profile_and_lifecycle_skip_config(runtime, config, monkeypatch, raw):
    reader = Mock(side_effect=AssertionError("Unexpected default-config read"))
    monkeypatch.setattr(swarm, "load_default_profile", reader)
    ctx = SimpleNamespace(state=runtime.state, submit_io=Mock(return_value="op_1"))
    swarm.SwarmController(config).command(ctx, raw_args=raw)
    ctx.submit_io.assert_called_once()
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


def settings_registry(config):
    from agent_utils import PipelineBuilder, Session
    from agent_zoo.features.settings import Setting, SettingsContext, SettingsRegistry

    session = Session()
    features = []
    swarm.register_features(SimpleNamespace(add=features.append), session=session, config=config)
    feature = next(feature for feature in features if feature.name == "swarm_settings")
    session.pipeline = PipelineBuilder().add(feature).build()
    session.ensure_initialized()
    registry = SettingsRegistry([item for item in session.pipeline if isinstance(item, Setting)])
    return registry, SettingsContext(state=session.state, emit=lambda _ev: None, config=config)


def settings_request(registry, ctx, value=None):
    request = dict(type="plugin_settings", request_id="test", action="list")
    if value is not None:
        request.update(action="set", binding=registry.binding, key="swarm.default_profile", value=value)
    return registry.handle(request, ctx)


def test_setting_persists_without_mutating_session_or_existing_pool(runtime, config, settings_file):
    original = {"unrelated": {"prompt": "leave verbatim", "fraction": 0.25}}
    settings_file.write_text(json.dumps(original))
    settings_file.chmod(0o640)
    before_pool = runtime.store.pool()
    config["llm"]["default"] = PROFILE  # A transient /model change.
    before_config = deepcopy(config)
    registry, ctx = settings_registry(config)
    result = settings_request(registry, ctx)
    row, = result["settings"]
    assert result["ok"] and row["error"] is None
    assert (row["key"], row["kind"], row["apply"], row["value"]) == (
        "swarm.default_profile", "choice", "live", "offline")
    assert row["choices"] == ["offline", PROFILE, "other"]
    assert row["writable"] and "new swarms only" in row["help"]
    result = settings_request(registry, ctx, "other")
    assert result["ok"] and result["settings"][0]["value"] == "other"
    assert json.loads(settings_file.read_text()) == dict(original, default_profile="other")
    assert settings_file.stat().st_mode & 0o777 == 0o640
    assert config == before_config and runtime.store.pool() == before_pool
    # A new session/pipeline sees the persisted choice, not its own active model.
    config["llm"]["default"] = "offline"
    fresh_registry, fresh_ctx = settings_registry(config)
    assert fresh_ctx.state is not ctx.state
    assert settings_request(fresh_registry, fresh_ctx)["settings"][0]["value"] == "other"


@pytest.mark.parametrize("explicit_file", [False, True, "bundled-alias"])
def test_saved_global_fallback_ignores_session_overrides(runtime, config, launches, tmp_path, explicit_file):
    config["llm"]["default"] = PROFILE
    config["_runtime_launch"] = {"config_sets": [f"llm.default={PROFILE}"], "model": PROFILE}
    expected = "offline"
    if explicit_file is True:
        path = tmp_path / "explicit.yaml"
        path.write_text(json.dumps({"llm": dict(config["llm"], default="other")}))
        config["_runtime_launch"]["config_path"] = str(path)
        expected = "other"
    elif explicit_file:
        config["_runtime_launch"]["config_path"] = str(swarm.CONFIG_PATH)
    registry, ctx = settings_registry(config)
    assert settings_request(registry, ctx)["settings"][0]["value"] == expected
    launch_ctx = SimpleNamespace(state=runtime.state, submit_io=Mock(return_value="op_1"))
    ctrl = swarm.SwarmController(config)
    ctrl.command(launch_ctx, raw_args="-n1")
    result = drain_first_request(ctrl, runtime.state, launch_ctx.submit_io.call_args.args[0])
    assert result["pool"]["model_profile"] == expected
    assert launches.prepare.call_args.args[0].settings.model == expected
    assert config["llm"]["default"] == PROFILE


def test_getter_is_read_only_and_setter_does_not_copy_source_defaults(config, tmp_path, monkeypatch):
    source = tmp_path / "source"
    bundled = source / "config/swarm.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text(json.dumps({"default_profile": PROFILE, "prompt": "source only", "future": 7}))
    monkeypatch.setattr(swarm, "__file__", str(source / "src/swarm.py"))
    registry, ctx = settings_registry(config)
    original = bundled.read_bytes()
    with monkeypatch.context() as guard:
        guard.setattr(Path, "mkdir", Mock(side_effect=AssertionError("getter wrote")))
        guard.setattr(swarm.os, "replace", Mock(side_effect=AssertionError("getter wrote")))
        assert settings_request(registry, ctx)["settings"][0]["value"] == PROFILE
    assert not swarm.swarm_config_path().exists()
    assert settings_request(registry, ctx, "other")["ok"]
    assert json.loads(swarm.swarm_config_path().read_text()) == {"default_profile": "other"}
    assert bundled.read_bytes() == original


@pytest.mark.parametrize("text", ["{SECRET", "[]", '{"default_profile":false}', '{"default_profile":"missing"}'])
def test_setting_malformed_or_unknown_default_is_safe_and_never_overwritten(config, settings_file, text):
    registry, ctx = settings_registry(config)
    settings_file.write_text(text)
    result = settings_request(registry, ctx, "other")
    assert not result["ok"] and result["settings"][0]["error"]
    assert "SECRET" not in json.dumps(result)
    assert settings_file.read_text() == text


@pytest.mark.parametrize("value", [False, 1, "missing", "inherit current session"])
def test_setting_rejects_non_choices_without_writing(config, value):
    registry, ctx = settings_registry(config)
    assert not settings_request(registry, ctx, value)["ok"]
    assert not swarm.swarm_config_path().exists()


def test_disabled_profiles_and_cli_only_profiles_are_not_choices(config):
    path = Path(swarm.resolve_state_root()) / "config.yaml"
    saved = deepcopy(config)
    saved["llm"]["models"][PROFILE]["enabled"] = False
    saved["llm"]["disabled_tags"] = ["blocked"]
    saved["llm"]["models"]["other"]["tags"] = ["blocked"]
    path.write_text(json.dumps(saved))
    config["llm"]["models"]["transient-only"] = {"enabled": True}
    registry, ctx = settings_registry(config)
    assert settings_request(registry, ctx)["settings"][0]["choices"] == ["offline"]
    from agent_zoo.features.settings import SettingError
    with pytest.raises(SettingError, match="enabled saved"):
        swarm.save_default_profile(config, PROFILE)
    assert not swarm.swarm_config_path().exists()


def test_saved_profile_disabled_after_registration_rejected(config):
    registry, ctx = settings_registry(config)
    path = Path(swarm.resolve_state_root()) / "config.yaml"
    config["llm"]["models"]["other"]["enabled"] = False
    path.write_text(json.dumps(config))
    result = settings_request(registry, ctx, "other")
    assert not result["ok"] and "enabled saved" in result["error"]["message"]
    assert not swarm.swarm_config_path().exists()


def test_malformed_saved_global_config_is_safe(config):
    registry, ctx = settings_registry(config)
    path = Path(swarm.resolve_state_root()) / "config.yaml"
    path.write_text("llm: [SECRET")
    result = settings_request(registry, ctx, "other")
    assert not result["ok"] and "SECRET" not in json.dumps(result)
    assert not swarm.swarm_config_path().exists()


def test_failed_atomic_save_preserves_existing_file(config, settings_file, monkeypatch):
    settings_file.write_text('{"default_profile":"offline","unknown":1}')
    original = settings_file.read_bytes()
    registry, ctx = settings_registry(config)
    monkeypatch.setattr(swarm.os, "replace", Mock(side_effect=OSError("sensitive detail")))
    result = settings_request(registry, ctx, "other")
    assert not result["ok"] and "sensitive detail" not in json.dumps(result)
    assert result["settings"][0]["value"] == "offline"
    assert settings_file.read_bytes() == original
    assert not list(settings_file.parent.glob(".swarm-*"))
