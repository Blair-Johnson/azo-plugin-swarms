"""Named swarm model profiles: offline admission, launch routing and recovery."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from agent_zoo.runtime.commands import CommandError

import swarm
from test_recovery import runtime, reserve, seed


PROFILE = "gpt-5.6-luna-max"


@pytest.fixture
def config():
    return {"llm": {"default": "offline", "models": {
        "offline": {"enabled": True},
        PROFILE: {"enabled": True},
        "other": {"enabled": True},
    }}}


@pytest.fixture
def launches(runtime, monkeypatch):
    """Keep actual RuntimeLaunch construction and durable attempt records, no spawn."""
    launcher = runtime.state.session_launcher
    launcher.prepare.side_effect = lambda request, **kwargs: SimpleNamespace(
        request=request, ready_file="/offline/ready")
    launcher.start.return_value = SimpleNamespace(
        process=SimpleNamespace(pid=100), stdout_path="offline.out", stderr_path="offline.err")

    async def ready(state, store, member, handle):
        return dict(label=member["label"], session_id=member["session_id"], state="ready_paused")

    monkeypatch.setattr(swarm, "await_member_ready", ready)
    return launcher


@pytest.mark.parametrize("raw", [
    f"-n4 --profile {PROFILE}", f"--profile={PROFILE} -n4",
    f"-n4 --profile={PROFILE} -p2 --name explore",
])
def test_parse_profile_preserves_named_key_with_dots(raw):
    assert swarm.parse_command(raw)["profile"] == PROFILE


def test_omitted_profile_preserves_exact_parse_result():
    assert swarm.parse_command("-n4") == dict(
        action="create", agents=4, pods=1, channels=("general",), name=None)


@pytest.mark.parametrize("raw", [
    "-n1 --profile", "-n1 --profile=", "-n1 --profile -p2",
    "-n1 --profile --name test", "--profile --channels general -n1",
])
def test_missing_profile_value(raw):
    with pytest.raises(ValueError, match="Missing value"):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw", [
    "-n1 --profile first --profile second", "-n1 --profile=first --profile=second",
    "--profile=first -n1 --profile second",
])
def test_duplicate_profile(raw):
    with pytest.raises(ValueError, match="Duplicate option: profile"):
        swarm.parse_command(raw)


@pytest.mark.parametrize("failure", ["unknown", "disabled", "string_disabled", "tag", "tag_alias", "no_models"])
def test_invalid_profile_is_status_only_before_defer(runtime, config, monkeypatch, failure):
    if failure == "unknown":
        del config["llm"]["models"][PROFILE]
    elif failure == "no_models":
        config["llm"].pop("models")
    elif failure in {"disabled", "string_disabled"}:
        config["llm"]["models"][PROFILE]["enabled"] = False if failure == "disabled" else "false"
    else:
        config["llm"]["models"][PROFILE]["tags"] = ["costly"]
        config["llm"]["disabled_tags" if failure == "tag" else "disable_tags"] = ["costly"]
    before = deepcopy(config)
    forbidden = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(swarm, "allocate_pool", forbidden)
    ctx = SimpleNamespace(state=runtime.state, session=SimpleNamespace(inject=Mock()), defer=Mock())
    with pytest.raises(CommandError, match="Unknown swarm model profile|disabled"):
        swarm.SwarmController(config).command(ctx, raw_args=f"-n2 --profile {PROFILE}")
    ctx.defer.assert_not_called()
    ctx.session.inject.assert_not_called()
    forbidden.assert_not_called()
    runtime.state.session_launcher.prepare.assert_not_called()
    assert config == before


@pytest.mark.parametrize("explicit", [True, False])
def test_creation_routes_and_saves_profile_without_parent_mutation(runtime, config, launches, explicit):
    original_config, original_settings = deepcopy(config), runtime.state.settings
    controller = swarm.SwarmController(config)
    ctx = SimpleNamespace(state=runtime.state, session=SimpleNamespace(inject=Mock()), defer=Mock())
    controller.command(ctx, raw_args="-n2 -p2" + (f" --profile={PROFILE}" if explicit else ""))
    ctx.defer.assert_called_once()
    result = asyncio.run(ctx.defer.call_args.args[0]())
    assert result["level"] == "info", result["summary"]
    selected = PROFILE if explicit else "offline"
    assert result["pool"]["model_profile"] == selected
    store = swarm.store_for(runtime.state, result["pool"]["id"])
    assert store.pool()["model_profile"] == selected
    requests = [call.args[0] for call in launches.prepare.call_args_list]
    assert len(requests) == 2
    assert all(request.settings.model == selected for request in requests)
    assert all(request.startup_mode == "paused" for request in requests)
    assert all(request.settings.project == original_settings.project for request in requests)
    assert all(request.settings.workdir == original_settings.workdir for request in requests)
    assert all(row["request"]["settings"]["model"] == selected for row in store.records().list("attempts"))
    assert config == original_config
    assert runtime.state.settings is original_settings
    ctx.session.inject.assert_not_called()


def test_explicit_profile_does_not_require_parent_default(runtime, config):
    config["llm"].pop("default")
    before = deepcopy(config)
    assert swarm.launch_settings(runtime.state, config, profile=PROFILE).model == PROFILE
    assert config == before


def test_queued_default_selection_is_pinned(runtime, config, launches):
    controller = swarm.SwarmController(config)
    ctx = SimpleNamespace(state=runtime.state, defer=Mock())
    controller.command(ctx, raw_args="-n1")
    config["llm"]["default"] = "other"
    result = asyncio.run(ctx.defer.call_args.args[0]())
    assert result["pool"]["model_profile"] == "offline"
    assert launches.prepare.call_args.args[0].settings.model == "offline"
    assert config["llm"]["default"] == "other"


def test_create_revalidates_before_allocation(runtime, config, monkeypatch):
    controller = swarm.SwarmController(config)
    ctx = SimpleNamespace(state=runtime.state, defer=Mock())
    controller.command(ctx, raw_args=f"-n1 --profile {PROFILE}")
    config["llm"]["models"][PROFILE]["enabled"] = False
    forbidden = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(swarm, "allocate_pool", forbidden)
    result = asyncio.run(ctx.defer.call_args.args[0]())
    assert result["level"] == "warning" and "disabled" in result["summary"]
    forbidden.assert_not_called()


def save_profile(store, profile):
    with store.mutation() as (records, pool):
        records.put("swarm", "pool", dict(pool, model_profile=profile))


@pytest.mark.parametrize("profile", [PROFILE, "offline"])
def test_recovery_uses_saved_selection_over_destination_default(runtime, config, launches, monkeypatch, profile):
    save_profile(runtime.store, profile)
    for member in runtime.members:
        seed(runtime, member)
    config["llm"]["default"] = "other"
    original_settings = replace(runtime.state.settings, model="other", config_path="destination.toml",
                                config_sets=("destination=true",))
    runtime.state.settings = original_settings
    before_config = deepcopy(config)
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "dead")
    result = asyncio.run(swarm.SwarmController(config).recover(runtime.state, runtime.store, {}))
    assert result["level"] == "info", result["summary"]
    requests = [call.args[0] for call in launches.prepare.call_args_list]
    assert len(requests) == 2
    assert all(request.settings == replace(original_settings, model=profile) for request in requests)
    assert all(isinstance(request.source, swarm.Resume) for request in requests)
    assert runtime.state.settings is original_settings
    assert runtime.store.pool()["model_profile"] == profile
    assert config == before_config


def test_recover_owned_keeps_each_pool_selection_and_legacy_default(runtime, config, launches, monkeypatch):
    save_profile(runtime.store, PROFILE)
    stores = [runtime.store]
    for flags in ("--profile other", ""):
        store, _ = swarm.allocate_pool(runtime.state, swarm.parse_command(f"-n1 {flags}"))
        stores.append(store)
    for store in stores:
        for pod in store.pool()["pods"]:
            for member in pod["members"]:
                seed(runtime, member)
    original_settings = runtime.state.settings
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "dead")
    result = asyncio.run(swarm.SwarmController(config).recover_owned(
        runtime.state, stores=[(store.pool()["id"], store, None) for store in stores]))
    assert "blocked" not in result["summary"], result["summary"]
    selected = {call.args[0].kind: call.args[0].settings.model for call in launches.prepare.call_args_list}
    assert selected == {"sw0p0a0": PROFILE, "sw0p1a0": PROFILE, "sw1p0a0": "other", "sw2p0a0": "offline"}
    assert runtime.state.settings is original_settings
    assert "model_profile" not in stores[-1].pool()


@pytest.mark.parametrize("failure", ["missing", "disabled", "tag", "empty_config"])
def test_missing_or_disabled_destination_fails_before_ownership_claim(runtime, config, monkeypatch, failure):
    save_profile(runtime.store, PROFILE)
    if failure == "missing":
        del config["llm"]["models"][PROFILE]
    elif failure == "disabled":
        config["llm"]["models"][PROFILE]["enabled"] = False
    elif failure == "tag":
        config["llm"]["models"][PROFILE]["tags"] = ["disabled"]
        config["llm"]["disabled_tags"] = ["disabled"]
    else:
        config.clear()
    before, grants = runtime.store.pool(), deepcopy(runtime.state.grants)
    claim = Mock(side_effect=AssertionError("must not claim ownership"))
    monkeypatch.setattr(swarm, "claim_recovery", claim)
    with pytest.raises(ValueError, match="Unknown swarm model profile|disabled"):
        asyncio.run(swarm.SwarmController(config).recover(runtime.state, runtime.store,
                    dict(expected_epoch=1, confirmed_stopped=True)))
    claim.assert_not_called()
    assert runtime.store.pool() == before
    assert runtime.state.grants == grants
    runtime.state.session_launcher.prepare.assert_not_called()


@pytest.mark.parametrize("bad", [None, "", "  ", 42, []])
def test_malformed_saved_profile_cannot_fall_back(runtime, bad):
    with pytest.raises(ValueError, match="Invalid swarm model profile"):
        swarm.validate_pool(dict(runtime.pool, model_profile=bad))


def test_missing_destination_profile_does_not_block_live_successor_interrupt(runtime, monkeypatch):
    save_profile(runtime.store, PROFILE)
    reserve(runtime, runtime.members[0], identity=runtime.identity.to_dict())
    before = runtime.store.pool()
    runtime.state._instance_id = "successor"
    runtime.state.grants = {}
    monkeypatch.setattr(swarm, "owner_replaced", lambda state, pool: True)
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "alive")
    controller = swarm.SwarmController({})
    controller.control = AsyncMock(return_value=dict(summary="pause requested", level="info"))
    result = asyncio.run(controller.recover(runtime.state, runtime.store, {}))
    assert "pause requested" in result["summary"]
    assert controller.control.call_args.args[1]["action"] == "interrupt"
    after = runtime.store.pool()
    assert after["owner"]["epoch"] == before["owner"]["epoch"]
    assert after["owner"]["token"] == before["owner"]["token"]
    assert after["model_profile"] == PROFILE
    runtime.state.session_launcher.prepare.assert_not_called()
