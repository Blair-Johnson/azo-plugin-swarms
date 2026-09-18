"""Offline /swarm controls: isolated stores, fake launchers, and fake sockets.

These tests never start models or connect to a live runtime.  Use the integrated
host interpreter, as for tests/test_plugin.py.
"""
import asyncio
from copy import deepcopy
import json
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_zoo.runtime.commands import Command, CommandError, CommandRegistry, CommandResult
from agent_zoo.runtime_launch import RuntimeSettings
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
    monkeypatch.chdir(tmp_path)
    (tmp_path / "state" / "projects" / "test-project").mkdir(parents=True)
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=AssertionError("real spawn forbidden")))
    monkeypatch.setattr(swarm.AzoWs, "attach", Mock(side_effect=AssertionError("live socket forbidden")))


@pytest.fixture
def settings(tmp_path):
    return RuntimeSettings(
        project="test-project", model="offline-test-model", workdir=str(tmp_path),
        state_root=str(tmp_path / "state"), store_root=str(tmp_path / "store"),
        local_state_root=str(tmp_path / "local"),
    )


@pytest.fixture
def state(tmp_path, settings):
    return SimpleNamespace(
        _session_id="parent-session", _agent_zoo_context={
            "project_name": "test-project", "project_store_root": str(tmp_path / "store"),
        },
        swarm_access={}, session_cwd=str(tmp_path), session_launcher=Mock(),
        settings=settings,
    )


@pytest.fixture
def controller():
    return swarm.SwarmController({"llm": {"default": "offline-test-model"}})


def context(state):
    return SimpleNamespace(state=state, session=SimpleNamespace(state=state, inject=Mock()), defer=Mock())


def allocate(state, raw="-n4 -p2"):
    return swarm.allocate_pool(state, swarm.parse_command(raw))


def members(pool):
    return [member for pod in pool["pods"] for member in pod["members"]]


def test_registration_exposes_one_raw_command_without_work(monkeypatch):
    features = []
    forbidden = Mock(side_effect=AssertionError("registration must not do work"))
    for attribute in ("allocate_pool", "resolve_pool", "launch_member", "store_for"):
        monkeypatch.setattr(swarm, attribute, forbidden)
    swarm.register_features(SimpleNamespace(add=features.append), session=SimpleNamespace(), config={})
    commands = [component for feature in features for component in feature.components
                if isinstance(component, Command)]
    assert len(commands) == 1
    command = commands[0]
    assert command.path == "/swarm" and command.raw and not command.args
    assert not command.cancels_lifecycle
    raw = '  bcast sw0 -p 0 "  first\n\t雪 last  "  '
    invocation = CommandRegistry(commands).parse("/swarm" + raw)
    assert invocation.arguments == {"raw_args": raw}
    forbidden.assert_not_called()


@pytest.mark.parametrize("raw", ["", "-n0", "-n3 -p2", 'bcast "unterminated', "cancel"])
def test_invalid_command_never_defers_or_touches_storage(state, controller, monkeypatch, raw):
    ctx = context(state)
    forbidden = Mock(side_effect=AssertionError("invalid ingress must have no side effects"))
    for attribute in ("allocate_pool", "resolve_pool", "store_for", "launch_member"):
        monkeypatch.setattr(swarm, attribute, forbidden)
    before = deepcopy(state.swarm_access)
    with pytest.raises(CommandError):
        controller.command(ctx, raw_args=raw)
    ctx.defer.assert_not_called()
    forbidden.assert_not_called()
    assert state.swarm_access == before


def test_valid_control_is_deferred_with_detached_state(state, controller, monkeypatch):
    ctx = context(state)
    execute = AsyncMock(return_value={"summary": "done"})
    monkeypatch.setattr(controller, "execute", execute)
    result = controller.command(ctx, raw_args='bcast sw0 "  first\nlast  "')
    assert isinstance(result, CommandResult)
    execute.assert_not_called()
    ctx.defer.assert_called_once()
    work = ctx.defer.call_args.args[0]
    assert ctx.defer.call_args.kwargs["then"] == controller.complete
    state._agent_zoo_context["project_name"] = "changed-after-ingress"
    state.swarm_access["other"] = "pod-1"
    asyncio.run(work())
    snapshot, request = execute.call_args.args
    assert snapshot is not state
    assert snapshot._session_id == "parent-session"
    assert snapshot._agent_zoo_context == {
        "project_name": "test-project", "project_store_root": state.settings.store_root,
    }
    assert snapshot.swarm_access == {}
    assert snapshot.session_launcher is state.session_launcher
    assert request == {"action": "bcast", "target": "sw0", "pods": None,
                       "message": "  first\nlast  "}


def test_allocation_has_stable_ids_unique_aliases_and_balanced_topology(state):
    store0, pool0 = allocate(state, "-n4 -p2 --name explore --channels comms,breakthroughs")
    store1, pool1 = allocate(state, "-n1")
    assert pool0["id"] == "sw0" and pool0["name"] == "explore"
    assert pool1["id"] == "sw1" and pool1["name"] == "sw1"
    assert pool0["owner_session_id"] == state._session_id
    assert pool0["host"]["host_id"]
    assert pool0["channels"] == ["comms", "breakthroughs"]
    assert [len(pod["members"]) for pod in pool0["pods"]] == [2, 2]
    assert [member["label"] for member in members(pool0)] == [
        "sw0p0a0", "sw0p0a1", "sw0p1a0", "sw0p1a1",
    ]
    assert len({member["session_id"] for member in members(pool0) + members(pool1)}) == 5
    assert store0.pool() == pool0 and store1.pool() == pool1
    assert swarm.resolve_pool(state, "explore")[1]["id"] == "sw0"
    assert swarm.resolve_pool(state, "sw1")[1]["id"] == "sw1"
    with pytest.raises(ValueError):
        swarm.resolve_pool(state, None)
    with pytest.raises(ValueError):
        allocate(state, "-n1 --name explore")
    assert store0.pool() == pool0


def test_default_pool_recovered_from_registry_after_restart_without_spawn(state, monkeypatch):
    store, pool = allocate(state)
    restarted = SimpleNamespace(**vars(state))
    restarted.swarm_access = {}
    restarted.session_launcher = Mock(side_effect=AssertionError("restart must not spawn"))
    forbidden = Mock(side_effect=AssertionError("lookup must not launch"))
    monkeypatch.setattr(swarm, "launch_member", forbidden)
    recovered_store, recovered = swarm.resolve_pool(restarted, None)
    assert recovered == pool
    assert recovered_store.durable == store.durable
    assert swarm.resolve_pool(restarted, pool["id"])[1] == pool
    forbidden.assert_not_called()
    assert store.attempts() == []


@pytest.mark.parametrize("actor", ["peer", "different-owner"])
def test_peer_and_cross_owner_cannot_resolve_control_target(state, controller, monkeypatch, actor):
    store, pool = allocate(state)
    actor_state = SimpleNamespace(**vars(state))
    actor_state._session_id = members(pool)[0]["session_id"] if actor == "peer" else "other-owner"
    # A forged checkpointed parent binding must not bypass durable ownership.
    actor_state.swarm_access = {pool["id"]: ""}
    for target in (pool["id"], pool["name"], None):
        with pytest.raises(ValueError):
            swarm.resolve_pool(actor_state, target)
    assert store.attempts() == []
    send = AsyncMock()
    monkeypatch.setattr(swarm, "send_to_member", send)
    with pytest.raises(CommandError):
        asyncio.run(controller.execute(actor_state, swarm.parse_command("interrupt sw0")))
    send.assert_not_awaited()
    assert store.records().list("operations") == []


def test_cancelled_pool_not_selected_by_default(state):
    store, pool = allocate(state)
    with store.records(write=True).transaction("swarm", "pool") as saved:
        saved["desired_state"] = "cancelled"
    with pytest.raises(ValueError):
        swarm.resolve_pool(state, None)
    _, replacement = allocate(state, "-n1")
    assert replacement["id"] != pool["id"]
    assert swarm.resolve_pool(state, None)[1]["id"] == replacement["id"]


def test_blocking_waits_for_thread_cleanup_when_cancelled():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        cleaned = threading.Event()

        def blocking_work():
            started.set()
            try:
                assert release.wait(5), "test failed to release worker thread"
            finally:
                cleaned.set()

        task = asyncio.create_task(swarm.blocking(blocking_work))
        try:
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            assert started.is_set()
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done(), "cancellation escaped while thread still owned side effects"
            assert not cleaned.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        assert cleaned.is_set()

    asyncio.run(scenario())


@pytest.fixture
def identities(monkeypatch):
    from tmux_pilot.process_identity import HostIdentity, ProcessIdentity

    host = HostIdentity("offline-host", "offline-boot", "offline-namespace", "offline")
    identity = ProcessIdentity(host.host_id, host.boot_id, host.pid_namespace, 4242, "start-1")
    monkeypatch.setattr(swarm, "local_host_identity", lambda: host)
    monkeypatch.setattr(swarm, "capture_process_identity", Mock(return_value=identity))
    return host, identity


@pytest.fixture
def target(identities):
    host, identity = identities
    return {
        "session_id": "member-session", "instance_id": "member-instance", "label": "sw0p0a0",
        "host": host.to_dict(), "process_identity": identity.to_dict(),
        "ready": {"format_version": 1, "session_id": "member-session",
                  "instance_id": "member-instance", "pid": identity.pid,
                  "websocket_url": "ws://127.0.0.1:12345/session",
                  "process_identity": identity.to_dict()},
    }


def fake_client(monkeypatch, target, **hello_changes):
    hello = {"type": "hello", "protocol_version": 1,
             "session_id": target["session_id"], "instance_id": target["instance_id"]}
    hello.update(hello_changes)
    client = SimpleNamespace(
        connect=AsyncMock(), close=AsyncMock(return_value=True), protocol_errors=(),
        send_user=AsyncMock(), slash=AsyncMock(), control=AsyncMock(),
        observation=Mock(return_value=SimpleNamespace(hello=hello)),
    )
    monkeypatch.setattr(swarm.AzoWs, "attach", Mock(return_value=client))
    return client


@pytest.mark.parametrize("action", ["bcast", "interrupt", "continue", "cancel"])
def test_wire_helper_sends_exact_protocol_action_and_closes(target, monkeypatch, action):
    client = fake_client(monkeypatch, target)
    payload = "  first\n\t雪 \\n literal\r\nlast  "
    request = {"action": action, "target": "sw0"}
    if action == "bcast":
        request.update(message=payload, pods=None)
    asyncio.run(swarm.send_to_member(target, request))
    assert swarm.AzoWs.attach.call_args.args[0] == target["ready"]["websocket_url"]
    client.connect.assert_awaited_once()
    if action == "bcast":
        assert client.send_user.await_args.args == (payload,)
        client.send_user.assert_awaited_once()
        client.slash.assert_not_awaited()
        client.control.assert_not_awaited()
    elif action in {"interrupt", "continue"}:
        assert client.slash.await_args.args == ("/" + action,)
        client.slash.assert_awaited_once()
        client.send_user.assert_not_awaited()
        client.control.assert_not_awaited()
    else:
        assert client.control.await_args.args == (
            "shutdown", {"save": True, "reason": "swarm-cancel"},
        )
        client.control.assert_awaited_once()
        client.send_user.assert_not_awaited()
        client.slash.assert_not_awaited()
    client.close.assert_awaited_once()


@pytest.mark.parametrize("field", ["session_id", "instance_id"])
def test_wire_helper_rejects_exact_hello_identity_mismatch_and_closes(target, monkeypatch, field):
    client = fake_client(monkeypatch, target, **{field: "different-identity"})
    with pytest.raises(ValueError):
        asyncio.run(swarm.send_to_member(target, {"action": "interrupt", "target": "sw0"}))
    client.send_user.assert_not_awaited()
    client.slash.assert_not_awaited()
    client.control.assert_not_awaited()
    client.close.assert_awaited_once()


def test_wire_helper_closes_after_send_failure_without_retry(target, monkeypatch):
    client = fake_client(monkeypatch, target)
    client.send_user.side_effect = OSError("connection lost after write")
    with pytest.raises(OSError, match="connection lost"):
        asyncio.run(swarm.send_to_member(target, {"action": "bcast", "message": "once"}))
    client.send_user.assert_awaited_once()
    client.close.assert_awaited_once()
    swarm.AzoWs.attach.assert_called_once()


def test_missing_endpoint_cannot_send(state, controller, monkeypatch):
    store, pool = allocate(state, "-n1")
    send = AsyncMock()
    monkeypatch.setattr(swarm, "send_to_member", send)
    result = asyncio.run(controller.execute(state, swarm.parse_command('bcast sw0 "no endpoint"')))
    send.assert_not_awaited()
    assert result["level"] != "info"
    assert store.records().list("operations")[0]["outcomes"][0]["state"] == "unavailable"
    assert store.attempts() == []
    with pytest.raises(ValueError):
        swarm.connection_target(store, members(pool)[0])


def test_all_pod_selectors_validated_before_any_send(state, controller, monkeypatch):
    store, pool = allocate(state, "-n2 -p2")
    send = AsyncMock()
    lookup = Mock(side_effect=lambda store, member: dict(member))
    monkeypatch.setattr(swarm, "send_to_member", send)
    monkeypatch.setattr(swarm, "connection_target", lookup)
    with pytest.raises(CommandError, match="Pod selection"):
        asyncio.run(controller.execute(state, swarm.parse_command('bcast sw0 -p0,2 "never"')))
    send.assert_not_awaited()
    lookup.assert_not_called()
    assert store.records().list("operations") == []
    assert store.attempts() == []


def test_broadcast_only_selected_pod_preserves_exact_multiline_payload(state, controller, monkeypatch):
    store, pool = allocate(state)
    payload = "  first\n\t雪 --name ignored\r\nlast  "
    sent = []

    async def send(target, request):
        sent.append((target["session_id"], request["message"]))

    monkeypatch.setattr(swarm, "connection_target", lambda store, member: dict(member))
    monkeypatch.setattr(swarm, "send_to_member", send)
    result = asyncio.run(controller.execute(state, swarm.parse_command(f'bcast sw0 -p1 "{payload}"')))
    assert sorted(sent) == sorted((member["session_id"], payload)
                                  for member in pool["pods"][1]["members"])
    assert result["level"] == "info"
    assert store.messages("pod-1", "general") == []
    assert store.messages("pod-2", "general") == []


def test_partial_delivery_is_durable_and_failed_targets_are_not_retried(state, controller, monkeypatch):
    store, pool = allocate(state, "-n3")
    selected = members(pool)
    calls = []

    async def send(target, request):
        # The operation must exist before a remote write becomes possible.
        assert len(store.records().list("operations")) == 1
        calls.append(target["session_id"])
        if target["session_id"] == selected[1]["session_id"]:
            raise OSError("uncertain remote delivery")

    monkeypatch.setattr(swarm, "connection_target", lambda store, member: dict(member))
    monkeypatch.setattr(swarm, "send_to_member", send)
    result = asyncio.run(controller.execute(state, swarm.parse_command('bcast sw0 "once per target"')))
    assert sorted(calls) == sorted(member["session_id"] for member in selected)
    assert result["level"] != "info"
    durable = store.records().list("operations")
    assert len(durable) == 1
    assert durable[0]["state"] == "finished"
    assert [row["state"] for row in durable[0]["outcomes"]] == ["sent", "unknown", "sent"]
    encoded = json.dumps(durable)
    assert "uncertain remote delivery" in encoded
    assert all(member["session_id"] in encoded for member in selected)
    reopened = swarm.store_for(state, pool["id"])
    assert reopened.records().list("operations") == durable
    # A new controller and durable lookup must not replay even uncertain outcomes.
    swarm.SwarmController({})
    swarm.resolve_pool(state, "sw0")
    assert sorted(calls) == sorted(member["session_id"] for member in selected)


def test_controller_serializes_inflight_commands(state, controller, monkeypatch):
    allocate(state, "-n1")
    monkeypatch.setattr(swarm, "connection_target", lambda store, member: dict(member))

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def send(target, request):
            calls.append(request["action"])
            if request["action"] == "interrupt":
                entered.set()
                await release.wait()

        monkeypatch.setattr(swarm, "send_to_member", send)
        first = asyncio.create_task(controller.execute(state, swarm.parse_command("interrupt sw0")))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            second = asyncio.create_task(controller.execute(state, swarm.parse_command("continue sw0")))
            for _ in range(10):
                await asyncio.sleep(0)
            assert calls == ["interrupt"]
            assert not second.done()
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(first, *([second] if second else [])), 2)
        assert calls == ["interrupt", "continue"]

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", ["model", "project", "store", "launcher"])
def test_invalid_creation_settings_have_no_deferred_or_durable_side_effects(
    state, controller, monkeypatch, invalid,
):
    if invalid == "model":
        controller.config = {"llm": {"default": ""}}
    elif invalid == "project":
        state._agent_zoo_context.pop("project_name")
    elif invalid == "store":
        state._agent_zoo_context.pop("project_store_root")
    else:
        state.session_launcher = None
    ctx = context(state)
    allocate_mock = Mock(side_effect=AssertionError("invalid settings must not allocate"))
    monkeypatch.setattr(swarm, "allocate_pool", allocate_mock)
    with pytest.raises(CommandError):
        controller.command(ctx, raw_args="-n1")
    ctx.defer.assert_not_called()
    allocate_mock.assert_not_called()
    assert state.swarm_access == {}


def test_broadcast_encoded_frame_limit_checked_before_deferral(state, controller, monkeypatch):
    ctx = context(state)
    forbidden = Mock(side_effect=AssertionError("oversized ingress must not touch storage"))
    monkeypatch.setattr(swarm, "resolve_pool", forbidden)
    # JSON escaping, not just character count, determines the wire size.
    payload = "\x00" * (swarm.MAX_BROADCAST_FRAME_BYTES // 6 + 1)
    assert len(payload.encode()) < swarm.MAX_BROADCAST_FRAME_BYTES
    with pytest.raises(CommandError, match="limit|shorter"):
        controller.command(ctx, raw_args=f'bcast sw0 "{payload}"')
    ctx.defer.assert_not_called()
    forbidden.assert_not_called()


@pytest.fixture
def ready_member(state, identities):
    host, identity = identities
    store, pool = allocate(state, "-n1")
    member = members(pool)[0]
    ready = {
        "format_version": 1, "session_id": member["session_id"],
        "instance_id": "member-instance", "pid": identity.pid,
        "websocket_url": "ws://127.0.0.1:12345/session",
        "process_identity": identity.to_dict(),
    }
    store.reserve_attempt(member["session_id"], {
        "session_id": member["session_id"], "instance_id": ready["instance_id"],
        "state": "ready", "host": host.to_dict(), "pid": identity.pid,
        "process_identity": identity.to_dict(), "ready": ready,
        "ready_file": "/deliberately-missing/ready.json",
    })
    return store, member, ready


def test_verified_durable_endpoint_survives_loss_of_temporary_ready_file(ready_member):
    store, member, ready = ready_member
    target = swarm.connection_target(store, member)
    assert target == {
        "session_id": member["session_id"], "instance_id": ready["instance_id"],
        "label": member["label"], "ready": ready, "process_identity": ready["process_identity"],
    }
    target["ready"]["session_id"] = "changed-local-copy"
    assert store.records().get("attempts", member["session_id"])["ready"] == ready


@pytest.mark.parametrize("damage", [
    "foreign-host", "missing-identity", "attempt-pid", "ready-session", "ready-instance",
    "ready-pid", "ready-identity", "missing-ready", "format", "remote-url", "wrong-path",
])
def test_endpoint_identity_and_location_checks_fail_closed(ready_member, damage):
    store, member, ready = ready_member
    with store.records(write=True).transaction("attempts", member["session_id"]) as attempt:
        if damage == "foreign-host":
            attempt["host"]["host_id"] = "other-host"
        elif damage == "missing-identity":
            attempt.pop("process_identity")
        elif damage == "attempt-pid":
            attempt["pid"] += 1
        elif damage == "missing-ready":
            attempt.pop("ready")
        elif damage == "ready-identity":
            attempt["ready"]["process_identity"]["start_token"] = "reused-pid"
        elif damage == "format":
            attempt["ready"]["format_version"] = True
        elif damage == "remote-url":
            attempt["ready"]["websocket_url"] = "ws://example.invalid:12345/session"
        elif damage == "wrong-path":
            attempt["ready"]["websocket_url"] = "ws://127.0.0.1:12345/not-session"
        else:
            field = {"ready-session": "session_id", "ready-instance": "instance_id",
                     "ready-pid": "pid"}[damage]
            attempt["ready"][field] = "not-the-owned-process"
    with pytest.raises(ValueError):
        swarm.connection_target(store, member)
    swarm.AzoWs.attach.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_stale_or_unobservable_process_identity_cannot_resolve_endpoint(
    ready_member, identities, monkeypatch, missing,
):
    from dataclasses import replace

    store, member, ready = ready_member
    identity = None if missing else replace(identities[1], start_token="new-process-same-pid")
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: identity)
    with pytest.raises(ValueError, match="identity"):
        swarm.connection_target(store, member)
    swarm.AzoWs.attach.assert_not_called()


def test_process_identity_rechecked_after_websocket_connect(target, monkeypatch):
    client = fake_client(monkeypatch, target)
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: None)
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(swarm.send_to_member(target, {"action": "continue"}))
    client.slash.assert_not_awaited()
    client.send_user.assert_not_awaited()
    client.control.assert_not_awaited()
    client.close.assert_awaited_once()


def test_creation_uses_fake_handles_records_readiness_and_keeps_models_idle(
    state, controller, identities,
):
    owner_thread = threading.get_ident()
    identity = identities[1]
    requests, polls = [], []

    def prepare(request, **kwargs):
        requests.append(request)
        return SimpleNamespace(request=request, ready_file="/missing/ready.json")

    def start(plan):
        request = plan.request
        store = swarm.store_for(state, "sw0")
        attempt = store.records().get("attempts", request.target.session_id)
        assert attempt["state"] == "reserved"
        assert attempt["instance_id"] == request.target.instance_id
        ready = {
            "format_version": 1, "session_id": request.target.session_id,
            "instance_id": request.target.instance_id, "pid": identity.pid,
            "process_identity": identity.to_dict(),
            "websocket_url": "ws://127.0.0.1:12345/session",
        }

        def poll():
            polls.append(threading.get_ident())
            return SimpleNamespace(status="ready", ready=ready, error="")

        return SimpleNamespace(
            request=request, ref=request.target, process=SimpleNamespace(pid=identity.pid),
            poll=poll, stdout_path="/unused/stdout", stderr_path="/unused/stderr",
        )

    state.session_launcher = SimpleNamespace(prepare=Mock(side_effect=prepare), start=Mock(side_effect=start))
    result = asyncio.run(controller.execute(state, swarm.parse_command("-n4 -p2 --name explore")))
    assert result["created"] and result["level"] == "info"
    assert len(requests) == 4 and len(controller.handles) == 4
    assert {request.kind for request in requests} == {"sw0p0a0", "sw0p0a1", "sw0p1a0", "sw0p1a1"}
    assert all(request.first_user_message == "" for request in requests)
    assert all(request.parent_session_id == state._session_id for request in requests)
    assert all(request.lifetime == "independent" for request in requests)
    assert polls and all(thread_id != owner_thread for thread_id in polls)
    assert not hasattr(state, "handles")
    attempts = swarm.store_for(state, "sw0").attempts()
    assert len(attempts) == 4
    assert all(attempt["state"] == "ready" for attempt in attempts)
    assert all(attempt["ready"]["process_identity"] == identity.to_dict() for attempt in attempts)
    swarm.AzoWs.attach.assert_not_called()


@pytest.mark.parametrize("created", [False, True])
def test_completion_binds_without_io_and_only_creation_injects_system_notification(
    state, controller, monkeypatch, created,
):
    store, pool = allocate(state)
    state.swarm_access = {}
    ctx = context(state)
    forbidden = Mock(side_effect=AssertionError("completion must remain on-owner without I/O"))
    monkeypatch.setattr(swarm, "store_for", forbidden)
    result = controller.complete(ctx, {
        "pool": pool, "created": created, "summary": "completed offline", "level": "warning",
    })
    assert state.swarm_access == {pool["id"]: ""}
    assert isinstance(result, CommandResult)
    assert result.outputs[0].text == "completed offline" and result.outputs[0].level == "warning"
    if created:
        ctx.session.inject.assert_called_once()
        assert ctx.session.inject.call_args.kwargs == {"role": "system", "system_generated": True}
        assert pool["id"] in ctx.session.inject.call_args.args[0]
    else:
        ctx.session.inject.assert_not_called()
    forbidden.assert_not_called()


def test_creation_command_captures_settings_but_does_not_allocate_at_ingress(
    state, controller, monkeypatch,
):
    ctx = context(state)
    execute = AsyncMock(return_value={})
    monkeypatch.setattr(controller, "execute", execute)
    allocate_mock = Mock(side_effect=AssertionError("ingress must not allocate"))
    monkeypatch.setattr(swarm, "allocate_pool", allocate_mock)
    result = controller.command(ctx, raw_args="-n1")
    assert isinstance(result, CommandResult)
    ctx.defer.assert_called_once()
    execute.assert_not_called()
    allocate_mock.assert_not_called()
    asyncio.run(ctx.defer.call_args.args[0]())
    snapshot, request = execute.call_args.args
    assert isinstance(snapshot.settings, RuntimeSettings)
    assert snapshot.settings.model == "offline-test-model"
    assert snapshot.settings.project == "test-project"
    assert snapshot.settings.store_root == state.settings.store_root
    assert snapshot.settings.state_root == state.settings.state_root
    assert snapshot.settings.port == 0
    assert request["action"] == "create"
    assert state.swarm_access == {}


def test_pool_ids_are_project_wide_but_default_selection_is_owner_scoped(state):
    _, first = allocate(state, "-n1 --name first-owner")
    other = SimpleNamespace(**vars(state))
    other._session_id = "another-parent"
    other.swarm_access = {}
    _, second = allocate(other, "-n1 --name second-owner")
    assert (first["id"], second["id"]) == ("sw0", "sw1")
    assert swarm.resolve_pool(state, None)[1]["id"] == "sw0"
    assert swarm.resolve_pool(other, None)[1]["id"] == "sw1"
    with pytest.raises(ValueError):
        allocate(other, "-n1 --name first-owner")
