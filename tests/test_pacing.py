"""Whole-swarm slash pacing/configuration and terminal cancellation contracts via AgentZooSessionDriver."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_zoo.cli.wsctl import AzoWs
from agent_zoo.features.commands import CommandError
from agent_zoo.features.commands.core import _afk
import swarm
from conftest import execute_swarm_request
from test_adoption import client_for, discover_proposal, lineage, reload, send_via_driver
from test_recovery import runtime


@pytest.mark.parametrize("action", ["cancel", "release", "capture"])
@pytest.mark.parametrize("target", [None, "sw0", "Research_1"])
def test_optional_whole_swarm_target(action, target):
    raw = " \n" + action + ("\t" + target if target else "") + " \n"
    assert swarm.parse_command(raw) == {"action": action, "target": target}


@pytest.mark.parametrize("raw,target,message", [
    ("afk", None, None), ("afk sw0", "sw0", None),
    ('afk "clear"', None, "clear"), ("afk sw0 -", "sw0", "-"),
    ('afk "  first\n\t雪 last  "', None, "  first\n\t雪 last  "),
    ("afk 'keep working'", None, "keep working"),
    ('afk sw0 "  first\nlast  "', "sw0", "  first\nlast  "),
    ('afk sw0 keep  working\nthen say "done"  ', "sw0", 'keep  working\nthen say "done"  '),
    (r'afk "say \"hello\" at C:\\tmp; literal \n"', None, 'say "hello" at C:\\tmp; literal \\n'),
    ('afk sw0 "-p 0 is literal text"', "sw0", "-p 0 is literal text"),
])
def test_afk_parsing_contract(raw, target, message):
    assert swarm.parse_command(raw) == {"action": "afk", "target": target, "message": message}


@pytest.mark.parametrize("raw", [
    "release sw0 sw1", "capture sw0 sw1", "cancel sw0 sw1",
    "release -p 0", "capture sw0 -p 0", "cancel sw0 -p 0",
    "afk -p 0", 'afk sw0 -p 0 "text"', 'afk sw0 --pods=0 "text"',
    "afk ../bad text", 'afk ""', 'afk " \n\t "', 'afk sw0 ""',
    'afk "unterminated', 'afk sw0 "unterminated', 'afk "text" trailing',
    'afk sw0 "text" -p 0',
])
def test_malformed_slash_input_is_status_only(runtime, raw):
    ctx = SimpleNamespace(state=runtime.state, submit_io=Mock(return_value="op_1"), emit=Mock())
    with pytest.raises(CommandError):
        swarm.SwarmController({}).command(ctx, raw_args=raw)
    ctx.submit_io.assert_not_called()
    ctx.emit.assert_not_called()
    assert not runtime.store.records().list("operations")


def test_afk_oversize_rejected_before_background_or_socket(runtime, monkeypatch):
    ctx = SimpleNamespace(state=runtime.state, submit_io=Mock(return_value="op_1"), emit=Mock())
    attach = Mock(side_effect=AssertionError("must validate before socket"))
    monkeypatch.setattr(AzoWs, "attach", attach)
    with pytest.raises(CommandError, match="1 MiB"):
        swarm.SwarmController({}).command(ctx, raw_args='afk "' + "雪" * 200000 + '"')
    ctx.submit_io.assert_not_called()
    ctx.emit.assert_not_called()
    attach.assert_not_called()


@pytest.mark.parametrize("message,expected", [
    (None, "saved instructions"), ("clear", ""), ("OFF", ""), ("none", ""), ("-", ""),
    ("Keep working\non the next task", "Keep working\non the next task"),
])
def test_native_afk_is_instructions_not_pace(message, expected):
    state = SimpleNamespace(afk_message="saved instructions", _afk_message="saved instructions")
    result = _afk(SimpleNamespace(state=state), message=message)
    assert not result.events and state.afk_message == expected and state._afk_message == expected
    assert swarm.afk_command(message) == ("/afk" if message is None else "/afk " + message)


def ready_members(runtime):
    with runtime.store.mutation() as (records, _):
        for member in runtime.members:
            records.put("members", member["session_id"], dict(recovery_status="ready_paused"))


def cancel_record(store):
    with store.mutation() as (records, pool):
        pool.update(desired_state="cancelled", phase="cancelled")
        pool["pod_states"] = {p["id"]: "cancelled" for p in pool["pods"]}
        records.put("swarm", "pool", pool)


@pytest.mark.parametrize("raw,expected", [
    ("release sw0", "/release"), ("capture sw0", "/capture"),
    ("afk sw0", "/afk"), ("afk sw0 clear", "/afk clear"),
    ('afk "  first\n\t雪 last  "', "/afk   first\n\t雪 last  "),
    ('afk sw0 say "hello"\nthen continue', '/afk say "hello"\nthen continue'),
])
def test_exact_owned_websocket_slash_payload(lineage, monkeypatch, raw, expected):
    l = lineage
    ready_members(l.runtime)
    client, attach = client_for(monkeypatch, l.a)
    result = execute_swarm_request(swarm.SwarmController({}), l.runtime.state, swarm.parse_command(raw))
    client.send_json.assert_awaited_once()
    frame = client.send_json.call_args.args[0]
    assert frame["channel"] == "slash_command" and frame["payload"] == {"raw": expected}
    assert attach.call_count == 1
    assert "1 of 2 agents" in result["summary"]
    l.runtime.state.session_launcher.start.assert_not_called()


@pytest.mark.parametrize("action", ["release", "capture", "afk"])
def test_pacing_requires_native_slash_capability_before_attach(lineage, monkeypatch, action):
    l = lineage
    l.rows[0] = replace(l.a, capabilities=("input:user_text", "control:shutdown"))
    target = discover_proposal(l.runtime.store, l.member)
    client, attach = client_for(monkeypatch, l.a)
    before = l.runtime.store.records().get("attempts", l.member["session_id"])
    with pytest.raises(ValueError, match="input:slash_command"):
        swarm.build_member_input_request(target, dict(action=action), store=l.runtime.store, member=l.member)
    attach.assert_not_called()
    client.send_json.assert_not_awaited()
    assert l.runtime.store.records().get("attempts", l.member["session_id"]) == before


@pytest.mark.parametrize("action", ["release", "capture", "afk"])
@pytest.mark.parametrize("desired", ["paused", "running", "mixed"])
def test_pacing_preserves_all_lifecycle_state(lineage, monkeypatch, action, desired):
    l = lineage; r = l.runtime
    ready_members(r)
    with r.store.mutation() as (records, pool):
        pool.update(desired_state=desired, phase="mixed")
        pool["pod_states"] = {p["id"]: ("paused" if i == 0 else "running")
                              for i, p in enumerate(pool["pods"])}
        records.put("swarm", "pool", pool)
    before = deepcopy(r.store.pool())
    client, _ = client_for(monkeypatch, l.a)
    result = execute_swarm_request(
        swarm.SwarmController({}), r.state, dict(action=action, target="sw0", message="keep working"),
    )
    assert client.send_json.await_count == 1
    assert r.store.pool() == before
    assert result["operation"]["action"] == action
    assert "sw0" in swarm.SwarmController.receipt(dict(action=action, target="sw0"))
    r.state.session_launcher.start.assert_not_called()


@pytest.mark.parametrize("action", ["release", "continue", "bcast"])
@pytest.mark.parametrize("status,blocked", [(None, False), ("ambiguous", False), ("ready_paused", True)])
def test_work_admission_checks_every_member_before_any_dispatch(runtime, monkeypatch, action, status, blocked):
    r = runtime
    ready_members(r)
    with r.store.mutation() as (records, _):
        records.put("members", r.members[-1]["session_id"], dict(recovery_status=status, work_blocked=blocked))
    discover = Mock(side_effect=AssertionError("must not discover"))
    monkeypatch.setattr(swarm, "build_member_discover_request", discover)
    result = execute_swarm_request(
        swarm.SwarmController({}), r.state, dict(action=action, target="sw0", message="text"),
    )
    assert result["level"] == "warning" and "no work sent" in result["summary"]
    discover.assert_not_called()
    assert not r.store.records().list("operations")
    r.state.session_launcher.start.assert_not_called()


@pytest.mark.parametrize("action", ["capture", "afk"])
def test_configuration_does_not_require_recovery_readiness(lineage, monkeypatch, action):
    l = lineage
    client, _ = client_for(monkeypatch, l.a)
    result = execute_swarm_request(swarm.SwarmController({}), l.runtime.state, dict(action=action, target="sw0"))
    assert client.send_json.await_count == 1
    assert "1 of 2 agents" in result["summary"]


@pytest.mark.parametrize("action", ["release", "capture", "afk"])
def test_pacing_is_slash_only_and_rejects_pod_scoped_requests(runtime, action):
    with pytest.raises(ValueError, match="model accessible"):
        swarm.ControlRequest(action, "sw0").as_dict()
    result = execute_swarm_request(
        swarm.SwarmController({}), runtime.state, dict(action=action, target="sw0", pods=(0,)),
    )
    assert "whole swarm" in result["summary"]
    assert not runtime.store.records().list("operations")


def test_default_cancel_resolves_only_unique_non_cancelled_pool(runtime):
    r = runtime
    r.store.post("pod-1", "general", "historical board", "parent", message_id="history")
    other, _ = swarm.allocate_pool(r.state, swarm.parse_command("-n1 --name remaining"))
    controller = swarm.SwarmController({})
    ambiguous = execute_swarm_request(controller, r.state, swarm.parse_command("cancel"))
    assert "exactly one" in ambiguous["summary"]
    assert not r.store.records().list("operations") and not other.records().list("operations")
    execute_swarm_request(controller, r.state, swarm.parse_command("cancel sw0"))
    assert swarm.resolve_pool(r.state, None)[1]["id"] == "sw1"
    execute_swarm_request(controller, r.state, swarm.parse_command("cancel"))
    for store in (r.store, other):
        assert store.pool()["desired_state"] == "cancelled"
        assert set(store.pool()["pod_states"].values()) == {"cancelled"}
    with pytest.raises(ValueError, match="exactly one"):
        swarm.resolve_pool(r.state, None)
    assert swarm.resolve_pool(r.state, "sw0")[1]["desired_state"] == "cancelled"
    assert r.store.messages("pod-1", "general")[0]["message"]["text"] == "historical board"
    r.state.session_launcher.start.assert_not_called()


@pytest.mark.parametrize("raw", [
    "release sw0", "capture sw0", "afk sw0 keep working", "continue sw0", "interrupt sw0",
    'bcast sw0 "work"', "recover sw0 --takeover --expected-epoch 1 --confirmed-stopped",
])
def test_cancelled_pool_cannot_resume_configure_broadcast_or_recover(runtime, monkeypatch, raw):
    cancel_record(runtime.store)
    before = runtime.store.pool()
    discover = Mock(side_effect=AssertionError("must not discover"))
    monkeypatch.setattr(swarm, "build_member_discover_request", discover)
    result = execute_swarm_request(swarm.SwarmController({}), runtime.state, swarm.parse_command(raw))
    assert result["level"] == "warning" and "cancelled" in result["summary"].lower()
    assert runtime.store.pool() == before and not runtime.store.records().list("operations")
    discover.assert_not_called()
    runtime.state.session_launcher.start.assert_not_called()


def test_cancelled_pool_cannot_adopt_even_for_shutdown(lineage):
    l = lineage
    reload(l)
    proposal = discover_proposal(l.runtime.store, l.member)
    cancel_record(l.runtime.store)
    before = l.runtime.store.records().get("attempts", l.member["session_id"])
    for allow_cancelled in (False, True):
        with pytest.raises(ValueError, match="[Cc]ancelled"):
            swarm.cas_adopt(l.runtime.store, l.member, proposal, allow_cancelled=allow_cancelled)
    assert l.runtime.store.records().get("attempts", l.member["session_id"]) == before


def test_cancel_still_shuts_down_exact_peer_without_adopting(lineage, monkeypatch):
    l = lineage
    client, _ = client_for(monkeypatch, l.a)
    before = l.runtime.store.records().get("attempts", l.member["session_id"])
    result = execute_swarm_request(swarm.SwarmController({}), l.runtime.state, swarm.parse_command("cancel"))
    client.send_json.assert_awaited_once()
    frame = client.send_json.call_args.args[0]
    assert frame["channel"] == "control"
    assert frame["payload"] == {"type": "shutdown", "save": True, "reason": "swarm-cancel", "spawned_policy": "leave"}
    assert l.runtime.store.records().get("attempts", l.member["session_id"]) == before
    assert l.runtime.store.pool()["desired_state"] == "cancelled"
    assert "shutdown requested" in result["summary"]


@pytest.mark.parametrize("action", ["release", "capture", "afk", "continue", "bcast"])
def test_cancellation_after_adoption_fences_final_dispatch(lineage, monkeypatch, action):
    l = lineage
    target = discover_proposal(l.runtime.store, l.member)
    client, _ = client_for(monkeypatch, l.a)
    real_adopt = swarm.cas_adopt

    def adopt_then_cancel(*args, **kwargs):
        revision = real_adopt(*args, **kwargs)
        cancel_record(l.runtime.store)
        return revision

    monkeypatch.setattr(swarm, "cas_adopt", adopt_then_cancel)
    res = send_via_driver(
        target, dict(action=action, message="text", operation_id="not-sent"),
        store=l.runtime.store, member=l.member,
    )
    assert res.outcome == "unavailable" and "cancelled" in res.error.lower()
    client.send_json.assert_not_awaited()
    assert not l.runtime.store.records().list("operations")


def test_pacing_operation_is_never_replayed(lineage, monkeypatch):
    l = lineage
    ready_members(l.runtime)
    client, _ = client_for(monkeypatch, l.a)
    request = dict(action="release", target="sw0", operation_id="one-operation")
    controller = swarm.SwarmController({})
    first = execute_swarm_request(controller, l.runtime.state, request)
    second = execute_swarm_request(controller, l.runtime.state, request)
    assert "1 of 2 agents" in first["summary"] and "replay forbidden" in second["summary"]
    assert client.send_json.await_count == 1
    l.runtime.state.session_launcher.start.assert_not_called()


def test_unknown_pacing_delivery_is_not_retried_or_relaunched(lineage, monkeypatch):
    l = lineage
    ready_members(l.runtime)
    client, attach = client_for(monkeypatch, l.a)
    client.send_json.side_effect = TimeoutError("delivery outcome lost")
    result = execute_swarm_request(swarm.SwarmController({}), l.runtime.state, swarm.parse_command("release"))
    assert "delivery unknown" in result["summary"]
    assert client.send_json.await_count == 1 and attach.call_count == 1
    l.runtime.state.session_launcher.start.assert_not_called()
