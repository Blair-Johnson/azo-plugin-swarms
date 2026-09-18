"""Exact lineage/native/hello adoption, no launcher or real sockets."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import uuid

import pytest
from agent_zoo.live_session_registry import LiveSessionRecord, LiveProcessObservation
from tmux_pilot.process_identity import ProcessIdentity, ObservationState
import swarm
from test_recovery import runtime, reserve


@pytest.fixture
def lineage(runtime, monkeypatch):
    r = runtime
    member = r.members[0]
    identity = ProcessIdentity("host-a", "boot-a", "namespace-a", 200, "native-child")
    old = reserve(r, member, state="ready_paused", identity=identity.to_dict())
    def row(iid, *, terminal=False, previous=None, native=identity, generation="opaque-generation", **metadata):
        meta = dict(project_name="test", parent_session_id="parent", generation=generation, **metadata)
        if previous:
            meta.update(reload_previous_instance_id=previous,
                        reload_request_id=json.dumps(dict(version=1, previous_instance_id=previous, instance_id=iid)))
        return LiveSessionRecord(session_id=member["session_id"], instance_id=iid, pid=native.pid,
            websocket_url=f"ws://127.0.0.1:{12000 + len(rows)}/session", status="stopped" if terminal else "running",
            kind=member["label"], process_identity=native, metadata=meta,
            capabilities=("input:user_text", "input:slash_command", "control:shutdown"),
            process_observation=LiveProcessObservation(ObservationState.UNKNOWN,
                "instance-retired" if terminal else "live", time.time()-1, 1))
    rows = []
    a = row(old["instance_id"])
    rows.append(a)
    r.store.update_attempt(member["session_id"], ready={"websocket_url": a.websocket_url}, pid=identity.pid)
    monkeypatch.setattr(swarm, "list_live_sessions", lambda *args, **kwargs: list(rows))
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: identity if pid == identity.pid else r.identity)
    return SimpleNamespace(runtime=r, member=member, identity=identity, rows=rows, row=row, a=a)


def retire(row):
    return replace(row, status="stopped", process_observation=LiveProcessObservation(
        ObservationState.UNKNOWN, "instance-retired", time.time()-1, 1))


def reload(l, predecessor=None, *, terminal=False):
    previous = predecessor or l.rows[-1]
    successor = l.row(uuid.uuid4().hex, previous=previous.instance_id, terminal=terminal,
                      generation="next-opaque-generation")
    l.rows[l.rows.index(previous)] = retire(previous)
    l.rows.append(successor)
    return successor


def client_for(monkeypatch, row, **hello_changes):
    hello = dict(type="hello", protocol_version=1, session_id=row.session_id, instance_id=row.instance_id)
    hello.update(hello_changes)
    client = SimpleNamespace(connect=AsyncMock(), close=AsyncMock(), send_user=AsyncMock(),
                             slash=AsyncMock(), control=AsyncMock(), protocol_errors=[],
                             observation=lambda: SimpleNamespace(hello=hello))
    attach = Mock(return_value=client)
    monkeypatch.setattr(swarm.AzoWs, "attach", attach)
    return client, attach


def operation(l):
    request = swarm.ControlRequest("bcast", "sw0", message=" exact\n'\\\u96ea  ").as_dict()
    with l.runtime.store.mutation() as (records, pool):
        records.put("operations", request["operation_id"], dict(id=request["operation_id"], state="pending"))
    return request


def test_unchanged_exact_peer_and_cas_no_unnecessary_revision(lineage, monkeypatch):
    l = lineage; store = l.runtime.store
    target = swarm.connection_target(store, l.member)
    first = swarm.cas_adopt(store, l.member, target)
    again = swarm.connection_target(store, l.member)
    assert swarm.cas_adopt(store, l.member, again) == first
    client, _ = client_for(monkeypatch, l.a)
    req = operation(l)
    asyncio.run(swarm.send_to_member(again, req, store=store, member=l.member))
    client.send_user.assert_awaited_once_with(req["message"], client_id=req["operation_id"])
    l.runtime.state.session_launcher.start.assert_not_called()


def test_repeated_reload_adopts_tip_without_old_ready_file(lineage, monkeypatch):
    l = lineage; store = l.runtime.store
    origin = store.records().get("attempts", l.member["session_id"])
    b = reload(l); c = reload(l, b)
    target = swarm.connection_target(store, l.member)
    assert target["instance_id"] == c.instance_id and len(target["path"]) == 2
    client, attach = client_for(monkeypatch, c)
    req = operation(l)
    asyncio.run(swarm.send_to_member(target, req, store=store, member=l.member))
    saved = store.records().get("attempts", l.member["session_id"])
    assert saved["current"]["instance_id"] == c.instance_id and saved["binding_revision"] == 1
    assert all(saved[key] == origin[key] for key in ("instance_id", "process_identity", "request", "ready"))
    assert attach.call_args.args[0] == c.websocket_url and client.send_user.await_count == 1
    assert store.records().get("operations", req["operation_id"])["dispatch"][l.member["session_id"]]["instance_id"] == c.instance_id


def test_unrelated_newer_sibling_never_adopted(lineage):
    l = lineage
    sibling = replace(l.row(uuid.uuid4().hex), updated_at=time.time()+1000)
    l.rows.append(sibling)
    assert swarm.connection_target(l.runtime.store, l.member)["instance_id"] == l.a.instance_id
    l.rows[0] = retire(l.a)
    with pytest.raises(ValueError, match="descendant"):
        swarm.connection_target(l.runtime.store, l.member)


def test_restart_ready_claim_with_new_native_identity(lineage, monkeypatch):
    l = lineage
    identity = ProcessIdentity("host-a", "boot-a", "namespace-a", 201, "restart-native")
    b = l.row(uuid.uuid4().hex, native=identity)
    claim = dict(status="ready", session_id=l.a.session_id, previous_instance_id=l.a.instance_id,
                 instance_id=b.instance_id, owner=identity.to_dict(), request_id="restart-request", revision_id="checkpoint")
    l.rows[0] = replace(retire(l.a), metadata=dict(l.a.metadata, recovery=claim))
    l.rows.append(b)
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: identity)
    target = swarm.connection_target(l.runtime.store, l.member)
    assert target["path"][0]["kind"] == "restart" and target["instance_id"] == b.instance_id
    claim["status"] = "starting"
    with pytest.raises(ValueError, match="claim"):
        swarm.connection_target(l.runtime.store, l.member)


@pytest.mark.parametrize("mutation", ["version-bool", "wrong-parent", "wrong-project", "wrong-session",
    "foreign-host", "changed-native", "wrong-previous", "forward-conflict", "branch", "active-predecessor"])
def test_bad_lineage_fails_closed(lineage, mutation):
    l = lineage
    b = reload(l)
    if mutation == "version-bool":
        b.metadata["reload_request_id"] = json.dumps(dict(version=True, previous_instance_id=l.a.instance_id, instance_id=b.instance_id))
    elif mutation == "wrong-parent":
        b.metadata["parent_session_id"] = "foreign"
    elif mutation == "wrong-project":
        b.metadata["project_name"] = "other"
    elif mutation == "wrong-session":
        l.rows[-1] = replace(b, session_id="other")
    elif mutation in {"foreign-host", "changed-native"}:
        identity = replace(l.identity, host_id="foreign") if mutation == "foreign-host" else replace(l.identity, start_token="other")
        l.rows[-1] = replace(b, process_identity=identity)
    elif mutation == "wrong-previous":
        b.metadata["reload_previous_instance_id"] = "other"
    elif mutation == "forward-conflict":
        l.rows[0].metadata["successor_instance_id"] = "another"
    elif mutation == "branch":
        l.rows.append(l.row(uuid.uuid4().hex, previous=l.a.instance_id))
    elif mutation == "active-predecessor":
        l.rows[0] = l.a
    with pytest.raises(ValueError):
        swarm.connection_target(l.runtime.store, l.member)


def test_generation_mutation_and_lost_cas_never_overwrite(lineage):
    l = lineage; store = l.runtime.store
    b = reload(l)
    proposal = swarm.connection_target(store, l.member)
    swarm.cas_adopt(store, l.member, proposal)
    with pytest.raises(swarm.PreSendRace):
        swarm.cas_adopt(store, l.member, proposal)
    b.metadata["generation"] = "same-iid-different-generation"
    with pytest.raises(ValueError, match="generation"):
        swarm.connection_target(store, l.member)
    assert store.records().get("attempts", l.member["session_id"])["current"]["registry_generation"] != b.metadata["generation"]


def test_revalidation_branch_race_sends_nothing(lineage, monkeypatch):
    l = lineage; b = reload(l)
    target = swarm.connection_target(l.runtime.store, l.member)
    client, _ = client_for(monkeypatch, b)
    async def connect():
        l.rows.append(l.row(uuid.uuid4().hex, previous=l.a.instance_id))
    client.connect.side_effect = connect
    with pytest.raises(ValueError, match="branches"):
        asyncio.run(swarm.send_to_member(target, operation(l), store=l.runtime.store, member=l.member))
    client.send_user.assert_not_awaited()
    assert l.runtime.store.records().get("attempts", l.member["session_id"])["binding_revision"] == 0


@pytest.mark.parametrize("changes", [{"instance_id": "sibling"}, {"session_id": "foreign"}, {"protocol_version": True}])
def test_wrong_hello_is_unavailable_before_send(lineage, monkeypatch, changes):
    l = lineage; target = swarm.connection_target(l.runtime.store, l.member)
    client, _ = client_for(monkeypatch, l.a, **changes)
    with pytest.raises(ValueError, match="identity"):
        asyncio.run(swarm.send_to_member(target, operation(l), store=l.runtime.store, member=l.member))
    client.send_user.assert_not_awaited()


def test_send_failure_unknown_never_chases_new_successor(lineage, monkeypatch):
    l = lineage; target = swarm.connection_target(l.runtime.store, l.member)
    client, attach = client_for(monkeypatch, l.a)
    async def ambiguous(*args, **kwargs):
        reload(l)
        raise TimeoutError("write outcome lost")
    client.send_user.side_effect = ambiguous
    with pytest.raises(swarm.DeliveryUnknown):
        asyncio.run(swarm.send_to_member(target, operation(l), store=l.runtime.store, member=l.member))
    assert client.send_user.await_count == 1 and attach.call_count == 1
    client.close.assert_awaited_once()


@pytest.mark.parametrize("url", ["ws://localhost:10/session", "ws://10.0.0.1:10/session", "ws://127.0.0.1:10/session?x=1",
    "ws://user@127.0.0.1:10/session", "wss://127.0.0.1:10/session", "ws://127.0.0.1:10/other"])
def test_bad_loopback_endpoint(url):
    with pytest.raises(ValueError):
        swarm.endpoint(url)


def test_recovery_does_not_duplicate_unadopted_live_restart(lineage, monkeypatch):
    l = lineage
    successor_identity = replace(l.identity, pid=201, start_token="successor")
    successor = l.row(uuid.uuid4().hex, native=successor_identity)
    claim = dict(status="ready", session_id=l.a.session_id, previous_instance_id=l.a.instance_id,
                 instance_id=successor.instance_id, owner=successor_identity.to_dict(),
                 request_id="recovery", revision_id="saved")
    l.rows[0] = replace(retire(l.a), metadata=dict(l.a.metadata, recovery=claim)); l.rows.append(successor)
    monkeypatch.setattr(swarm, "process_evidence", lambda identity: "alive" if identity == successor_identity.to_dict() else "dead")
    attempt = l.runtime.store.records().get("attempts", l.member["session_id"])
    assert swarm.reconcile_attempt(l.runtime.state, attempt) == "alive"
    swarm.claim_recovery(l.runtime.state, l.runtime.store, expected_epoch=1, confirmed_stopped=True)
    assert l.runtime.store.records().get("members", l.member["session_id"])["recovery_status"] == "ambiguous"
    with pytest.raises(ValueError, match="reconciliation"):
        reserve(l.runtime, l.member)


def test_missing_replacement_history_is_unknown_not_dead(lineage, monkeypatch):
    l = lineage
    l.rows.clear()
    monkeypatch.setattr(swarm, "process_evidence", lambda identity: "dead")
    attempt = l.runtime.store.records().get("attempts", l.member["session_id"])
    assert swarm.reconcile_attempt(l.runtime.state, attempt) == "unknown"


def test_dispatch_gate_prevents_takeover_until_write_finishes(lineage, monkeypatch):
    l = lineage; store = l.runtime.store
    target = swarm.connection_target(store, l.member)
    client, _ = client_for(monkeypatch, l.a)
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "dead")
    async def scenario():
        sending, release = asyncio.Event(), asyncio.Event()
        async def blocked_send(*args, **kwargs):
            sending.set(); await release.wait()
            assert store.pool()["owner"]["epoch"] == 1
        client.send_user.side_effect = blocked_send
        send = asyncio.create_task(swarm.send_to_member(target, operation(l), store=store, member=l.member))
        await asyncio.wait_for(sending.wait(), 2)
        recovery = asyncio.create_task(asyncio.to_thread(swarm.claim_recovery, l.runtime.state, store,
                                                       expected_epoch=1, confirmed_stopped=True))
        await asyncio.sleep(.05)
        assert not recovery.done() and store.pool()["owner"]["epoch"] == 1
        release.set()
        await asyncio.wait_for(send, 2)
        await asyncio.wait_for(recovery, 2)
        assert store.pool()["owner"]["epoch"] == 2
    asyncio.run(scenario())


def test_cleanup_failure_after_send_is_unknown(lineage, monkeypatch):
    l = lineage
    target = swarm.connection_target(l.runtime.store, l.member)
    client, _ = client_for(monkeypatch, l.a)
    client.close.side_effect = OSError("close failed")
    with pytest.raises(swarm.DeliveryUnknown):
        asyncio.run(swarm.send_to_member(target, operation(l), store=l.runtime.store, member=l.member))
    assert client.send_user.await_count == 1


def test_parent_adopted_replacement_can_bootstrap_normal_mode(lineage, monkeypatch):
    from test_recovery import seed
    l = lineage; r = l.runtime
    b = reload(l)
    saved = seed(r, l.member, instance=b.instance_id)
    swarm.cas_adopt(r.store, l.member, swarm.connection_target(r.store, l.member))
    state = SimpleNamespace(**vars(r.state))
    state._session_id = l.member["session_id"]; state._instance_id = b.instance_id; state.grants = {}
    monkeypatch.setattr(swarm, "native_self", lambda: l.identity.to_dict())
    client_for(monkeypatch, b)
    result = asyncio.run(swarm.bootstrap_member(state, dict(session_id=state._session_id, instance_id=b.instance_id,
                                restored=True, startup_mode="normal", paused=False, revision_id=saved.commit_id)))
    assert result["grants"]["sw0"]["instance_id"] == b.instance_id
    assert result["access"] == {"sw0": "pod-1"}
    assert r.store.records().get("attempts", state._session_id)["bootstrap"]["paused"] is False
