"""Fenced recovery on real RecordStores/journals; no model or runtime processes."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest
from agent_utils import Session
from agent_utils.session_repository import SessionRepository
from agent_zoo.runtime_launch import RuntimeSettings
from tmux_pilot.process_identity import HostIdentity, ProcessIdentity
import swarm


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    for key in ("AZO_SWARM_ID", "AZO_SWARM_POD_ID", "AZO_SWARM_GRANT"):
        monkeypatch.delenv(key, raising=False)
    host = HostIdentity("host-a", "boot-a", "namespace-a", "a")
    identity = ProcessIdentity("host-a", "boot-a", "namespace-a", 100, "native-parent")
    monkeypatch.setattr(swarm, "local_host_identity", lambda: host)
    monkeypatch.setattr(swarm, "native_self", lambda: identity.to_dict())
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: identity if pid == 100 else None)
    monkeypatch.setattr(swarm, "list_live_sessions", lambda *a, **k: [])
    settings = RuntimeSettings(project="test", model="offline", workdir=str(tmp_path),
        state_root=str(tmp_path / "state"), store_root=str(tmp_path / "registry"),
        local_state_root=str(tmp_path / "local"))
    state = SimpleNamespace(_session_id="parent", _instance_id="parent-a", swarm_access={}, grants={},
        _agent_zoo_context={"project_name": "test", "project_store_root": settings.store_root},
        session_cwd=str(tmp_path), settings=settings, session_launcher=Mock(), pending_interrupts=[])
    store, pool = swarm.allocate_pool(state, swarm.parse_command("-n2 -p2"))
    return SimpleNamespace(state=state, store=store, pool=pool, host=host, identity=identity,
                           members=[m for p in pool["pods"] for m in p["members"]], tmp=tmp_path)


def seed(runtime, member, *, instance=None):
    sid = member["session_id"]
    instance = instance or uuid.uuid4().hex
    repository = SessionRepository(swarm.project_directory(runtime.state) / "sessions" / sid)
    session = Session(system_prompt="offline seed")
    session.state._session_id = sid
    session.state._instance_id = instance
    session.inject("saved durable text", role="user")
    writer = repository.create_instance(session_id=sid, instance_id=instance, origin_host_id="host-a")
    commit = writer.save(session)
    return repository.load(session_id=sid, commit_id=commit.commit_id)


def reserve(runtime, member, *, state="spawned", identity=None):
    request = dict(settings=json.loads(json.dumps(asdict(runtime.state.settings))), kind=member["label"])
    return runtime.store.reserve_attempt(member["session_id"], dict(
        session_id=member["session_id"], instance_id=uuid.uuid4().hex,
        pod_id=runtime.store.member_pod(member["session_id"]), state=state, request=request,
        process_identity=identity, created_ns=1))


def test_stale_owner_board_and_attempt_writes_fenced(runtime, monkeypatch):
    r = runtime
    member = r.members[0]
    attempt = reserve(r, member, identity=r.identity.to_dict())
    stale = swarm.SwarmStore(r.store.durable, r.store.cache, grant=r.store.grant)
    r.store.post("pod-1", "general", "before", "parent", message_id="before")
    before = r.store.records().get("boards/pod-1", "general", resolve_refs=False)
    r.state._instance_id = "parent-b"
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "dead")
    recovered = swarm.claim_recovery(r.state, r.store)
    assert recovered["owner"]["epoch"] == 2
    with pytest.raises(ValueError, match="Stale"):
        stale.post("pod-1", "general", "late", "parent")
    with pytest.raises(ValueError, match="Stale"):
        stale.update_attempt(member["session_id"], expected_attempt=attempt["attempt_id"], state="ready_paused")
    assert r.store.records().get("boards/pod-1", "general", resolve_refs=False) == before
    assert r.store.records().get("attempt_history", attempt["attempt_id"])["state"] == "spawned"


@pytest.mark.parametrize("evidence,confirmed", [("alive", False), ("alive", True), ("unknown", False)])
def test_foreign_or_live_owner_does_not_grant_takeover(runtime, monkeypatch, evidence, confirmed):
    r = runtime
    r.state._instance_id = "destination"
    monkeypatch.setattr(swarm, "process_evidence", lambda _: evidence)
    before = r.store.pool()
    with pytest.raises(ValueError, match="owner|takeover"):
        swarm.claim_recovery(r.state, r.store, expected_epoch=1, confirmed_stopped=confirmed)
    assert r.store.pool() == before
    r.state.session_launcher.start.assert_not_called()


def test_foreign_takeover_requires_exact_epoch_and_records_subjects(runtime, monkeypatch):
    r = runtime
    old = reserve(r, r.members[0], state="unknown")
    r.state._instance_id = "destination"
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "unknown")
    with pytest.raises(ValueError, match="epoch"):
        swarm.claim_recovery(r.state, r.store, expected_epoch=0, confirmed_stopped=True)
    pool = swarm.claim_recovery(r.state, r.store, expected_epoch=1, confirmed_stopped=True)
    evidence = pool["recovery"]["takeover_evidence"]
    assert evidence["confirmed_stopped"] and evidence["subjects"][0]["attempt_id"] == old["attempt_id"]
    assert "external jobs" in evidence["scope"]
    assert r.store.records().get("attempts", r.members[0]["session_id"])["state"] == "fenced"


def test_live_or_unknown_orphan_never_gets_duplicate_slot(runtime, monkeypatch):
    r = runtime
    for member in r.members:
        reserve(r, member, state="unknown")
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "unknown")
    pool = swarm.claim_recovery(r.state, r.store)
    for member in r.members:
        assert r.store.records().get("members", member["session_id"])["recovery_status"] == "ambiguous"
        with pytest.raises(ValueError, match="reconciliation"):
            reserve(r, member)
    assert pool["desired_state"] == "paused"


def test_cancelled_pool_never_recovers(runtime):
    r = runtime
    with r.store.mutation() as (records, pool):
        pool.update(desired_state="cancelled", phase="cancelled")
        records.put("swarm", "pool", pool)
    with pytest.raises(ValueError, match="Cancelled"):
        swarm.claim_recovery(r.state, r.store, expected_epoch=1, confirmed_stopped=True)
    assert r.store.pool()["owner"]["epoch"] == 1


def test_v1_requires_stopped_migration_preserving_board_bytes(runtime, monkeypatch):
    r = runtime
    r.store.post("pod-1", "general", "original", "parent", message_id="one")
    before = r.store.records().get("boards/pod-1", "general", resolve_refs=False)
    records = r.store.records(write=True)
    old = r.store.pool(); old["version"] = 1
    old.pop("owner"); old.pop("resource_uid")
    r.state.swarm_resource_uids = {}  # a genuine v1 checkpoint has no UID reference
    records.put("swarm", "pool", old)
    with pytest.raises(ValueError, match="v1 migration"):
        swarm.claim_recovery(r.state, r.store)
    pool = swarm.claim_recovery(r.state, r.store, expected_epoch=0, confirmed_stopped=True)
    assert pool["version"] == 2 and pool["owner"]["epoch"] == 1
    assert r.store.records().get("boards/pod-1", "general", resolve_refs=False) == before
    assert r.store.messages("pod-1", "general")[0]["message"]["text"] == "original"


def test_verified_shared_checkpoint_survives_all_local_cache_loss(runtime):
    r = runtime
    saved = seed(r, r.members[0])
    r.store.post("pod-1", "general", "durable board", "parent")
    r.store.messages("pod-1", "general")
    shutil.rmtree(r.tmp / "local", ignore_errors=True)
    transcript = swarm.saved_transcript(r.state, r.members[0]["session_id"])
    assert saved.commit_id in transcript and "saved durable text" in transcript
    assert "journal" in transcript and "not a live stream" in transcript
    assert r.store.messages("pod-1", "general")[0]["message"]["text"] == "durable board"
    assert not (saved.repository / "session.json").exists()


def test_missing_revision_never_falls_back_to_fresh_or_latest(runtime):
    r = runtime
    seed(r, r.members[0])
    with pytest.raises(FileNotFoundError):
        swarm.shared_checkpoint(r.state, r.members[0]["session_id"], "f" * 64)
    with pytest.raises(FileNotFoundError):
        swarm.shared_checkpoint(r.state, r.members[1]["session_id"])
    r.state.session_launcher.start.assert_not_called()


def test_pin_shared_revision_not_temp_or_old_argv(runtime, monkeypatch):
    r = runtime
    saved = seed(r, r.members[0])
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "dead")
    swarm.claim_recovery(r.state, r.store)
    ref = swarm.pin_checkpoint(r.state, r.store, r.members[0])
    assert ref.session_id == saved.session_id and ref.revision_id == saved.commit_id
    selected = r.store.pool()["recovery"]["selected_checkpoints"][saved.session_id]
    assert selected["revision_id"] == saved.commit_id


def test_paused_resume_same_sid_new_iid_and_destination_settings(runtime, monkeypatch):
    r = runtime
    member = r.members[0]
    saved = seed(r, member)
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "dead")
    swarm.claim_recovery(r.state, r.store)
    ref = swarm.pin_checkpoint(r.state, r.store, member)
    launcher = r.state.session_launcher
    launcher.prepare.side_effect = lambda request, **kwargs: SimpleNamespace(request=request, ready_file="/lost/ready")
    launcher.start.return_value = SimpleNamespace(process=SimpleNamespace(pid=100), stdout_path="new.out", stderr_path="new.err")
    swarm.launch_member(r.state, "sw0", member["session_id"], r.state.settings, handles={}, checkpoint=ref)
    request = launcher.prepare.call_args.args[0]
    assert request.startup_mode == "paused" and request.first_user_message == ""
    assert request.target.session_id == saved.session_id and request.target.instance_id != saved.instance_id
    assert request.source.checkpoint == ref and request.settings == r.state.settings
    assert request.parent_session_id == "parent" and request.lifetime == "independent"
    assert json.loads(launcher.prepare.call_args.kwargs["env"]["AZO_SWARM_GRANT"])["epoch"] == 2


def test_uncertain_start_retained_and_cannot_retry(runtime):
    r = runtime
    launcher = r.state.session_launcher
    launcher.prepare.side_effect = lambda request, **kwargs: SimpleNamespace(request=request, ready_file="missing")
    launcher.start.side_effect = OSError("spawn acknowledgement lost")
    member = r.members[0]
    with pytest.raises(OSError):
        swarm.launch_member(r.state, "sw0", member["session_id"], r.state.settings, handles={})
    assert r.store.records().get("attempts", member["session_id"])["state"] == "unknown"
    with pytest.raises(ValueError, match="reconciliation"):
        swarm.launch_member(r.state, "sw0", member["session_id"], r.state.settings, handles={})
    assert launcher.start.call_count == 1


def test_partial_recovery_does_not_replay_or_continue(runtime, monkeypatch):
    r = runtime
    saved = seed(r, r.members[0])
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "dead")
    launches = []
    def launch(state, swarm_id, sid, settings, *, handles, checkpoint):
        launches.append((sid, checkpoint))
        return object()
    async def ready(state, store, member, handle):
        return dict(label=member["label"], session_id=member["session_id"], state="ready_paused")
    monkeypatch.setattr(swarm, "launch_member", launch)
    monkeypatch.setattr(swarm, "await_member_ready", ready)
    controller = swarm.SwarmController({})
    result = asyncio.run(controller.recover(r.state, r.store, {}))
    assert launches == [(saved.session_id, swarm.CheckpointRef(saved.session_id, saved.commit_id))]
    assert "1 missing checkpoint" in result["summary"] and "1 of 2 agents restored (paused)" in result["summary"]
    assert r.store.pool()["phase"] == "mixed"
    assert all(op["action"] == "recover" for op in r.store.records().list("operations"))


def test_owned_discovery_survives_missing_saved_access_and_index(runtime):
    r = runtime
    r.state.swarm_access = {}
    registry = swarm.RecordStore(swarm.project_directory(r.state) / "swarms")
    registry.put("registry", "pools", {"pools": {}, "next": 1})
    matches = swarm.owned_stores(r.state)
    assert [(sid, error) for sid, store, error in matches] == [("sw0", None)]


def test_checkpoint_pending_tool_outcome_blocks_continue():
    doc = {"entries": [{"messages": [{"role": "assistant", "tool_calls": [{"id": "external"}]}]}]}
    assert swarm.unresolved_tools(doc)
    doc["entries"][0]["messages"].append({"role": "tool", "tool_call_id": "external"})
    assert not swarm.unresolved_tools(doc)


def test_pinned_recovery_ignores_later_shared_selection(runtime, monkeypatch):
    r = runtime
    saved = seed(r, r.members[0])
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "dead")
    swarm.claim_recovery(r.state, r.store)
    original = swarm.pin_checkpoint(r.state, r.store, r.members[0])
    # Independent divergent branch is real journal data; an existing recovery pin is exact.
    other = seed(r, r.members[0])
    repository = SessionRepository(saved.repository)
    repository.select(other.commit_id, session_id=other.session_id)
    with r.store.mutation() as (records, pool):
        pool["phase"] = "mixed"; records.put("swarm", "pool", pool)
    previous_operation = r.store.pool()["recovery"]["operation_id"]
    swarm.claim_recovery(r.state, r.store)
    assert swarm.pin_checkpoint(r.state, r.store, r.members[0]) == original
    assert r.store.records().get("recovery_history", previous_operation)["selected_checkpoints"][saved.session_id]["revision_id"] == saved.commit_id


def test_stale_post_does_not_even_publish_an_orphan_blob(runtime, monkeypatch):
    r = runtime
    stale = swarm.SwarmStore(r.store.durable, r.store.cache, grant=dict(r.store.grant, token="f"*32))
    blob = Mock(side_effect=AssertionError("stale writer must not create blobs"))
    monkeypatch.setattr(swarm.RecordStore, "put_immutable_json", blob)
    with pytest.raises(ValueError, match="Stale"):
        stale.post("pod-1", "general", "late", "parent")
    blob.assert_not_called()


def test_child_bootstrap_proves_shared_seed_before_parent_readiness(runtime, monkeypatch):
    r = runtime; member = r.members[0]
    child_identity = ProcessIdentity("host-a", "boot-a", "namespace-a", 200, "child-start")
    attempt = reserve(r, member, identity=child_identity.to_dict())
    saved = seed(r, member, instance=attempt["instance_id"])
    child = SimpleNamespace(**vars(r.state))
    child._session_id = member["session_id"]; child._instance_id = attempt["instance_id"]; child.grants = {}
    child.swarm_access = {"sw0": ""}  # deliberately stale checkpoint scope
    monkeypatch.setattr(swarm, "native_self", lambda: child_identity.to_dict())
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: child_identity)
    event = dict(session_id=child._session_id, instance_id=child._instance_id, restored=False,
                 startup_mode="paused", paused=True, revision_id=saved.commit_id)
    result = asyncio.run(swarm.bootstrap_member(child, event))
    assert result["access"] == {"sw0": "pod-1"}
    ready = dict(format_version=1, session_id=child._session_id, instance_id=child._instance_id,
                 pid=200, process_identity=child_identity.to_dict(), websocket_url="ws://127.0.0.1:12000/session",
                 startup_mode="paused", paused=True, revision_id=saved.commit_id)
    handle = SimpleNamespace(ref=SimpleNamespace(instance_id=child._instance_id), process=SimpleNamespace(pid=200),
                             poll=lambda: SimpleNamespace(status="ready", ready=ready))
    outcome = asyncio.run(swarm.await_member_ready(r.state, r.store, member, handle))
    assert outcome["state"] == "ready_paused"
    assert r.store.records().get("members", child._session_id)["shared_checkpoint"]["revision_id"] == saved.commit_id
    assert r.store.records().get("attempts", child._session_id)["ready"] == ready


def test_transport_ready_without_plugin_admission_is_not_recoverable(runtime, monkeypatch):
    r = runtime; member = r.members[0]
    identity = ProcessIdentity("host-a", "boot-a", "namespace-a", 200, "child-start")
    attempt = reserve(r, member, identity=identity.to_dict())
    saved = seed(r, member, instance=attempt["instance_id"])
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: identity)
    ready = dict(format_version=1, session_id=member["session_id"], instance_id=attempt["instance_id"],
                 pid=200, process_identity=identity.to_dict(), websocket_url="ws://127.0.0.1:12000/session",
                 startup_mode="paused", paused=True, revision_id=saved.commit_id)
    handle = SimpleNamespace(ref=SimpleNamespace(instance_id=attempt["instance_id"]), process=SimpleNamespace(pid=200),
                             poll=lambda: SimpleNamespace(status="ready", ready=ready))
    with pytest.raises(TimeoutError, match="Agent startup timed out"):
        asyncio.run(swarm.await_member_ready(r.state, r.store, member, handle, timeout=0))
    assert r.store.records().get("members", member["session_id"])["recovery_status"] == "pending"


def test_interrupted_takeover_reuses_confirmation_only_for_same_attempt(runtime, monkeypatch):
    r = runtime
    old = reserve(r, r.members[0], state="unknown")
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "unknown")
    swarm.claim_recovery(r.state, r.store, expected_epoch=1, confirmed_stopped=True)
    # No new process/attempt was admitted: this is still the explicitly stopped old slot.
    same = r.store.records().get("attempts", r.members[0]["session_id"])
    assert swarm.reconcile_attempt(r.state, same, r.store.pool()) == "confirmed_stopped"
    swarm.claim_recovery(r.state, r.store)
    new = reserve(r, r.members[0], state="unknown")
    assert new["attempt_id"] != old["attempt_id"]
    assert swarm.reconcile_attempt(r.state, new, r.store.pool()) == "unknown"
    swarm.claim_recovery(r.state, r.store)
    assert r.store.records().get("members", r.members[0]["session_id"])["recovery_status"] == "ambiguous"


def test_historical_attempt_snapshot_never_overwritten_by_reconciliation(runtime, monkeypatch):
    r = runtime
    original = reserve(r, r.members[0], state="unknown")
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "unknown")
    swarm.claim_recovery(r.state, r.store, expected_epoch=1, confirmed_stopped=True)
    archived = r.store.records().get("attempt_history", original["attempt_id"])
    assert archived == original
    reserve(r, r.members[0], state="reserved")
    assert r.store.records().get("attempt_history", original["attempt_id"]) == original


def test_same_iid_different_native_process_is_not_same_owner(runtime, monkeypatch):
    r = runtime
    foreign = ProcessIdentity("host-a", "boot-a", "namespace-a", 999, "other-native")
    monkeypatch.setattr(swarm, "native_self", lambda: foreign.to_dict())
    monkeypatch.setattr(swarm, "process_evidence", lambda _: "alive")
    with pytest.raises(ValueError, match="Different live owner"):
        swarm.claim_recovery(r.state, r.store, expected_epoch=1, confirmed_stopped=True)
    assert r.store.pool()["owner"]["epoch"] == 1


def test_crosshost_confirmed_recovery_uses_destination_not_origin_argv(runtime, monkeypatch):
    from dataclasses import replace
    r = runtime; member = r.members[0]
    source_child = ProcessIdentity("host-a", "boot-a", "namespace-a", 200, "source-child")
    original = reserve(r, member, identity=source_child.to_dict())
    saved = seed(r, member, instance=original["instance_id"])
    source_request = deepcopy(original["request"])
    destination_host = HostIdentity("host-b", "boot-b", "namespace-b", "b")
    destination_parent = ProcessIdentity("host-b", "boot-b", "namespace-b", 500, "destination-parent")
    destination_child = ProcessIdentity("host-b", "boot-b", "namespace-b", 501, "destination-child")
    monkeypatch.setattr(swarm, "local_host_identity", lambda: destination_host)
    monkeypatch.setattr(swarm, "native_self", lambda: destination_parent.to_dict())
    monkeypatch.setattr(swarm, "observe_process", lambda identity: SimpleNamespace(
        state=swarm.ObservationState.UNKNOWN if identity.host_id == "host-a" else swarm.ObservationState.ALIVE))
    monkeypatch.setattr(swarm, "capture_process_identity", lambda pid: destination_child if pid == 501 else destination_parent)
    monkeypatch.setattr(swarm.AzoWs, "attach", Mock(side_effect=AssertionError("never contact source loopback")))
    r.state._instance_id = "parent-on-b"
    r.state.grants = {}
    destination_workdir = r.tmp / "destination-work"; destination_workdir.mkdir()
    r.state.settings = replace(r.state.settings, workdir=str(destination_workdir),
                               store_root=str(r.tmp / "destination-registry"), config_path=str(r.tmp / "destination.toml"))
    r.state._agent_zoo_context["project_store_root"] = r.state.settings.store_root
    with pytest.raises(ValueError, match="unknown"):
        swarm.claim_recovery(r.state, r.store)
    swarm.claim_recovery(r.state, r.store, expected_epoch=1, confirmed_stopped=True)
    shutil.rmtree(r.tmp / "local", ignore_errors=True)
    ref = swarm.pin_checkpoint(r.state, r.store, member)
    launcher = r.state.session_launcher
    launcher.prepare.side_effect = lambda request, **kwargs: SimpleNamespace(request=request, ready_file="destination-ready")
    launcher.start.return_value = SimpleNamespace(process=SimpleNamespace(pid=501), stdout_path="destination.out", stderr_path="destination.err")
    swarm.launch_member(r.state, "sw0", member["session_id"], r.state.settings, handles={}, checkpoint=ref)
    launched = launcher.prepare.call_args.args[0]
    assert launched.settings.workdir == str(destination_workdir)
    assert launched.settings.store_root == str(r.tmp / "destination-registry")
    assert launched.settings.config_path == str(r.tmp / "destination.toml")
    assert launched.target.session_id == saved.session_id and launched.target.instance_id != saved.instance_id
    assert launched.source.checkpoint.revision_id == saved.commit_id and launched.startup_mode == "paused"
    assert r.store.records().get("attempt_history", original["attempt_id"])["request"] == source_request
    assert r.store.pool()["owner"]["host"]["host_id"] == "host-b"
    swarm.AzoWs.attach.assert_not_called()


def test_watermark_cannot_advance_to_unrelated_shared_branch(runtime):
    r = runtime
    first = seed(r, r.members[0]); other = seed(r, r.members[0])
    with pytest.raises(ValueError, match="branch reconciliation"):
        swarm.require_checkpoint_ancestry(r.state, first.session_id, other,
                                           dict(revision_id=first.commit_id))
    swarm.require_checkpoint_ancestry(r.state, first.session_id, first, dict(revision_id=first.commit_id))
