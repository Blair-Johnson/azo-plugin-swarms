"""Offline transfer tests: real journals/stores, no backend/LLM/tool launches."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
import zipfile

import pytest
from agent_utils import Session
from agent_utils import checkpoint_revisions as revisions
from agent_utils.session_repository import SessionRepository
from agent_zoo import projects
from tmux_pilot.fs_store import RecordStore, StoreError, file_lock
from tmux_pilot.process_identity import capture_process_identity

SPEC = importlib.util.spec_from_file_location(
    "swarm_transfer", Path(__file__).resolve().parents[1] / "scripts" / "swarm_transfer.py")
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


def checkpoint(project, sid, *, parent="", marker="selected"):
    projects.ensure_project(project)
    projects.ensure_session_dir(project, sid)
    session = Session(system_prompt="offline transfer test")
    session.state.marker = marker
    session.state.session_cwd = "/old/root/work"
    session.state.terminal_cwd = "/old/root/work"
    session.state.runner = {"workdir": "/old/root/jobs"}
    session.state.swarm_access = {"sw0": "pod-1" if parent else ""}
    repo = Path(projects.resolve_session_paths(project, sid)["session_dir"])
    try:
        previous = revisions.load_revision(repo).revision_id
    except FileNotFoundError:
        previous = None
    publication = revisions.publish_revision(session, repo, session_id=sid,
        execution_id=f"test-{uuid.uuid4().hex}", parent_revision_id=previous,
        expected_preferred_revision_id=previous)
    projects.register_session(project, sid, title=sid, parent_session_id=parent,
                              kind="rlm" if parent else "default")
    return publication.revision.revision_id


@pytest.fixture
def source(tmp_path, monkeypatch):
    home = tmp_path / "source"
    monkeypatch.setenv(projects.HOME_ENV, str(home))
    monkeypatch.delenv(projects.STATE_ROOT_ENV, raising=False)
    parent, child = str(uuid.uuid4()), str(uuid.uuid4())
    exact = {parent: checkpoint("source", parent),
             child: checkpoint("source", child, parent=parent)}
    root = home / "projects" / "source" / "swarms"
    registry = RecordStore(root)
    registry.put("registry", "pools", {"next": 1, "pools": {"sw0": {"name": "crew", "owner": parent}}})
    records = RecordStore(root / "sw0")
    pool = dict(version=1, id="sw0", name="crew", owner_session_id=parent,
                created_ns=1234, desired_state="paused", host={"historical": True},
                channels=["general", "empty"], pods=[dict(id="pod-1", index=0,
                members=[dict(session_id=child, index=0, label="sw0p0a0")])])
    records.put("swarm", "pool", pool)
    message = dict(id="m1", pod_id="pod-1", channel="general", sender=child,
                   text="Historical bytes: café\n\u0000 literal /old/root/path")
    reference = records.put_immutable_json(message)
    records.put("boards/pod-1", "general", {"entries": [dict(id="m1", created_ns=10, message=reference)]})
    records.put("attempts", child, dict(session_id=child, instance_id="old-instance",
                pod_id="pod-1", state="unknown", host={"historical": True},
                request={"settings": {"state_root": "/old/state"}}, pid=123))
    records.put("operations", "operation-old", dict(action="bcast", state="pending", message="never replay"))
    archive = tmp_path / "swarm.zip"
    return SimpleNamespace(home=home, parent=parent, child=child, exact=exact, root=root,
                           records=records, pool=pool, message=message, reference=reference,
                           archive=archive, tmp=tmp_path)


def export(source, **kwargs):
    return transfer.export_swarm(project="source", swarm="crew", output=source.archive,
                                 home=source.home, revision_ids=source.exact,
                                 confirmed_stopped=True, **kwargs)


def imported(source, **kwargs):
    return transfer.import_swarm(source.archive, project="destination", home=source.tmp / "destination",
                                 confirmed_stopped=True, **kwargs)


def rewrite_archive(path, changes):
    with zipfile.ZipFile(path) as source:
        files = {name: source.read(name) for name in source.namelist()}
    changes(files)
    with zipfile.ZipFile(path, "w") as target:
        for name, data in files.items():
            target.writestr(name, data)


def rewrite_payload(path, change):
    def apply(files):
        payload = json.loads(files["swarm.json"])
        change(payload)
        files["swarm.json"] = transfer._json_bytes(payload)
        manifest = json.loads(files["manifest.json"])
        manifest["files"]["swarm.json"] = dict(size=len(files["swarm.json"]),
            sha256=hashlib.sha256(files["swarm.json"]).hexdigest())
        files["manifest.json"] = transfer._json_bytes(manifest)
    rewrite_archive(path, apply)


def test_real_journal_roundtrip_and_destination_revisions(source):
    exported = export(source)
    assert exported["revision_ids"] == source.exact
    destination = source.tmp / "destination"
    result = imported(source, workdir="/new/work", path_maps=[("/old/root", "/new/root")])
    root = destination / "projects" / "destination" / "swarms"
    pool_store = RecordStore(root / "sw0", create=False)
    pool = pool_store.get("swarm", "pool")
    commits = result["sessions"]["checkpoint_revisions"]
    assert set(commits) == {source.parent, source.child}
    assert any(commits[sid] != source.exact[sid] for sid in commits)
    assert pool["recovery"]["selected_checkpoints"] == {
        sid: dict(session_id=sid, revision_id=revision) for sid, revision in commits.items()}
    assert pool["owner"]["parent_instance_id"] == ""
    assert pool["owner"]["process_identity"] is None
    assert pool["host"] == {} and pool["phase"] == "paused"
    assert pool_store.get("members", source.child)["current_attempt_id"] is None
    assert pool_store.get("members", source.child)["shared_checkpoint"]["revision_id"] == commits[source.child]
    assert pool_store.get("attempts", source.child)["state"] == "fenced"
    assert pool_store.get("operations", "operation-old")["message"] == "never replay"
    raw_board = pool_store.get("boards/pod-1", "general", resolve_refs=False)
    assert raw_board["entries"][0]["message"] == source.reference
    assert pool_store.get("boards/pod-1", "general")["entries"][0]["message"] == source.message
    registry = RecordStore(root, create=False)
    assert registry.get("registry", "pools")["pools"]["sw0"]["owner"] == source.parent
    assert registry.get("transfer-imports", result["operation_id"])["phase"] == "completed"
    for sid in commits:
        state = SessionRepository(destination / "projects" / "destination" / "sessions" / sid).load(
            session_id=sid, commit_id=commits[sid])
        assert state.document["attrs"]["session_cwd"] == "/new/work"
        assert state.document["attrs"]["runner"]["workdir"] == "/new/root/jobs"
        assert state.document["attrs"]["marker"] == "selected"
    with transfer.selected_home(destination):
        catalog = {row["session_id"]: row for row in projects.list_sessions("destination")}
    assert catalog[source.child]["parent_session_id"] == source.parent
    assert os.environ[projects.HOME_ENV] == str(source.home)


def test_dry_runs_publish_no_resources(source):
    plan = export(source, dry_run=True)
    assert plan["dry_run"] and not source.archive.exists()
    export(source)
    plan = imported(source, dry_run=True)
    assert plan["dry_run"]
    root = source.tmp / "destination" / "projects" / "destination"
    assert not (root / "swarms").exists()
    assert not (root / "sessions" / source.child).exists()


def test_missing_child_checkpoint_fails_export(source):
    source.exact.pop(source.child)
    with pytest.raises(transfer.TransferError, match="every member"):
        export(source)
    assert not source.archive.exists()


def test_missing_child_in_bundle_fails_before_import(source):
    export(source)
    rewrite_payload(source.archive, lambda payload: payload["revision_ids"].pop(source.child))
    with pytest.raises(transfer.TransferError, match="every member|Incomplete"):
        imported(source)
    assert not (source.tmp / "destination" / "projects").exists()


def test_corrupt_immutable_reference_fails(source):
    immutable = source.records.records_root / "immutable_json"
    next(immutable.glob("*.json")).unlink()
    with pytest.raises((ValueError, RuntimeError), match="missing|corrupt"):
        export(source)
    assert not source.archive.exists()


def test_tampering_fails_before_session_import(source, monkeypatch):
    export(source)
    rewrite_archive(source.archive, lambda files: files.__setitem__("swarm.json", files["swarm.json"].replace(b"crew", b"evil")))
    monkeypatch.setattr(transfer, "import_sessions", lambda *a, **k: pytest.fail("import called before checksum check"))
    with pytest.raises(transfer.TransferError, match="Checksum"):
        imported(source)


def test_live_source_cannot_be_overridden(source):
    identity = capture_process_identity(os.getpid())
    assert identity is not None
    attempt = source.records.get("attempts", source.child)
    attempt["process_identity"] = identity.to_dict()
    source.records.put("attempts", source.child, attempt)
    with pytest.raises(transfer.TransferError, match="Known live source"):
        export(source)


def test_live_registry_cannot_be_overridden(source, monkeypatch):
    identity = capture_process_identity(os.getpid())
    registry_path = Path(projects.resolve_session_paths("source", source.parent)["store_root"])
    RecordStore(registry_path)
    record = SimpleNamespace(session_id=source.parent, instance_id="registry-live",
                             process_identity=identity, is_confirmed_stopped=lambda: False)
    monkeypatch.setattr(transfer, "list_live_sessions", lambda *a, **k: [record])
    with pytest.raises(transfer.TransferError, match="registry/registry-live"):
        export(source)


def test_attestation_required(source):
    with pytest.raises(transfer.TransferError, match="confirmed-stopped"):
        transfer.export_swarm(project="source", swarm="sw0", output=source.archive, home=source.home)
    export(source)
    with pytest.raises(transfer.TransferError, match="confirmed-stopped"):
        transfer.import_swarm(source.archive, project="destination", home=source.tmp / "destination")


def test_alias_collision_precedes_session_import(source, monkeypatch):
    export(source)
    root = source.tmp / "destination" / "projects" / "destination" / "swarms"
    registry = RecordStore(root)
    registry.put("registry", "pools", {"next": 2, "pools": {"sw1": {"name": "crew", "owner": str(uuid.uuid4())}}})
    monkeypatch.setattr(transfer, "import_sessions", lambda *a, **k: pytest.fail("collision checked too late"))
    with pytest.raises(transfer.TransferError, match="collision"):
        imported(source)
    assert not (root / "sw0").exists()


def test_session_collision_rejected(source):
    export(source)
    with transfer.selected_home(source.tmp / "destination"):
        checkpoint("destination", source.child)
    with pytest.raises(RuntimeError, match="collis|exist|contains"):
        imported(source)
    assert not (source.tmp / "destination" / "projects" / "destination" / "swarms").exists()


@pytest.mark.parametrize("policy", ["remap", "replace", "skip"])
def test_unsupported_policies_are_explicit(source, policy):
    with pytest.raises(transfer.TransferError, match="unsupported"):
        imported(source, on_conflict=policy)


def test_partial_publication_retains_journal_and_imported_revisions(source, monkeypatch):
    export(source)
    def fail(*args):
        raise OSError("injected pool publication failure")
    monkeypatch.setattr(transfer, "_publish_pool", fail)
    with pytest.raises(transfer.TransferError, match="reconcile transfer-imports"):
        imported(source)
    root = source.tmp / "destination" / "projects" / "destination" / "swarms"
    registry = RecordStore(root, create=False)
    progress = registry.list("transfer-imports")
    assert len(progress) == 1 and progress[0]["phase"] == "reconcile_required"
    assert progress[0]["failed_phase"] == "sessions_imported"
    assert set(progress[0]["session_result"]["checkpoint_revisions"]) == {source.parent, source.child}
    assert Path(progress[0]["staging"]).is_dir()
    assert registry.get("registry", "pools", {"pools": {}})["pools"] == {}
    assert not (root / "sw0").exists()
    for sid in (source.parent, source.child):
        assert (root.parent / "sessions" / sid).is_dir()


def test_snapshot_uses_runtime_gate(source, monkeypatch):
    original = transfer.snapshot_records
    def gated(records):
        with pytest.raises(StoreError, match="reentrant"):
            with file_lock(records.lock_path("swarm-gates", "mutation"), timeout=0):
                pass
        return original(records)
    monkeypatch.setattr(transfer, "snapshot_records", gated)
    export(source)


@pytest.mark.parametrize("filename", ["../escape", "/absolute", "a\\b", "a/../b", "a//b"])
def test_archive_traversal_rejected(source, filename):
    export(source)
    with zipfile.ZipFile(source.archive, "a") as archive:
        archive.writestr(filename, b"x")
    with pytest.raises(transfer.TransferError, match="Unsafe"):
        imported(source)


def test_duplicate_member_and_symlink_rejected(source):
    export(source)
    with zipfile.ZipFile(source.archive, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("swarm.json", b"{}")
    with pytest.raises(transfer.TransferError, match="duplicate"):
        imported(source)
    source.archive.unlink()
    export(source)
    def link(files):
        files["link"] = b"ignored"
    rewrite_archive(source.archive, link)
    with zipfile.ZipFile(source.archive, "a") as archive:
        info = zipfile.ZipInfo("symlink")
        info.create_system = 3
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, b"/tmp/unsafe")
    with pytest.raises(transfer.TransferError, match="Unsafe"):
        imported(source)


def test_source_symlink_rejected(source):
    (source.records.root / "unrelated-link").symlink_to(source.tmp / "missing")
    with pytest.raises(transfer.TransferError, match="Symlink"):
        export(source)


def test_missing_current_attempt_rejected(source):
    source.records.put("members", source.child, {"current_attempt_id": "missing"})
    with pytest.raises(transfer.TransferError, match="Missing current attempt"):
        export(source)


def test_cli_parser_and_help():
    with pytest.raises(SystemExit) as exit_info:
        transfer.main(["--help"])
    assert exit_info.value.code == 0


def test_import_strips_attempt_authority_preserves_original_evidence(source):
    attempt = source.records.get("attempts", source.child)
    attempt["current"] = dict(instance_id="source-runtime", pid=88, websocket_url="ws://127.0.0.1:999/session")
    source.records.put("attempts", source.child, attempt)
    export(source)
    result = imported(source)
    records = RecordStore(source.tmp / "destination" / "projects" / "destination" / "swarms" / "sw0", create=False)
    inert = records.get("attempts", source.child)
    assert not {"current", "ready", "request", "host", "pid", "process_identity"} & set(inert)
    assert inert["instance_id"] == "" and inert["state"] == "fenced"
    assert records.get(f"transfer-history/{result['operation_id']}/attempts", source.child) == attempt
    assert records.get("operations", "operation-old")["state"] == "historical"


def test_resource_collision_without_registry_binding(source):
    export(source)
    root = source.tmp / "destination" / "projects" / "destination" / "swarms"
    RecordStore(root)
    existing = dict(source.pool, id="sw7", name="other", version=2,
        resource_uid=transfer.resource_uid(source.pool), phase="paused",
        owner=dict(epoch=1, token=uuid.uuid4().hex, parent_instance_id="", host={}, process_identity=None))
    RecordStore(root / "sw7").put("swarm", "pool", existing)
    with pytest.raises(transfer.TransferError, match="authoritative swarm collides"):
        imported(source)


def test_cancelled_pool_stays_cancelled(source):
    source.pool["desired_state"] = "cancelled"
    source.records.put("swarm", "pool", source.pool)
    export(source)
    imported(source)
    pool = RecordStore(source.tmp / "destination" / "projects" / "destination" / "swarms" / "sw0", create=False).get("swarm", "pool")
    assert pool["desired_state"] == pool["phase"] == "cancelled"


def test_inner_archive_expansion_bound(source, monkeypatch):
    export(source)
    monkeypatch.setattr(transfer, "MAX_EXPANDED", 1)
    # Explicit _safe_zip argument avoids relying on a default captured at definition.
    with zipfile.ZipFile(source.archive) as archive:
        with pytest.raises(transfer.TransferError, match="size"):
            transfer._safe_zip(archive, maximum=1)


def test_version2_session_keyed_attempt_and_revision_watermarks(source):
    identity = capture_process_identity(os.getpid()).to_dict()
    identity["host_id"] = str(uuid.uuid4())  # foreign host is UNKNOWN, covered by attestation
    source.pool.update(version=2, resource_uid=uuid.uuid4().hex, phase="paused",
        owner=dict(epoch=8, token=uuid.uuid4().hex, parent_instance_id="old-parent",
                   host={}, process_identity=identity),
        shared_checkpoint=dict(session_id=source.parent, revision_id=source.exact[source.parent]))
    source.records.put("swarm", "pool", source.pool)
    attempt = source.records.get("attempts", source.child)
    attempt["attempt_id"] = uuid.uuid4().hex
    source.records.put("attempts", source.child, attempt)
    source.records.put("members", source.child, dict(current_attempt_id=attempt["attempt_id"],
        shared_checkpoint=dict(session_id=source.child, revision_id=source.exact[source.child])))
    result = transfer.export_swarm(project="source", swarm="sw0", output=source.archive,
                                   home=source.home, confirmed_stopped=True)
    assert result["revision_ids"] == source.exact
    assert any(row["state"] == "unknown" for row in result["observations"])
    imported(source)
    pool = RecordStore(source.tmp / "destination" / "projects" / "destination" / "swarms" / "sw0", create=False).get("swarm", "pool")
    assert pool["resource_uid"] == source.pool["resource_uid"]
    assert pool["owner"]["epoch"] == 9


def test_project_symlink_rejected(source):
    export(source)
    home = source.tmp / "destination"
    outside = source.tmp / "outside"
    outside.mkdir()
    (home / "projects").mkdir(parents=True)
    (home / "projects" / "destination").symlink_to(outside, target_is_directory=True)
    with pytest.raises(transfer.TransferError, match="Symlink"):
        imported(source)
    assert list(outside.iterdir()) == []



def test_exact_old_checkpoint_not_newer_head(source):
    for sid in (source.parent, source.child):
        newer = checkpoint("source", sid, parent=source.parent if sid == source.child else "", marker="newer")
        assert newer != source.exact[sid]
    export(source)
    result = imported(source)
    for sid, revision in result["sessions"]["checkpoint_revisions"].items():
        root = source.tmp / "destination" / "projects" / "destination" / "sessions" / sid
        saved = SessionRepository(root).load(session_id=sid, commit_id=revision)
        assert saved.document["attrs"]["marker"] == "selected"


def test_missing_child_in_inner_session_archive(source, monkeypatch):
    export(source)
    def remove_child(files):
        with zipfile.ZipFile(io.BytesIO(files["sessions.zip"])) as sessions:
            nested = {name: sessions.read(name) for name in sessions.namelist()}
        manifest = json.loads(nested["manifest.json"])
        manifest["sessions"] = [row for row in manifest["sessions"] if row["session_id"] != source.child]
        nested["manifest.json"] = transfer._json_bytes(manifest)
        result = io.BytesIO()
        with zipfile.ZipFile(result, "w") as archive:
            for name, data in nested.items():
                archive.writestr(name, data)
        files["sessions.zip"] = result.getvalue()
        wrapper = json.loads(files["manifest.json"])
        wrapper["files"]["sessions.zip"] = dict(size=len(files["sessions.zip"]),
            sha256=hashlib.sha256(files["sessions.zip"]).hexdigest())
        files["manifest.json"] = transfer._json_bytes(wrapper)
    rewrite_archive(source.archive, remove_child)
    monkeypatch.setattr(transfer, "import_sessions", lambda *a, **k: pytest.fail("missing child reached importer"))
    with pytest.raises(transfer.TransferError, match="required parent/member"):
        imported(source)


def test_registry_publication_failure_leaves_only_unbound_pool(source, monkeypatch):
    export(source)
    root = source.tmp / "destination" / "projects" / "destination" / "swarms"
    original = RecordStore.transaction
    @contextmanager
    def fail_binding(self, namespace, key, *args, **kwargs):
        if self.root == root and (namespace, key) == ("registry", "pools"):
            raise OSError("injected registry binding failure")
        with original(self, namespace, key, *args, **kwargs) as record:
            yield record
    monkeypatch.setattr(RecordStore, "transaction", fail_binding)
    with pytest.raises(transfer.TransferError, match="reconcile transfer-imports"):
        imported(source)
    registry = RecordStore(root, create=False)
    progress = registry.list("transfer-imports")[0]
    assert progress["phase"] == "reconcile_required"
    assert progress["failed_phase"] == "pool_published"
    pool = RecordStore(root / "sw0", create=False).get("swarm", "pool")
    assert pool["owner"]["parent_instance_id"] == "" and pool["owner"]["process_identity"] is None
    assert pool["transfer"]["requires_recovery"] is True
    assert not registry.get("registry", "pools", {"pools": {}})["pools"]


def test_explicit_coordination_store_roots(source):
    custom_source = source.tmp / "source-coordination"
    RecordStore(custom_source)
    export(source, store_root=custom_source)
    custom_destination = source.tmp / "destination-coordination"
    result = imported(source, store_root=custom_destination)
    assert result["sessions"]["store_root"] == str(custom_destination)
    assert (custom_destination / "format.json").is_file()
