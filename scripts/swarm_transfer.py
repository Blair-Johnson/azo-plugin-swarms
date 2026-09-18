#!/usr/bin/env python3
"""Offline, identity-preserving swarm transfer (run in the Agent Zoo environment).

Export/import BOTH require --confirmed-stopped: the operator confirms the old
parent, children and detached external work are stopped/isolated and the source
copy will not run again. This is an attestation, not a distributed fence. A known
live source always blocks export/import. Import never launches or resumes work.

Only collision policy 'error' is supported. Archives contain sensitive session
and historical launch data; checksums detect corruption, not malicious authors.
Public functions are synchronous; explicit home selection temporarily changes the
process environment and must not be used concurrently with runtime code.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile

from agent_zoo import projects
from agent_zoo.session_archive import export_sessions, import_sessions
from agent_zoo.live_session_registry import list_live_sessions
from tmux_pilot.atomic_publication import rename_exclusive
from tmux_pilot.fs_store import RecordStore, file_lock
from tmux_pilot.process_identity import ObservationState, ProcessIdentity, observe_process

FORMAT = "azo-swarm-transfer-v1"
MAX_PAYLOAD = 64 * 1024 * 1024
MAX_ARCHIVE = 1024 * 1024 * 1024
MAX_EXPANDED = 2 * MAX_ARCHIVE
MAX_FILES = 100000
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
ATTESTATION = ("Source parent, all children and detached external work are stopped or "
               "isolated; the source copy is retired and will not run simultaneously.")


class TransferError(ValueError):
    """Invalid, unsafe or incomplete transfer; retain reported recovery evidence."""


def _name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise TransferError(f"Invalid name: {value!r}")
    return value


def _sid(value):
    if not isinstance(value, str):
        raise TransferError("Session ID must be a UUID")
    try:
        if str(uuid.UUID(value)) != value or not uuid.UUID(value).int:
            raise ValueError(value)
    except ValueError as exc:
        raise TransferError(f"Session ID must be a canonical nonzero UUID: {value!r}") from exc
    return value


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TransferError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              TransferError(f"Non-finite JSON value: {value}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TransferError(f"Invalid JSON: {exc}") from exc


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_tree(root):
    for directory, _, files in os.walk(root, topdown=False):
        for filename in files:
            with (Path(directory) / filename).open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(directory)


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def selected_home(home=None):
    old = os.environ.get(projects.HOME_ENV)
    if home is not None:
        os.environ[projects.HOME_ENV] = str(Path(home).expanduser().resolve())
    try:
        yield projects.resolve_state_root()
    finally:
        if old is None:
            os.environ.pop(projects.HOME_ENV, None)
        else:
            os.environ[projects.HOME_ENV] = old


def validate_pool(pool):
    """Validate portable topology without importing the runtime plugin."""
    if not isinstance(pool, dict) or type(pool.get("version")) is not int or pool["version"] not in {1, 2}:
        raise TransferError("Missing or unsupported swarm pool")
    _name(pool.get("id")); _name(pool.get("name")); _sid(pool.get("owner_session_id"))
    channels, pods = pool.get("channels"), pool.get("pods")
    if (not isinstance(channels, list) or not 1 <= len(channels) <= 32
            or not all(isinstance(c, str) for c in channels) or len(set(channels)) != len(channels)
            or not isinstance(pods, list) or not 1 <= len(pods) <= 32):
        raise TransferError("Invalid swarm topology")
    for channel in channels:
        _name(channel)
    ids, pod_ids = {pool["owner_session_id"]}, set()
    for index, pod in enumerate(pods):
        if not isinstance(pod, dict) or type(pod.get("index")) is not int or pod["index"] != index:
            raise TransferError("Invalid pod index")
        pod_id = _name(pod.get("id"))
        if pod_id in pod_ids:
            raise TransferError("Duplicate pod")
        pod_ids.add(pod_id)
        members = pod.get("members")
        if not isinstance(members, list) or not 1 <= len(members) <= 64:
            raise TransferError("Invalid members")
        for ordinal, member in enumerate(members):
            if not isinstance(member, dict) or type(member.get("index")) is not int or member["index"] != ordinal:
                raise TransferError("Invalid member index")
            sid = _sid(member.get("session_id")); _name(member.get("label"))
            if sid in ids:
                raise TransferError("Duplicate owner/member session identity")
            ids.add(sid)
    if pool.get("desired_state") not in {"paused", "running", "mixed", "cancelled"}:
        raise TransferError("Invalid desired state")
    if pool["version"] == 2:
        owner = pool.get("owner")
        if (not isinstance(owner, dict) or type(owner.get("epoch")) is not int or owner["epoch"] < 1
                or not isinstance(owner.get("parent_instance_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", str(owner.get("token", "")))
                or not re.fullmatch(r"[0-9a-f]{32}", str(pool.get("resource_uid", "")))
                or pool.get("phase") not in {"recovering", "paused", "running", "mixed", "cancelled"}):
            raise TransferError("Invalid version-2 authority")
        if owner.get("process_identity") is not None:
            ProcessIdentity.from_dict(owner["process_identity"])
        elif owner["parent_instance_id"]:
            raise TransferError("Bound owner is missing process identity")
    return pool


def session_ids(pool):
    return [pool["owner_session_id"]] + [m["session_id"] for p in pool["pods"] for m in p["members"]]


def resource_uid(pool):
    return pool.get("resource_uid") or uuid.uuid5(
        uuid.NAMESPACE_OID, f"azo-swarm:{pool['owner_session_id']}:{pool['id']}:{pool.get('created_ns', '')}").hex


def _regular_tree(root):
    """Reject links/special files before any traversal/copy of a source tree."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise TransferError(f"Not a real store directory: {root}")
    count = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise TransferError(f"Symlink or special file in store: {path}")
            count += 1
            if count > MAX_FILES:
                raise TransferError("Store contains too many files")


def snapshot_records(records):
    """Caller holds the swarm mutation gate. Return raw logical values, not caches."""
    _regular_tree(records.root)
    if any(path.is_file() for path in records.pending_root.rglob("*")):
        raise TransferError("Unresolved RecordStore pending operation; reconcile before transfer")
    namespaces = set()
    total = 0
    for path in records.records_root.rglob("*"):
        if path.is_file():
            if path.suffix != ".json":
                raise TransferError(f"Unrecognized authoritative record file: {path}")
            namespace = path.parent.relative_to(records.records_root).as_posix()
            if namespace == ".":
                raise TransferError("Record has no namespace")
            namespaces.add(namespace)
            total += path.stat().st_size
            if total > MAX_PAYLOAD:
                raise TransferError("Swarm records exceed transfer size limit")
    rows = []
    for namespace in sorted(namespaces):
        for key, value in records.items(namespace, resolve_refs=False):
            records.resolve_value_refs(value)  # missing/corrupt references fail closed
            rows.append(dict(namespace=namespace, key=key, value=value))
    validate_records(rows)
    return rows


def validate_records(rows):
    if not isinstance(rows, list) or not rows or len(rows) > MAX_FILES:
        raise TransferError("Invalid record inventory")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"namespace", "key", "value"}:
            raise TransferError("Malformed record inventory entry")
        namespace, key = row["namespace"], row["key"]
        if (not isinstance(namespace, str) or not namespace or "\\" in namespace
                or any(part in {"", ".", ".."} for part in namespace.split("/"))
                or not isinstance(key, str) or "\x00" in key):
            raise TransferError("Invalid record namespace/key")
        identity = namespace, key
        if identity in indexed:
            raise TransferError("Duplicate record")
        indexed[identity] = row["value"]
    pool = validate_pool(indexed.get(("swarm", "pool")))
    members = set(session_ids(pool)[1:])
    boards = {(f"boards/{pod['id']}", channel) for pod in pool["pods"] for channel in pool["channels"]}
    for (namespace, key), value in indexed.items():
        if namespace.startswith("boards/"):
            if (namespace, key) not in boards or not isinstance(value, dict) or not isinstance(value.get("entries"), list):
                raise TransferError("Malformed or unknown board")
            seen = set()
            for entry in value["entries"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("message"), dict):
                    raise TransferError("Malformed board entry")
                message_id = _name(entry.get("id"))
                if message_id in seen or type(entry.get("created_ns")) is not int:
                    raise TransferError("Duplicate or malformed board entry")
                seen.add(message_id)
        if namespace == "members":
            if key not in members or not isinstance(value, dict):
                raise TransferError("Malformed member record")
            attempt = value.get("current_attempt_id")
            if attempt:
                target = indexed.get(("attempts", attempt))
                if target is None:
                    target = indexed.get(("attempts", key))
                    if not isinstance(target, dict) or target.get("attempt_id") != attempt:
                        raise TransferError("Missing current attempt record")
                if not isinstance(target, dict) or target.get("session_id") != key:
                    raise TransferError("Current attempt belongs to another member")
        if namespace == "attempts":
            if not isinstance(value, dict) or value.get("session_id") not in members:
                raise TransferError("Malformed attempt record")
        if namespace == "operations" and not isinstance(value, dict):
            raise TransferError("Malformed operation record")
    return pool


def restore_records(rows, root):
    """Recreate a private store via public APIs; never extract record paths."""
    validate_records(rows)
    records = RecordStore(root)
    for row in rows:
        if row["namespace"] == "immutable_json":
            blob = row["value"]
            if not isinstance(blob, dict) or blob.get("sha256") != row["key"] or "value" not in blob:
                raise TransferError("Invalid immutable blob")
            reference = records.put_immutable_json(blob["value"])
            actual = records.get("immutable_json", row["key"], resolve_refs=False)
            if actual != blob or not reference:
                raise TransferError("Immutable blob hash/size mismatch")
    for row in rows:
        if row["namespace"] != "immutable_json":
            records.put(row["namespace"], row["key"], row["value"])
    for row in rows:
        records.get(row["namespace"], row["key"])  # validate all references
    pool = validate_pool(records.get("swarm", "pool"))
    for pod in pool["pods"]:
        for channel in pool["channels"]:
            board = records.get(f"boards/{pod['id']}", channel, {"entries": []})
            for entry in board["entries"]:
                message = entry["message"]
                if (not isinstance(message, dict) or message.get("id") != entry["id"]
                        or message.get("pod_id") != pod["id"] or message.get("channel") != channel
                        or not isinstance(message.get("sender"), str) or not message["sender"]
                        or not isinstance(message.get("text"), str)):
                    raise TransferError("Board message does not match its immutable entry")
    return records


def _observations(rows, project, ids, store_root=None, *, include_registry=True):
    observations = []

    def observe(raw, subject):
        identity = ProcessIdentity.from_dict(raw)
        observation = observe_process(identity)
        observations.append(dict(subject=subject, **observation.to_dict()))
        if observation.state == ObservationState.ALIVE:
            raise TransferError(f"Known live source process: {subject}; attestation cannot override it")

    def visit(value, subject):
        if isinstance(value, dict):
            identity = value.get("process_identity")
            if identity is not None:
                observe(identity, subject)
            for key, child in value.items():
                if key != "process_identity":
                    visit(child, f"{subject}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{subject}/{index}")

    for row in rows:
        if row["namespace"] != "immutable_json" and not row["namespace"].startswith("boards/"):
            visit(row["value"], f"{row['namespace']}/{row['key']}")
    if include_registry:
        root = Path(store_root).expanduser() if store_root is not None else Path(
            projects.resolve_session_paths(project, ids[0])["store_root"])
        if root.is_symlink():
            raise TransferError("Coordination store must not be a symlink")
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            registry = RecordStore(root, create=False)
            for record in list_live_sessions(registry, include_stale=True):
                if record.session_id in ids:
                    if record.process_identity:
                        observe(record.process_identity.to_dict(), f"registry/{record.instance_id}")
                    elif not record.is_confirmed_stopped():
                        observations.append(dict(subject=f"registry/{record.instance_id}",
                                                 state="unknown", reason="missing_process_identity"))
    return observations


def select_revisions(pool, rows, revision_ids=None):
    indexed = {(r["namespace"], r["key"]): r["value"] for r in rows}
    selected = dict(revision_ids or {})
    required = set(session_ids(pool))
    if set(selected) - required:
        raise TransferError("Revision supplied for session outside swarm topology")
    saved = (pool.get("recovery") or {}).get("selected_checkpoints", {})
    for sid in required:
        ref = indexed.get(("members", sid), {}).get("shared_checkpoint") or saved.get(sid)
        if sid == pool["owner_session_id"]:
            ref = pool.get("shared_checkpoint") or (pool.get("owner") or {}).get("shared_checkpoint") or ref
        if sid not in selected and isinstance(ref, dict) and ref.get("session_id") == sid:
            selected[sid] = ref.get("revision_id")
    if set(selected) != required or any(not isinstance(v, str) or not v for v in selected.values()):
        raise TransferError("Exact checkpoint revision required for parent AND every member; use --revision SID=COMMIT")
    return selected


def _swarm_root(state_root, project):
    root = state_root / "projects" / project / "swarms"
    for path in (state_root / "projects", root.parent, root):
        if path.is_symlink():
            raise TransferError(f"Symlink in project state path: {path}")
    return root


def _registry(root):
    if root.is_symlink():
        raise TransferError("Swarm registry must not be a symlink")
    return RecordStore(root, create=False) if root.exists() else None


def _index(registry):
    index = registry.get("registry", "pools", {"next": 0, "pools": {}}) if registry else {"next": 0, "pools": {}}
    if (not isinstance(index, dict) or not isinstance(index.get("pools"), dict)
            or type(index.get("next", 0)) is not int):
        raise TransferError("Corrupt swarm registry")
    for alias, row in index["pools"].items():
        _name(alias)
        if not isinstance(row, dict):
            raise TransferError("Corrupt swarm registry entry")
        _name(row.get("name")); _sid(row.get("owner"))
    return index


def _resolve_source(root, swarm):
    _name(swarm)
    registry = _registry(root)
    if registry is None:
        raise TransferError("Missing source swarm registry")
    index = _index(registry)
    matches = [key for key, row in index["pools"].items() if swarm in {key, row["name"]}]
    if len(matches) != 1:
        raise TransferError("Source alias must resolve to exactly one registered swarm")
    target = root / matches[0]
    if target.is_symlink():
        raise TransferError("Swarm directory must not be a symlink")
    records = RecordStore(target, create=False)
    return records, index["pools"][matches[0]]


def _require_attestation(confirmed_stopped):
    if confirmed_stopped is not True:
        raise TransferError("--confirmed-stopped is required: " + ATTESTATION)


def export_swarm(*, project, swarm, output, home=None, store_root=None,
                 confirmed_stopped=False, revision_ids=None, dry_run=False):
    """Export exactly one quiescent swarm and pinned parent/member checkpoints."""
    _name(project); _require_attestation(confirmed_stopped)
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise TransferError(f"Output already exists: {output}")
    with selected_home(home) as state_root:
        root = _swarm_root(state_root, project)
        records, registry_row = _resolve_source(root, swarm)
        # EXACTLY the runtime mutation gate; never a separate transfer-only snapshot lock.
        with file_lock(records.lock_path("swarm-gates", "mutation")):
            rows = snapshot_records(records)
            pool = validate_records(rows)
            if (pool["id"] != records.root.name or registry_row["owner"] != pool["owner_session_id"]
                    or registry_row["name"] != pool["name"]
                    or registry_row.get("resource_uid", resource_uid(pool)) != resource_uid(pool)):
                raise TransferError("Registry and authoritative pool disagree")
            ids = session_ids(pool)
            revisions = select_revisions(pool, rows, revision_ids)
            observations = _observations(rows, project, ids, store_root)
            with tempfile.TemporaryDirectory(prefix="azo-swarm-export-") as tmp:
                temporary = Path(tmp)
                restore_records(rows, temporary / "validation")
                session_path = temporary / "sessions.zip"
                plan = export_sessions(project=project, queries=ids, output=session_path,
                                       revision_ids=revisions, database="state", store_root=store_root,
                                       dry_run=dry_run)
                if set(plan["sessions"]) != set(ids) or plan["revision_ids"] != revisions:
                    raise TransferError("Session exporter did not include the exact checkpoint set")
                result = dict(project=project, swarm=pool["id"], sessions=ids, revision_ids=revisions,
                              resource_uid=resource_uid(pool), output=str(output), dry_run=dry_run,
                              attestation=ATTESTATION, observations=observations)
                if dry_run:
                    return result
                payload = dict(format=FORMAT, source_project=project, resource_uid=resource_uid(pool),
                               records=rows, revision_ids=revisions,
                               attestation=dict(confirmed_stopped=True, statement=ATTESTATION,
                                                recorded_ns=time.time_ns(), observations=observations))
                swarm_bytes = _json_bytes(payload)
                if len(swarm_bytes) > MAX_PAYLOAD or session_path.stat().st_size > MAX_ARCHIVE:
                    raise TransferError("Transfer payload exceeds size limit")
                manifest = dict(format=FORMAT, files={
                    "swarm.json": dict(size=len(swarm_bytes), sha256=hashlib.sha256(swarm_bytes).hexdigest()),
                    "sessions.zip": dict(size=session_path.stat().st_size, sha256=_sha(session_path))})
                output.parent.mkdir(parents=True, exist_ok=True)
                fd, staged = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
                os.close(fd)
                staged = Path(staged)
                try:
                    with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr("manifest.json", _json_bytes(manifest))
                        archive.writestr("swarm.json", swarm_bytes)
                        archive.write(session_path, "sessions.zip")
                    with staged.open("rb") as stream:
                        os.fsync(stream.fileno())
                    rename_exclusive(staged, output)
                    _fsync_directory(output.parent)
                except BaseException as exc:
                    raise TransferError(f"Archive publication failed/uncertain; inspect {output} and {staged}: {exc}") from exc
                result["sha256"] = _sha(output)
                return result


def _safe_zip(archive, *, maximum=MAX_EXPANDED):
    infos = archive.infolist()
    if len(infos) > MAX_FILES:
        raise TransferError("Too many archive members")
    seen, total = set(), 0
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (not info.filename or "\\" in info.filename or "\x00" in info.orig_filename
                or path.is_absolute() or any(p in {"", ".", ".."} for p in info.filename.split("/"))
                or ":" in info.filename or info.filename in seen or info.is_dir()
                or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG})):
            raise TransferError(f"Unsafe/duplicate archive member: {info.filename!r}")
        if info.flag_bits & 1:
            raise TransferError("Encrypted archive entries are not supported")
        seen.add(info.filename)
        total += info.file_size
        if info.file_size > maximum or total > maximum:
            raise TransferError("Archive expanded size exceeds limit")
    return {info.filename: info for info in infos}


def read_bundle(archive, staging):
    """Validate then write ONLY fixed private filenames; never call extractall."""
    archive, staging = Path(archive), Path(staging)
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE + MAX_PAYLOAD:
        raise TransferError("Invalid or oversized wrapper archive")
    with zipfile.ZipFile(archive) as source:
        inventory = _safe_zip(source, maximum=MAX_ARCHIVE + MAX_PAYLOAD)
        if set(inventory) != {"manifest.json", "swarm.json", "sessions.zip"}:
            raise TransferError("Wrapper must contain exactly manifest.json, swarm.json and sessions.zip")
        if inventory["manifest.json"].file_size > 65536 or inventory["swarm.json"].file_size > MAX_PAYLOAD:
            raise TransferError("Oversized transfer metadata")
        manifest = _json(source.read("manifest.json"))
        if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
                or set(manifest.get("files", {})) != {"swarm.json", "sessions.zip"}):
            raise TransferError("Invalid transfer manifest")
        staging.mkdir(parents=True, exist_ok=True, mode=0o700)
        for filename in ("swarm.json", "sessions.zip"):
            detail = manifest["files"][filename]
            if not isinstance(detail, dict) or detail.get("size") != inventory[filename].file_size:
                raise TransferError("Archive inventory size mismatch")
            target = staging / filename
            digest = hashlib.sha256()
            with source.open(filename) as reader, target.open("xb") as writer:
                os.chmod(target, 0o600)
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(block); digest.update(block)
                writer.flush(); os.fsync(writer.fileno())
            if digest.hexdigest() != detail.get("sha256"):
                raise TransferError(f"Checksum mismatch: {filename}")
    payload = _json((staging / "swarm.json").read_bytes())
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise TransferError("Invalid swarm payload")
    pool = validate_records(payload.get("records"))
    if payload.get("resource_uid") != resource_uid(pool):
        raise TransferError("Resource identity mismatch")
    _name(payload.get("source_project"))
    attestation = payload.get("attestation")
    if not isinstance(attestation, dict) or attestation.get("confirmed_stopped") is not True:
        raise TransferError("Missing source quiescence attestation")
    revisions = select_revisions(pool, payload["records"], payload.get("revision_ids"))
    if payload.get("revision_ids") != revisions:
        raise TransferError("Incomplete exact checkpoint manifest")
    with zipfile.ZipFile(staging / "sessions.zip") as sessions:
        inventory = _safe_zip(sessions)
        if "manifest.json" not in inventory or inventory["manifest.json"].file_size > MAX_PAYLOAD:
            raise TransferError("Missing/oversized session manifest")
        session_manifest = _json(sessions.read("manifest.json"))
        actual = session_manifest.get("sessions", [])
        if (not isinstance(actual, list) or len(actual) != len(revisions)
                or any(not isinstance(row, dict) for row in actual)
                or {row.get("session_id"): row.get("revision_id") for row in actual} != revisions):
            raise TransferError("Session archive omits or changes a required parent/member checkpoint")
    restore_records(payload["records"], staging / "pool")
    return payload


def _alias_collision(index, pool):
    wanted = {pool["id"], pool["name"]}
    for alias, row in index["pools"].items():
        if wanted & {alias, row["name"]} or row.get("resource_uid") == resource_uid(pool):
            raise TransferError(f"Existing swarm alias/resource collision: {alias}")


def check_destination(root, pool):
    registry = _registry(root)
    index = _index(registry)
    _alias_collision(index, pool)
    if (root / pool["id"]).exists() or (root / pool["id"]).is_symlink():
        raise TransferError("Destination swarm path already exists")
    # Check authoritative resources too: registry publication can have failed earlier.
    candidates = set(index["pools"])
    if root.exists():
        candidates.update(p.name for p in root.iterdir()
                          if NAME.fullmatch(p.name) and p.name not in {"records", "locks", "pending"}
                          and (p / "format.json").exists())
    for alias in candidates:
        path = root / _name(alias)
        if path.is_symlink():
            raise TransferError("Destination swarm is a symlink")
        existing = validate_pool(RecordStore(path, create=False).get("swarm", "pool"))
        if resource_uid(existing) == resource_uid(pool) or {existing["id"], existing["name"]} & {pool["id"], pool["name"]}:
            raise TransferError("Destination authoritative swarm collides")
    return registry


def _normalize_import(records, payload, result, operation_id, project):
    source_pool = validate_pool(records.get("swarm", "pool"))
    ids = session_ids(source_pool)
    if (result.get("id_map") != {sid: sid for sid in ids} or set(result.get("imported", [])) != set(ids)
            or result.get("skipped") or result.get("replaced") or result.get("remapped")):
        raise TransferError("Importer did not preserve the complete session identity set")
    revisions = result.get("checkpoint_revisions", {})
    if set(revisions) != set(ids) or any(not isinstance(v, str) or not v for v in revisions.values()):
        raise TransferError("Importer did not return every destination checkpoint revision")
    pool = deepcopy(source_pool)
    pool.update(version=2, resource_uid=resource_uid(source_pool), host={},
                owner=dict(epoch=(source_pool.get("owner") or {}).get("epoch", 0) + 1,
                           token=uuid.uuid4().hex, parent_instance_id="", host={}, process_identity=None),
                phase="cancelled" if source_pool["desired_state"] == "cancelled" else "paused",
                desired_state="cancelled" if source_pool["desired_state"] == "cancelled" else "paused")
    pool["pod_states"] = {pod["id"]: pool["desired_state"] for pod in pool["pods"]}
    pool["recovery"] = dict(phase="imported", operation_id=operation_id, parent_instance_id="",
                            selected_checkpoints={sid: dict(session_id=sid, revision_id=revision)
                                                  for sid, revision in revisions.items()})
    pool["transfer"] = dict(version=1, operation_id=operation_id, source_project=payload["source_project"],
                            destination_project=project, source_resource_uid=payload["resource_uid"],
                            source_revision_ids=payload["revision_ids"], checkpoint_revisions=revisions,
                            source_retired_attestation=True, requires_recovery=True)
    records.put("transfer-history/pool", operation_id, source_pool)
    for key, attempt in records.items("attempts", resolve_refs=False):
        records.put(f"transfer-history/{operation_id}/attempts", key, attempt)
        records.put("attempts", key, dict(session_id=attempt["session_id"],
            attempt_id=attempt.get("attempt_id", key), pod_id=attempt.get("pod_id", ""),
            instance_id="", epoch=0, state="fenced", historical_only=True,
            transfer_operation_id=operation_id))
    for key, operation in records.items("operations", resolve_refs=False):
        records.put(f"transfer-history/{operation_id}/operations", key, operation)
        records.put("operations", key, dict(operation, state="historical", historical_only=True))
    for sid in ids[1:]:
        old_member = records.get("members", sid)
        if old_member is not None:
            records.put(f"transfer-history/{operation_id}/members", sid, old_member)
        records.put("members", sid, dict(session_id=sid, current_attempt_id=None,
                    shared_checkpoint=dict(session_id=sid, revision_id=revisions[sid], instance_id=""),
                    recovery_status="pending"))
    pool["shared_checkpoint"] = dict(session_id=ids[0], revision_id=revisions[ids[0]], instance_id="")
    records.put("swarm", "pool", pool)
    validate_pool(pool)
    return pool


def _publish_pool(staged, target):
    rename_exclusive(staged, target)
    _fsync_directory(target.parent)


def import_swarm(archive, *, project, home=None, store_root=None, workdir=None,
                 path_maps=(), confirmed_stopped=False, dry_run=False, on_conflict="error"):
    """Import a complete, unbound pool; failures retain explicit reconciliation evidence."""
    _name(project); _require_attestation(confirmed_stopped)
    if on_conflict != "error":
        raise TransferError("Swarm transfer supports ONLY on_conflict='error'; remap/skip/replace are unsupported")
    archive = Path(archive).expanduser().absolute()
    with selected_home(home) as state_root, tempfile.TemporaryDirectory(prefix="azo-swarm-validate-") as tmp:
        checked = Path(tmp)
        payload = read_bundle(archive, checked)
        pool = validate_records(payload["records"])
        _observations(payload["records"], payload["source_project"], session_ids(pool), include_registry=False)
        root = _swarm_root(state_root, project)
        check_destination(root, pool)
        kwargs = dict(project=project, on_conflict="error", path_maps=path_maps,
                      workdir=workdir, store_root=store_root)
        plan = import_sessions(checked / "sessions.zip", dry_run=True, **kwargs)
        ids = session_ids(pool)
        if plan.get("id_map") != {sid: sid for sid in ids} or set(plan.get("imported", [])) != set(ids):
            raise TransferError("Session preflight did not preserve complete swarm identities")
        result = dict(project=project, swarm=pool["id"], resource_uid=resource_uid(pool),
                      dry_run=dry_run, sessions=plan, attestation=ATTESTATION,
                      warning="Imported pool is unbound. Recover paused explicitly; never restart the source copy.")
        if dry_run:
            return result
        # All payload and real-importer checks passed before any resource publication.
        registry = RecordStore(root) if not root.exists() else RecordStore.open_existing_writable(root)
        with file_lock(registry.lock_path("swarm-gates", "transfer")):
            check_destination(root, pool)
            operation_id = uuid.uuid4().hex
            staging = root.parent / f".swarm-transfer-{operation_id}"
            shutil.copytree(checked, staging)
            _fsync_tree(staging)
            _fsync_directory(staging.parent)
            marker = dict(version=1, operation_id=operation_id, phase="prepared", archive=str(archive),
                          archive_sha256=_sha(archive), staging=str(staging), target=str(root / pool["id"]),
                          source_project=payload["source_project"], project=project,
                          source_session_ids=ids, source_revision_ids=payload["revision_ids"],
                          attestation=ATTESTATION, created_ns=time.time_ns())
            registry.put("transfer-imports", operation_id, marker)
            try:
                marker["phase"] = "sessions_importing"
                registry.put("transfer-imports", operation_id, marker)
                session_result = import_sessions(staging / "sessions.zip", **kwargs)
                marker.update(phase="sessions_imported", session_result=session_result)
                registry.put("transfer-imports", operation_id, marker)
                staged_records = RecordStore.open_existing_writable(staging / "pool")
                imported_pool = _normalize_import(staged_records, payload, session_result, operation_id, project)
                check_destination(root, imported_pool)
                _publish_pool(staging / "pool", root / imported_pool["id"])
                marker["phase"] = "pool_published"
                registry.put("transfer-imports", operation_id, marker)
                # No external effects inside this pure registry read/modify/write.
                with registry.transaction("registry", "pools", default={"next": 0, "pools": {}}) as index:
                    _alias_collision(index, imported_pool)
                    index["pools"][imported_pool["id"]] = dict(name=imported_pool["name"],
                        owner=imported_pool["owner_session_id"], resource_uid=imported_pool["resource_uid"])
                marker["phase"] = "completed"
                registry.put("transfer-imports", operation_id, marker)
                result.update(sessions=session_result, operation_id=operation_id, progress_store=str(root),
                              retained_staging=str(staging))
                return result
            except BaseException as exc:
                previous = marker["phase"]
                marker.update(phase="reconcile_required", failed_phase=previous, error=str(exc))
                try:
                    registry.put("transfer-imports", operation_id, marker)
                except BaseException:
                    pass  # earlier durable marker and staging remain; never delete imported sessions
                raise TransferError(f"Transfer failed/uncertain; reconcile transfer-imports/{operation_id} "
                                    f"in {root}; staging retained at {staging}: {exc}") from exc


def _pair(value):
    first, equal, second = value.partition("=")
    if not equal or not first or not second:
        raise argparse.ArgumentTypeError("Expected OLD=NEW (or SESSION_ID=REVISION)")
    return first, second


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export", help="Export a stopped swarm and exact checkpoints")
    importing = commands.add_parser("import", help="Import without launching; source must remain retired")
    for command in (exporting, importing):
        command.add_argument("--project", required=True)
        command.add_argument("--home", type=Path)
        command.add_argument("--store-root", type=Path)
        command.add_argument("--confirmed-stopped", action="store_true", help=ATTESTATION)
        command.add_argument("--dry-run", action="store_true")
    exporting.add_argument("--swarm", required=True)
    exporting.add_argument("--output", required=True, type=Path)
    exporting.add_argument("--revision", action="append", type=_pair, default=[], metavar="SID=COMMIT")
    importing.add_argument("archive", type=Path)
    importing.add_argument("--workdir", type=Path)
    importing.add_argument("--path-map", action="append", type=_pair, default=[], metavar="OLD=NEW")
    importing.add_argument("--on-conflict", default="error", choices=["error"],
                           help="Only error is supported; IDs and board authorship are preserved")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "export":
            pairs = args.pop("revision")
            if len(dict(pairs)) != len(pairs):
                raise TransferError("Duplicate --revision session")
            result = export_swarm(revision_ids=dict(pairs), **args)
        else:
            args["path_maps"] = args.pop("path_map")
            result = import_swarm(**args)
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"swarm-transfer: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
