"""Storage-only checks: no model, network, or runtime-agent launch."""
import multiprocessing
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import swarm
from swarm import SwarmStore


@pytest.fixture
def store(tmp_path):
    result = SwarmStore(tmp_path / "shared", tmp_path / "host-a" / "messages.sqlite3")
    result.create("demo", pods=2, agents_per_pod=2, boards=2)
    return result


def test_create_topology_without_launch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Creating swarm metadata must not launch a subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = SwarmStore(tmp_path / "shared", tmp_path / "local" / "cache.sqlite3")
    pool = result.create("demo", pods=4, agents_per_pod=4, boards=2)
    assert pool["id"] == "demo"
    assert pool["desired_state"] == "paused"
    assert pool["channels"] == ["general", "channel-2"]
    assert [pod["id"] for pod in pool["pods"]] == [f"pod-{i}" for i in range(1, 5)]
    members = [member["session_id"] for pod in pool["pods"] for member in pod["members"]]
    assert all(len(pod["members"]) == 4 for pod in pool["pods"])
    assert len(set(members)) == 16
    assert all(swarm.name(member) == member for member in members)
    assert result.pool() == pool
    assert result.attempts() == []
    with pytest.raises(ValueError, match="already exists"):
        result.create("demo")
    assert result.pool() == pool


def test_append_preserves_immutable_references_and_detached_reads(store):
    store.post("pod-1", "general", "first", "sender", message_id="one")
    original = store.records().get("boards/pod-1", "general", resolve_refs=False)
    reference = original["entries"][0]["message"]
    assert reference != {"text": "first"}
    assert "text" not in reference
    first = store.messages("pod-1", "general")
    store.post("pod-1", "general", "second", "sender", message_id="two")
    raw = store.records().get("boards/pod-1", "general", resolve_refs=False)
    assert raw["entries"][:1] == original["entries"]
    assert store.records().resolve_value_refs(reference) == first[0]["message"]
    rows = store.messages("pod-1", "general")
    assert rows[:1] == first
    assert [row["id"] for row in rows] == ["one", "two"]
    rows[0]["message"]["text"] = "local mutation"
    assert store.messages("pod-1", "general")[0]["message"]["text"] == "first"


def test_retry_is_idempotent_but_conflicting_id_fails(store):
    assert store.post("pod-1", "general", "first", "sender", message_id="stable") == "stable"
    before = store.messages("pod-1", "general")
    for _ in range(2):
        assert store.post("pod-1", "general", "first", "sender", message_id="stable") == "stable"
    for text, sender in [("changed", "sender"), ("first", "someone-else")]:
        with pytest.raises(ValueError, match="different content"):
            store.post("pod-1", "general", text, sender, message_id="stable")
    assert store.messages("pod-1", "general") == before


def test_same_named_boards_and_ids_are_isolated_per_pod_and_channel(store):
    for pod, channel, text in [
        ("pod-1", "general", "one"),
        ("pod-2", "general", "two"),
        ("pod-1", "channel-2", "other board"),
    ]:
        store.post(pod, channel, text, "sender", message_id="same-id")
        rows = store.messages(pod, channel)
        assert len(rows) == 1
        assert rows[0]["message"] == dict(
            id="same-id", pod_id=pod, channel=channel, sender="sender", text=text)
    assert store.messages("pod-1", "general")[0]["message"]["text"] == "one"
    assert store.messages("pod-2", "channel-2") == []


def _write_messages(durable, cache, worker, barrier):
    """Spawn target only writes records; it never invokes the runtime launcher."""
    child = SwarmStore(Path(durable), Path(cache))
    barrier.wait(timeout=30)
    for number in range(8):
        identity = f"worker-{worker}-{number}"
        child.post("pod-1", "general", identity, f"writer-{worker}", message_id=identity)


def test_concurrent_process_writers_lose_no_messages(store, tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    processes = [context.Process(target=_write_messages, args=(
        str(store.durable), str(tmp_path / f"writer-{worker}" / "cache.sqlite3"), worker, barrier,
    )) for worker in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
        assert all(not process.is_alive() for process in processes), "writer timed out"
        assert [process.exitcode for process in processes] == [0] * 4
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    rows = store.messages("pod-1", "general")
    expected = {f"worker-{worker}-{number}" for worker in range(4) for number in range(8)}
    assert len(rows) == len(expected)
    assert {row["id"] for row in rows} == expected
    assert {row["message"]["text"] for row in rows} == expected


def test_deleted_cache_and_second_host_rebuild_identical_data(store, tmp_path):
    store.post("pod-1", "general", "durable", "sender", message_id="kept")
    metadata, rows = store.pool(), store.messages("pod-1", "general")
    assert store.cache.exists()
    shutil.rmtree(store.cache.parent)
    reopened = SwarmStore(store.durable, store.cache)
    assert reopened.pool() == metadata
    assert reopened.messages("pod-1", "general") == rows
    assert reopened.cache.exists()
    other_host = SwarmStore(store.durable, tmp_path / "host-b" / "messages.sqlite3")
    assert other_host.cache != reopened.cache
    assert other_host.pool() == metadata
    assert other_host.messages("pod-1", "general") == rows
    other_host.post("pod-1", "general", "new", "other-host", message_id="new")
    assert reopened.messages("pod-1", "general") == other_host.messages("pod-1", "general")
    assert not list(store.durable.rglob("*.sqlite*"))


def test_corrupt_local_cache_returns_authoritative_data(store):
    store.post("pod-1", "general", "first", "sender", message_id="one")
    store.messages("pod-1", "general")
    store.cache.write_bytes(b"not a SQLite database\x00" * 100)
    store.post("pod-1", "general", "second", "sender", message_id="two")
    authoritative = store.records().get("boards/pod-1", "general")["entries"]
    assert store.messages("pod-1", "general") == authoritative
    assert [row["id"] for row in authoritative] == ["one", "two"]


def test_missing_durable_root_never_serves_stale_cache_or_recreates(store, tmp_path):
    store.post("pod-1", "general", "cached", "sender", message_id="old")
    store.messages("pod-1", "general")
    store.durable.rename(tmp_path / "disconnected-shared")
    reopened = SwarmStore(store.durable, store.cache)
    for candidate in (store, reopened):
        for operation in (
            candidate.pool,
            lambda: candidate.messages("pod-1", "general"),
            lambda: candidate.post("pod-1", "general", "new", "sender", message_id="new"),
        ):
            with pytest.raises((OSError, RuntimeError, ValueError)):
                operation()
            assert not store.durable.exists()
    assert store.cache.exists()


@pytest.mark.parametrize("publication", ["blob", "board"])
def test_precommit_publication_failure_is_not_acknowledged(store, monkeypatch, publication):
    original = swarm.RecordStore._publish_record

    def fail_publication(self, path, key, value, data=None):
        is_board = isinstance(value, dict) and "entries" in value
        if is_board == (publication == "board"):
            raise OSError("injected precommit publication failure")
        return original(self, path, key, value, data)

    state = SimpleNamespace(swarm_access={"demo": ""}, _session_id="parent")
    monkeypatch.setattr(swarm, "store_for", lambda state, swarm_id: store)
    with monkeypatch.context() as fault:
        fault.setattr(swarm.RecordStore, "_publish_record", fail_publication)
        with pytest.raises(RuntimeError, match="not acknowledged"):
            swarm.post_message("demo", "pod-1", "general", "retry me", "stable", state)
    assert store.messages("pod-1", "general") == []
    assert "Posted stable" in swarm.post_message(
        "demo", "pod-1", "general", "retry me", "stable", state)
    assert len(store.messages("pod-1", "general")) == 1


@pytest.mark.parametrize("status", ["reserved", "unknown", "spawned"])
def test_persisted_resume_attempt_blocks_new_reservation(store, tmp_path, status):
    member = store.pool()["pods"][0]["members"][0]["session_id"]
    attempt = dict(session_id=member, instance_id="attempt-1", state="reserved",
                   request={"source": {"kind": "resume", "session_id": member}})
    store.reserve_attempt(member, attempt)
    store.update_attempt(member, state=status)
    reopened = SwarmStore(store.durable, tmp_path / "host-b" / "cache.sqlite3")
    expected = dict(attempt, state=status)
    assert reopened.attempts() == [expected]
    with pytest.raises(ValueError, match="reconciliation"):
        reopened.reserve_attempt(member, dict(attempt, instance_id="attempt-2"))
    assert reopened.attempts() == [expected]


@pytest.mark.parametrize("field,bad", [
    ("pods", 0), ("pods", 33), ("pods", True), ("pods", 1.5),
    ("agents_per_pod", 0), ("agents_per_pod", 65), ("agents_per_pod", False),
    ("boards", 0), ("boards", 33), ("boards", True), ("boards", "2"),
])
def test_invalid_counts_do_not_create_storage(tmp_path, field, bad):
    target = SwarmStore(tmp_path / "shared", tmp_path / "cache.sqlite3")
    with pytest.raises(ValueError):
        target.create("demo", **{field: bad})
    assert not target.durable.exists()


@pytest.mark.parametrize("bad", ["", "../escape", "/absolute", "a/b", "space name", "x" * 65, None])
def test_invalid_swarm_names_do_not_create_storage(tmp_path, bad):
    target = SwarmStore(tmp_path / "shared", tmp_path / "cache.sqlite3")
    with pytest.raises(ValueError):
        target.create(bad)
    assert not target.durable.exists()


@pytest.mark.parametrize("pod,channel,identity", [
    ("../pod", "general", "okay"), ("pod-99", "general", "okay"),
    ("pod-1", "../board", "okay"), ("pod-1", "missing", "okay"),
    ("pod-1", "general", "../message"), ("pod-1", "general", "x" * 65),
])
def test_invalid_post_names_leave_board_empty(store, pod, channel, identity):
    with pytest.raises(ValueError):
        store.post(pod, channel, "hello", "sender", message_id=identity)
    assert store.messages("pod-1", "general") == []


@pytest.mark.parametrize("text", [None, 7, "", " \n\t", "x" * 65537, "é" * 32769],
                         ids=["none", "integer", "empty", "blank", "ascii-too-large", "utf8-too-large"])
def test_invalid_message_text_is_rejected(store, text):
    with pytest.raises(ValueError):
        store.post("pod-1", "general", text, "sender")
    assert store.messages("pod-1", "general") == []


def test_utf8_byte_limit_and_required_sender(store):
    with pytest.raises(ValueError, match="Sender"):
        store.post("pod-1", "general", "hello", "")
    text = "é" * 32768
    store.post("pod-1", "general", text, "sender", message_id="boundary")
    assert store.messages("pod-1", "general")[0]["message"]["text"] == text


def test_cache_cannot_be_inside_shared_root(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        SwarmStore(tmp_path / "shared", tmp_path / "shared" / "cache.sqlite3")


def test_ambiguous_committed_post_can_be_retried_without_duplication(store, monkeypatch):
    original = swarm.RecordStore._publish_record

    def commit_then_fail(self, path, key, value, data=None):
        original(self, path, key, value, data)
        if isinstance(value, dict) and "entries" in value:
            raise OSError("reply lost after durable board publication")

    state = SimpleNamespace(swarm_access={"demo": ""}, _session_id="parent")
    monkeypatch.setattr(swarm, "store_for", lambda state, swarm_id: store)
    with monkeypatch.context() as fault:
        fault.setattr(swarm.RecordStore, "_publish_record", commit_then_fail)
        with pytest.raises(RuntimeError, match="not acknowledged"):
            swarm.post_message("demo", "pod-1", "general", "committed", "stable", state)
    assert len(store.messages("pod-1", "general")) == 1
    assert "Posted stable" in swarm.post_message(
        "demo", "pod-1", "general", "committed", "stable", state)
    assert len(store.messages("pod-1", "general")) == 1
