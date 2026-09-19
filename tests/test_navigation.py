"""Lazy navigation and all-message projections over real durable swarm journals."""
from copy import deepcopy
from dataclasses import replace
import json
import uuid

import pytest
from agent_utils.files.buffer_manager import BufferManager
from agent_utils.session_repository import SessionRepository
import swarm
from test_recovery import runtime, seed


@pytest.fixture
def navigation(runtime):
    runtime.state.buffer_manager = BufferManager()
    swarm.SwarmBuffers()(runtime.state)
    return runtime


def read(runtime, identifier):
    return runtime.state.buffer_manager.resolve_for_read(runtime.state, identifier)


def forbidden(*args, **kwargs):
    raise AssertionError("directory/authorization must not load content")


def checkpoint(runtime, entries, *, previous=None):
    """Extend seed's actual canonical journal rather than substituting a reader."""
    previous = previous or seed(runtime, runtime.members[0])
    document = deepcopy(previous.document)
    document["entries"] = entries
    repository = SessionRepository(previous.repository)
    writer = repository.create_instance(session_id=previous.session_id,
        instance_id=uuid.uuid4().hex, origin_host_id="host-a", parent_commit_id=previous.commit_id)
    commit = writer.commit(document)
    return repository.load(session_id=previous.session_id, commit_id=commit.commit_id)


def cancel(runtime):
    with runtime.store.mutation() as (records, pool):
        pool.update(desired_state="cancelled", phase="cancelled")
        records.put("swarm", "pool", pool)


def peer(runtime, *, scope=""):
    runtime.state._session_id = runtime.members[0]["session_id"]
    runtime.state._instance_id = "peer-instance"
    runtime.state.grants = {}
    runtime.state.swarm_access = {"sw0": scope}


def test_indexes_are_directories_without_loading_content(navigation, monkeypatch):
    r = navigation
    seed(r, r.members[0])
    r.store.records(write=True).put("operations", "private-op", {
        "id": "private-op", "action": "bcast", "message": "private operation body", "created_ns": 1})
    monkeypatch.setattr(swarm, "shared_checkpoint", forbidden)
    monkeypatch.setattr(SessionRepository, "load", forbidden)
    monkeypatch.setattr(swarm.RecordStore, "list", forbidden)
    monkeypatch.setattr(swarm.SwarmStore, "messages", forbidden)
    for identifier in ("swarm:index", "swarm:sw0:index"):
        view = read(r, identifier)
        assert view.readonly
        assert "swarm:sw0p0a0" in view.text and "swarm:sw0p1a0" in view.text
        assert "swarm:sw0:pod-1:board:general" in view.text
        assert "swarm:sw0:operations" in view.text
        for value in (r.pool["resource_uid"], *(m["session_id"] for m in r.members),
                      "private operation body", "Recovery restores", "not evidence", "Operation:"):
            assert value not in view.text
    assert r.state.buffer_manager.buffers() == []


def test_short_alias_is_lazy_and_refreshes_saved_journal(navigation, monkeypatch):
    r = navigation
    calls = []
    original = swarm.shared_checkpoint
    def tracked(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(swarm, "shared_checkpoint", tracked)
    read(r, "swarm:index")
    assert calls == []
    assert "No shared checkpoint" in read(r, "swarm:sw0p0a0").text
    first = checkpoint(r, [{"index": 3, "messages": [{"role": "user", "content": "first saved"}]}])
    initial = read(r, "swarm:sw0p0a0")
    assert "first saved" in initial.text and first.commit_id in initial.text
    second = checkpoint(r, [{"index": 4, "messages": [{"role": "user", "content": "second saved"}]}],
                        previous=first)
    latest = read(r, "swarm:sw0p0a0")
    assert "second saved" in latest.text and "second saved" not in initial.text
    assert second.commit_id in latest.text and first.commit_id not in latest.text
    long_id = f"swarm:sw0:pod-1:session:{r.members[0]['session_id']}"
    assert read(r, long_id).text == latest.text
    assert calls == [r.members[0]["session_id"]] * 4
    assert r.state.buffer_manager.buffers() == []


def test_all_messages_tools_and_turn_chronology_from_shared_journal(navigation):
    r = navigation
    raw_call = {"id": "call-1", "type": "function", "function": {
        "name": "read_file", "arguments": '{"path":"notes.txt","line":2}'}}
    entries = [
        {"index": 0, "messages": [{"role": "system", "content": "system preamble"}]},
        {"index": 10, "messages": [{"role": "user", "content": "older real request"}]},
        {"index": 11, "messages": [{"role": "assistant", "channel": "commentary",
          "content": [{"type": "text", "text": "older commentary"}], "tool_calls": [raw_call]},
          {"role": "tool", "tool_call_id": "call-1", "content": "older tool result"}],
         "tool_calls": [{"id": "call-1", "name": "read_file", "raw_args": raw_call["function"]["arguments"],
                         "result": "older tool result"}]},
        {"index": 12, "system_generated": True, "messages": [
            {"role": "user", "content": "generated interruption"}]},
        {"index": 13, "messages": [{"role": "developer", "content": "developer instruction"},
            {"role": "system", "content": "mid-turn system instruction"},
            {"role": "assistant", "content": "older final"}]},
        {"index": 20, "messages": [{"role": "user", "content": "newer real request"},
            {"role": "assistant", "channel": "commentary", "content": "newer commentary"},
            {"role": "assistant", "channel": "final", "content": "newer final"}]},
    ]
    saved = checkpoint(r, entries)
    text = read(r, "swarm:sw0p0a0").text
    assert saved.commit_id in text and str(saved.repository) in text and saved.instance_id in text
    assert "Shared saved transcript, not a live stream." in text
    assert "Source Format: journal" in text and "not journal file lines" in text
    assert "### ASSISTANT (commentary) [entry 11; message 1]" in text
    assert "### USER (system-generated) [entry 12; message 1]" in text
    assert "### TOOL RESULT [entry 11; message 2; call call-1]" in text
    ordered = ["newer real request", "newer commentary", "newer final", "older real request",
               "older commentary", "Tool call: read_file", '"path": "notes.txt"', "older tool result",
               "generated interruption", "developer instruction", "mid-turn system instruction", "older final",
               "system preamble"]
    assert [text.index(value) for value in ordered] == sorted(text.index(value) for value in ordered)
    assert text.count("older tool result") == 1 and text.count("Tool call: read_file") == 1
    assert text.count("## Turn [") == 2


def test_provider_blocks_native_calls_and_opaque_metadata(navigation):
    r = navigation
    entries = [{"index": 1, "messages": [{"role": "user", "content": "inspect all blocks"}]},
        {"index": 2, "messages": [{"role": "assistant", "content": [
            {"type": "text", "text": "readable content", "signature": "SECRET-TEXT-SIGNATURE"},
            {"type": "thinking", "thinking": "readable thinking", "signature": "SECRET-SIGNATURE"},
            {"type": "redacted_thinking", "data": "SECRET-REDACTED"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "readable summary"}],
             "encrypted_content": "SECRET-ENCRYPTED"},
            {"type": "image", "source": {"data": "SECRET-IMAGE"}},
            {"type": "tool_use", "id": "block-call", "name": "lookup", "input": {"query": "search terms"}}],
            "reasoning_content": "plaintext reasoning",
            "_responses_reasoning_items": [{"item": {"encrypted_content": "SECRET-RESPONSES"}}],
            "provider_metadata": {"payload": "SECRET-PROVIDER"},
            "provider_tool_calls": [{"id": "provider-call", "function": {
                "name": "provider_lookup", "arguments": '{"needle":"provider args"}'}}]},
            {"role": "user", "system_generated": True, "content": [
                {"type": "tool_result", "tool_use_id": "block-call", "content": [
                    {"type": "text", "text": "block result"}]}]},
            {"role": "tool", "tool_call_id": "provider-call", "content": {"text": "structured result", "rows": [1, 2]},
             "encrypted_content": "SECRET-RESULT-METADATA"}]},
        {"index": 3, "messages": [{"role": "assistant", "content": "native call checkpoint"}],
         "tool_calls": [{"id": "native-call", "name": "native_tool", "raw_args": '{"x":true}',
                         "result": "native durable result", "error": True}]},
        {"index": 4, "messages": [{"type": "function_call", "name": "direct_tool",
                                    "call_id": "direct-call", "arguments": '{"n":4}'},
                                   {"type": "function_call_output", "call_id": "direct-call",
                                    "output": "direct result"}]},
    ]
    checkpoint(r, entries)
    text = read(r, "swarm:sw0p0a0").text
    for readable in ("readable content", "readable thinking", "readable summary", "plaintext reasoning",
                     "lookup", "search terms", "block result", "provider_lookup", "provider args",
                     '"rows": [', "native_tool", '"x": true', "native durable result",
                     "TOOL RESULT (error)", "direct_tool", '"n": 4', "direct result"):
        assert readable in text
    for omitted in ("SECRET-", "encrypted_content", "_responses_reasoning_items", "provider_metadata",
                    '"signature"', "redacted_thinking"):
        assert omitted not in text
    assert text.count("## Turn [") == 1


def test_nested_entries_projection_adapter_and_top_level_precedence(navigation, monkeypatch):
    # Current SessionRepository validates top-level entries. This projection-only
    # adapter test covers state.entries without claiming that schema can be committed.
    r = navigation
    saved = seed(r, r.members[0])
    entries = [{"index": "bad-index", "messages": [{"role": "system", "content": "nested text"}]}]
    document = {"state": {"entries": entries}}
    monkeypatch.setattr(swarm, "shared_checkpoint", lambda *args: replace(saved, document=document))
    text = read(r, "swarm:sw0p0a0").text
    assert "nested text" in text and "SYSTEM [entry 0; message 1]" in text
    document["entries"] = []
    assert "nested text" not in read(r, "swarm:sw0p0a0").text
    assert "No saved messages" in read(r, "swarm:sw0p0a0").text


def test_cancelled_global_index_collapses_but_history_remains_readable(navigation, monkeypatch):
    r = navigation
    seed(r, r.members[0])
    r.store.post("pod-1", "general", "archived board", "parent")
    cancel(r)
    with monkeypatch.context() as guard:
        guard.setattr(swarm, "shared_checkpoint", forbidden)
        guard.setattr(swarm.RecordStore, "list", forbidden)
        guard.setattr(swarm.SwarmStore, "messages", forbidden)
        index = read(r, "swarm:index").text
        assert "cancelled (archived): swarm:sw0:index" in index
        assert "pod-1" not in index and "swarm:sw0p0a0" not in index
        history = read(r, "swarm:sw0:index").text
        assert "swarm:sw0p0a0" in history and "swarm:sw0:pod-1:board:general" in history
    assert "saved durable text" in read(r, "swarm:sw0p0a0").text
    assert "archived board" in read(r, "swarm:sw0:pod-1:board:general").text


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("persisted_scope", ["", "pod-2"])
def test_worker_scope_on_short_aliases_history_and_operations(navigation, monkeypatch, cancelled, persisted_scope):
    r = navigation
    seed(r, r.members[0])
    if cancelled:
        cancel(r)
    peer(r, scope=persisted_scope)
    for identifier in ("swarm:index", "swarm:sw0:index"):
        text = read(r, identifier).text
        assert "swarm:sw0p1a0" not in text and "pod-2" not in text
        assert "operations" not in text
    assert "saved durable text" in read(r, "swarm:sw0p0a0").text
    monkeypatch.setattr(swarm, "shared_checkpoint", forbidden)
    monkeypatch.setattr(swarm.RecordStore, "list", forbidden)
    for identifier in ("swarm:sw0p1a0", "swarm:sw0:pod-2:board:general",
                       f"swarm:sw0:pod-2:session:{r.members[1]['session_id']}"):
        with pytest.raises(ValueError, match="scope"):
            read(r, identifier)
    with pytest.raises(ValueError, match="parent"):
        read(r, "swarm:sw0:operations")


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("forgery", ["owner", "uid"])
def test_every_path_revalidates_durable_authority(navigation, monkeypatch, cancelled, forgery):
    r = navigation
    if cancelled:
        cancel(r)
    if forgery == "owner":
        r.state._session_id = "unrelated"
        r.state.grants = {}
        error = "neither owner"
    else:
        r.state.swarm_resource_uids = {"sw0": "f" * 32}
        error = "resource UID"
    monkeypatch.setattr(swarm, "shared_checkpoint", forbidden)
    monkeypatch.setattr(swarm.RecordStore, "list", forbidden)
    for identifier in ("swarm:index", "swarm:sw0:index", "swarm:sw0:operations", "swarm:sw0p0a0",
                       "swarm:sw0:pod-1:board:general",
                       f"swarm:sw0:pod-1:session:{r.members[0]['session_id']}"):
        with pytest.raises(ValueError, match=error):
            read(r, identifier)


def test_alias_requires_exact_durable_label_and_membership(navigation, monkeypatch):
    r = navigation
    monkeypatch.setattr(swarm, "shared_checkpoint", forbidden)
    for identifier in ("swarm:sw0p00a0", "swarm:sw0p0a00", "swarm:sw0p0a99", "swarm:sw0p99a0"):
        with pytest.raises(KeyError):
            read(r, identifier)
    with pytest.raises(ValueError, match="not bound"):
        read(r, "swarm:sw99p0a0")
    with pytest.raises(KeyError):
        read(r, f"swarm:sw0:pod-1:session:{r.members[1]['session_id']}")
    with r.store.mutation() as (records, pool):
        pool["pods"][0]["members"][0]["label"] = "different-label"
        records.put("swarm", "pool", pool)
    with pytest.raises(KeyError):
        read(r, "swarm:sw0p0a0")


def test_operations_are_separate_parent_only_saved_records(navigation, monkeypatch):
    r = navigation
    for ordinal in (1, 2):
        r.store.records(write=True).put("operations", f"operation-{ordinal}", {
            "id": f"operation-{ordinal}", "created_ns": ordinal,
            "action": "bcast", "message": f"operation details {ordinal}"})
    monkeypatch.setattr(swarm, "shared_checkpoint", forbidden)
    text = read(r, "swarm:sw0:operations").text
    assert text.index("operation details 2") < text.index("operation details 1")
    assert "operation details" not in read(r, "swarm:index").text
    cancel(r)
    assert "operation details 1" in read(r, "swarm:sw0:operations").text
    peer(r)
    with pytest.raises(ValueError, match="parent"):
        read(r, "swarm:sw0:operations")


def test_same_pod_teammate_alias_is_readable_without_other_pod_access(navigation):
    r = navigation
    _, pool = swarm.allocate_pool(r.state, swarm.parse_command("-n4 -p2"))
    own, teammate = pool["pods"][0]["members"]
    other = pool["pods"][1]["members"][0]
    saved = seed(r, teammate)
    r.state._session_id = own["session_id"]
    r.state._instance_id = "peer-instance"
    r.state.grants = {}
    r.state.swarm_access = {pool["id"]: ""}  # forged parent scope must not widen access
    short_id = f"swarm:{teammate['label']}"
    long_id = f"swarm:{pool['id']}:pod-1:session:{teammate['session_id']}"
    assert saved.commit_id in read(r, short_id).text
    assert read(r, short_id).text == read(r, long_id).text
    assert short_id in read(r, f"swarm:{pool['id']}:index").text
    with pytest.raises(ValueError, match="scope"):
        read(r, f"swarm:{other['label']}")


def test_tool_payloads_are_not_filtered_as_provider_metadata(navigation):
    arguments = {"signature": "actual argument", "_private": "private field",
                 "usage": {"encrypted_content": "literal user value"}}
    checkpoint(navigation, [{"index": 1, "messages": [{"role": "assistant",
        "tool_calls": [{"id": "args", "type": "function", "function": {
            "name": "inspect", "arguments": json.dumps(arguments)}}],
        "provider_metadata": {"signature": "opaque envelope"}},
        {"role": "tool", "tool_call_id": "args", "content": arguments}]}])
    text = read(navigation, "swarm:sw0p0a0").text
    for value in ("actual argument", "private field", "literal user value"):
        assert text.count(value) == 2
    assert "opaque envelope" not in text
