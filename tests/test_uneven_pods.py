"""Uneven pod sizes preserve every agent through allocation and transfer."""
from copy import deepcopy
import uuid

import pytest
import swarm
from test_recovery import runtime
from test_transfer import source, checkpoint, export, imported, transfer


@pytest.mark.parametrize("agents,pods,sizes", [
    (16, 3, [6, 5, 5]), (3, 2, [2, 1]), (7, 4, [2, 2, 2, 1]),
    (16, 4, [4, 4, 4, 4]), (5, 5, [1] * 5), (64, 1, [64]),
    (191, 3, [64, 64, 63]),
])
def test_balanced_allocation_keeps_every_agent(runtime, agents, pods, sizes):
    runtime.state._session_id = str(uuid.uuid4())
    request = swarm.parse_command(f"-n{agents} -p{pods}")
    store, pool = swarm.allocate_pool(runtime.state, request)
    assert [len(p["members"]) for p in pool["pods"]] == sizes
    members = [m for p in pool["pods"] for m in p["members"]]
    assert len({m["session_id"] for m in members}) == agents
    assert swarm.validate_pool(store.pool()) == pool
    assert transfer.validate_pool(pool) == pool
    for i, pod in enumerate(pool["pods"]):
        assert [m["index"] for m in pod["members"]] == list(range(sizes[i]))
        assert [m["label"] for m in pod["members"]] == [f"{pool['id']}p{i}a{j}" for j in range(sizes[i])]
        selected = swarm.control_members(pool, {"action": "bcast", "pods": [i]})
        assert selected == pod["members"]
    assert len(swarm.control_members(pool, {"action": "bcast"})) == agents


@pytest.mark.parametrize("raw", ["-n1 -p2", "-n0 -p1", "-n193 -p3", "-n33 -p33"])
def test_invalid_nonempty_pod_limits(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("agents", [0, 1, 129, True, 3.5, "3"])
def test_store_rejects_invalid_total_without_writes(tmp_path, agents):
    store = swarm.SwarmStore(tmp_path / "durable", tmp_path / "cache.sqlite")
    with pytest.raises(ValueError):
        store.create("sw0", pods=2, agents=agents,
                     owner_session_id="parent", owner_instance_id="instance")
    assert not store.durable.exists()


def test_uneven_topology_survives_transfer(source):
    pool = deepcopy(source.pool)
    for pod_index, agent_index in [(0, 1), (1, 0)]:
        sid = str(uuid.uuid4())
        source.exact[sid] = checkpoint("source", sid, parent=source.parent)
        if pod_index == len(pool["pods"]):
            pool["pods"].append(dict(id="pod-2", index=1, members=[]))
        pool["pods"][pod_index]["members"].append(dict(
            session_id=sid, index=agent_index, label=f"sw0p{pod_index}a{agent_index}"))
        source.records.put("attempts", sid, dict(session_id=sid, instance_id=f"old-{sid}",
            pod_id=pool["pods"][pod_index]["id"], state="unknown", host={}))
    source.records.put("swarm", "pool", pool)
    export(source)
    result = imported(source)
    root = source.tmp / "destination" / "projects" / "destination" / "swarms" / "sw0"
    restored = swarm.RecordStore(root, create=False).get("swarm", "pool")
    assert restored["pods"] == pool["pods"]
    assert [len(p["members"]) for p in restored["pods"]] == [2, 1]
    assert len(result["sessions"]["checkpoint_revisions"]) == 4
