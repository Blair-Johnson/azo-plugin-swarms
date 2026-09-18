"""The archive publication contract must be consumable by the real runtime."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from agent_zoo import projects
from tmux_pilot.fs_store import RecordStore

import swarm
from test_transfer import source, export, imported, checkpoint


@pytest.fixture
def destination(source, monkeypatch):
    export(source)
    result = imported(source, workdir="/new/work", path_maps=[("/old/root", "/new/root")])
    home = source.tmp / "destination"
    monkeypatch.setenv(projects.HOME_ENV, str(home))
    monkeypatch.setattr(swarm, "resolve_local_root", lambda: source.tmp / "destination-cache")
    state = SimpleNamespace(_session_id=source.parent, _instance_id="destination-parent",
        _agent_zoo_context={"project_name": "destination"}, swarm_access={"sw0": ""},
        swarm_resource_uids={}, grants={})
    store = swarm.store_for(state, "sw0")
    registry = RecordStore(home / "projects" / "destination" / "swarms")
    return SimpleNamespace(source=source, result=result, state=state, store=store, registry=registry)


def test_completed_import_is_valid_but_unbound(destination):
    d = destination
    pool = d.store.pool()
    assert pool["owner"]["process_identity"] is None
    d.store.grant = swarm.parent_grant(pool, "")
    with pytest.raises(ValueError, match="[Ii]nstance|[Uu]nbound"):
        with d.store.mutation():
            pytest.fail("Imported authority must not permit writes")


@pytest.mark.parametrize("phase", [None, "prepared", "sessions_importing", "sessions_imported",
                                   "pool_published", "reconcile_required"])
def test_incomplete_import_never_claims_runtime_authority(destination, phase):
    d = destination
    operation = d.result["operation_id"]
    marker = d.registry.get("transfer-imports", operation)
    marker["phase"] = phase
    d.registry.put("transfer-imports", operation, marker)
    before = deepcopy(d.store.records().get("swarm", "pool"))
    with pytest.raises(ValueError, match="[Ii]mport|[Tt]ransfer|publication"):
        swarm.claim_recovery(d.state, d.store, expected_epoch=before["owner"]["epoch"],
                             confirmed_stopped=True)
    assert d.store.records().get("swarm", "pool") == before
    assert d.state.grants == {}


def test_imported_checkpoint_pin_survives_takeover_and_newer_head(destination):
    d = destination
    sid = d.source.child
    exact = d.result["sessions"]["checkpoint_revisions"][sid]
    newer = checkpoint("destination", sid, parent=d.source.parent, marker="do not choose")
    assert newer != exact
    pool = swarm.claim_recovery(d.state, d.store, expected_epoch=d.store.pool()["owner"]["epoch"],
                                confirmed_stopped=True)
    member = pool["pods"][0]["members"][0]
    selected = swarm.pin_checkpoint(d.state, d.store, member)
    assert selected.revision_id == exact
    assert swarm.shared_checkpoint(d.state, sid, selected.revision_id).document["attrs"]["marker"] == "selected"
    assert d.store.records().get("members", sid)["recovery_status"] == "pending"
    assert pool["owner"]["parent_instance_id"] == d.state._instance_id


def test_unbound_owner_requires_explicit_transfer_metadata(destination):
    pool = destination.store.records().get("swarm", "pool")
    pool.pop("transfer")
    with pytest.raises(ValueError):
        swarm.validate_pool(pool)
