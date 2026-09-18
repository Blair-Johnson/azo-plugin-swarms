"""Core/AU/plugin startup boundaries; isolated journals, no providers or processes."""
import asyncio
from concurrent.futures import Future
from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent_utils import Session
from agent_utils.components import HarnessEventHandler
from agent_utils.serialization import capture_session_document
from agent_utils.session_repository import SessionRepository
from agent_utils.types import Entry, ToolCall
from agent_zoo.runtime import live_backend as live
from agent_zoo.runtime.wire_live import _UnavailableRunDB
from agent_zoo.session_load import LoadedSession
import swarm
from test_recovery import runtime, reserve


class WorkCounter:
    def __init__(self, work):
        self.work = work

    def __call__(self, state):
        self.work.append('pipeline')
        if any(e.tool_calls for e in state.entries):
            self.work.append('historical-tools')
        return state


class BeforeTransport(HarnessEventHandler):
    event_kinds = {'runtime.session_ready'}

    def __init__(self, seen):
        self.seen = seen

    def handle_harness_event(self, event, state):
        assert not getattr(state, '_session_websocket_server', None)
        self.seen.append(dict(event.payload))


@contextmanager
def context(r, monkeypatch, *, sid, iid, checkpoint=None):
    """Real live context and Session, with only provider/transport setup substituted."""
    monkeypatch.chdir(r.tmp)
    work, seen, controllers = [], [], []
    def build(session, **kwargs):
        c = swarm.SwarmController(kwargs.get('config', {}), session)
        controllers.append(c)
        return [WorkCounter(work), swarm.SwarmBuffers(), BeforeTransport(seen),
                swarm.SwarmRuntimeReady(c), swarm.SwarmCompletionCheck(c)]
    monkeypatch.setattr(live, 'build_session_pipeline', build)
    monkeypatch.setattr(live, 'ensure_enabled_llm_default', lambda _: 'fake')
    for name in ('format_model_display', 'resolve_model_id'):
        monkeypatch.setattr(live, name, lambda *a, **k: 'fake')
    monkeypatch.setattr(live, 'resolve_model_context_limit', lambda *a, **k: 10000)
    def provider(*a, **k):
        work.append('provider')
        raise AssertionError('provider must not run')
    monkeypatch.setattr(live, 'make_llm_fn', lambda _: provider)
    import agents.llm
    monkeypatch.setattr(agents.llm, 'make_llm_fn', lambda _: provider)
    supervisor = Mock()
    supervisor.submit.side_effect = provider
    monkeypatch.setattr('agents.llm_subprocess.LLMProcessSupervisor', Mock(return_value=supervisor))
    monkeypatch.setattr(live.LiveBackendClient, 'start', lambda self: None)
    monkeypatch.setattr(live.SessionWebSocketServer, 'start_background', lambda self: 'ws://127.0.0.1:1234/session')
    monkeypatch.setattr(live.SessionWebSocketServer, 'stop_background', lambda self: True)
    # Execute the actual async startup coroutine; do not mock plugin startup or AU poll.
    def submit(self, factory):
        future = Future()
        try:
            future.set_result(asyncio.run(factory()))
        except Exception as exc:
            future.set_exception(exc)
        return future
    monkeypatch.setattr(live.SessionWebSocketServer, 'submit_background', submit)
    params = dict(config={'llm': {'default': 'fake', 'models': {'fake': {
        'provider': 'openai', 'model': 'fake', 'enabled': True}}},
        'default': {'system_prompt': 'SYS'}, 'pilot': {'scheduler': 'local'}},
        session_name=sid, window='test', restore=checkpoint is not None,
        user_cwd=str(r.tmp), session_id=sid, instance_id=iid,
        parent_session_id='' if sid == 'parent' else 'parent', project_name='test',
        session_paths={'state_file': str(r.tmp / 'state' / 'projects' / 'test' / 'sessions' / sid / 'session.json')},
        local_state_root=str(r.tmp / 'local'), state_root=str(r.tmp / 'state'),
        store_root=str(r.tmp / 'registry'), terminal_backend='headless',
        startup_mode='normal' if checkpoint else 'paused', startup_checkpoint=checkpoint,
        startup_system_message='boundary guidance', first_user_message='')
    try:
        with live.live_backend_context(_UnavailableRunDB(sid), params) as ctx:
            yield SimpleNamespace(ctx=ctx, work=work, seen=seen, controller=controllers[0])
    finally:
        supervisor.submit.assert_not_called()


def test_real_fresh_paused_member_bootstrap_publishes_exact_shared_seed(runtime, monkeypatch):
    r = runtime
    member = r.members[0]
    attempt = reserve(r, member, state='reserved', identity=r.identity.to_dict())
    monkeypatch.setenv('AZO_SWARM_ID', 'sw0')
    monkeypatch.setenv('AZO_SWARM_POD_ID', r.store.member_pod(member['session_id']))
    with context(r, monkeypatch, sid=member['session_id'], iid=attempt['instance_id']) as b:
        state = b.ctx.session.state
        assert b.work == []
        assert len(b.seen) == 1 and b.seen[0]['paused'] is True
        revision = b.seen[0]['revision_id']
        assert revision
        assert swarm.member_resource(b.controller.snapshot(state)) is not None, state._agent_zoo_context
        b.ctx.client.backend.run_iteration()  # actual AU check-only interrupt polling
        assert b.work == []
        assert not b.controller.pending and not b.controller.ready_events
        row = r.store.records().get('attempts', member['session_id'])
        assert row.get('bootstrap', {}).get('revision_id') == revision, state.pending_interrupts
        assert row['bootstrap']['paused'] is True
        shared = swarm.shared_checkpoint(b.controller.snapshot(state), member['session_id'], revision)
        assert shared.commit_id == revision
        assert b.controller.grants['sw0']['instance_id'] == attempt['instance_id']
        assert b.ctx.client.backend.startup_paused
        assert state.pending_interrupts  # completion retained, not injected/delivered while paused


def parent_checkpoint(r):
    session = Session(system_prompt='saved parent')
    session.ensure_initialized()
    session.state._session_id = 'parent'
    session.state._instance_id = 'old-parent'
    session.inject('historical turn')
    call = ToolCall('old-call', 'never_execute', '{}', {})
    session.state.entries.append(Entry(messages=[{'role': 'assistant', 'content': '',
        'tool_calls': [{'id': 'old-call', 'type': 'function', 'function': {
            'name': 'never_execute', 'arguments': '{}'}}]}],
        index=len(session.state.entries), step=0, tool_calls=[call]))
    session.state.pending_tool_calls = [call]
    session.state.last_tool_calls = [call]
    session.state._await_requested = True
    repo = SessionRepository(r.tmp / 'parent-source')
    writer = repo.create_instance(session_id='parent', instance_id='old-parent', origin_host_id='host-a')
    saved = writer.commit(capture_session_document(session.state, session.pipeline, session_id='parent'))
    return LoadedSession(repo.load(session_id='parent', commit_id=saved.commit_id), 'local', 'exact')


def test_real_restored_parent_holds_saved_work_through_recovery_poll(runtime, monkeypatch):
    r = runtime
    checkpoint = parent_checkpoint(r)
    # A provably live prior owner blocks takeover. Discovery and recovery still run for real.
    monkeypatch.setattr(swarm, 'process_evidence', lambda _: 'alive')
    with context(r, monkeypatch, sid='parent', iid='restored-parent', checkpoint=checkpoint) as b:
        assert b.seen[0]['restored'] is True
        assert b.seen[0]['revision_id'] == checkpoint.state.commit_id
        assert b.ctx.client.backend.startup_paused
        assert b.work == []
        for _ in range(3):
            b.ctx.client.backend.run_iteration()
        assert b.work == []
        assert b.ctx.client.backend.startup_paused
        assert b.ctx.session.state.entries[-1].tool_calls[0].id == 'old-call'
        assert not b.controller.pending and not b.controller.ready_events
        assert any('blocked' in str(x) for x in b.ctx.session.state.pending_interrupts)
        r.state.session_launcher.start.assert_not_called()
