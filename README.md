# azo-plugin-swarms

Pod-based swarms with durable message boards, explicit agent tools, and paused recovery. Requires the Agent Zoo integration stack with lifecycle-managed background submissions, `RuntimeLaunch.startup_mode`, and the `runtime.session_ready` feature event. No additional runtime dependencies.

## User commands

```text
/swarm -n 16 -p 4 --channels comms,breakthroughs --name exploration
/swarm -n 4
/swarm -n 4 --profile gpt-5.6-luna-max
/swarm bcast "A message to the only active swarm"
/swarm bcast sw0 "A message to every pod"
/swarm bcast exploration -p 0-2,3 "A message to selected pods"
/swarm interrupt sw0
/swarm continue sw0
/swarm cancel
/swarm cancel sw0
/swarm release
/swarm capture sw0
/swarm afk sw0 Keep reviewing until the assigned scope is covered.
/swarm afk "Keep reviewing.
Post findings before stopping."
```

`-n` is the total number of agents. `-p` defaults to one; pods are equal-sized. Bounds are 1–32 pods, 1–64 agents per pod, and 1–32 distinct channels. Channels default to `general`. Project-scoped IDs (`sw0`, `sw1`, …) are never reused; `--name` adds a unique alias. Omitting the target for `bcast`, `cancel`, `release`, `capture`, or `afk` selects the sole owned, non-cancelled swarm; otherwise specify its ID or name.

`--profile NAME` (or `--profile=NAME`) selects an enabled named entry in `llm.models` for every peer, without changing the parent model. New pools save the selected profile, including the current default when omitted. Recovery uses each pool's saved profile and blocks relaunch if it is missing or disabled on the destination; legacy pools without a saved profile use the current default. Profile validation does not prevent interrupting preserved live peers after a verified parent replacement.

Pod selectors are zero-based indices and inclusive ranges. Quoted messages may span lines and preserve whitespace. Escape the matching quote or backslash; other escape sequences remain literal. Broadcasts arrive as ordinary user messages, even when their contents begin with `/`. Encoded websocket frames are limited to 1 MiB; oversized messages are rejected, never truncated or split.

Peers have stable attach-menu labels such as `sw0p0a0`. Use the ordinary attach menu to inspect or message an individual peer. New peers start paused and idle; an explicit user message or continuation releases the startup gate. Board messages alone do not authorize work.

Swarm-member pipelines exclude the agent-facing RLM tools (`submit_rlm`, `rlm_status`, and `cancel_rlm`) so peers cannot delegate additional RLM work. Automatic context compaction remains enabled, including its shared internal RLM queue and poller; occasional maintenance jobs are separate from agent-directed delegation. This restriction follows the member's durable session kind through reload and recovery; parent and ordinary sessions retain their RLM tools. It does not cancel descendants launched by older pipelines.

Malformed slash commands report through the TUI status bar. Accepted operations run asynchronously and report their outcome. Cancel requests graceful save-and-shutdown and permanently retires the resource: it cannot be recovered, continued, or assigned more work. Historical boards and transcripts remain readable. Interrupt cannot preempt an arbitrary synchronous tool. Continue does not invent a new prompt or bypass an independent provider-error gate.

`release` and `capture` forward the native commands to every worker, selecting autonomous or interactive pace. `afk` sets the native continuation instructions without changing pace; with no message it queries those instructions, and `clear` clears them. Quote an AFK message when omitting the swarm target. After an explicit target, the message may be quoted or free text, including multiple lines. These commands address the whole swarm and never create replacement workers.

## Agent tools

```python
swarm_broadcast(swarm_id: str, message: str, pods: list[int] | None = None)
swarm_interrupt(swarm_id: str)
swarm_continue(swarm_id: str)
swarm_cancel(swarm_id: str)
swarm_post(swarm_id: str, pod: int, channel: str, message: str,
           message_id: str | None = None)
```

Creation and takeover remain user-only. Tools accept named values, not command strings. Pod numbers are zero-based; omitted broadcast `pods` means all pods, while an empty list is invalid. A multiline message is a normal string, with no extra command quoting. Tools return a correlated receipt and automatically deliver compact completion feedback with outcome counts, affected peer labels, and actionable errors. No polling is required. A completed transport send is not an acknowledgement of peer acceptance or execution. Uncertain sends are never automatically replayed.

`swarm_post` appends to a pod board without waking agents. Board text is limited to 65,536 UTF-8 bytes. An optional stable `message_id` permits idempotent retries of the same sender/content; reusing it for different content is rejected. Peers can access only their own pod; the parent can address every pod it owns.

## Worker guidance and parent onboarding

Workers receive a persistent system-prompt addendum identifying their swarm, pod, and worker label, plus guidance on assignments, `swarm:index`, pod boards, and working without spawning more agents. This is rendered onto the current system entry, not injected as a one-shot transcript message, so compaction and reload retain it. Worker startup and background bookkeeping still run, but their swarm notices are not injected as worker interrupts; errors remain in the logs. Parent lifecycle feedback remains enabled.

After installation, edit `<state-home>/plugin-configs/azo-plugin-swarms/config/worker_prompt.md` to change worker guidance. The plugin reads it once per pipeline build; reload a worker to apply edits to that worker. Reload the parent to rebuild its own pipeline, not to silently change already-running workers. The fixed identity header is generated from the durable session kind, not environment variables or the session display name.

Non-worker sessions receive the bundled `Swarms` skill, including how the user can assign work directly via `/swarm bcast` or `/attach`. Launch feedback explicitly tells the parent that the user created the pool. User broadcasts through `/swarm bcast` notify the parent with target worker labels and exact message content; the skill tells it not to resend or override those assignments. Ordinary agent tool completions retain compact summaries.

The skill is installed at `<state-home>/plugin-configs/azo-plugin-swarms/skills/swarm/SKILL.md`. Both files use the installer's config-preservation mechanism: normal reinstalls preserve user edits, and `--force-config` replaces them with bundled defaults. Skill registration runs once per registry; it adds no per-turn filesystem scanning. Use `/skills refresh` for later skill-file catalog changes. The coordinator skill is excluded from worker catalogs, including restored catalogs that previously contained it.

## Views and durable state

`swarm:index` is a compact directory of resources, pods, board buffers, and worker transcript aliases such as `swarm:sw0p0a0`. Cancelled swarms occupy one archive line. Open `swarm:sw0:index` for a specific swarm's directory, including historical members, or `swarm:sw0:operations` for its operation history. Operation details and transcripts are not loaded into the index.

Board IDs retain the storage form `swarm:<id>:pod-1:board:<channel>`; `pod-1` is pod number 0. Existing long session buffer IDs remain valid. Transcript views render all saved messages on demand, with the newest turn first and chronological messages within each turn. Readable user and assistant text, tool calls, and tool results replace raw provider JSON; opaque provider metadata is omitted. Each view identifies its shared checkpoint revision and source. Attach remains the live inspection interface.

Authoritative records live under `<state-home>/projects/<project>/swarms/`. Immutable message records and append-only references preserve board history. Host-local SQLite is a disposable projection; losing temporary files or caches does not erase acknowledged boards. Shared-storage failures are surfaced rather than hidden behind a stale cache. Ownership checks and mutations share a per-swarm lock; stale runtime grants cannot publish new board entries or overwrite successor outcomes.

## Recovery and replacement instances

Parent restoration holds saved work paused while it discovers owned resources, without requiring a model turn. Sessions with no resources are released automatically; a swarm-owning parent stays paused for review and explicit continuation. Recoverable children retain their session IDs, receive new runtime instance IDs, and resume exact shared checkpoints with the startup pause gate engaged. Successful children remain paused when another member fails. Missing checkpoints, divergent histories, unresolved tool outcomes, and uncertain launches remain explicit blocked states; recovery never substitutes a fresh session or replays an old broadcast. Initial readiness also requires plugin admission and a shared checkpoint, not merely an open websocket.

A verified same-host parent successor can preserve positively live child bindings and their grants instead of rotating the owner epoch and relaunching them. It requests interruption of those existing peers; this is not proof of tool quiescence. The old parent instance loses mutation authority. Mixed or uncertain execution sets take the conservative recovery path and may require explicit reconciliation.

Reload/restart adoption follows verified harness successor lineage, including chains while the parent was offline. It checks the exact session, instance, native process identity, and websocket hello before changing the current binding. A newer unrelated process with the same session ID is not adopted. Original launch provenance is retained. An uncertain send is not redirected to the successor.

Cross-host recovery requires the durable swarm records and verified parent/member checkpoints on the destination, plus valid destination configuration and working paths. A foreign or unreachable host is not proof that prior executions stopped. When automatic reconciliation cannot establish safety, a user can explicitly attest that the old parent, all children, uncertain launches, and external jobs are stopped or isolated:

```text
/swarm recover sw0 --takeover --expected-epoch 3 --confirmed-stopped
```

The expected epoch prevents taking over a changed resource; a positively identified live competing owner still blocks takeover. This is cooperative ownership fencing, not an ability to stop remote processes or undo external tool effects. Independent peers and detached jobs may outlive their parent. Shared-filesystem safety depends on working cross-host advisory locks and atomic durable publication; local tests do not qualify an arbitrary NFS deployment.

## Disconnected transfer

The standalone wrapper includes swarm records, board history, and exact parent/member revisions using the harness session archive format. Ordinary parent-session export alone is insufficient. Both export and import require source-stopped confirmation. Checksums, topology, session coverage, and collisions are validated before publication. IDs are preserved; remap, skip, and replace are not supported. Import does not launch agents or reuse source process identities as destination authority. Runtime recovery requires the matching import journal to be completed and uses its exact destination checkpoint revisions, even if newer revisions exist. Failed partial imports retain reconciliation evidence rather than deleting imported sessions; their unbound owners cannot obtain runtime authority.

```sh
python scripts/swarm_transfer.py export --project research --swarm sw0 \
  --output /private/path/swarm.zip --home /shared/azo --store-root /shared/coord \
  --confirmed-stopped --revision PARENT_ID=PARENT_COMMIT
python scripts/swarm_transfer.py import /private/path/swarm.zip --project research \
  --home /destination/azo --store-root /destination/coord \
  --workdir /destination/work --confirmed-stopped --dry-run
```

Provide `--revision SESSION_ID=COMMIT` for every session without a recorded recoverable revision, including the parent. Inspect the dry-run before importing; omit `--dry-run` to publish. Use explicit `--path-map OLD=NEW` where required. Archives contain private transcripts and operational data; keep them private and retire the source before using the destination.

## Development

```sh
export AZO_HOST_MANIFEST=/absolute/path/to/agent-zoo-worktrees/swarm-integration/pixi.toml
pixi run test
```

The plugin installs in the common scope. Tests use isolated durable roots, deterministic lifecycle fixtures, and local websocket checks; they do not install into or launch model work in user sessions.
