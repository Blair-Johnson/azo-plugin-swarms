# azo-plugin-swarm

Pod-based swarms with durable message boards, raw slash commands and scoped agent tools. Requires the Agent Zoo integration stack with `CommandContext.defer` (agent-zoo `f4a1e6a` or later) and the integrated runtime launcher. No additional runtime dependencies.

## Commands

```text
/swarm -n 16 -p 4 --channels comms,breakthroughs --name exploration
/swarm -n 4
/swarm bcast "A message to the only active swarm"
/swarm bcast sw0 "A message to every pod"
/swarm bcast exploration -p 0-2,3 "A message to selected pods"
/swarm interrupt sw0
/swarm continue sw0
/swarm cancel sw0
```

`-n` is the total number of agents. `-p` defaults to one; pods must be equal-sized. Initial bounds are 1–32 pods, 1–64 agents per pod and 1–32 distinct channels. Channels default to `general`. IDs are project-scoped `sw0`, `sw1`, etc., durably allocated and never reused. `--name` adds a unique alias; either the alias or ID addresses the resource. Omitted broadcast targets require exactly one owned, non-cancelled swarm.

Pod selectors are zero-based comma-separated indices and inclusive ranges, such as `0-3,5`. Quoted broadcast messages may span lines and preserve their contents, including surrounding whitespace. Escape the matching quote or backslash with a backslash; other escape sequences remain literal. Broadcasts are ordinary user messages, not board posts or system instructions. The complete encoded websocket frame is limited to 1 MiB; oversized messages are rejected without truncation or splitting.

Creation records all membership before launching idle, independent peer sessions with no initial user message. Their attach-menu types are stable labels such as `sw0p0a0`, `sw0p0a1`, and `sw0p1a0`. Use the normal attach menu to inspect a peer or message it directly. The parent receives a system notification describing the resource after creation completes. A partial launch retains its member identities, attempts and process handles; it is never automatically retried or rolled back.

Interrupt and continue forward the existing `/interrupt` and `/continue` commands. Cancel requests graceful shutdown with checkpoint saving; it does not mean interrupt, kill by PID, or prove that the peer has exited. Cancellation is terminal intent for the resource: later broadcast/continue is refused, but cancel may be explicitly requested again. Interrupt cannot preempt an arbitrary synchronous tool; continue does not create a new prompt or recover a provider error requiring `/retry`.

Malformed slash commands and operational feedback use the TUI status bar. Valid operations run asynchronously through the harness hook, with blocking storage/launch calls off the shared I/O loop. A parent's operations are serialized, with bounded member concurrency. Sends are reported as sent, not accepted or executed. Uncertain sends are not retried automatically.

## Agent tools and views

The parent gets `swarm_control(command=...)`, which queues the same command tail, such as `bcast sw0 -p 0 "Investigate this"` or `interrupt sw0`, through the existing session input channel. It shares the slash parser, dispatcher and status feedback. Creation remains user-only through this tool interface. `swarm_post` appends to a named pod board and does not wake agents or dispatch work.

The read-only `swarm:index` buffer lists bound resources, pod indices, member labels, board buffers and saved transcript buffers. Parents also see recent operation outcomes. Boards use `swarm:<id>:<pod>:board:<channel>` and transcripts use `swarm:<id>:<pod>:session:<session-id>`. Internal pod IDs remain `pod-1`, `pod-2`, etc.; command indices are zero-based. Transcript views show shared saved checkpoints, not live activity. Use attach for live inspection.

Peers bootstrap from `AZO_SWARM_ID` and `AZO_SWARM_POD_ID`. Their tools and buffers expose only their own pod; parents can see all pods they own. This is plugin-level scope enforcement, not filesystem, process or adversarial security isolation. Registration and reading buffers never launch agents or submit model work.

## Durability and recovery

Authoritative records live under `<state-home>/projects/<project>/swarms/`, using the filesystem `RecordStore`. Pools have durable ownership and topology; launch attempts are reserved before spawning. Broadcast/control records retain the requested operation and outcomes. A pending record after interruption is ambiguous and is not replayed. Desired state records intent, not confirmed runtime state.

Board messages are immutable content-addressed records with an append-only reference list. SQLite under `<host-local-state>/swarm-cache/` is only a disposable projection. Losing temporary/local data cannot erase acknowledged board messages or durable control records. Shared-storage errors are surfaced rather than masked with stale cache data. A stable board `message_id` supports idempotent retries of identical content.

A restored parent can address its durable owned resource records even without a saved binding. Verified readiness snapshots are durable too: same-host/same-boot control may reconnect after temporary launch files are lost. Every send requires the recorded native process identity and an exact session/instance websocket hello; endpoints must be loopback. Missing identities, changed PIDs, foreign hosts/boots and unrelated session instances fail closed. The plugin never adopts an arbitrary latest session or signals a stored PID.

Full cross-host recovery, stopped-peer relaunch and adoption of a peer's replacement `/reload` instance are not implemented. Restoring metadata does not launch, resume, pause, or replay peers. Independent peers may outlive their parent; explicitly interrupt or cancel them as appropriate. Portable storage requires the durable swarm directory and relevant member checkpoints, not merely the parent's checkpoint. Shared-filesystem guarantees remain those of RecordStore and the underlying filesystem.

## Development

```sh
export AZO_HOST_MANIFEST=/absolute/path/to/agent-zoo-worktrees/swarm-integration/pixi.toml
pixi run test
```

The manifest installs `src/swarm.py` in the common plugin scope so parents and peers load the same feature. Commands inherit the current model/environment, configuration provenance, project/store roots and working directory; peers use the default pipeline. Tests use isolated durable roots and mocked launches/transports, with real local websocket integration where noted. They do not install the plugin or launch model-backed agents.
