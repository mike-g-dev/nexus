# Shutdown Sequence

## Triggering shutdown

`Runtime::block_on` takes `&mut self` and runs until the future
returns. While `block_on` is executing, the runtime handle is
exclusively borrowed — you cannot call `shutdown_quiesce` or `drop`
from inside a spawned task. The shutdown sequence must be triggered
from outside `block_on` (or from inside the future passed to
`block_on`, which then returns to release the borrow).

**The handles you need are `&self` accessors.** Clone them BEFORE
calling `block_on`:

```rust,ignore
let mut runtime = Runtime::builder(&mut world).build();
let shutdown = runtime.shutdown_handle();   // Clone — Send + Sync
let stats    = runtime.shutdown_stats();    // Arc clone — survives drop

// Hand `shutdown` to whatever can trigger it (signal handler, RPC,
// parent thread, supervised process). Standard pattern: ctrl-c
// installs a handler that calls shutdown.trigger().
std::thread::spawn({
    let shutdown = shutdown.clone();
    move || {
        wait_for_sigterm();
        shutdown.trigger();              // sets the flag, wakes block_on
    }
});

// The future passed to block_on awaits the signal:
runtime.block_on(async move {
    // Spawn long-running work as separate tasks; the main future
    // just awaits shutdown.
    spawn_boxed(my_main_loop());
    spawn_boxed(my_metrics_reporter());

    // Resolves when shutdown.trigger() fires from anywhere → block_on returns.
    shutdown.signal().await;
});

// block_on returned → &mut self is back. Now run the sequence below.
runtime.shutdown_quiesce(std::time::Duration::from_millis(500))?;
drop(runtime);

// Optional post-mortem on the abnormal-shutdown counters:
let final_stats = stats.snapshot();
if final_stats.aborted_unwinds != 0 || final_stats.leaked_box_tasks != 0 {
    my_logger::warn!("nexus runtime shutdown: {final_stats:?}");
}
```

**Key API touchpoints:**

- `Runtime::shutdown_handle() -> ShutdownHandle` (`&self`) — `Clone`,
  `Send`, `Sync`. Hand to anywhere that needs to trigger.
- `ShutdownHandle::trigger()` — sets the flag, wakes the registered
  task waker, breaks epoll_wait. Safe to call from any thread.
- `ShutdownHandle::is_shutdown() -> bool` — non-blocking check.
- `ShutdownHandle::signal() -> ShutdownSignal` — `Future<Output = ()>`
  that resolves when triggered. Use inside the `block_on` future to
  await shutdown.
- `Runtime::shutdown_stats() -> Arc<ShutdownStatsAtomics>` (`&self`) —
  Arc clone. Holds the underlying counters; survives Runtime drop.
  Read post-drop via `.snapshot()`.

For SIGINT/SIGTERM auto-wiring, `nexus_async_rt::shutdown::install_signal_handlers`
handles the boilerplate.

## Shutting down cleanly

Once `block_on` has returned (because the future awaiting
`shutdown.signal()` completed), the runtime should shut down in this
order:

1. **Stop producers of cross-thread refs.** Anything outside the
   runtime that may push to its cross-thread queue or hold a
   `JoinHandle` / channel sender / waker reference must be torn down
   first.

   - Drop the tokio runtime (or call `tokio_runtime().shutdown_timeout(...)`).
   - Stop the Aeron driver thread.
   - Drop external channel senders and waker handles.
   - Drop any `tokio_compat::TokioJoinHandle`s held in user code.

2. **Quiesce.** Call `Runtime::shutdown_quiesce(timeout)` to drive the
   executor until the cross-thread queue is drained and no local
   ready work remains.

   ```rust,ignore
   runtime.shutdown_quiesce(std::time::Duration::from_millis(500))?;
   ```

   Choose a timeout appropriate for the producer landscape — a
   trading-system shutdown sequence with multiple Aeron drivers +
   tokio futures + channel senders has very different settling
   characteristics than a unit test. PR 2 §2.4 deliberately
   ships `shutdown_quiesce` with **no default timeout** to force the
   user to pick one.

3. **Drop the Runtime.** With producers stopped and the executor
   quiesced, the outstanding-ref panic paths in `Executor::drop`
   should be unreachable in normal operation.

   ```rust,ignore
   drop(runtime);
   ```

If step 2 returns `Err(QuiesceTimeout)`, a producer hasn't released
its refs. Investigate before letting the Runtime drop — the
unwind-abort path in `Executor::drop` is **defensive, not desired**.

## ShutdownStats integration

After dropping the Runtime, inspect the abnormal-shutdown counters
via the `Arc<ShutdownStatsAtomics>` handle obtained pre-drop:

```rust,ignore
let stats_handle = runtime.shutdown_stats();
runtime.shutdown_quiesce(Duration::from_millis(500))?;
drop(runtime);

let stats = stats_handle.snapshot();
if stats.aborted_unwinds != 0
    || stats.leaked_box_tasks != 0
    || stats.unbalanced_normal_shutdowns != 0
    || stats.cross_queue_undrained != 0
{
    // Your own observability — log to wherever you want.
    my_logger::warn!("nexus runtime shutdown: {stats:?}");
}
```

The runtime emits no log events of its own when these counters fire
(per PR 2's design — counters give zero-cost observability and let
you own the logging policy). The PR 1a `eprintln!` calls in the
slab-unwinding-abort path remain (only signal at the moment of
process abort) but new abnormal paths are pure counter increments.

### What each counter signifies

- `aborted_unwinds`: The slab-unwinding 100ms-wait-then-abort path
  fired. A producer thread held a slab task ref past Runtime drop
  during a panic. The process aborted to avoid UAF on slab backing
  storage. **Non-zero usually means a previous run aborted** — read
  this counter only if your code somehow survives the abort (e.g.,
  a parent process inspecting via shared memory).
- `leaked_box_tasks`: Box-allocated tasks the executor couldn't free
  during shutdown unwinding (outstanding cross-thread refs, leaked
  to avoid double-panic). Memory leak, not UAF; reclaimed at process
  exit.
- `unbalanced_normal_shutdowns`: Normal shutdown found an `all_tasks`
  entry with `rc > 0`. Debug builds panic. Release builds increment
  the counter and leak. Indicates a producer didn't release refs
  before Runtime drop — `shutdown_quiesce` would have surfaced this
  as `Err(QuiesceTimeout)` instead.
- `cross_queue_undrained`: Cross-thread queue entries still in the
  Runtime's cross-thread queue at the moment `Executor::drop` runs
  its final tally. These are wakes that landed too late to be polled
  but before drop completed. Pure memory leak (PR 1a's inherited
  leak path); the entries are popped during drop to clear `QUEUED`
  flags so off-thread holders' future pushes still get the dedup
  signal, but no executor will dequeue them after this point. A
  non-zero count indicates producers were still pushing at shutdown —
  consider a longer `shutdown_quiesce` timeout.

## What about panic-during-shutdown?

The 100ms slab-unwinding wait + abort path in `Executor::drop` is
defense-in-depth for the panic case. Production shutdown should
**NOT** rely on it. Use `shutdown_quiesce` for clean teardown; the
100ms path exists so a panic mid-shutdown doesn't UAF.

If a `shutdown_quiesce` returns `Err(QuiesceTimeout)`, the *correct*
response is to log the diagnostic counts, decide whether to extend
the timeout, and try again — not to drop the Runtime and rely on the
abort path.

## Future hardening (post-PR-2)

The PR 1a `eprintln!` calls in the abort path are **tagged for
removal once `shutdown_quiesce` is the canonical path**. At that
point the abort branch becomes unreachable in normal operation, and
the eprintln is dead weight. Removal candidate for a future PR.
