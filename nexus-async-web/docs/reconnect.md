# Self-healing Reconnect

The pool types (`ClientPool`, `AtomicClientPool`) heal dead
connections automatically. This document explains what triggers
reconnect, how it's scheduled, and what the caller observes.

## Lifecycle of a dead slot

```text
  healthy          send() fails             try_acquire sees dead
  ┌─────┐  ─────►  ┌──────────┐  ─────►  ┌──────────────────┐
  │slot │          │poisoned  │          │ spawn_reconnect  │
  │ok   │          │conn dead │          │ slot out of pool │
  └─────┘          └──────────┘          └──────────────────┘
     ▲                                             │
     │  reconnect task writes                      │
     │  fresh HttpConnection,                      │
     │  drops Pooled guard,                        │
     │  slot rejoins pool                          │
     └─────────────────────────────────────────────┘
```

## What poisons a connection

- `HttpConnection::send()` returns `RestError::Io(_)` mid-request or
  mid-response — could be EPIPE, ECONNRESET, or a read of zero on
  a write-side timeout
- `RestError::ReadTimeout` — elapsed without any response bytes,
  treat as a dead peer
- `RestError::Http(HttpError::...)` where the parser failed on
  bytes from the wire
- `RestError::ConnectionClosed(_)` — server sent `Connection: close`
  or TCP FIN before the response body was complete

In every case, the connection's internal `poisoned` bit is set
and `needs_reconnect()` on the slot returns `true`.

## What triggers `spawn_reconnect`

Only **`try_acquire` / `acquire`** ever spawn reconnect tasks. The
trigger is the call walking the LIFO pool and finding a slot whose
`needs_reconnect()` returns true:

```text
try_acquire():
    loop:
        slot <- pool.try_acquire() or return None
        if !slot.needs_reconnect():
            return Some(slot)           # healthy, return it
        spawn_reconnect(slot)           # eject and heal
        # loop — check the next slot
```

A slot that's in use (held by a live `Pooled` guard) can't be
healed until it's returned — which is the right behavior, because
the in-use slot might still be mid-request and not actually dead.
Once dropped, the next `try_acquire` picks it up and evaluates
its health.

## Reconnect task

The reconnect task is spawned with `spawn_local` (`ClientPool`) or
`spawn` (`AtomicClientPool`). It owns the `Pooled<ClientSlot>`
guard — this is the critical invariant: the guard returns the
slot to the pool when dropped, so the task gets free
"rejoin-on-success" semantics by letting the guard drop on the
happy path.

```text
task loop:
    attempt connect
    on success:
        slot.conn = Some(conn)
        slot.reader.reset()
        return                # drop(Pooled) -> slot back in pool
    on failure:
        sleep(backoff)
        backoff = min(backoff * 2, 5s)
        retry
```

**The task never gives up.** If the target endpoint stays
unreachable for an hour, the task keeps retrying at a 5-second
interval. The slot stays out of rotation the whole time.

## Backoff parameters

- **Initial delay:** 100 ms
- **Multiplier:** 2x per failure
- **Cap:** 5000 ms (5 s)
- **Max attempts:** unbounded (keeps retrying)

These are currently hardcoded. If you need to configure them,
file an issue — it's a one-line builder addition.

## What the caller sees

### `try_acquire()`

- **Healthy slot available:** `Some(slot)` immediately.
- **Dead slots on top:** they're ejected, reconnect tasks spawn,
  loop continues until a healthy slot is found OR the pool runs
  out of slots to scan. If all slots are unhealthy-and-just-spawned
  or already in use, returns `None`.
- **Cost:** O(dead slots) per call, with the dead slots leaving
  the pool as a side effect. Amortized O(1) once the pool is warm.

### `acquire().await`

- Loops over `try_acquire()` with 1ms, 2ms, 4ms, ..., 1000ms
  backoff.
- Returns `Ok(slot)` as soon as one becomes healthy (either from
  another task returning one, or from a reconnect task finishing).
- Returns `Err(RestError::ConnectionClosed("pool acquire timed out..."))`
  after ~20 attempts. That's roughly 15 seconds of backoff before
  giving up — if you need longer, call `acquire` again in an outer
  loop.

## What the caller should do

### On `try_acquire() == None`

On a trading hot path, **fail fast**. Don't wait — reject the
order, log the outage, let the supervisor handle it. Waiting for
a reconnect pushes you over exchange rate limits and leaves you
with stale market state when you do come back.

```rust
let Some(mut slot) = pool.try_acquire() else {
    metrics::increment!("rest.pool.exhausted");
    return Err(TradingError::RestUnavailable);
};
```

### On `acquire().await` timeout

This means reconnect has been failing for 15 seconds and every
slot is dead. The endpoint is down or DNS is broken or TLS is
mis-configured. You should:

1. Log at error level with the endpoint
2. Trip a circuit breaker if you have one
3. Optionally poll `pool.available()` before the next call

### On transient send errors

Let the error propagate. Drop the slot. Next `try_acquire` ejects
it and the reconnect task takes over. **Do not manually reconnect
or reset the slot** — you don't have the lifecycle hooks and the
pool doesn't expect it.

## WebSocket reconnect

`WsStream` has **no** automatic reconnect. WebSocket connections
are session-stateful — subscriptions, auth, sequence numbers — and
blindly reconnecting without replaying that state is worse than
failing.

For WebSocket, reconnect is the application's job. The pattern is
always the same:

```rust
loop {
    match run_feed().await {
        Ok(()) => { /* clean close, restart */ }
        Err(e) => {
            tracing::warn!(?e, "feed died");
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }
}
```

See [patterns.md — Exchange client with reconnect](./patterns.md#exchange-client-with-reconnect).

## Inspecting pool state

```rust
pool.available()    // slots currently in the free list (not in-flight)
```

Neither `available()` nor the pool type exposes "how many slots
are reconnecting." That's intentional — the reconnect task owns
the slot and you don't need to know. If you do need observability,
wrap the pool in your own type and increment metrics at the
call sites where you spawn tasks / receive errors.
