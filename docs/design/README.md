# Design Documents

Pre-implementation architecture planning for upcoming nexus crates.
These are living documents — they evolve through discussion before
code is written.

## How to use these docs

Each document lays out:
- What's decided (philosophy, constraints, invariants)
- What's open (design tensions, both paths presented)
- Key questions that need answers before implementation

Where prior design notes exist in different versions, both paths
are presented with their tradeoffs so we can discuss the merits.

## Current designs

- [nexus-shm](nexus-shm.md) — Shared memory primitives (foundation,
  ring buffers, journal, slot, map)
- [nexus-fix](nexus-fix.md) — FIX protocol codec and engine
