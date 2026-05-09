# nexus-collections

High-performance, slab-backed collections for latency-critical systems.

## Why This Crate?

Node-based data structures (linked lists, heaps, trees) offer
operations that contiguous structures can't — O(1) unlink/re-link, stable
handles to interior elements, and movement between collections without
copying. The trade-off is normally heap fragmentation and allocator overhead
on every node allocation.

This crate eliminates that trade-off by using
[`nexus-slab`](https://crates.io/crates/nexus-slab) — a SLUB-style slab
allocator — as dedicated backing storage for all nodes. Nodes live in
contiguous, type-homogeneous slabs rather than scattered across the global
heap, giving you:

- **Global allocator isolation** — your hot path doesn't compete with
  logging, serialization, or background tasks for allocator resources
- **LIFO cache locality** — recently freed nodes are reused first, staying
  hot in L1
- **Zero fragmentation** — every slot is the same size, freed slots are
  immediately reusable
- **Stable handles** — `RcSlot`-based references that survive unlink,
  re-link, and movement between collections
- **Bounded or unbounded** — fixed capacity with `try_` methods, or
  growable with infallible methods

## Important

All collections require `clear(&slab)` before drop. In debug builds, dropping a non-empty collection panics (leak detection). This is intentional -- the slab owns the memory, and the collection must return its nodes before going out of scope.

Comparator functions passed to RbTree and BTree must not panic. A panic during comparison leaves the tree in an inconsistent state. Use total orderings only.

## Quick Start

```rust
use nexus_slab::rc::bounded::Slab;
use nexus_collections::list::{List, ListNode};

// Create slab — user owns the allocator
// SAFETY: caller accepts manual memory management contract
let slab = unsafe { Slab::<ListNode<u64>>::with_capacity(1000) };

let mut list = List::new();
let handle = list.try_push_back(&slab, 42).unwrap();

// Access through borrow guards
{
    let node = handle.borrow();
    assert_eq!(node.value, 42);
}

// Unlink — works with bounded or unbounded slabs
list.unlink(&handle, &slab);
slab.free(handle);
```

## Collections

### List — Doubly-Linked List

O(1) push/pop/unlink anywhere. `RcSlot` handles enable O(1) access by
identity. Elements can move between lists without deallocation.

```rust
use nexus_slab::rc::bounded::Slab;
use nexus_collections::list::{List, ListNode};

let slab = unsafe { Slab::<ListNode<Order>>::with_capacity(1000) };

let mut list = List::new();

// Bounded: try_push returns Result
let handle = list.try_push_back(&slab, Order { id: 1, price: 100.0 }).unwrap();

// Access via borrow guard
{
    let mut node = handle.borrow_mut();
    node.value.price = 105.0;
}

// O(1) unlink and re-link
list.unlink(&handle, &slab);
list.link_back(&handle);  // no slab needed — just pointer wiring

// Clean up
list.clear(&slab);
slab.free(handle);
```

### Heap — Pairing Heap

O(1) push, O(log n) pop, O(1) peek. `RcSlot` handles enable O(log n)
removal of arbitrary elements by handle.

```rust
use nexus_slab::rc::bounded::Slab;
use nexus_collections::heap::{Heap, HeapNode};

let slab = unsafe { Slab::<HeapNode<u64>>::with_capacity(1000) };

let mut heap = Heap::new();
let handle = heap.try_push(&slab, 42).unwrap();

// O(1) peek
assert_eq!(heap.peek().unwrap().value, 42);

// O(log n) pop — returns owned handle, no slab needed
if let Some(popped) = heap.pop() {
    slab.free(popped);
}
```

### RbTree — Red-Black Tree Sorted Map

Deterministic O(log n) sorted map. Uses raw `Slot` handles (tree owns
all nodes — no shared ownership needed).

```rust
use nexus_slab::bounded::Slab;
use nexus_collections::rbtree::{RbTree, RbNode};
use nexus_collections::Natural;

let slab = unsafe { Slab::<RbNode<u64, String>>::with_capacity(1000) };

let mut map = RbTree::new(Natural);
map.try_insert(&slab, 100, "hello".into()).unwrap();

assert_eq!(map.get(&100), Some(&"hello".into()));

// Entry API (bounded slab)
map.entry(&slab, 200).or_try_insert("world".into()).unwrap();
```

### BTree — B-Tree Sorted Map

Cache-friendly sorted map with tunable branching factor. Uses raw `Slot`
handles like RbTree.

```rust
use nexus_slab::bounded::Slab;
use nexus_collections::btree::{BTree, BTreeNode};
use nexus_collections::Natural;

let slab = unsafe { Slab::<BTreeNode<u64, String, 8>>::with_capacity(1000) };

let mut map: BTree<u64, String, 8> = BTree::new(Natural);
map.try_insert(&slab, 100, "hello".into()).unwrap();

assert_eq!(map.get(&100), Some(&"hello".into()));
```

## Slab Types

Collections accept both bounded and unbounded slabs:

| Method | Slab type | Behavior |
|--------|-----------|----------|
| `push_back(slab, val)` | unbounded | Never fails |
| `try_push_back(slab, val)` | bounded | Returns `Result<_, Full<T>>` |
| `insert(slab, k, v)` | unbounded | Never fails |
| `try_insert(slab, k, v)` | bounded | Returns `Result` |
| `unlink(handle, slab)` | either | Via `RcFree` trait |
| `clear(slab)` | either | Via `RcFree` / `SlabOps` trait |
| `remove(slab, key)` | either | Via `SlabOps` trait |
| `pop()` / `pop_front()` | none | Transfers ownership |
| `link_back(handle)` | none | Just pointer wiring |

Choose bounded or unbounded at setup and stick with it per collection.

## Ownership Model

### List / Heap (Rc handles)

User and collection both hold references to the same node:
- User holds `RcSlot<ListNode<T>>` — their ownership token
- Collection increments refcount on link, decrements on unlink
- Node freed when last handle is freed via `slab.free()`

### RbTree / BTree (raw handles)

Tree owns all nodes internally:
- Insert allocates from slab, tree holds the `Slot`
- Remove frees the slot back to the slab
- No shared ownership — simpler, faster

## Performance

Batched timing (100 ops per rdtsc pair), pinned to core 0.

### List (p50 cycles)

| Operation | Cycles |
|-----------|--------|
| link_back | 2-3 |
| pop_front | 3 |
| unlink | 3-4 |
| try_push_back (alloc+link) | 4 |
| peek (front/back) | <1 |

### Heap (p50 cycles)

| Operation | Cycles |
|-----------|--------|
| link (push) | 6 |
| pop | ~106 |
| unlink | 6-41 |
| try_push (alloc+link) | 8 |
| peek | <1 |

### Sorted Maps (p50 cycles, @10K population)

p50 floors, taskset-pinned P-cores, turbo on, best-of-5.

| Operation | RbTree | BTree | std BTreeMap |
|-----------|--------|-------|-------------|
| get (hit) | **21** | 31 | 36 |
| insert (steady) | 303 | **237** | 186 |
| remove | 283 | **220** | 195 |
| pop_first | **24** | 43 | 51 |
| entry (occupied) | **31** | 41 | 35 |
| insert (growing) p999 | **612** | 698 | 3714 |

RbTree wins on lookups and pops. BTree wins on insert and remove. std
wins p50 on mutation but explodes at p999 on growing insert (3714 cycles
vs nexus RbTree's 612) — global allocator pressure from node splits.
The slab eliminates allocation jitter where it matters: tail latency.

## When to Choose What

**List**: Order queues at a price level, LRU caches, any linked structure
where you need O(1) insert/remove by handle.

**Heap**: Timer wheels, priority scheduling, any min/max extraction pattern.

**RbTree**: Entry-heavy workloads (order books), pop-heavy workloads (timer
wheels), check-then-insert via the entry API.

**BTree**: Read-heavy lookups, existence checking, range scans. Tunable
branching factor via const generic `B`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
