//! DAG pipeline examples — typed, by-reference dataflow graphs.
//!
//! DAGs extend linear Pipelines with fan-out and merge. Use DAG when
//! data needs to flow to multiple processing arms and optionally merge
//! back. For purely sequential chains, prefer `PipelineBuilder`.
//!
//! Key design points:
//! - **Root** takes the event by value (`E → T`). All subsequent nodes
//!   take input by shared reference (`&T`).
//! - **Fork** shares `&T` to all arms — no `Clone` needed. Arms produce
//!   independent output types.
//! - **Merge** combines arm outputs into a new value. **Join** terminates
//!   a fork without merging (all arms return `()`).
//! - After `.build()`, the DAG implements `Handler<E>`. The concrete type
//!   is deeply nested and unnameable — **box it** for storage:
//!   `Box::new(dag)` or `Virtual<E>`. The vtable dispatch at the handler
//!   boundary is the only cost; all internal dispatch remains monomorphized.
//!
//! Sections:
//! 1. Linear chain — root → process → store
//! 2. Diamond — fork into 2 arms, merge
//! 3. Fan-out with join — independent sinks
//! 4. Route — conditional branching
//! 5. Tap and tee — inline observation
//! 6. Dedup — suppress consecutive duplicates
//! 7. Guard — filtering via predicate
//! 8. Boxing into `Box<dyn Handler<E>>`
//! 9. Splat — destructure tuple output into individual `&T` args
//!
//! Run with:
//! ```bash
//! cargo run -p nexus-rt --example dag
//! ```

use nexus_rt::dag::{DagArmSeed, DagBuilder};
use nexus_rt::{Handler, Res, ResMut, Resource, WorldBuilder, new_resource};

new_resource!(Spread(f64));
new_resource!(Counter(u64));

// =============================================================================
// Domain types
// =============================================================================

#[derive(Clone, PartialEq)]
struct Tick {
    symbol: &'static str,
    price: f64,
    size: u64,
}

#[derive(Resource)]
struct PriceCache {
    latest: f64,
    updates: u64,
}

impl PriceCache {
    fn new() -> Self {
        Self {
            latest: 0.0,
            updates: 0,
        }
    }
}

#[derive(Resource)]
struct TradeLog {
    entries: Vec<String>,
}

impl TradeLog {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

// =============================================================================
// Root steps — take event E by value
// =============================================================================

#[allow(clippy::needless_pass_by_value)]
fn extract_price(tick: Tick) -> f64 {
    tick.price
}

// =============================================================================
// Chain steps — take &T by reference (params first, input last)
// =============================================================================

#[allow(clippy::needless_pass_by_value, clippy::trivially_copy_pass_by_ref)]
fn apply_spread(spread: Res<Spread>, price: &f64) -> f64 {
    *price * (1.0 + spread.0)
}

#[allow(clippy::needless_pass_by_value, clippy::trivially_copy_pass_by_ref)]
fn store_price(mut cache: ResMut<PriceCache>, price: &f64) {
    println!("  [store] price={:.2}", price);
    cache.latest = *price;
    cache.updates += 1;
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn get_price(tick: &Tick) -> f64 {
    tick.price
}

#[allow(clippy::needless_pass_by_value)]
fn log_trade(mut log: ResMut<TradeLog>, tick: &Tick) {
    let entry = format!("{} {}@{:.2}", tick.symbol, tick.size, tick.price);
    println!("  [log] {entry}");
    log.entries.push(entry);
}

#[allow(clippy::needless_pass_by_value, clippy::trivially_copy_pass_by_ref)]
fn log_price(mut log: ResMut<TradeLog>, price: &f64) {
    let entry = format!("price={price:.2}");
    println!("  [log] {entry}");
    log.entries.push(entry);
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn merge_sum(a: &f64, b: &f64) -> f64 {
    println!("  [merge] {a:.2} + {b:.2} = {:.2}", a + b);
    a + b
}

#[allow(clippy::needless_pass_by_value, clippy::trivially_copy_pass_by_ref)]
fn count_update(mut ctr: ResMut<Counter>, _val: &u32) {
    **ctr += 1;
}

#[allow(clippy::needless_pass_by_value, clippy::trivially_copy_pass_by_ref)]
fn count_and_print(mut ctr: ResMut<Counter>, x: &u32) {
    println!("  [guard] passed: {x}");
    **ctr += 1;
}

// =============================================================================
// Examples
// =============================================================================

fn main() {
    // --- 1. Linear chain: extract → apply spread → store ---

    println!("=== 1. Linear Chain ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(PriceCache::new());
    wb.register(Spread(0.001)); // spread
    let mut world = wb.build();
    let reg = world.registry();

    let mut linear = DagBuilder::<Tick>::new()
        .root(extract_price, reg)
        .then(apply_spread, reg)
        .then(store_price, reg)
        .build();

    linear.run(
        &mut world,
        Tick {
            symbol: "BTC",
            price: 50_000.0,
            size: 10,
        },
    );

    let cache = world.resource::<PriceCache>();
    println!(
        "  cache: latest={:.2}, updates={}\n",
        cache.latest, cache.updates
    );
    assert_eq!(cache.updates, 1);
    assert!((cache.latest - 50_050.0).abs() < 0.01);

    // --- 2. Diamond: fork into spread + fee arms, merge ---
    //
    // Both arms observe the same &f64 — no cloning needed.

    println!("=== 2. Diamond (fork/merge) ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(Spread(0.001)); // spread
    let mut world = wb.build();
    let reg = world.registry();

    let mut diamond = DagBuilder::<Tick>::new()
        .root(extract_price, reg)
        .fork()
        .arm(|a| a.then(apply_spread, reg))
        .arm(|b| b.then(|p: &f64| *p * 0.1, reg))
        .merge(merge_sum, reg)
        .then(|_v: &f64| {}, reg)
        .build();

    diamond.run(
        &mut world,
        Tick {
            symbol: "ETH",
            price: 3_000.0,
            size: 5,
        },
    );

    // --- 3. Fan-out: broadcast to independent sinks ---
    //
    // Fork with join() — both arms produce () independently.

    println!("\n=== 3. Fan-out (join) ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(PriceCache::new());
    wb.register(TradeLog::new());
    let mut world = wb.build();
    let reg = world.registry();

    let mut fanout = DagBuilder::<Tick>::new()
        .root(extract_price, reg)
        .fork()
        .arm(|a| a.then(store_price, reg))
        .arm(|b| b.then(log_price, reg))
        .join()
        .build();

    for tick in [
        Tick {
            symbol: "BTC",
            price: 50_000.0,
            size: 10,
        },
        Tick {
            symbol: "ETH",
            price: 3_000.0,
            size: 100,
        },
    ] {
        fanout.run(&mut world, tick);
    }

    let cache = world.resource::<PriceCache>();
    let log = world.resource::<TradeLog>();
    println!(
        "\n  cache: latest={:.2}, updates={}",
        cache.latest, cache.updates
    );
    println!("  log: {} entries", log.entries.len());
    assert_eq!(cache.updates, 2);
    assert_eq!(log.entries.len(), 2);

    // --- 4. Route: conditional branching ---
    //
    // Arms are pre-built, predicate selects which runs.

    println!("\n=== 4. Route (conditional) ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(PriceCache::new());
    let mut world = wb.build();
    let reg = world.registry();

    let high_value = DagArmSeed::<f64>::new().then(store_price, reg);
    let low_value =
        DagArmSeed::<f64>::new().then(|p: &f64| println!("  [skip] low-value price={p:.2}"), reg);

    let mut routed = DagBuilder::<Tick>::new()
        .root(extract_price, reg)
        .route(|price: &f64| *price > 10_000.0, reg, high_value, low_value)
        .build();

    for tick in [
        Tick {
            symbol: "BTC",
            price: 50_000.0,
            size: 1,
        },
        Tick {
            symbol: "DOGE",
            price: 0.08,
            size: 1_000_000,
        },
        Tick {
            symbol: "ETH",
            price: 3_000.0,
            size: 10,
        },
    ] {
        println!("  routing {} @ {:.2}...", tick.symbol, tick.price);
        routed.run(&mut world, tick);
    }

    let cache = world.resource::<PriceCache>();
    println!("  cache: updates={} (only >10k)\n", cache.updates);
    assert_eq!(cache.updates, 1);

    // --- 5a. Tap: inline observation without consuming ---

    println!("=== 5a. Tap (inline observation) ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(PriceCache::new());
    let mut world = wb.build();
    let reg = world.registry();

    let mut tapped = DagBuilder::<Tick>::new()
        .root(extract_price, reg)
        .tap(|price: &f64| println!("  [tap] saw price={price:.2}"), reg)
        .then(store_price, reg)
        .build();

    tapped.run(
        &mut world,
        Tick {
            symbol: "BTC",
            price: 55_000.0,
            size: 5,
        },
    );
    assert_eq!(world.resource::<PriceCache>().updates, 1);

    // --- 5b. Tee: fork off a multi-step side-effect chain ---

    println!("\n=== 5b. Tee (side-effect arm) ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(PriceCache::new());
    wb.register(TradeLog::new());
    let mut world = wb.build();
    let reg = world.registry();

    let log_side = DagArmSeed::<Tick>::new().then(log_trade, reg);

    let mut teed = DagBuilder::<Tick>::new()
        .root(|t: Tick| t, reg)
        .tee(log_side)
        .then(get_price, reg)
        .then(store_price, reg)
        .build();

    teed.run(
        &mut world,
        Tick {
            symbol: "ETH",
            price: 4_000.0,
            size: 50,
        },
    );

    let cache = world.resource::<PriceCache>();
    let log = world.resource::<TradeLog>();
    println!("  cache: latest={:.2}", cache.latest);
    println!("  log: {:?}", log.entries);
    assert_eq!(cache.updates, 1);
    assert_eq!(log.entries.len(), 1);

    // --- 6. Dedup: suppress consecutive unchanged values ---

    println!("\n=== 6. Dedup ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(Counter(0));
    let mut world = wb.build();
    let reg = world.registry();

    let mut deduped = DagBuilder::<u32>::new()
        .root(|x: u32| x, reg)
        .dedup()
        .inspect(|val: &u32| println!("  [dedup] passed: {val:?}"), reg)
        .map(count_update, reg)
        .unwrap_or(())
        .build();

    for &v in &[1, 1, 2, 2, 2, 3, 1] {
        deduped.run(&mut world, v);
    }

    let count = world.resource::<Counter>().0;
    println!("  updates: {count} (4 unique runs from 7 inputs)\n");
    assert_eq!(count, 4);

    // --- 7. Guard: filtering ---

    println!("=== 7. Guard ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(Counter(0));
    let mut world = wb.build();
    let reg = world.registry();

    let mut guarded = DagBuilder::<u32>::new()
        .root(|x: u32| x, reg)
        .guard(|x: &u32| (*x).is_multiple_of(2), reg)
        .map(count_and_print, reg)
        .unwrap_or(())
        .build();

    for v in 0..6u32 {
        guarded.run(&mut world, v);
    }

    let count = world.resource::<Counter>().0;
    println!("  even count: {count}");
    assert_eq!(count, 3);

    // --- 8. Box<dyn Handler>: type erasure ---

    println!("\n=== 8. Box<dyn Handler> ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(PriceCache::new());
    wb.register(Spread(0.001));
    let mut world = wb.build();
    let reg = world.registry();

    let dag: Box<dyn Handler<Tick>> = Box::new(
        DagBuilder::<Tick>::new()
            .root(extract_price, reg)
            .then(apply_spread, reg)
            .then(store_price, reg)
            .build(),
    );

    let mut handlers: Vec<Box<dyn Handler<Tick>>> = vec![dag];

    for h in &mut handlers {
        h.run(
            &mut world,
            Tick {
                symbol: "BTC",
                price: 60_000.0,
                size: 1,
            },
        );
    }

    let cache = world.resource::<PriceCache>();
    println!("  cache: latest={:.2}", cache.latest);
    assert!((cache.latest - 60_060.0).abs() < 0.01);

    // --- 9. Splat — destructure tuple into individual &T arguments ---

    println!("\n=== 9. Splat ===\n");

    let mut wb = WorldBuilder::new();
    wb.register(PriceCache::new());
    let mut world = wb.build();
    let reg = world.registry();

    #[allow(clippy::items_after_statements, clippy::needless_pass_by_value)]
    fn split_tick(t: Tick) -> (f64, u64) {
        (t.price, t.size)
    }
    #[allow(clippy::items_after_statements, clippy::trivially_copy_pass_by_ref)]
    fn weighted(price: &f64, size: &u64) -> f64 {
        *price * *size as f64
    }
    #[allow(
        clippy::items_after_statements,
        clippy::needless_pass_by_value,
        clippy::trivially_copy_pass_by_ref
    )]
    fn store_weighted(mut cache: ResMut<PriceCache>, val: &f64) {
        cache.latest = *val;
    }

    let mut dag = DagBuilder::<Tick>::new()
        .root(split_tick, reg)
        .splat()
        .then(weighted, reg)
        .then(store_weighted, reg)
        .build();

    dag.run(
        &mut world,
        Tick {
            symbol: "ETH",
            price: 3_500.0,
            size: 2,
        },
    );
    let weighted_val = world.resource::<PriceCache>().latest;
    println!("  weighted: {weighted_val:.2}");
    assert!((weighted_val - 7_000.0).abs() < 0.01);

    println!("\nDone.");
}
