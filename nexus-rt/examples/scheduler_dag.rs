//! Staged scheduler — reconciliation systems with boolean propagation.
//!
//! The staged scheduler executes `System`s (not `Handler`s) in stage
//! order. Key differences from handlers:
//!
//! - **System** returns `bool` — `true` means "my outputs changed, run
//!   downstream", `false` means "nothing changed, skip downstream".
//! - **No event parameter** — systems read shared state, not per-event data.
//! - **Boolean propagation** — root stage always runs; subsequent stages
//!   run only if the previous stage returned at least one `true` (OR
//!   semantics within a stage).
//!
//! Typical pattern: event handlers write resources, then the scheduler
//! runs reconciliation after each event (or batch of events).
//!
//! Run with:
//! ```bash
//! cargo run --release -p nexus-rt --example scheduler_dag
//! ```

#![allow(clippy::needless_pass_by_value)]

use nexus_rt::scheduler::SchedulerBuilder;
use nexus_rt::{Handler, IntoHandler, Res, ResMut, Resource, WorldBuilder};

// ── Domain types ────────────────────────────────────────────────────────

#[derive(Resource)]
struct MidPrice(f64);

#[derive(Resource)]
struct TheoreticalValue(f64);

#[derive(Resource)]
struct SpreadBps(f64);

#[derive(Resource)]
struct QuoteState {
    bid: f64,
    ask: f64,
}

#[derive(Resource)]
struct RiskFlag(bool);

// ── Systems ─────────────────────────────────────────────────────────────

/// Recompute theoretical value from mid price. Always propagates —
/// the system's bool return controls downstream execution.
fn compute_theo(mid: Res<MidPrice>, mut theo: ResMut<TheoreticalValue>) -> bool {
    let new_theo = mid.0 * 1.001; // trivial model
    if (new_theo - theo.0).abs() > f64::EPSILON {
        theo.0 = new_theo;
        true
    } else {
        false
    }
}

/// Recompute quotes from theoretical value and spread.
fn compute_quotes(
    theo: Res<TheoreticalValue>,
    spread: Res<SpreadBps>,
    mut quotes: ResMut<QuoteState>,
) -> bool {
    let half_spread = theo.0 * spread.0 / 10_000.0 / 2.0;
    quotes.bid = theo.0 - half_spread;
    quotes.ask = theo.0 + half_spread;
    true
}

/// Check risk limits after quote update. Could gate downstream publishing.
fn check_risk(quotes: Res<QuoteState>, mut flag: ResMut<RiskFlag>) -> bool {
    let spread = quotes.ask - quotes.bid;
    flag.0 = spread < 100.0; // within limits
    flag.0
}

// ── Event handler (simulated market data feed) ──────────────────────────

fn on_market_tick(mut mid: ResMut<MidPrice>, new_price: f64) {
    mid.0 = new_price;
}

// ── main ────────────────────────────────────────────────────────────────

fn main() {
    // -- Build ----------------------------------------------------------------

    let mut wb = WorldBuilder::new();
    wb.register(MidPrice(50_000.0));
    wb.register(TheoreticalValue(0.0));
    wb.register(SpreadBps(10.0)); // 10 bps
    wb.register(QuoteState { bid: 0.0, ask: 0.0 });
    wb.register(RiskFlag(false));

    let reg = wb.registry();

    // Build the staged chain: compute_theo → compute_quotes → check_risk
    let mut scheduler = wb.install_driver(
        SchedulerBuilder::new()
            .root(compute_theo, reg)
            .then(compute_quotes, reg)
            .then(check_risk, reg),
    );

    // Event handler for market data
    let mut market_handler = on_market_tick.into_handler(wb.registry());

    let mut world = wb.build();

    // -- Simulate event loop --------------------------------------------------

    let prices = [50_100.0, 50_200.0, 50_200.0, 50_300.0];

    for (i, &price) in prices.iter().enumerate() {
        // Event phase: driver delivers market data
        world.next_sequence();
        market_handler.run(&mut world, price);

        // Reconciliation phase: scheduler runs after events
        let ran = scheduler.run(&mut world);

        let quotes = world.resource::<QuoteState>();
        let risk = world.resource::<RiskFlag>();

        println!(
            "pass {}: mid={:.0}, ran={}/{}, bid={:.2}, ask={:.2}, risk_ok={}",
            i + 1,
            price,
            ran,
            3,
            quotes.bid,
            quotes.ask,
            risk.0,
        );
    }

    // Note: pass 3 has mid_price 50_200 → 50_200 (no change).
    // compute_theo compares old vs new and returns false, stopping
    // downstream — only 1 system runs instead of 3.

    println!("\nDone. The system's bool return controls propagation —");
    println!("return false when outputs didn't change to skip downstream.");
}
