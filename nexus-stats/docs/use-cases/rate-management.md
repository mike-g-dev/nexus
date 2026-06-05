# Use Case: Rate Management

Tracking event rates, detecting rate changes, and monitoring against limits.

Note: active rate *limiting* (token bucket, GCRA, sliding window counter)
lives in the `nexus-rate` crate. This covers the *measurement* side.

## Recipe: Monitor Against Exchange Rate Limits

```rust
use nexus_stats::Condition;
use nexus_stats::monitoring::{EventRateU64, SaturationF64};

// Track our own order rate
let mut order_rate = EventRateU64::builder().span(15).build().unwrap();

// Detect if we're approaching the limit
let mut limit_sat = SaturationF64::builder()
    .span(10)
    .threshold(0.80)  // warn at 80% of exchange limit
    .build().unwrap();

// On each order sent:
order_rate.update(now_ns);

if let Some(rate) = order_rate.rate() {
    let utilization = rate / exchange_rate_limit;
    if let Some(Condition::Degraded) = limit_sat.update(utilization) {
        throttle_order_flow();
    }
}
```

## Recipe: Detect Rate Anomalies

```rust
use nexus_stats::Direction;
use nexus_stats::monitoring::EventRateU64;
use nexus_stats::detection::CusumF64;

let mut rate = EventRateU64::builder().span(31).build().unwrap();
let mut cusum = CusumF64::builder(expected_rate)
    .slack(expected_rate * 0.1)
    .threshold(expected_rate * 2.0)
    .build().unwrap();

rate.update(now_ns);
if let Some(r) = rate.rate() {
    if let Some(shift) = cusum.update(r) {
        match shift {
            Direction::Rising => log::warn!("rate spike detected"),
            Direction::Falling => log::warn!("rate drop detected"),
            Direction::Neutral => {}
        }
    }
}
```

## Primitives Used

| Primitive | Role |
|-----------|------|
| [EventRate](../algorithms/event-rate.md) | Smoothed rate measurement |
| [Saturation](../algorithms/saturation.md) | Rate limit utilization |
| [CUSUM](../algorithms/cusum.md) | Rate shift detection |
| [LevelCrossing](../algorithms/level-crossing.md) | Count rate limit breaches |
