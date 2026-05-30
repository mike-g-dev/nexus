#![allow(
    unused_must_use,
    dead_code,
    clippy::float_cmp,
    clippy::used_underscore_binding,
    clippy::items_after_statements
)]
//! Integration tests for #[derive(Param)].

use nexus_rt::{Handler, IntoHandler, Local, Param, Res, ResMut, Resource, WorldBuilder, no_event};

// =========================================================================
// Test types
// =========================================================================

#[derive(Resource, Default)]
struct OrderBook {
    best_bid: f64,
    best_ask: f64,
}

#[derive(Resource, Default)]
struct RiskState {
    exposure: f64,
}

#[derive(Resource, Default)]
struct Config {
    max_exposure: f64,
}

// =========================================================================
// Basic: Res + ResMut
// =========================================================================

#[derive(Param)]
struct BasicParams<'w> {
    book: Res<'w, OrderBook>,
    risk: ResMut<'w, RiskState>,
}

fn handler_basic(mut params: BasicParams<'_>, _event: u32) {
    let spread = params.book.best_ask - params.book.best_bid;
    params.risk.exposure += spread;
}

#[test]
fn basic_res_resmut() {
    let mut wb = WorldBuilder::new();
    wb.register(OrderBook {
        best_bid: 100.0,
        best_ask: 101.0,
    });
    wb.register(RiskState::default());
    let mut world = wb.build();

    let mut h = handler_basic.into_handler(world.registry());
    h.run(&mut world, 0u32);

    assert_eq!(world.resource::<RiskState>().exposure, 1.0);
}

// =========================================================================
// With Local
// =========================================================================

#[derive(Param)]
struct ParamsWithLocal<'w> {
    config: Res<'w, Config>,
    call_count: Local<'w, u64>,
}

fn handler_with_local(mut params: ParamsWithLocal<'_>) {
    *params.call_count += 1;
    let _ = params.config.max_exposure;
}

#[test]
fn with_local() {
    let mut wb = WorldBuilder::new();
    wb.register(Config {
        max_exposure: 1000.0,
    });
    let mut world = wb.build();

    let mut h = no_event(handler_with_local).into_handler(world.registry());
    h.run(&mut world, ());
    h.run(&mut world, ());
    h.run(&mut world, ());
    // Local state persists across calls — 3 invocations
    // (We can't directly inspect Local from outside, but it compiles and runs)
}

// =========================================================================
// With Option<Res<T>>
// =========================================================================

#[derive(Param)]
struct OptionalParams<'w> {
    config: Option<Res<'w, Config>>,
    risk: ResMut<'w, RiskState>,
}

fn handler_optional(mut params: OptionalParams<'_>) {
    if let Some(config) = params.config {
        params.risk.exposure = config.max_exposure;
    }
}

#[test]
fn with_optional_res() {
    // Config NOT registered — should still work
    let mut wb = WorldBuilder::new();
    wb.register(RiskState::default());
    let mut world = wb.build();

    let mut h = no_event(handler_optional).into_handler(world.registry());
    h.run(&mut world, ());
    assert_eq!(world.resource::<RiskState>().exposure, 0.0); // no config

    // Now with Config registered
    let mut wb2 = WorldBuilder::new();
    wb2.register(RiskState::default());
    wb2.register(Config {
        max_exposure: 500.0,
    });
    let mut world2 = wb2.build();

    let mut h2 = no_event(handler_optional).into_handler(world2.registry());
    h2.run(&mut world2, ());
    assert_eq!(world2.resource::<RiskState>().exposure, 500.0);
}

// =========================================================================
// With #[param(ignore)] field
// =========================================================================

#[derive(Param)]
struct ParamsWithIgnored<'w> {
    risk: ResMut<'w, RiskState>,
    #[param(ignore)]
    _marker: std::marker::PhantomData<u32>,
}

fn handler_ignored(mut params: ParamsWithIgnored<'_>) {
    params.risk.exposure += 1.0;
}

#[test]
fn with_ignored_field() {
    let mut wb = WorldBuilder::new();
    wb.register(RiskState::default());
    let mut world = wb.build();

    let mut h = no_event(handler_ignored).into_handler(world.registry());
    h.run(&mut world, ());
    assert_eq!(world.resource::<RiskState>().exposure, 1.0);
}

// =========================================================================
// Nested Param structs
// =========================================================================

#[derive(Param)]
struct InnerParams<'w> {
    risk: ResMut<'w, RiskState>,
}

#[derive(Param)]
struct OuterParams<'w> {
    inner: InnerParams<'w>,
    config: Res<'w, Config>,
}

fn handler_nested(mut params: OuterParams<'_>) {
    params.inner.risk.exposure = params.config.max_exposure;
}

#[test]
fn nested_params() {
    let mut wb = WorldBuilder::new();
    wb.register(RiskState::default());
    wb.register(Config {
        max_exposure: 999.0,
    });
    let mut world = wb.build();

    let mut h = no_event(handler_nested).into_handler(world.registry());
    h.run(&mut world, ());
    assert_eq!(world.resource::<RiskState>().exposure, 999.0);
}

// =========================================================================
// Param + additional resource params (higher arity)
// =========================================================================

#[derive(Param)]
struct TradingParams<'w> {
    book: Res<'w, OrderBook>,
    risk: ResMut<'w, RiskState>,
}

fn handler_mixed(mut params: TradingParams<'_>, config: Res<Config>) {
    params.risk.exposure = params.book.best_bid * config.max_exposure;
}

#[test]
fn param_plus_additional_resources() {
    let mut wb = WorldBuilder::new();
    wb.register(OrderBook {
        best_bid: 50.0,
        best_ask: 51.0,
    });
    wb.register(RiskState::default());
    wb.register(Config { max_exposure: 2.0 });
    let mut world = wb.build();

    let mut h = no_event(handler_mixed).into_handler(world.registry());
    h.run(&mut world, ());
    assert_eq!(world.resource::<RiskState>().exposure, 100.0);
}
