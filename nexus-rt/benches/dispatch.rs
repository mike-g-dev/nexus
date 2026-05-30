#![allow(
    unused_must_use,
    dead_code,
    clippy::float_cmp,
    clippy::used_underscore_binding,
    clippy::items_after_statements
)]
//! Criterion benchmarks for nexus-rt dispatch hot paths.
//!
//! Run:
//! ```bash
//! cargo bench -p nexus-rt --bench dispatch
//! ```
//!
//! Pin to a core for stable results:
//! ```bash
//! taskset -c 0 cargo bench -p nexus-rt --bench dispatch
//! ```

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nexus_rt::{
    Handler, IntoCallback, IntoHandler, IntoSystem, PipelineBuilder, Res, ResMut, System,
    WorldBuilder,
};

// =============================================================================
// Resources
// =============================================================================

nexus_rt::new_resource!(ValA(u64));
nexus_rt::new_resource!(ValB(u32));
nexus_rt::new_resource!(Out(u64));

// =============================================================================
// Handler dispatch
// =============================================================================

fn bench_handler_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("handler_dispatch");
    group.throughput(Throughput::Elements(1));

    // 0 params
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut handler = (|_event: u64| {}).into_handler(world.registry());

        group.bench_function("0_params", |b| {
            b.iter(|| handler.run(black_box(&mut world), black_box(42u64)));
        });
    }

    // 1 param (Res)
    {
        let mut wb = WorldBuilder::new();
        wb.register(ValA(100));
        let mut world = wb.build();

        fn step(val: Res<ValA>, _event: u64) {
            std::hint::black_box(val.0);
        }

        let mut handler = step.into_handler(world.registry());

        group.bench_function("1_param_res", |b| {
            b.iter(|| handler.run(black_box(&mut world), black_box(42u64)));
        });
    }

    // 2 params (Res + ResMut)
    {
        let mut wb = WorldBuilder::new();
        wb.register(ValA(100));
        wb.register(Out(0));
        let mut world = wb.build();

        fn step(val: Res<ValA>, mut out: ResMut<Out>, _event: u64) {
            out.0 = val.0;
        }

        let mut handler = step.into_handler(world.registry());

        group.bench_function("2_params_res_resmut", |b| {
            b.iter(|| handler.run(black_box(&mut world), black_box(42u64)));
        });
    }

    group.finish();
}

// =============================================================================
// Callback dispatch (context-owning)
// =============================================================================

fn bench_callback_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("callback_dispatch");
    group.throughput(Throughput::Elements(1));

    struct Ctx {
        count: u64,
    }

    // 0 params
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();

        fn step(ctx: &mut Ctx, _event: u64) {
            ctx.count += 1;
        }

        let mut cb = step.into_callback(Ctx { count: 0 }, world.registry());

        group.bench_function("0_params", |b| {
            b.iter(|| cb.run(black_box(&mut world), black_box(42u64)));
        });
    }

    // 2 params
    {
        let mut wb = WorldBuilder::new();
        wb.register(ValA(100));
        wb.register(Out(0));
        let mut world = wb.build();

        fn step(ctx: &mut Ctx, val: Res<ValA>, mut out: ResMut<Out>, _event: u64) {
            out.0 = val.0 + ctx.count;
            ctx.count += 1;
        }

        let mut cb = step.into_callback(Ctx { count: 0 }, world.registry());

        group.bench_function("2_params", |b| {
            b.iter(|| cb.run(black_box(&mut world), black_box(42u64)));
        });
    }

    group.finish();
}

// =============================================================================
// Pipeline dispatch
// =============================================================================

fn bench_pipeline_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_dispatch");
    group.throughput(Throughput::Elements(1));

    // 3-stage bare (no World access)
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let reg = world.registry();

        let mut p = PipelineBuilder::<u64>::new()
            .then(|x: u64| x.wrapping_mul(3), reg)
            .then(|x: u64| x.wrapping_add(7), reg)
            .then(|x: u64| x >> 1, reg);

        group.bench_function("3_stage_bare", |b| {
            b.iter(|| black_box(p.run(black_box(&mut world), black_box(42u64))));
        });
    }

    // 3-stage with World resources
    {
        let mut wb = WorldBuilder::new();
        wb.register(ValA(100));
        wb.register(ValB(7));
        wb.register(Out(0));
        let mut world = wb.build();
        let reg = world.registry();

        fn add_a(val: Res<ValA>, x: u64) -> u64 {
            x.wrapping_add(val.0)
        }

        fn mul_b(val: Res<ValB>, x: u64) -> u64 {
            x.wrapping_mul(val.0 as u64)
        }

        fn store(mut out: ResMut<Out>, x: u64) {
            out.0 = x;
        }

        let mut p = PipelineBuilder::<u64>::new()
            .then(add_a, reg)
            .then(mul_b, reg)
            .then(store, reg);

        group.bench_function("3_stage_world_access", |b| {
            b.iter(|| {
                p.run(black_box(&mut world), black_box(42u64));
                black_box(())
            });
        });
    }

    // 5-stage with guard
    {
        let mut wb = WorldBuilder::new();
        wb.register(ValA(100));
        wb.register(Out(0));
        let mut world = wb.build();
        let reg = world.registry();

        fn add_a(val: Res<ValA>, x: u64) -> u64 {
            x.wrapping_add(val.0)
        }

        let mut p = PipelineBuilder::<u64>::new()
            .then(|x: u64| x.wrapping_mul(3), reg)
            .guard(|x: &u64| *x > 0, reg)
            .map(add_a, reg)
            .map(|x: u64| x >> 1, reg)
            .map(|x: u64| x.wrapping_add(1), reg);

        group.bench_function("5_stage_with_guard", |b| {
            b.iter(|| black_box(p.run(black_box(&mut world), black_box(42u64))));
        });
    }

    group.finish();
}

// =============================================================================
// System dispatch
// =============================================================================

fn bench_system_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("system_dispatch");
    group.throughput(Throughput::Elements(1));

    let mut wb = WorldBuilder::new();
    wb.register(ValA(0));
    wb.register(Out(0));
    let mut world = wb.build();

    fn reconcile(val: Res<ValA>, mut out: ResMut<Out>) -> bool {
        out.0 = val.0.wrapping_add(1);
        true
    }

    let mut sys = reconcile.into_system(world.registry());

    group.bench_function("2_params_bool_return", |b| {
        b.iter(|| black_box(sys.run(black_box(&mut world))));
    });

    group.finish();
}

// =============================================================================
// World resource access (cold path)
// =============================================================================

fn bench_world_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_access");
    group.throughput(Throughput::Elements(1));

    let mut wb = WorldBuilder::new();
    wb.register(ValA(42));
    wb.register(Out(0));
    let mut world = wb.build();

    group.bench_function("resource_read", |b| {
        b.iter(|| {
            let val = world.resource::<ValA>();
            std::hint::black_box(val.0);
        });
    });

    group.bench_function("resource_mut", |b| {
        b.iter(|| {
            let val = world.resource_mut::<Out>();
            val.0 += 1;
        });
    });

    group.finish();
}

// =============================================================================
// Template stamping
// =============================================================================

fn bench_template(c: &mut Criterion) {
    use nexus_rt::template::HandlerTemplate;

    let mut group = c.benchmark_group("template");
    group.throughput(Throughput::Elements(1));

    let mut wb = WorldBuilder::new();
    wb.register(Out(0));
    let mut world = wb.build();
    let reg = world.registry();

    nexus_rt::handler_blueprint!(SinkBlueprint, Event = u64, Params = (ResMut<'static, Out>,));

    fn sink(mut out: ResMut<Out>, event: u64) {
        out.0 = event;
    }

    let template = HandlerTemplate::<SinkBlueprint>::new(sink, reg);

    group.bench_function("stamp_handler", |b| {
        b.iter(|| {
            let mut h = template.generate();
            h.run(&mut world, 42u64);
        });
    });

    group.finish();
}

// =============================================================================
// Reactor dispatch
// =============================================================================

#[cfg(feature = "reactors")]
fn bench_reactor_dispatch(c: &mut Criterion) {
    use nexus_notify::Token;
    use nexus_rt::ReactorNotify;

    let mut group = c.benchmark_group("reactor_dispatch");
    group.throughput(Throughput::Elements(1));

    struct Ctx {
        _reactor_id: Token,
    }

    fn noop_step(_ctx: &mut Ctx) {}

    fn one_res_step(_ctx: &mut Ctx, _val: Res<ValA>) {}

    // 1 reactor, noop
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();

        let src = world.register_source();
        world
            .spawn_reactor(|id| Ctx { _reactor_id: id }, noop_step)
            .subscribe(src);

        group.bench_function("1_reactor_noop", |b| {
            b.iter(|| {
                world.resource_mut::<ReactorNotify>().mark(black_box(src));
                black_box(world.dispatch_reactors());
            });
        });
    }

    // 10 reactors, noop
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();

        let src = world.register_source();
        for _ in 0..10 {
            world
                .spawn_reactor(|id| Ctx { _reactor_id: id }, noop_step)
                .subscribe(src);
        }

        group.bench_function("10_reactors_noop", |b| {
            b.iter(|| {
                world.resource_mut::<ReactorNotify>().mark(black_box(src));
                black_box(world.dispatch_reactors());
            });
        });
    }

    // 50 reactors, noop
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();

        let src = world.register_source();
        for _ in 0..50 {
            world
                .spawn_reactor(|id| Ctx { _reactor_id: id }, noop_step)
                .subscribe(src);
        }

        group.bench_function("50_reactors_noop", |b| {
            b.iter(|| {
                world.resource_mut::<ReactorNotify>().mark(black_box(src));
                black_box(world.dispatch_reactors());
            });
        });
    }

    // 10 reactors, 1 Res param
    {
        let mut wb = WorldBuilder::new();
        wb.register(ValA(42));
        let mut world = wb.build();

        let src = world.register_source();
        for _ in 0..10 {
            world
                .spawn_reactor(|id| Ctx { _reactor_id: id }, one_res_step)
                .subscribe(src);
        }

        group.bench_function("10_reactors_1_res", |b| {
            b.iter(|| {
                world.resource_mut::<ReactorNotify>().mark(black_box(src));
                black_box(world.dispatch_reactors());
            });
        });
    }

    group.finish();
}

// =============================================================================
// Entry point
// =============================================================================

#[cfg(not(feature = "reactors"))]
criterion_group!(
    benches,
    bench_handler_dispatch,
    bench_callback_dispatch,
    bench_pipeline_dispatch,
    bench_system_dispatch,
    bench_world_access,
    bench_template,
);

#[cfg(feature = "reactors")]
criterion_group!(
    benches,
    bench_handler_dispatch,
    bench_callback_dispatch,
    bench_pipeline_dispatch,
    bench_system_dispatch,
    bench_world_access,
    bench_template,
    bench_reactor_dispatch,
);

criterion_main!(benches);
