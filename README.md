# hyperopt-rs

An **Optuna-shaped hyperparameter optimization framework for Rust**: pluggable
search algorithms, pruning / early-stopping, a persistence layer, and local
parallel trial execution — composing with the wider Rust ML ecosystem rather
than duplicating it.

It is built around **define-by-run**: the search space isn't declared up front
as a static description, it is *discovered by calling* `suggest_*` methods inside
the objective, which makes conditional / dynamic search spaces natural.

```rust
use hyperopt::prelude::*;

let study = StudyBuilder::new("quadratic")
    .direction(Direction::Minimize)
    .sampler(TpeSampler::seeded(42))
    .build()?;

study.optimize(|trial| {
    let x = trial.suggest_float("x", -10.0, 10.0);
    let y = trial.suggest_float("y", -10.0, 10.0);
    Ok((x - 2.0).powi(2) + (y + 3.0).powi(2))
}, 200)?;

println!("best = {:?}", study.best_trial()?);
```

Run the full quickstart (define-by-run, a conditional search space, and median
pruning): `cargo run -p hyperopt --example quickstart`.

## Workspace layout

The framework is split into sub-crates because `Sampler` / `Pruner` / `Storage`
are the real extension points — a third party can depend on just
`hyperopt-core` and implement a new sampler without pulling in SQLite or
`rayon`.

| Crate | Contents |
|---|---|
| [`hyperopt-core`](hyperopt-core) | `Study`, `Trial`, `TrialContext`, `Value`, `Distribution`, and the `Sampler` / `Pruner` / `Storage` traits. Optional `parallel` feature adds `Study::optimize_parallel`. |
| [`hyperopt-samplers`](hyperopt-samplers) | `RandomSampler`, `GridSampler`, `TpeSampler` (wraps the [`tpe`](https://crates.io/crates/tpe) crate). |
| [`hyperopt-pruners`](hyperopt-pruners) | `NopPruner`, `MedianPruner`, `SuccessiveHalvingPruner` (ASHA). |
| [`hyperopt-storage`](hyperopt-storage) | `InMemoryStorage`, `SqliteStorage` (resumable studies; feature `sqlite`). |
| [`hyperopt`](hyperopt) | Ergonomic facade: re-exports + `StudyBuilder` + `prelude`. |
| [`hyperopt-viz`](hyperopt-viz) | Optional: optimization-history plot + a parameter-importance proxy. |

## Features

- **Samplers** — random (baseline), exhaustive grid, and adaptive TPE. All
  three implement the same `Sampler` trait and are interchangeable through one
  `Study` API. On a 3-D sphere with an 80-trial budget, TPE reaches a mean best
  of ~0.5 versus random's ~7.6 (see `hyperopt/tests/phase15_samplers.rs`).
- **Pruning** — report intermediate values with `trial.report(step, value)` and
  check `trial.should_prune()`. On a synthetic 30-step benchmark, `MedianPruner`
  cuts ~50% of objective evaluations with no loss in best value
  (`hyperopt/tests/phase2_pruning_savings.rs`).
- **Persistence** — `SqliteStorage` (schema-versioned) lets a study be optimized
  partway, dropped, and resumed in a fresh process; adaptive samplers pick up
  the loaded history (`hyperopt/tests/phase3_storage.rs`).
- **Local parallelism** — `Study::optimize_parallel(obj, n_trials, n_workers)`
  behind the `parallel` feature (rayon). ~3.9x speedup with 4 workers on a
  sleep-bound objective (`hyperopt/tests/phase4_parallel.rs`).

  Under parallelism a sampler necessarily works from a *partial, slightly stale*
  view of study history (several trials may be in flight before earlier ones are
  saved). This matches Optuna's behaviour and is **by design, not a bug** — it is
  why parallel and sequential runs of the same study can diverge somewhat.
  Compare best-values, not exact trajectories.

- **Visualization** — `optimization_history_svg(...)` renders the
  best-value-so-far curve; `parameter_importance(...)` gives a lightweight
  proxy for which parameters matter.

## Building and testing

```bash
cargo build --workspace
cargo test  --workspace                      # core, samplers, pruners, storage, viz
cargo test  -p hyperopt --features parallel  # includes the parallel-execution tests
```

## Scope (what's implemented vs. deliberately not)

Implemented and tested: **Phases 1–5** of [`PLAN.md`](PLAN.md) — the core
define-by-run API, all three samplers, all three pruners, both storage backends,
local parallel execution, and the visualization / importance layer.

Explicitly **out of scope for v1** (stated up front, not discovered later):

- **True multi-machine distributed execution** — v1 covers local multi-threaded
  parallelism only. `SqliteStorage` does allow several local processes to share
  one study file as a lighter-weight middle ground.
- **fANOVA-based parameter importance** — `hyperopt-viz` ships a simpler,
  clearly-documented decision-stump *proxy* (variance explained by the best
  single split) instead. It reliably ranks an influential parameter above an
  irrelevant one but is **not** the same statistical method as Optuna's fANOVA
  and ignores interaction effects. A `smartcore`-random-forest importance would
  be the natural richer follow-up.
- **CMA-ES sampling** — a plausible v2 addition, not required for the framework
  to be useful.

Not done in this build (they depend on external repos / a publish step, tracked
in `PLAN.md`):

- **Phase 6** — swapping `hyperopt-rs` into the `rust-ml-guide` /
  `model-selection-rs` notebooks (needs those companion repos).
- **Phase 7** — publishing to crates.io (name reservation + `cargo publish`).
  Publish-readiness prep: run `cargo publish --dry-run` per member in dependency
  order (`hyperopt-core` first, `hyperopt` facade last) and verify names on
  crates.io before committing to them.

## License

MIT © mi7plus
