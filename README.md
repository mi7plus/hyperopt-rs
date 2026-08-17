# hyperopt-rs

An **Optuna-shaped hyperparameter optimization framework for Rust**: pluggable
search algorithms, pruning / early-stopping, a persistence layer, and local
parallel trial execution — composing with the wider Rust ML ecosystem rather
than duplicating it.

It is built around **define-by-run**: the search space isn't declared up front
as a static description, it is *discovered by calling* `suggest_*` methods inside
the objective, which makes conditional / dynamic search spaces natural.

```rust
use hyperopt_rs::prelude::*;

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
pruning): `cargo run -p hyperopt-rs --example quickstart`.

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
| [`hyperopt-rs`](hyperopt) | Ergonomic facade (crate `hyperopt-rs`, imported as `hyperopt_rs`): re-exports + `StudyBuilder` + `prelude`. |
| [`hyperopt-viz`](hyperopt-viz) | Optional: optimization-history plot, a parameter-importance proxy, and full random-forest **fANOVA** importance. |
| [`hyperopt-distributed`](hyperopt-distributed) | Optional: a `Coordinator` server + `Worker` client for **multi-machine** distributed execution over TCP. |

## Features

- **Samplers** — random (baseline), exhaustive grid, adaptive TPE, and
  **CMA-ES**. All implement the same `Sampler` trait and are interchangeable
  through one `Study` API. On a 3-D sphere with an 80-trial budget, both TPE and
  CMA-ES reach a mean best of ~0.5 versus random's ~7.6 (see
  `hyperopt/tests/phase15_samplers.rs` and `hyperopt/tests/phase16_cmaes.rs`).
  CMA-ES is a from-scratch (μ/μ_w, λ) implementation — including its own
  symmetric eigensolver — that adapts a full covariance to the objective's local
  geometry; it optimizes the numeric parameters and leaves categoricals to
  independent sampling.
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

- **Visualization & importance** — `optimization_history_svg(...)` renders the
  best-value-so-far curve; `parameter_importance(...)` gives a lightweight
  decision-stump proxy for which parameters matter, and `fanova_importance(...)`
  runs the full random-forest **fANOVA** (Hutter et al. 2014), attributing
  objective variance to each parameter's main effect by an exact functional-ANOVA
  decomposition of the forest (see `hyperopt-viz/tests/viz.rs`).

- **Distributed execution** — `hyperopt-distributed` adds a `Coordinator` server
  and a `Worker` client so trials run across **many machines** against one
  authoritative study. The coordinator owns the sampler/pruner/storage and
  assigns every trial number atomically; workers run the objective locally and
  round-trip each `suggest_*` call back. Because the remote trial implements the
  same `Suggest` trait as `TrialContext`, one objective closure runs unchanged
  locally or distributed (`cargo run -p hyperopt-distributed --example distributed`).

## Building and testing

```bash
cargo build --workspace
cargo test  --workspace                      # core, samplers, pruners, storage, viz, distributed
cargo test  -p hyperopt-rs --features parallel  # includes the parallel-execution tests
```

## Scope (what's implemented vs. deliberately not)

Implemented and tested: the core
define-by-run API, all samplers, all three pruners, both storage backends,
local parallel execution, and the visualization / importance layer — plus the
three items originally deferred past v1, now built:

- **True multi-machine distributed execution** — `hyperopt-distributed` adds a
  coordinator/worker system (see the Distributed-execution feature above). The
  earlier `SqliteStorage` shared-file approach remains as a lighter-weight,
  local-only middle ground.
- **fANOVA-based parameter importance** — `fanova_importance(...)` implements the
  full random-forest functional-ANOVA method alongside the original decision-stump
  proxy, which is kept as the cheap default. fANOVA marginalizes over the other
  parameters (the proxy does not) and so is far less fooled by correlated
  sampling; its main effects deliberately do not fold in interaction variance.
- **CMA-ES sampling** — `CmaEsSampler` is a from-scratch (μ/μ_w, λ) implementation
  interchangeable with the other samplers through the `Sampler` trait. Bound
  handling defaults to reflection (`BoundHandling::Reflect`) — folding out-of-box
  draws back inside with a tent map instead of piling them on the boundary — with
  clamping selectable.

And the follow-ups to those, also now built:

- **fANOVA interaction effects** — `fanova_interactions(...)` reports the
  second-order terms: how much variance each *pair* of parameters explains
  together beyond their individual main effects.
- **Secured distributed transport** — the coordinator supports an optional
  shared-secret token (`Coordinator::require_token` / `Worker::authenticate`,
  constant-time compared) and, behind the `tls` feature, TLS via `rustls`
  (`listen_tls` / `connect_tls`; `ring` provider, no OpenSSL). Plain TCP remains
  the default for a trusted network.

Not done in this build (they depend on external repos / a publish step):

- **Notebook integration** — swapping `hyperopt-rs` into the `rust-ml-guide` /
  `model-selection-rs` notebooks (needs those companion repos).
- **Publishing to crates.io** (name reservation + `cargo publish`).
  Publish-readiness prep: run `cargo publish --dry-run` per member in dependency
  order (`hyperopt-core` first, `hyperopt-rs` facade last) and verify names on
  crates.io before committing to them.

## License

MIT © mi7plus
