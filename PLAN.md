# Project Plan: `hyperopt-rs` — A Hyperparameter Optimization Framework for Rust

Phased, multi-crate build plan formatted for Claude Code. This is
explicitly the largest item in the ecosystem-gaps review — scoped here as
a **Cargo workspace of several crates**, built in phases rather than
milestones alone, since a single linear milestone list would understate
how much of this is genuine framework design versus incremental feature
addition. Drop this file in the workspace root as `PLAN.md` (or
`CLAUDE.md`) for persistent context across sessions, and treat each phase
as a separate work block — don't attempt Phase 4+ before Phase 1–3 are
solid and actually in use.

**Goal**: an Optuna-shaped framework — pluggable search algorithms,
pruning/early-stopping, a persistence layer, and (local) parallel trial
execution — composing with `model-selection-rs` (for the evaluation loop
each trial calls into) and `tpe` (as one pluggable sampler among several,
not reimplemented from scratch) rather than duplicating either.

**Explicitly out of scope for v1**, stated up front rather than
discovered mid-project: true multi-machine distributed execution (v1
covers local multi-threaded parallelism only), fANOVA-based parameter
importance (a simpler proxy is scoped in Phase 5 instead), and CMA-ES
sampling (a plausible v2 addition, not required for the framework to be
genuinely useful).

---

## Workspace structure

```
hyperopt-rs/
├── Cargo.toml                  # workspace root
├── hyperopt-core/                # Study, Trial, Sampler/Pruner/Storage traits
├── hyperopt-samplers/            # RandomSampler, GridSampler, TpeSampler
├── hyperopt-pruners/              # MedianPruner, SuccessiveHalvingPruner, NopPruner
├── hyperopt-storage/              # InMemoryStorage, SqliteStorage
├── hyperopt/                      # facade crate — re-exports the above with an ergonomic top-level API
└── hyperopt-viz/                   # optional: optimization-history / param-importance plots
```

Splitting into sub-crates now (rather than one big crate later) matters
because `Sampler`/`Pruner`/`Storage` are the actual extension points —
third parties should eventually be able to depend on just
`hyperopt-core` and implement a new sampler without pulling in SQLite or
`rayon`.

---

## Phase 1 — Core abstractions and the define-by-run API

**Goal:** the foundational design decision this entire framework rests
on — get this right before writing any sampler or pruner, since changing
it later means rewriting everything built on top.

### Design decision: define-by-run, not define-and-run
Optuna's key departure from earlier tools (Hyperopt, scikit-optimize) is
that the search space isn't declared upfront as a static dict — it's
discovered by *calling* `trial.suggest_*()` methods inside the objective
function itself, which allows conditional/dynamic search spaces (e.g.
suggesting a different set of hyperparameters depending on which model
type an earlier suggestion picked). Adopt this pattern deliberately; it's
more work to implement (samplers must handle a possibly-different
parameter set per trial) but is the right foundation, matching what
`hyperopt-rs` is explicitly positioning itself against.

Tasks (`hyperopt-core`):
- [ ] `Value` enum: `Float(f64)`, `Int(i64)`, `Categorical(String)` (or a
      generic categorical value type — decide during implementation
      whether to support arbitrary types or restrict to
      string/enum-like categoricals for v1 simplicity).
- [ ] `Distribution` enum: `Uniform { low, high }`, `LogUniform { low,
      high }`, `IntUniform { low, high }`, `Categorical { choices }`.
- [ ] `TrialState` enum: `Running`, `Complete`, `Pruned`, `Failed`.
- [ ] `Trial` struct: trial number, suggested params so far (ordered
      map), intermediate values (`Vec<(step, value)>`, populated via
      `.report()`, needed for pruning in Phase 2), final objective value,
      state.
- [ ] `TrialContext` — the handle passed into the user's objective
      closure, exposing `suggest_float`, `suggest_int`,
      `suggest_categorical`, `report(step, value)`, and (stubbed until
      Phase 2) `should_prune()`. Each `suggest_*` call both records the
      distribution+value on the trial *and* asks the active `Sampler` what
      value to use — this dual responsibility (recording + delegating) is
      the crux of the define-by-run mechanism, worth getting the ordering
      right (record distribution → ask sampler → return value) since
      later phases depend on the recorded history being accurate.
- [ ] `Sampler` trait: `fn suggest(&mut self, study_state: &StudyState,
      trial: &Trial, param_name: &str, distribution: &Distribution) ->
      Value`. Takes `study_state` (read access to prior completed trials)
      so adaptive samplers (Phase 1.5/TPE) have what they need without a
      separate channel.
- [ ] `Study` struct: holds a `Box<dyn Sampler>`, direction
      (`Minimize`/`Maximize`), and (stubbed until Phase 3) a storage
      backend; `optimize(objective_fn, n_trials)` runs trials
      sequentially for now (parallel execution is Phase 4).
- [ ] `ObjectiveError` handling: if the objective closure panics or
      returns an error, mark the trial `Failed` and continue rather than
      aborting the whole study — a single bad trial shouldn't kill a
      long-running optimization run.

### First sampler: `RandomSampler`
Implement in `hyperopt-samplers` alongside the core design, specifically
because it's the simplest possible `Sampler` implementation and is the
right thing to validate the whole `Trial`/`TrialContext`/`Study` plumbing
against before building anything more complex on top.

Tasks:
- [ ] `RandomSampler`: uniform/log-uniform/categorical random sampling
      per distribution type, no adaptive state needed.
- [ ] End-to-end test: optimize a simple known function (e.g. a 2D
      quadratic bowl with a known minimum) with `RandomSampler` across
      enough trials that the best found value converges close to the true
      minimum — this is the first real proof the core plumbing works.

**Definition of done:** a user can write an objective closure using
`trial.suggest_float(...)` calls, run `study.optimize(objective, 100)`
with `RandomSampler`, and get a sensible best-trial result on a toy
problem.

---

## Phase 1.5 — `GridSampler` and `TpeSampler`

**Goal:** round out the sampler family to match what the guide's
Optimization chapter addenda already established conceptually (grid,
random, TPE), now as real pluggable `Sampler` implementations rather than
hand-rolled loops.

Tasks:
- [ ] `GridSampler`: exhaustive enumeration over a provided discrete grid
      per parameter — note explicitly that this only makes sense for
      parameters with a finite, pre-specified set of values (unlike
      `RandomSampler`, which can sample continuous distributions
      directly), and document this constraint clearly in the type's docs.
- [ ] `TpeSampler`: **wraps the existing `tpe` crate rather than
      reimplementing Tree-structured Parzen Estimation from scratch** —
      this is a real, correct integration point, not a missing-crate gap
      in itself. The implementation work here is translating between
      `hyperopt-core`'s `Distribution`/`Trial` history representation and
      whatever input shape `tpe` expects, and handling the case where
      `tpe`'s API doesn't map 1:1 onto every `Distribution` variant this
      framework supports (e.g. if `tpe` only handles certain distribution
      shapes, document that as a `TpeSampler`-specific limitation rather
      than silently under-supporting it).
- [ ] Verify `tpe`'s current API surface against the pinned version before
      finalizing `TpeSampler`'s implementation — same diligence applied
      to every other crate-integration point across this whole project.
- [ ] Comparison test: run all three samplers (`Random`, `Grid`, `Tpe`) on
      the same toy objective with the same trial budget, confirm `Tpe`
      converges to a better best-value on average than `Random` given
      enough trials — this is the actual point of having an adaptive
      sampler, worth a real test proving it rather than assuming.

**Definition of done:** all three samplers implement `Sampler` and are
interchangeable via the same `Study` API; the TPE-vs-random comparison
test passes.

---

## Phase 2 — Pruning and intermediate reporting

**Goal:** early-stopping for expensive trials — a trial reports its
progress at intermediate steps (e.g. epoch-by-epoch validation score for
an iterative model), and a `Pruner` decides whether continuing is worth
it based on how other trials performed at the same step.

Tasks (`hyperopt-pruners`):
- [ ] `Pruner` trait: `fn should_prune(&self, study_state: &StudyState,
      trial: &Trial) -> bool` — called from `TrialContext::should_prune()`
      inside the user's objective, after a `.report()` call; the user's
      loop is expected to check this and break early if it returns true.
- [ ] `NopPruner`: always returns false — the default/no-op, useful when
      pruning isn't wanted but the API shape should stay consistent.
- [ ] `MedianPruner`: at a given step, compare the current trial's
      intermediate value against the median of other trials' values at
      that same step (among trials that reached at least that step);
      prune if sufficiently worse. Requires read access to other trials'
      full intermediate-value history via `study_state` — this is why
      `Sampler`/`Pruner` both take `study_state` rather than only their
      own trial.
- [ ] `SuccessiveHalvingPruner` (ASHA-style): allocate a shrinking budget
      across rungs, promoting only the top fraction of trials at each
      rung to continue — more involved than `MedianPruner`, implement
      second and treat as the "advanced" pruner option, documented as
      such.
- [ ] Wire `TrialContext::should_prune()` into the real pruner (replacing
      Phase 1's stub) and update `Study::optimize` to mark pruned trials
      with `TrialState::Pruned` rather than `Complete`.
- [ ] Test: an objective simulating an iterative process (a synthetic
      "epoch loop" with a known bad-trajectory case) correctly gets pruned
      by `MedianPruner` before completing all steps, while a good-
      trajectory trial isn't.

**Definition of done:** pruning demonstrably saves total objective-
function evaluations on a synthetic multi-step benchmark, measured and
reported (evaluation count with pruning on vs. off), not just
implemented and assumed to help.

---

## Phase 3 — Persistence: `InMemoryStorage` and `SqliteStorage`

**Goal:** studies need to survive process restarts for anything beyond a
single-session optimization run — this is a real gap between "toy
framework" and "actually useful for long-running HPO work."

Tasks (`hyperopt-storage`):
- [ ] `Storage` trait: `save_trial`, `load_trials`, `save_study_metadata`,
      `load_study_metadata` — abstracts over where trial history lives so
      `Study` doesn't need to know or care.
- [ ] `InMemoryStorage`: the Phase 1–2 default, `Vec<Trial>` behind a
      lock — no persistence across restarts, but zero setup cost.
- [ ] `SqliteStorage` (via `rusqlite`): mirrors Optuna's RDB storage
      pattern — trials, their parameters, and intermediate values
      persisted to a SQLite file, so a study can be resumed (`Study::load
      ("study.db", study_name)`) after a process restart, and multiple
      processes can in principle share the same storage file for basic
      local-multi-process coordination (a lighter-weight alternative to
      full distributed execution, worth noting explicitly as a practical
      middle ground rather than requiring Phase 4's full parallel
      execution machinery for this specific use case).
- [ ] Schema versioning: even at v1, include a schema-version field in the
      SQLite file so future storage-format changes don't silently corrupt
      or misread old study files — cheap to add now, expensive to retrofit
      later.

**Definition of done:** a study optimized partway, persisted to
`SqliteStorage`, then reloaded in a fresh process, continues suggesting
trials that account for the already-completed history (verified by
confirming `TpeSampler`'s suggestions after reload differ from a cold
start, proving the historical trials actually got loaded and used).

---

## Phase 4 — Local parallel trial execution

**Goal:** multiple trials running concurrently on one machine — explicitly
**not** the multi-machine distributed execution named as out of scope for
v1, but the more tractable and still genuinely valuable local version.

Tasks:
- [ ] `Study::optimize_parallel(objective_fn, n_trials, n_workers)` using
      `rayon`, gated behind a `parallel` feature flag (same convention as
      `model-selection-rs`'s `parallel` feature) so single-threaded users
      don't pay for the dependency.
- [ ] **Concurrency design, resolved explicitly rather than left
      implicit**: adaptive samplers like `TpeSampler` want to see prior
      completed trials to make good suggestions, but with true parallel
      execution, several trials may be suggested before earlier ones
      finish — this means samplers necessarily work with a *partial,
      slightly stale* view of study history under parallelism. This is
      standard practice (Optuna itself works this way under parallel
      execution) and not a bug to "fix," but document it clearly so users
      understand why parallel and sequential runs of the same study can
      diverge somewhat, rather than assuming a bug when they compare
      results.
- [ ] Thread-safety: `Sampler` and `Storage` implementations need
      interior mutability (`Mutex`/`RwLock`) to be shared safely across
      worker threads — audit `RandomSampler`, `TpeSampler`,
      `InMemoryStorage`, and `SqliteStorage` specifically for this rather
      than assuming the Phase 1–3 implementations are automatically
      thread-safe.
- [ ] Benchmark: wall-clock time for N trials sequential vs. parallel
      (matching the "back the claim with a real number" convention
      established in the Multithreading chapter and `model-selection-rs`
      benchmarks), on an objective with an artificial per-trial cost so
      the parallelism benefit is measurable and not dominated by
      per-trial overhead.

**Definition of done:** parallel execution produces valid results (no data
races, confirmed via `cargo test` under `--features parallel` plus a
stress test with many concurrent trials) and the benchmark shows a real
speedup; the staleness/divergence behavior under parallelism is
documented, not just present.

---

## Phase 5 — Visualization and a simple parameter-importance proxy

**Goal:** the analysis layer that turns a completed study into insight,
not just a best-trial number.

Tasks (`hyperopt-viz`, optional crate — depends on `plotters` and,
optionally, `plotters-statistical` if that crate exists in this
ecosystem by the time this phase is reached):
- [ ] **Optimization history plot**: best-value-so-far vs. trial number —
      directly the same chart shape already established in the guide's
      Optimization chapter addenda (`search-strategy-comparison.ipynb`'s
      best-score-vs-evaluations plots); this crate should make that a
      one-call function rather than something re-hand-rolled per project.
- [ ] **Parameter-importance proxy**: full fANOVA is out of scope for v1
      (genuinely complex to implement correctly), but a simpler, still
      useful proxy is tractable — fit a quick `smartcore` `RandomForest`
      regressor on (trial parameters → objective value) across completed
      trials, and use its feature-importance output as a stand-in for
      parameter importance. Document plainly that this is an
      approximation, not the same statistical method Optuna's fANOVA
      implementation uses, but note it reuses infrastructure this whole
      ecosystem already has (the guide's own Ensemble & Forest Models
      chapter) rather than requiring new statistical machinery.
- [ ] **Parallel coordinates plot** (optional/stretch within this phase):
      one line per trial across parameter axes, colored by objective
      value — a standard Optuna visualization; include if time allows,
      not required for Phase 5 to be considered done.

**Definition of done:** optimization-history and parameter-importance
outputs render correctly on a completed study from Phase 1–4's toy
benchmarks; the parameter-importance proxy correctly ranks a synthetic
objective's genuinely-important parameter above an irrelevant one it was
constructed to include.

---

## Phase 6 — Integration with `rust-ml-guide` and `model-selection-rs`

**Goal:** the concrete proof this framework is more than an isolated
exercise — it needs to actually replace what the guide's Optimization
chapter addenda were doing by hand or via bare `tpe`.

Tasks:
- [ ] Add `hyperopt` (the facade crate) to the guide's pre-warmed evcxr
      cache.
- [ ] Rework `05b-optimization/hyperparameter-search.ipynb` and
      `search-strategy-comparison.ipynb` to use `hyperopt-rs`'s
      `Study`/`Sampler` API directly instead of the manually-implemented
      grid/random search and raw `tpe` calls — the three-way sampler
      comparison these notebooks already wanted becomes a direct
      `RandomSampler` vs. `GridSampler` vs. `TpeSampler` comparison
      through one consistent API.
- [ ] Wire the Evaluation chapter's `NestedCV` (from `model-selection-rs`)
      as the "tune" closure `NestedCV::new(...)` expects — a `hyperopt-rs`
      `Study::optimize` call is exactly the kind of tuning procedure that
      closure was designed to accept generically, so this is a genuine
      composition of the two companion crates, not just two features
      sitting side by side.
- [ ] Add a pruning example to the guide specifically — none of the
      original Optimization addenda covered early-stopping, since it
      wasn't tractable without a real framework; this is new content this
      crate's existence enables, worth adding rather than treating Phase
      6 as pure replacement of old content.
- [ ] Confirm every touched notebook still executes cleanly via
      `jupyter-book build` after these changes.

**Definition of done:** no hand-rolled grid/random search loop or bare
`tpe` call remains in the guide's Optimization chapters; at least one
notebook demonstrates pruning, which wasn't previously covered anywhere
in the guide.

---

## Phase 7 — Publish

Tasks:
- [ ] Verify crate names are available on crates.io before committing to
      them across the workspace (`hyperopt-rs` and sub-crate names) —
      confirm rather than assume, given this is a more visible/ambitious
      project name than the earlier companion crates.
- [ ] `cargo publish --dry-run` for each workspace member in dependency
      order (`hyperopt-core` first, `hyperopt` facade last).
- [ ] Publish all crates at `v0.1.0`.
- [ ] Tag the GitHub release with the Phase 4 parallel-speedup benchmark
      and Phase 1.5 sampler-comparison results linked for visibility.
- [ ] README (workspace root) stating plainly what's implemented (Phases
      1–6) versus explicitly out of scope (true distributed execution,
      fANOVA, CMA-ES) — same honesty convention as every other crate in
      this series, especially important here given how directly this
      project positions itself against Optuna/Hyperopt/Ray Tune by name.

---

## Sequencing note

Given the scope, treat Phases 1–3 as the minimum viable framework worth
using in the guide at all (core API + samplers + persistence). Phase 4
(parallelism) and Phase 5 (visualization) are genuinely valuable but
separable — if time is constrained, a v0.1 release after Phase 3 with
Phases 4–5 as a documented v0.2 roadmap is a reasonable stopping point,
consistent with how `plotters-statistical`'s heatmap/pair-plot chart types
were deliberately deferred to its own v0.2 rather than bloating v0.1.
