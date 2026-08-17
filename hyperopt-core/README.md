# hyperopt-core

Core abstractions for [`hyperopt-rs`](https://crates.io/crates/hyperopt-rs), an
Optuna-shaped, **define-by-run** hyperparameter optimization framework for Rust.

This crate defines the foundational types and the three extension traits
everything else plugs into, so a third party can depend on just `hyperopt-core`
to implement a new sampler or pruner without pulling in SQLite or `rayon`:

- `Study`, `Trial`, `TrialContext`, `Value`, `Distribution`, `StudyState`
- `Sampler` — a pluggable search algorithm
- `Pruner` — a pluggable early-stopping policy
- `Storage` — where trial history lives
- `Suggest` — the portable objective interface (runs local or distributed)

Most users want the [`hyperopt-rs`](https://crates.io/crates/hyperopt-rs) facade
instead, which re-exports this crate alongside the samplers, pruners, and storage
backends behind a `StudyBuilder`.

See the [repository](https://github.com/mi7plus/hyperopt-rs) for the full guide.

## License

MIT © mi7plus
