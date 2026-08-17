# hyperopt-samplers

Pluggable `Sampler` implementations for
[`hyperopt-rs`](https://crates.io/crates/hyperopt-rs), the Optuna-shaped
hyperparameter optimization framework. All are interchangeable through one
`Study` API:

- `RandomSampler` — independent random draws; the baseline.
- `GridSampler` — exhaustive enumeration over a caller-provided grid.
- `TpeSampler` — adaptive Tree-structured Parzen Estimator (wraps the
  [`tpe`](https://crates.io/crates/tpe) crate).
- `CmaEsSampler` — Covariance Matrix Adaptation Evolution Strategy: a
  from-scratch (μ/μ_w, λ) implementation with its own symmetric eigensolver and
  reflective bound handling. On a 3-D sphere with an 80-trial budget it reaches a
  mean best of ~0.5 versus random's ~7.6.

Most users want the [`hyperopt-rs`](https://crates.io/crates/hyperopt-rs) facade,
which re-exports these. See the
[repository](https://github.com/mi7plus/hyperopt-rs) for the full guide.

## License

MIT © mi7plus
