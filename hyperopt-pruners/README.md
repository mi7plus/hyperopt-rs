# hyperopt-pruners

Pluggable `Pruner` (early-stopping) policies for
[`hyperopt-rs`](https://crates.io/crates/hyperopt-rs), the Optuna-shaped
hyperparameter optimization framework:

- `NopPruner` — never prunes; the baseline.
- `MedianPruner` — prunes a trial whose latest intermediate value is worse than
  the median of other trials at the same step (mirrors Optuna). On a synthetic
  30-step benchmark it cuts ~50% of objective evaluations with no loss in best
  value.
- `SuccessiveHalvingPruner` — asynchronous successive halving (ASHA).

Report intermediate values with `trial.report(step, value)` and check
`trial.should_prune()` inside the objective.

Most users want the [`hyperopt-rs`](https://crates.io/crates/hyperopt-rs) facade,
which re-exports these. See the
[repository](https://github.com/mi7plus/hyperopt-rs) for the full guide.

## License

MIT © mi7plus
