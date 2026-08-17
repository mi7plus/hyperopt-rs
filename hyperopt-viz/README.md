# hyperopt-viz

Optional visualization and parameter-importance analysis for
[`hyperopt-rs`](https://crates.io/crates/hyperopt-rs) studies — turning a
completed study into insight rather than just a best-trial number:

- `optimization_history_svg(...)` / `best_so_far(...)` — the best-value-so-far
  curve, rendered as an SVG.
- `parameter_importance(...)` — a lightweight, documented decision-stump
  importance proxy; cheap and model-free.
- `fanova_importance(...)` — the full random-forest **fANOVA** (Hutter et al.
  2014): per-parameter main-effect importance from an exact functional-ANOVA
  decomposition of the forest, marginalizing over the other parameters.
- `fanova_interactions(...)` — the second-order (pairwise) terms: how much
  variance each pair explains together beyond their main effects.

Depends only on `plotters` (SVG backend) and `rand`, so it stays cheap to build.
See the [repository](https://github.com/mi7plus/hyperopt-rs) for the full guide.

## License

MIT © mi7plus
