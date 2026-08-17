//! fANOVA parameter importance (Hutter, Hoos & Leyton-Brown, ICML 2014).
//!
//! Where [`crate::parameter_importance`] fits one decision *stump* per parameter
//! as a cheap proxy, this module implements the real thing: it trains a random
//! forest of regression trees on `parameters -> objective`, then performs a
//! **functional ANOVA** decomposition of the forest's prediction function to
//! attribute objective variance to each parameter.
//!
//! For a piecewise-constant tree the marginal of the prediction over all but one
//! parameter — and the variance of that marginal — can be computed *exactly* by
//! walking the leaves, so no Monte-Carlo integration is needed. The reported
//! importance of a parameter is the fraction of total prediction variance
//! explained by its **main effect**, averaged across the trees.
//!
//! Unlike the stump proxy this accounts for a parameter's effect *after
//! marginalizing over the others* rather than a single univariate split, so it
//! is far less fooled by correlated sampling. Main effects need not sum to 1:
//! the remainder is variance carried by interaction terms, which `fanova`
//! deliberately does not fold into any single parameter.
//!
//! ```
//! use hyperopt_core::{Distribution, Trial, TrialState, Value};
//! use hyperopt_viz::fanova_importance;
//!
//! let mut trials = Vec::new();
//! for i in 0..80 {
//!     let x = (i as f64 % 10.0) - 5.0;
//!     let noise = ((i * 7) % 11) as f64;
//!     let mut t = Trial::new(i);
//!     t.record("x", Distribution::Uniform { low: -5.0, high: 5.0 }, Value::Float(x));
//!     t.record("noise", Distribution::Uniform { low: 0.0, high: 10.0 }, Value::Float(noise));
//!     t.value = Some((x - 2.0).powi(2)); // depends only on x
//!     t.state = TrialState::Complete;
//!     trials.push(t);
//! }
//! let imp = fanova_importance(&trials);
//! assert_eq!(imp[0].name, "x");
//! ```

use crate::Importance;
use hyperopt_core::{Distribution, Trial, TrialState, Value};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Tunables for the random forest fANOVA. [`Default`] mirrors the settings the
/// reference fANOVA implementation uses for small studies.
#[derive(Debug, Clone)]
pub struct FanovaOptions {
    /// Number of regression trees in the forest.
    pub n_trees: usize,
    /// Candidate features considered at each split. `None` means "all of them"
    /// (bagging alone supplies the tree-to-tree diversity), which is stable when
    /// the number of parameters is small.
    pub max_features: Option<usize>,
    /// Minimum samples a node must hold to be eligible for a further split.
    pub min_samples_split: usize,
    /// Hard cap on tree depth (a guard against pathological deep trees).
    pub max_depth: usize,
    /// Seed for the bootstrap sampling, so importances are reproducible.
    pub seed: u64,
}

impl Default for FanovaOptions {
    fn default() -> Self {
        FanovaOptions {
            n_trees: 64,
            max_features: None,
            min_samples_split: 2,
            max_depth: 64,
            seed: 0,
        }
    }
}

/// Per-parameter fANOVA importance with default [`FanovaOptions`].
///
/// Returns importances sorted most-important first. `variance_explained` is the
/// forest-averaged fraction of objective variance carried by that parameter's
/// main effect; `normalized` rescales those so they sum to 1 across parameters.
/// A study with fewer than two usable trials, or one whose objective never
/// varies, yields all-zero importances.
pub fn fanova_importance(trials: &[Trial]) -> Vec<Importance> {
    fanova_importance_with(trials, &FanovaOptions::default())
}

/// Per-parameter fANOVA importance with explicit [`FanovaOptions`].
pub fn fanova_importance_with(trials: &[Trial], opts: &FanovaOptions) -> Vec<Importance> {
    let Some(data) = Dataset::from_trials(trials) else {
        return Vec::new();
    };
    let d = data.names.len();
    let forest = forest_stats(&data, opts);

    // Accumulate each feature's per-tree main-effect fraction.
    let mut totals = vec![0.0f64; d];
    for stats in &forest {
        for (j, total) in totals.iter_mut().enumerate() {
            *total += stats.marginal_variance(&[j]) / stats.total_var;
        }
    }
    let n = forest.len().max(1) as f64;

    let mut raw: Vec<Importance> = data
        .names
        .iter()
        .enumerate()
        .map(|(j, name)| Importance {
            name: name.clone(),
            variance_explained: if forest.is_empty() { 0.0 } else { totals[j] / n },
            normalized: 0.0,
        })
        .collect();

    let total: f64 = raw.iter().map(|i| i.variance_explained).sum();
    if total > 0.0 {
        for imp in &mut raw {
            imp.normalized = imp.variance_explained / total;
        }
    }

    raw.sort_by(|a, b| b.variance_explained.total_cmp(&a.variance_explained));
    raw
}

/// A pair of parameters' estimated **interaction** importance: the fraction of
/// objective variance explained by their joint effect *beyond* the sum of their
/// individual main effects (the pure second-order fANOVA term).
#[derive(Debug, Clone, PartialEq)]
pub struct Interaction {
    /// First parameter of the pair (names are ordered as first-seen).
    pub first: String,
    /// Second parameter of the pair.
    pub second: String,
    /// Forest-averaged fraction of objective variance carried by the pair's
    /// pure interaction, in `[0, 1]`.
    pub variance_explained: f64,
    /// `variance_explained` normalized so all reported interactions sum to 1.
    pub normalized: f64,
}

/// Pairwise fANOVA interaction importance with default [`FanovaOptions`].
///
/// A parameter's *main effect* ([`fanova_importance`]) captures its influence
/// averaged over the others; this captures what two parameters do *together*
/// that neither explains alone — e.g. a learning-rate/batch-size pairing where
/// the good region of one depends on the other. Returns every pair, sorted
/// most-interacting first. Objectives that are additive in their parameters
/// yield near-zero interactions.
pub fn fanova_interactions(trials: &[Trial]) -> Vec<Interaction> {
    fanova_interactions_with(trials, &FanovaOptions::default())
}

/// Pairwise fANOVA interaction importance with explicit [`FanovaOptions`].
pub fn fanova_interactions_with(trials: &[Trial], opts: &FanovaOptions) -> Vec<Interaction> {
    let Some(data) = Dataset::from_trials(trials) else {
        return Vec::new();
    };
    let d = data.names.len();
    if d < 2 {
        return Vec::new();
    }
    let forest = forest_stats(&data, opts);

    // Accumulate each pair's pure-interaction fraction across the forest.
    let mut totals = vec![0.0f64; d * d];
    for stats in &forest {
        // Main-effect variances, reused across every pair for this tree.
        let mains: Vec<f64> = (0..d).map(|j| stats.marginal_variance(&[j])).collect();
        for i in 0..d {
            for j in (i + 1)..d {
                let joint = stats.marginal_variance(&[i, j]);
                let pure = (joint - mains[i] - mains[j]).max(0.0);
                totals[i * d + j] += pure / stats.total_var;
            }
        }
    }
    let n = forest.len().max(1) as f64;

    let mut raw: Vec<Interaction> = Vec::new();
    for i in 0..d {
        for j in (i + 1)..d {
            raw.push(Interaction {
                first: data.names[i].clone(),
                second: data.names[j].clone(),
                variance_explained: if forest.is_empty() { 0.0 } else { totals[i * d + j] / n },
                normalized: 0.0,
            });
        }
    }

    let total: f64 = raw.iter().map(|it| it.variance_explained).sum();
    if total > 0.0 {
        for it in &mut raw {
            it.normalized = it.variance_explained / total;
        }
    }

    raw.sort_by(|a, b| b.variance_explained.total_cmp(&a.variance_explained));
    raw
}

/// Train the bagged forest and keep the fANOVA statistics of every tree whose
/// prediction actually varies.
fn forest_stats(data: &Dataset, opts: &FanovaOptions) -> Vec<TreeStats> {
    let mut rng = StdRng::seed_from_u64(opts.seed);
    let n = data.y.len();
    let mut forest = Vec::new();
    for _ in 0..opts.n_trees {
        let idx: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
        let tree = Tree::fit(data, &idx, opts, &mut rng);
        if let Some(stats) = TreeStats::from_tree(&tree, &data.bounds) {
            forest.push(stats);
        }
    }
    forest
}

/// The rectangular design matrix fANOVA needs, extracted from a study.
struct Dataset {
    names: Vec<String>,
    /// Per-feature marginalization domain `[lo, hi]` (in the encoded space).
    bounds: Vec<(f64, f64)>,
    /// Row-major encoded features, one row per usable trial.
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
}

impl Dataset {
    /// Build a design matrix from completed trials. Parameters are the set that
    /// appears across completed trials (first-seen order); rows are the trials
    /// that carry *every* such parameter (a mostly-rectangular / flat search
    /// space). Returns `None` if fewer than two usable rows survive.
    fn from_trials(trials: &[Trial]) -> Option<Dataset> {
        let completed: Vec<&Trial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .collect();
        if completed.len() < 2 {
            return None;
        }

        // Parameter names (first-seen) and the distribution recorded for each.
        let mut names: Vec<String> = Vec::new();
        let mut dists: Vec<Distribution> = Vec::new();
        for t in &completed {
            for p in &t.params {
                if !names.contains(&p.name) {
                    names.push(p.name.clone());
                    dists.push(p.distribution.clone());
                }
            }
        }
        if names.is_empty() {
            return None;
        }

        let bounds: Vec<(f64, f64)> = dists.iter().map(encode_bounds).collect();

        // Rows: trials that carry every parameter, encoded per its distribution.
        let mut x: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for t in &completed {
            let mut row = Vec::with_capacity(names.len());
            let mut complete_row = true;
            for (name, dist) in names.iter().zip(&dists) {
                match t.param_value(name).and_then(|v| encode(dist, v)) {
                    Some(f) => row.push(f),
                    None => {
                        complete_row = false;
                        break;
                    }
                }
            }
            if complete_row {
                x.push(row);
                y.push(t.value.unwrap());
            }
        }

        if x.len() < 2 {
            return None;
        }
        Some(Dataset { names, bounds, x, y })
    }
}

/// The marginalization domain for a distribution, in encoded coordinates.
/// Discrete distributions are widened by half a unit on each side so every
/// level occupies equal measure and split thresholds land between levels.
fn encode_bounds(dist: &Distribution) -> (f64, f64) {
    match dist {
        Distribution::Uniform { low, high } => (*low, *high),
        Distribution::LogUniform { low, high } => {
            if *low > 0.0 {
                (low.ln(), high.ln())
            } else {
                (0.0, 0.0)
            }
        }
        Distribution::IntUniform { low, high } => (*low as f64 - 0.5, *high as f64 + 0.5),
        Distribution::Categorical { choices } => {
            (-0.5, choices.len() as f64 - 1.0 + 0.5)
        }
    }
}

/// Encode a recorded value into the fANOVA feature space for its distribution.
fn encode(dist: &Distribution, value: &Value) -> Option<f64> {
    match dist {
        Distribution::Uniform { .. } => value.as_float(),
        Distribution::LogUniform { .. } => value.as_float().filter(|v| *v > 0.0).map(f64::ln),
        Distribution::IntUniform { .. } => value.as_int().map(|i| i as f64),
        Distribution::Categorical { choices } => {
            let label = value.as_categorical()?;
            choices.iter().position(|c| c == label).map(|i| i as f64)
        }
    }
}

/// A CART regression tree with axis-aligned splits.
enum Tree {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Tree>,
        right: Box<Tree>,
    },
}

/// A leaf paired with the hyper-rectangle of feature space it owns.
struct LeafBox {
    value: f64,
    /// Per-feature `[lo, hi)` extent of this leaf, clipped to the domain.
    bounds: Vec<(f64, f64)>,
}

impl Tree {
    fn fit(data: &Dataset, indices: &[usize], opts: &FanovaOptions, rng: &mut StdRng) -> Tree {
        Tree::build(data, indices, &data.bounds.clone(), opts, 0, rng)
    }

    fn build(
        data: &Dataset,
        indices: &[usize],
        node_box: &[(f64, f64)],
        opts: &FanovaOptions,
        depth: usize,
        rng: &mut StdRng,
    ) -> Tree {
        let mean = mean_of(&data.y, indices);
        if indices.len() < opts.min_samples_split
            || depth >= opts.max_depth
            || variance_of(&data.y, indices) <= 0.0
        {
            return Tree::Leaf { value: mean };
        }

        let d = data.names.len();
        let candidates = choose_features(d, opts.max_features, rng);

        let mut best: Option<(usize, f64, f64)> = None; // (feature, threshold, weighted_var)
        for &f in &candidates {
            // Sort the node's samples by this feature.
            let mut vals: Vec<(f64, f64)> = indices
                .iter()
                .map(|&i| (data.x[i][f], data.y[i]))
                .collect();
            vals.sort_by(|a, b| a.0.total_cmp(&b.0));

            // Prefix sums to evaluate every threshold in O(n).
            let n = vals.len();
            let mut prefix_sum = vec![0.0; n + 1];
            let mut prefix_sq = vec![0.0; n + 1];
            for (k, &(_, y)) in vals.iter().enumerate() {
                prefix_sum[k + 1] = prefix_sum[k] + y;
                prefix_sq[k + 1] = prefix_sq[k] + y * y;
            }
            let total_sum = prefix_sum[n];
            let total_sq = prefix_sq[n];

            for k in 1..n {
                // Only split between distinct feature values.
                if vals[k].0 == vals[k - 1].0 {
                    continue;
                }
                let left_n = k as f64;
                let right_n = (n - k) as f64;
                let left_var =
                    prefix_sq[k] - prefix_sum[k] * prefix_sum[k] / left_n;
                let right_var = (total_sq - prefix_sq[k])
                    - (total_sum - prefix_sum[k]) * (total_sum - prefix_sum[k]) / right_n;
                // Sum of squared errors (proportional to weighted variance).
                let weighted = left_var + right_var;
                let threshold = 0.5 * (vals[k].0 + vals[k - 1].0);
                if best.is_none_or(|(_, _, bw)| weighted < bw) {
                    best = Some((f, threshold, weighted));
                }
            }
        }

        let Some((feature, threshold, _)) = best else {
            return Tree::Leaf { value: mean };
        };

        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data.x[i][feature] <= threshold);
        if left_idx.is_empty() || right_idx.is_empty() {
            return Tree::Leaf { value: mean };
        }

        let mut left_box = node_box.to_vec();
        let mut right_box = node_box.to_vec();
        left_box[feature].1 = threshold;
        right_box[feature].0 = threshold;

        Tree::Split {
            feature,
            threshold,
            left: Box::new(Tree::build(data, &left_idx, &left_box, opts, depth + 1, rng)),
            right: Box::new(Tree::build(data, &right_idx, &right_box, opts, depth + 1, rng)),
        }
    }

    /// Collect every leaf together with the box of feature space it covers,
    /// tightening `current` along each split on the way down.
    fn collect(&self, current: &mut Vec<(f64, f64)>, out: &mut Vec<LeafBox>) {
        match self {
            Tree::Leaf { value } => out.push(LeafBox {
                value: *value,
                bounds: current.clone(),
            }),
            Tree::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let saved = current[*feature];
                current[*feature] = (saved.0, *threshold);
                left.collect(current, out);
                current[*feature] = (*threshold, saved.1);
                right.collect(current, out);
                current[*feature] = saved;
            }
        }
    }

}

/// Per-tree fANOVA statistics: the leaf partition, plus the mean and total
/// variance of the tree's (piecewise-constant) prediction under the
/// product-uniform domain measure. The variance of the marginal over any subset
/// of features is then computed exactly on demand — main effects for a single
/// feature, interactions for a pair.
struct TreeStats {
    leaves: Vec<LeafBox>,
    /// Per-dimension domain widths (0 for a degenerate dimension).
    widths: Vec<f64>,
    fbar: f64,
    total_var: f64,
}

impl TreeStats {
    /// Build the statistics for one tree over `domain`, or `None` if the tree's
    /// prediction has zero variance (nothing to attribute).
    fn from_tree(tree: &Tree, domain: &[(f64, f64)]) -> Option<TreeStats> {
        let mut current: Vec<(f64, f64)> = domain.to_vec();
        let mut leaves: Vec<LeafBox> = Vec::new();
        tree.collect(&mut current, &mut leaves);

        // A zero-width (degenerate) dimension gets fraction 1 so it drops out of
        // the volume product without dividing by 0.
        let widths: Vec<f64> = domain.iter().map(|(lo, hi)| hi - lo).collect();
        let stats = TreeStats {
            leaves,
            widths,
            fbar: 0.0,
            total_var: 0.0,
        };

        let fbar: f64 = stats
            .leaves
            .iter()
            .map(|leaf| stats.volume(leaf) * leaf.value)
            .sum();
        let total_var: f64 = stats
            .leaves
            .iter()
            .map(|leaf| stats.volume(leaf) * (leaf.value - fbar).powi(2))
            .sum();
        if total_var <= 0.0 {
            return None;
        }
        Some(TreeStats {
            fbar,
            total_var,
            ..stats
        })
    }

    fn frac(&self, b: (f64, f64), dim: usize) -> f64 {
        if self.widths[dim] > 0.0 {
            (b.1 - b.0) / self.widths[dim]
        } else {
            1.0
        }
    }

    /// Fraction of the whole domain volume a leaf's box covers.
    fn volume(&self, leaf: &LeafBox) -> f64 {
        (0..self.widths.len())
            .map(|dim| self.frac(leaf.bounds[dim], dim))
            .product()
    }

    /// Variance of the marginal prediction over `subset` (every other dimension
    /// integrated out), under the product-uniform domain measure. For a single
    /// feature this is its main-effect variance; for a pair it is the total
    /// variance of the 2-D marginal (main effects *plus* interaction).
    fn marginal_variance(&self, subset: &[usize]) -> f64 {
        // A degenerate dimension in the subset carries no variance.
        if subset.iter().any(|&d| self.widths[d] <= 0.0) {
            return 0.0;
        }
        let dcount = self.widths.len();

        // Complement weight of each leaf: product of fractions over dims not in
        // `subset` (the volume integrated out to form the marginal).
        let comp_w: Vec<f64> = self
            .leaves
            .iter()
            .map(|leaf| {
                (0..dcount)
                    .filter(|d| !subset.contains(d))
                    .map(|d| self.frac(leaf.bounds[d], d))
                    .product::<f64>()
            })
            .collect();

        // The marginal is piecewise-constant on the grid of leaf boundaries in
        // each subset dimension. (The outermost boundaries coincide with the
        // domain endpoints, since the leaves tile the box.)
        let intervals: Vec<Vec<(f64, f64)>> = subset
            .iter()
            .map(|&d| {
                let mut cuts: Vec<f64> = self
                    .leaves
                    .iter()
                    .flat_map(|leaf| [leaf.bounds[d].0, leaf.bounds[d].1])
                    .collect();
                cuts.sort_by(f64::total_cmp);
                cuts.dedup();
                cuts.windows(2)
                    .filter(|w| w[1] > w[0])
                    .map(|w| (w[0], w[1]))
                    .collect()
            })
            .collect();
        if intervals.iter().any(|iv| iv.is_empty()) {
            return 0.0;
        }

        // Sum over the Cartesian product of intervals (one cell of constant
        // marginal value), accumulating variance under the cell's measure.
        let sizes: Vec<usize> = intervals.iter().map(|iv| iv.len()).collect();
        let mut idx = vec![0usize; subset.len()];
        let mut var = 0.0;
        loop {
            let mut measure = 1.0;
            let mut mids = vec![0.0; subset.len()];
            for (k, &d) in subset.iter().enumerate() {
                let (p, q) = intervals[k][idx[k]];
                mids[k] = 0.5 * (p + q);
                measure *= (q - p) / self.widths[d];
            }
            // Marginal value on this cell: leaves whose box contains the cell in
            // every subset dimension, weighted by their complement volume.
            let mut a = 0.0;
            for (leaf, &cw) in self.leaves.iter().zip(&comp_w) {
                let inside = subset
                    .iter()
                    .enumerate()
                    .all(|(k, &d)| leaf.bounds[d].0 <= mids[k] && mids[k] <= leaf.bounds[d].1);
                if inside {
                    a += leaf.value * cw;
                }
            }
            var += measure * (a - self.fbar).powi(2);

            // Odometer over the interval grid.
            let mut k = 0;
            loop {
                if k == subset.len() {
                    return var;
                }
                idx[k] += 1;
                if idx[k] < sizes[k] {
                    break;
                }
                idx[k] = 0;
                k += 1;
            }
        }
    }
}

fn choose_features(d: usize, max_features: Option<usize>, rng: &mut StdRng) -> Vec<usize> {
    let k = max_features.unwrap_or(d).clamp(1, d);
    if k >= d {
        return (0..d).collect();
    }
    // Partial Fisher-Yates for a random size-k subset.
    let mut pool: Vec<usize> = (0..d).collect();
    for i in 0..k {
        let j = rng.random_range(i..d);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

fn mean_of(y: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64
}

fn variance_of(y: &[f64], indices: &[usize]) -> f64 {
    let n = indices.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean_of(y, indices);
    indices.iter().map(|&i| (y[i] - m).powi(2)).sum::<f64>() / n as f64
}
