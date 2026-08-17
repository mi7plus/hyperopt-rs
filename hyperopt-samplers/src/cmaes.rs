//! Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
//!
//! CMA-ES is a strong derivative-free optimizer for continuous search spaces: it
//! samples each generation from a multivariate normal and adapts that normal's
//! mean, step size, and full covariance from the ranking of the points it saw.
//! This module implements the standard (μ/μ_w, λ) CMA-ES (Hansen) from scratch —
//! including a Jacobi symmetric-eigensolver for the covariance update — so it
//! pulls in no dependency beyond `rand`.
//!
//! ## Fitting CMA-ES into define-by-run
//!
//! CMA-ES is inherently *generational and vector-valued*: it wants λ full
//! parameter vectors, their objective values, then an update. The framework's
//! [`Sampler`] trait is instead per-parameter and history-driven. [`CmaEsSampler`]
//! bridges the two exactly the way [`crate::TpeSampler`] does — statelessly, by
//! rebuilding from the study snapshot on demand:
//!
//! - The **search space** is the set of numeric parameters (`Uniform`,
//!   `LogUniform`, `IntUniform`) common to all completed trials, taken in sorted
//!   name order for a stable coordinate layout. Categorical parameters — and any
//!   parameter not yet seen in a completed trial — fall back to an independent
//!   random draw, so a mixed space still works.
//! - Each parameter is mapped to an internal `[0, 1]` coordinate; completed
//!   trials, sorted by number, are chunked into generations of λ and replayed
//!   through the engine (`tell`) to reconstruct the current distribution. A new
//!   trial's whole vector is then drawn once (`ask`), cached by trial number, and
//!   decoded per parameter on demand.
//! - Box constraints are handled by repairing a drawn coordinate back into
//!   `[0, 1]` before decoding. The default [`BoundHandling::Reflect`] folds an
//!   out-of-box draw back inside with a tent map (mirror at each bound), which
//!   keeps the sampling density smooth across a boundary; [`BoundHandling::Clamp`]
//!   is the simpler heuristic that snaps to the nearest bound but piles
//!   probability mass on it (biasing the covariance for a bound-adjacent
//!   optimum). Both keep every suggestion valid. Documented, not silent.
//!
//! Like TPE, the per-suggestion cost is `O(generations · n³)` for the eigensolve,
//! negligible next to a real objective evaluation, and the rebuild makes the
//! sampler correct under both parallel execution and reload from storage.

// This module is dense matrix/vector arithmetic (evolution paths, covariance
// updates, Jacobi rotations) where indexed loops that touch several arrays or
// two matrix columns at once read far closer to the textbook formulas than the
// equivalent iterator chains would.
#![allow(clippy::needless_range_loop)]

use hyperopt_core::{Direction, Distribution, Sampler, StudyState, Trial, Value};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;

use crate::random::sample_value;

/// How CMA-ES repairs a drawn coordinate that lands outside a parameter's
/// `[0, 1]` normalized box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundHandling {
    /// Fold the coordinate back into `[0, 1]` by mirroring at each bound (a
    /// period-2 tent map). Continuous across the boundary and free of the
    /// boundary pile-up that clamping causes — the default.
    Reflect,
    /// Snap the coordinate to the nearest bound. Simple, but concentrates
    /// probability on the boundary when the optimum sits near one.
    Clamp,
}

impl BoundHandling {
    /// Map an arbitrary real coordinate into `[0, 1]`.
    fn repair(self, v: f64) -> f64 {
        match self {
            BoundHandling::Clamp => v.clamp(0.0, 1.0),
            BoundHandling::Reflect => {
                // Tent map with period 2: [0,1] is identity, (1,2) mirrors back.
                let t = v.rem_euclid(2.0);
                if t > 1.0 {
                    2.0 - t
                } else {
                    t
                }
            }
        }
    }
}

/// A [`Sampler`] driven by CMA-ES over the study's numeric parameters.
///
/// Construct with [`CmaEsSampler::new`] (OS-seeded) or
/// [`CmaEsSampler::seeded`] (reproducible). The population size λ defaults to
/// the CMA-ES rule `4 + ⌊3·ln n⌋` and can be overridden with
/// [`CmaEsSampler::population_size`]; the first generation is sampled at random
/// as warmup (see [`CmaEsSampler::n_startup_trials`]).
pub struct CmaEsSampler {
    seed: u64,
    rng: StdRng,
    popsize: Option<usize>,
    n_startup_trials: Option<usize>,
    sigma0: f64,
    bound_handling: BoundHandling,
    /// Decoded candidate vector per trial number: `name -> value`.
    cache: HashMap<usize, HashMap<String, Value>>,
}

impl CmaEsSampler {
    /// A CMA-ES sampler seeded from OS entropy (non-deterministic across runs).
    pub fn new() -> Self {
        let mut seeder = rand::rng();
        Self::seeded(seeder.random())
    }

    /// A CMA-ES sampler with a fixed seed — reproducible for tests/benchmarks.
    pub fn seeded(seed: u64) -> Self {
        CmaEsSampler {
            seed,
            rng: StdRng::seed_from_u64(seed),
            popsize: None,
            n_startup_trials: None,
            sigma0: 0.2,
            bound_handling: BoundHandling::Reflect,
            cache: HashMap::new(),
        }
    }

    /// Override the population size λ (default `4 + ⌊3·ln n⌋`).
    pub fn population_size(mut self, lambda: usize) -> Self {
        self.popsize = Some(lambda.max(2));
        self
    }

    /// Number of initial trials drawn at random before CMA-ES takes over
    /// (default: one population, λ). A full generation of spread-out points
    /// makes the first covariance estimate meaningful.
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = Some(n);
        self
    }

    /// Initial step size σ₀, as a fraction of the normalized `[0, 1]` range
    /// (default `0.2`).
    pub fn sigma0(mut self, sigma0: f64) -> Self {
        self.sigma0 = sigma0;
        self
    }

    /// How to repair out-of-box draws (default [`BoundHandling::Reflect`]).
    pub fn bound_handling(mut self, handling: BoundHandling) -> Self {
        self.bound_handling = handling;
        self
    }

    /// The numeric search space common to all completed trials, sorted by name.
    /// Returns `(name, distribution)` pairs.
    fn search_space(study_state: &StudyState) -> Vec<(String, Distribution)> {
        let mut per_name: HashMap<String, Distribution> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut n_completed = 0usize;

        for t in study_state.completed_trials() {
            n_completed += 1;
            for p in &t.params {
                if is_numeric(&p.distribution) {
                    per_name.entry(p.name.clone()).or_insert_with(|| p.distribution.clone());
                    *counts.entry(p.name.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut space: Vec<(String, Distribution)> = per_name
            .into_iter()
            .filter(|(name, _)| counts.get(name).copied().unwrap_or(0) == n_completed)
            .collect();
        space.sort_by(|a, b| a.0.cmp(&b.0));
        space
    }

    /// Build (or fetch) this trial's full decoded candidate vector.
    fn candidate_for(
        &mut self,
        study_state: &StudyState,
        trial: &Trial,
    ) -> Option<HashMap<String, Value>> {
        if let Some(cached) = self.cache.get(&trial.number) {
            return Some(cached.clone());
        }

        let space = Self::search_space(study_state);
        if space.is_empty() {
            return None;
        }
        let n = space.len();

        let popsize = self.popsize.unwrap_or_else(|| default_popsize(n));
        let n_startup = self.n_startup_trials.unwrap_or(popsize);

        // Reconstruct each completed trial's normalized vector + fitness, in
        // trial-number order.
        let mut samples: Vec<(Vec<f64>, f64)> = Vec::new();
        let mut completed: Vec<&Trial> = study_state.completed_trials().collect();
        completed.sort_by_key(|t| t.number);
        for t in &completed {
            if let Some(x) = encode_vector(&space, t) {
                let fitness = match study_state.direction() {
                    Direction::Minimize => t.value.unwrap(),
                    Direction::Maximize => -t.value.unwrap(),
                };
                samples.push((x, fitness));
            }
        }

        // Warmup: not enough completed trials to trust a covariance yet.
        if samples.len() < n_startup {
            return None;
        }

        // Replay full generations of λ through the engine.
        let mut engine = Engine::new(n, self.sigma0);
        for gen in samples.chunks(popsize) {
            if gen.len() == popsize {
                engine.tell(gen);
            }
        }

        // Draw this trial's candidate deterministically from its number, so a
        // re-suggest (or a stale parallel re-read) yields the same vector.
        let mut ask_rng = StdRng::seed_from_u64(self.seed ^ (trial.number as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let raw = engine.ask(&mut ask_rng);

        // Repair to [0, 1] (reflect or clamp) and decode.
        let mut decoded = HashMap::new();
        for (i, (name, dist)) in space.iter().enumerate() {
            let z = self.bound_handling.repair(raw[i]);
            decoded.insert(name.clone(), decode(dist, z));
        }
        self.cache.insert(trial.number, decoded.clone());
        Some(decoded)
    }
}

impl Default for CmaEsSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for CmaEsSampler {
    fn suggest(
        &mut self,
        study_state: &StudyState,
        trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Value {
        // Non-numeric parameters are outside CMA-ES's space.
        if is_numeric(distribution) {
            if let Some(vector) = self.candidate_for(study_state, trial) {
                if let Some(v) = vector.get(param_name) {
                    // Guard: only trust the cached vector if its variant matches
                    // the distribution requested this trial.
                    if variant_matches(distribution, v) {
                        return v.clone();
                    }
                }
            }
        }
        sample_value(&mut self.rng, distribution)
    }
}

fn is_numeric(dist: &Distribution) -> bool {
    matches!(
        dist,
        Distribution::Uniform { .. }
            | Distribution::LogUniform { .. }
            | Distribution::IntUniform { .. }
    )
}

fn variant_matches(dist: &Distribution, value: &Value) -> bool {
    matches!(
        (dist, value),
        (Distribution::Uniform { .. }, Value::Float(_))
            | (Distribution::LogUniform { .. }, Value::Float(_))
            | (Distribution::IntUniform { .. }, Value::Int(_))
    )
}

/// CMA-ES population-size default `λ = 4 + ⌊3·ln n⌋`.
fn default_popsize(n: usize) -> usize {
    (4.0 + (3.0 * (n as f64).ln()).floor()) as usize
}

/// Encode a completed trial's parameters into the normalized `[0, 1]^n` vector
/// for `space`, or `None` if any parameter is missing / wrong-typed.
fn encode_vector(space: &[(String, Distribution)], trial: &Trial) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(space.len());
    for (name, dist) in space {
        let v = trial.param_value(name)?;
        out.push(encode(dist, v)?);
    }
    Some(out)
}

/// Map a parameter value to `[0, 1]`, or `None` if it can't be projected.
fn encode(dist: &Distribution, value: &Value) -> Option<f64> {
    match dist {
        Distribution::Uniform { low, high } => {
            let x = value.as_float()?;
            unit(x, *low, *high)
        }
        Distribution::LogUniform { low, high } => {
            let x = value.as_float()?;
            if *low <= 0.0 || x <= 0.0 {
                return None;
            }
            unit(x.ln(), low.ln(), high.ln())
        }
        Distribution::IntUniform { low, high } => {
            let x = value.as_int()? as f64;
            unit(x, *low as f64, *high as f64)
        }
        Distribution::Categorical { .. } => None,
    }
}

/// Decode a normalized `[0, 1]` coordinate back into a parameter value.
fn decode(dist: &Distribution, z: f64) -> Value {
    match dist {
        Distribution::Uniform { low, high } => Value::Float(low + z * (high - low)),
        Distribution::LogUniform { low, high } => {
            let l = low.ln();
            let h = high.ln();
            Value::Float((l + z * (h - l)).exp())
        }
        Distribution::IntUniform { low, high } => {
            let raw = *low as f64 + z * (*high as f64 - *low as f64);
            Value::Int(raw.round() as i64)
        }
        Distribution::Categorical { .. } => Value::Categorical(String::new()),
    }
}

fn unit(x: f64, low: f64, high: f64) -> Option<f64> {
    if high > low {
        Some(((x - low) / (high - low)).clamp(0.0, 1.0))
    } else {
        Some(0.5) // degenerate range: everything maps to the middle
    }
}

/// The pure CMA-ES engine, operating in normalized coordinates.
struct Engine {
    n: usize,
    mean: Vec<f64>,
    sigma: f64,
    cov: Vec<Vec<f64>>, // C
    p_c: Vec<f64>,
    p_s: Vec<f64>,
    b: Vec<Vec<f64>>, // eigenvectors of C
    d: Vec<f64>,      // sqrt eigenvalues of C
    generation: usize,
    // Strategy constants.
    mu: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    c_c: f64,
    c_s: f64,
    c_1: f64,
    c_mu: f64,
    damps: f64,
    chi_n: f64,
}

impl Engine {
    fn new(n: usize, sigma0: f64) -> Self {
        let lambda = default_popsize(n);
        let mu = lambda / 2;

        // Recombination weights (log-decreasing), normalized to sum to 1.
        let mut weights: Vec<f64> = (0..mu)
            .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln())
            .collect();
        let wsum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= wsum;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        let nf = n as f64;
        let c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
        let c_s = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
        let c_1 = 2.0 / ((nf + 1.3).powi(2) + mu_eff);
        let c_mu = ((2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).powi(2) + mu_eff))
            .min(1.0 - c_1))
        .max(0.0);
        let damps = 1.0
            + 2.0 * (((mu_eff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0)
            + c_s;
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        Engine {
            n,
            mean: vec![0.5; n], // center of the normalized box
            sigma: sigma0,
            cov: identity(n),
            p_c: vec![0.0; n],
            p_s: vec![0.0; n],
            b: identity(n),
            d: vec![1.0; n],
            generation: 0,
            mu,
            weights,
            mu_eff,
            c_c,
            c_s,
            c_1,
            c_mu,
            damps,
            chi_n,
        }
    }

    /// Draw a candidate `x = mean + sigma · B·(d ⊙ z)`, `z ~ N(0, I)`.
    fn ask(&self, rng: &mut StdRng) -> Vec<f64> {
        let z: Vec<f64> = (0..self.n).map(|_| standard_normal(rng)).collect();
        let dz: Vec<f64> = (0..self.n).map(|i| self.d[i] * z[i]).collect();
        let bdz = mat_vec(&self.b, &dz);
        (0..self.n).map(|i| self.mean[i] + self.sigma * bdz[i]).collect()
    }

    /// Fold one full generation of `(candidate, fitness)` pairs into the
    /// distribution. `candidates` is minimized over `fitness`.
    fn tell(&mut self, candidates: &[(Vec<f64>, f64)]) {
        self.generation += 1;
        let n = self.n;

        // Rank ascending by fitness and keep the best mu.
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| candidates[a].1.total_cmp(&candidates[b].1));

        let mean_old = self.mean.clone();

        // y_i = (x_i - mean_old) / sigma for the selected candidates.
        let ys: Vec<Vec<f64>> = order
            .iter()
            .take(self.mu)
            .map(|&idx| {
                (0..n)
                    .map(|k| (candidates[idx].0[k] - mean_old[k]) / self.sigma)
                    .collect()
            })
            .collect();

        // Weighted recombination: yw = Σ w_i y_i; mean_new = mean_old + sigma·yw.
        let mut yw = vec![0.0; n];
        for (w, y) in self.weights.iter().zip(&ys) {
            for k in 0..n {
                yw[k] += w * y[k];
            }
        }
        for k in 0..n {
            self.mean[k] = mean_old[k] + self.sigma * yw[k];
        }

        // C^{-1/2} · yw  via B·diag(1/d)·Bᵀ.
        let bt_yw = mat_vec(&transpose(&self.b), &yw);
        let scaled: Vec<f64> = (0..n).map(|i| bt_yw[i] / self.d[i]).collect();
        let c_inv_sqrt_yw = mat_vec(&self.b, &scaled);

        // Step-size evolution path p_s.
        let cs_factor = (self.c_s * (2.0 - self.c_s) * self.mu_eff).sqrt();
        for k in 0..n {
            self.p_s[k] = (1.0 - self.c_s) * self.p_s[k] + cs_factor * c_inv_sqrt_yw[k];
        }
        let ps_norm = norm(&self.p_s);

        // Heaviside step to stall the rank-one update when the path is long.
        let hsig = if ps_norm
            / (1.0 - (1.0 - self.c_s).powi(2 * self.generation as i32)).sqrt()
            / self.chi_n
            < 1.4 + 2.0 / (n as f64 + 1.0)
        {
            1.0
        } else {
            0.0
        };

        // Covariance evolution path p_c.
        let cc_factor = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for k in 0..n {
            self.p_c[k] = (1.0 - self.c_c) * self.p_c[k] + hsig * cc_factor * yw[k];
        }

        // Covariance update: C = (1-c1-cmu)·C + c1·(pc pcᵀ + δ·C) + cmu·Σ w_i y_i y_iᵀ.
        let delta = (1.0 - hsig) * self.c_c * (2.0 - self.c_c);
        for r in 0..n {
            for c in 0..n {
                let rank_one = self.p_c[r] * self.p_c[c] + delta * self.cov[r][c];
                let mut rank_mu = 0.0;
                for (w, y) in self.weights.iter().zip(&ys) {
                    rank_mu += w * y[r] * y[c];
                }
                self.cov[r][c] = (1.0 - self.c_1 - self.c_mu) * self.cov[r][c]
                    + self.c_1 * rank_one
                    + self.c_mu * rank_mu;
            }
        }

        // Step-size update.
        self.sigma *= ((self.c_s / self.damps) * (ps_norm / self.chi_n - 1.0)).exp();
        // Guard against under/overflow of the step size.
        self.sigma = self.sigma.clamp(1e-12, 1e6);

        // Refresh the eigendecomposition used by the next ask / C^{-1/2}.
        self.update_eigen();
    }

    fn update_eigen(&mut self) {
        // Symmetrize defensively, then eigendecompose.
        for r in 0..self.n {
            for c in (r + 1)..self.n {
                let avg = 0.5 * (self.cov[r][c] + self.cov[c][r]);
                self.cov[r][c] = avg;
                self.cov[c][r] = avg;
            }
        }
        let (evals, evecs) = jacobi_eigen(&self.cov);
        self.b = evecs;
        self.d = evals.iter().map(|&e| e.max(1e-20).sqrt()).collect();
    }
}

// --- small linear-algebra helpers -----------------------------------------

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut t = vec![vec![0.0; n]; n];
    for r in 0..n {
        for c in 0..n {
            t[c][r] = m[r][c];
        }
    }
    t
}

fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Box–Muller standard-normal draw from a uniform RNG.
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix. Returns
/// `(eigenvalues, eigenvectors)` where column `k` of the eigenvector matrix is
/// the eigenvector for `eigenvalues[k]`. Adequate and robust for the small,
/// symmetric covariance matrices CMA-ES produces.
fn jacobi_eigen(a_in: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a_in.len();
    let mut a = a_in.to_vec();
    let mut v = identity(n);
    if n == 1 {
        return (vec![a[0][0]], v);
    }

    for _sweep in 0..100 {
        // Sum of off-diagonal magnitudes; stop once negligible.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q].abs();
            }
        }
        if off < 1e-14 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-18 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Rotate rows/cols p, q.
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                // Accumulate the rotation into the eigenvectors.
                for k in 0..n {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let evals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (evals, v)
}
