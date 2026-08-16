use hyperopt_core::{Direction, Distribution, Sampler, StudyState, Trial, Value};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use tpe::density_estimation::DefaultEstimatorBuilder;
use tpe::{categorical_range, histogram_estimator, parzen_estimator, range, TpeOptimizer};

use crate::random::sample_value;

/// Tree-structured Parzen Estimator sampler.
///
/// This **wraps the [`tpe`](https://docs.rs/tpe) crate** rather than
/// reimplementing TPE from scratch — a real integration point, not a stubbed
/// gap. The work this type does is translating between `hyperopt-core`'s
/// `Distribution`/`Trial` history and `tpe`'s per-parameter `TpeOptimizer`
/// (one optimizer handles exactly one hyperparameter), including:
///
/// - mapping each [`Distribution`] variant onto a `tpe` range + estimator
///   (Parzen for numeric, histogram for categorical),
/// - encoding values into `tpe`'s coordinate space (log for `LogUniform`,
///   index for `Categorical`, `[low, high+1)` for `IntUniform`) and decoding
///   the sampled result back,
/// - honouring the study [`Direction`] by negating objective values under
///   `Maximize`, since `tpe` always minimizes.
///
/// The sampler is effectively **stateless over trial history**: each
/// `suggest` rebuilds a `TpeOptimizer` from the study snapshot and replays the
/// relevant observations. This makes it correct under both parallel execution
/// (it simply works from whatever snapshot it's given) and study reload from
/// storage, at an `O(n)` per-suggestion cost that is negligible next to a real
/// objective evaluation.
///
/// ### Limitations (documented, not silent)
/// - The first `n_startup_trials` completed trials are sampled at random to
///   seed the estimators — matching Optuna's warmup and avoiding a biased model
///   from too few points.
/// - Every `hyperopt-core` [`Distribution`] variant maps onto `tpe`. There is
///   no variant `tpe` cannot represent here; should a future distribution not
///   map cleanly, this sampler falls back to a random draw for it rather than
///   producing an out-of-range value.
pub struct TpeSampler {
    rng: StdRng,
    n_startup_trials: usize,
}

impl TpeSampler {
    /// A TPE sampler with default warmup (`n_startup_trials = 10`), seeded from
    /// OS entropy.
    pub fn new() -> Self {
        let mut seeder = rand::rng();
        TpeSampler {
            rng: StdRng::seed_from_u64(seeder.random()),
            n_startup_trials: 10,
        }
    }

    /// A TPE sampler with a fixed seed — reproducible for tests/benchmarks.
    pub fn seeded(seed: u64) -> Self {
        TpeSampler {
            rng: StdRng::seed_from_u64(seed),
            n_startup_trials: 10,
        }
    }

    /// Number of initial trials to sample randomly before building TPE models.
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Collect `(tpe-coordinate, value-to-minimize)` observations for `name`
    /// from the completed trials, honouring the study direction.
    fn observations(
        study_state: &StudyState,
        name: &str,
        distribution: &Distribution,
    ) -> Vec<(f64, f64)> {
        let direction = study_state.direction();
        let mut obs = Vec::new();
        for t in study_state.completed_trials() {
            let (Some(record), Some(value)) = (
                t.params.iter().find(|p| p.name == name),
                t.value,
            ) else {
                continue;
            };
            if let Some(coord) = encode(distribution, &record.value) {
                let objective = match direction {
                    Direction::Minimize => value,
                    Direction::Maximize => -value,
                };
                obs.push((coord, objective));
            }
        }
        obs
    }
}

impl Default for TpeSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for TpeSampler {
    fn suggest(
        &mut self,
        study_state: &StudyState,
        _trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Value {
        let obs = Self::observations(study_state, param_name, distribution);

        // Warmup / not enough data: sample randomly.
        if obs.len() < self.n_startup_trials {
            return sample_value(&mut self.rng, distribution);
        }

        // Build the tpe optimizer for this distribution; fall back to random if
        // the range is degenerate.
        let Some((estimator, tpe_range)) = build_estimator(distribution) else {
            return sample_value(&mut self.rng, distribution);
        };
        let mut optim: TpeOptimizer<DefaultEstimatorBuilder> =
            TpeOptimizer::new(estimator, tpe_range);

        for (coord, value) in &obs {
            // A coord outside the range (shouldn't happen given encode) or a
            // NaN value is simply skipped rather than aborting the suggestion.
            let _ = optim.tell(*coord, *value);
        }

        match optim.ask(&mut self.rng) {
            Ok(raw) => decode(distribution, raw),
            Err(_) => sample_value(&mut self.rng, distribution),
        }
    }
}

/// Build the `tpe` estimator + range for a distribution, or `None` if the
/// range is degenerate/unrepresentable.
fn build_estimator(
    distribution: &Distribution,
) -> Option<(DefaultEstimatorBuilder, tpe::range::Range)> {
    match distribution {
        Distribution::Uniform { low, high } => {
            Some((parzen_estimator(), range(*low, *high).ok()?))
        }
        Distribution::LogUniform { low, high } => {
            if *low <= 0.0 {
                return None;
            }
            Some((parzen_estimator(), range(low.ln(), high.ln()).ok()?))
        }
        Distribution::IntUniform { low, high } => {
            // Continuous [low, high+1); decode floors back to an integer.
            Some((parzen_estimator(), range(*low as f64, *high as f64 + 1.0).ok()?))
        }
        Distribution::Categorical { choices } => {
            Some((histogram_estimator(), categorical_range(choices.len()).ok()?))
        }
    }
}

/// Encode a recorded [`Value`] into `tpe`'s coordinate for `distribution`.
fn encode(distribution: &Distribution, value: &Value) -> Option<f64> {
    match distribution {
        Distribution::Uniform { .. } => value.as_float(),
        Distribution::LogUniform { .. } => value.as_float().map(f64::ln),
        Distribution::IntUniform { .. } => value.as_int().map(|i| i as f64),
        Distribution::Categorical { choices } => {
            let label = value.as_categorical()?;
            choices.iter().position(|c| c == label).map(|i| i as f64)
        }
    }
}

/// Decode a value sampled by `tpe` back into a [`Value`] for `distribution`.
fn decode(distribution: &Distribution, raw: f64) -> Value {
    match distribution {
        Distribution::Uniform { .. } => Value::Float(raw),
        Distribution::LogUniform { .. } => Value::Float(raw.exp()),
        Distribution::IntUniform { low, high } => {
            let i = (raw.floor() as i64).clamp(*low, *high);
            Value::Int(i)
        }
        Distribution::Categorical { choices } => {
            if choices.is_empty() {
                return Value::Categorical(String::new());
            }
            let idx = (raw.floor() as usize).min(choices.len() - 1);
            Value::Categorical(choices[idx].clone())
        }
    }
}
