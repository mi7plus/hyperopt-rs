use hyperopt_core::{Distribution, Sampler, StudyState, Trial, Value};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Draw a single value from a distribution with no adaptive state. Shared by
/// [`RandomSampler`] and used by [`crate::TpeSampler`] during its startup
/// (pre-model) phase.
pub(crate) fn sample_value(rng: &mut StdRng, distribution: &Distribution) -> Value {
    match distribution {
        Distribution::Uniform { low, high } => {
            if low < high {
                Value::Float(rng.random_range(*low..*high))
            } else {
                Value::Float(*low)
            }
        }
        Distribution::LogUniform { low, high } => {
            if *low > 0.0 && low < high {
                let l = low.ln();
                let h = high.ln();
                Value::Float(rng.random_range(l..h).exp())
            } else {
                Value::Float(*low)
            }
        }
        Distribution::IntUniform { low, high } => {
            if low <= high {
                Value::Int(rng.random_range(*low..=*high))
            } else {
                Value::Int(*low)
            }
        }
        Distribution::Categorical { choices } => {
            if choices.is_empty() {
                Value::Categorical(String::new())
            } else {
                let i = rng.random_range(0..choices.len());
                Value::Categorical(choices[i].clone())
            }
        }
    }
}

/// The simplest possible [`Sampler`]: every parameter is drawn independently
/// and uniformly (or log-uniformly / categorically) from its distribution, with
/// no learning from prior trials. It is the right baseline to validate the
/// whole `Trial`/`TrialContext`/`Study` plumbing against, and the reference
/// every adaptive sampler is compared to.
pub struct RandomSampler {
    rng: StdRng,
}

impl RandomSampler {
    /// A random sampler seeded from OS entropy (non-deterministic across runs).
    pub fn new() -> Self {
        let mut seeder = rand::rng();
        RandomSampler {
            rng: StdRng::seed_from_u64(seeder.random()),
        }
    }

    /// A random sampler with a fixed seed — reproducible across runs, which is
    /// what tests and benchmarks want.
    pub fn seeded(seed: u64) -> Self {
        RandomSampler {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for RandomSampler {
    fn suggest(
        &mut self,
        _study_state: &StudyState,
        _trial: &Trial,
        _param_name: &str,
        distribution: &Distribution,
    ) -> Value {
        sample_value(&mut self.rng, distribution)
    }
}
