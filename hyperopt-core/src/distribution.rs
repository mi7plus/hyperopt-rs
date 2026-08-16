use serde::{Deserialize, Serialize};

/// The search-space shape recorded for a single suggested parameter.
///
/// In the define-by-run model a `Distribution` is produced by each
/// `trial.suggest_*` call and handed to the active [`crate::Sampler`], which
/// returns a matching [`crate::Value`]. The distribution is also recorded on
/// the [`crate::Trial`] so later phases (pruning, adaptive sampling, storage,
/// importance analysis) can reconstruct exactly what was searched.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Distribution {
    /// Continuous uniform over `[low, high]`.
    Uniform { low: f64, high: f64 },
    /// Log-uniform over `[low, high]` (both must be `> 0`). Sampling is uniform
    /// in log-space — appropriate for scale parameters like learning rates.
    LogUniform { low: f64, high: f64 },
    /// Integer uniform over the inclusive range `[low, high]`.
    IntUniform { low: i64, high: i64 },
    /// Categorical over a fixed set of string labels.
    Categorical { choices: Vec<String> },
}

impl Distribution {
    /// Returns `true` if `value` lies inside this distribution's support.
    pub fn contains(&self, value: &crate::Value) -> bool {
        use crate::Value;
        match (self, value) {
            (Distribution::Uniform { low, high }, Value::Float(x))
            | (Distribution::LogUniform { low, high }, Value::Float(x)) => *low <= *x && *x <= *high,
            (Distribution::IntUniform { low, high }, Value::Int(x)) => *low <= *x && *x <= *high,
            (Distribution::Categorical { choices }, Value::Categorical(s)) => choices.contains(s),
            _ => false,
        }
    }
}
