use crate::{Distribution, Value};
use serde::{Deserialize, Serialize};

/// Lifecycle state of a single trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialState {
    /// The objective is currently being evaluated.
    Running,
    /// The objective returned a value successfully.
    Complete,
    /// A [`crate::Pruner`] stopped the trial early.
    Pruned,
    /// The objective panicked or returned an error.
    Failed,
}

/// One recorded `(name, distribution, value)` triple, in suggestion order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamRecord {
    pub name: String,
    pub distribution: Distribution,
    pub value: Value,
}

/// A single optimization trial: its suggested parameters, intermediate
/// reports, and final objective value.
///
/// `params` is kept in suggestion order (an ordered list rather than a hash
/// map) so that define-by-run search spaces — where the set of parameters can
/// differ between trials — round-trip faithfully through storage and are
/// reproducible for grid enumeration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    /// Zero-based trial index within its study.
    pub number: usize,
    /// Parameters suggested so far, in the order they were requested.
    pub params: Vec<ParamRecord>,
    /// Intermediate `(step, value)` reports, populated via
    /// [`crate::TrialContext::report`]; consumed by pruners.
    pub intermediate_values: Vec<(usize, f64)>,
    /// Final objective value, once the trial completes.
    pub value: Option<f64>,
    /// Current lifecycle state.
    pub state: TrialState,
}

impl Trial {
    /// Creates a fresh `Running` trial with the given number and no params.
    pub fn new(number: usize) -> Self {
        Trial {
            number,
            params: Vec::new(),
            intermediate_values: Vec::new(),
            value: None,
            state: TrialState::Running,
        }
    }

    /// Records a suggested parameter. Re-suggesting the same name overwrites
    /// the previous record (mirrors Optuna, where repeated `suggest_*` calls
    /// for one name within a trial are idempotent).
    pub fn record(&mut self, name: &str, distribution: Distribution, value: Value) {
        if let Some(existing) = self.params.iter_mut().find(|p| p.name == name) {
            existing.distribution = distribution;
            existing.value = value;
        } else {
            self.params.push(ParamRecord {
                name: name.to_string(),
                distribution,
                value,
            });
        }
    }

    /// Looks up a previously recorded parameter value by name.
    pub fn param_value(&self, name: &str) -> Option<&Value> {
        self.params.iter().find(|p| p.name == name).map(|p| &p.value)
    }

    /// The intermediate value reported at exactly `step`, if any.
    pub fn value_at_step(&self, step: usize) -> Option<f64> {
        self.intermediate_values
            .iter()
            .find(|(s, _)| *s == step)
            .map(|(_, v)| *v)
    }

    /// The intermediate value at the smallest reported step `>= step`, if any.
    /// Used by rung-based pruners that compare trials at a resource budget.
    pub fn value_at_or_after(&self, step: usize) -> Option<f64> {
        self.intermediate_values
            .iter()
            .filter(|(s, _)| *s >= step)
            .min_by_key(|(s, _)| *s)
            .map(|(_, v)| *v)
    }

    /// The most recent `(step, value)` report, if the trial has reported.
    pub fn last_intermediate(&self) -> Option<(usize, f64)> {
        self.intermediate_values.last().copied()
    }
}
