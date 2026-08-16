use crate::{Distribution, Pruner, Sampler, StudyState, Trial, Value};
use std::sync::Mutex;

/// The handle passed into the user's objective closure.
///
/// This is where define-by-run happens: the search space is discovered by
/// *calling* `suggest_*` methods here, not declared up front. Each call
/// (1) asks the active [`Sampler`] for a value given the study history and the
/// distribution, then (2) records the `(name, distribution, value)` on the
/// trial. Recording after asking keeps the trial's history accurate for the
/// storage, pruning, and importance phases that depend on it.
///
/// The sampler is held behind a shared [`Mutex`] so the exact same context type
/// serves both sequential and parallel execution; under sequential runs the
/// lock is uncontended.
pub struct TrialContext<'a> {
    trial: Trial,
    sampler: &'a Mutex<Box<dyn Sampler>>,
    study_state: &'a StudyState,
    pruner: &'a dyn Pruner,
}

impl<'a> TrialContext<'a> {
    /// Internal: build a context for one trial. Used by [`crate::Study`].
    pub(crate) fn new(
        number: usize,
        sampler: &'a Mutex<Box<dyn Sampler>>,
        study_state: &'a StudyState,
        pruner: &'a dyn Pruner,
    ) -> Self {
        TrialContext {
            trial: Trial::new(number),
            sampler,
            study_state,
            pruner,
        }
    }

    /// This trial's number within the study.
    pub fn number(&self) -> usize {
        self.trial.number
    }

    fn suggest(&mut self, name: &str, distribution: Distribution) -> Value {
        let value = {
            // Recover from poisoning: a prior trial that panicked mid-suggest
            // shouldn't take down the rest of the study.
            let mut sampler = self
                .sampler
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            sampler.suggest(self.study_state, &self.trial, name, &distribution)
        };
        self.trial.record(name, distribution, value.clone());
        value
    }

    /// Suggest a continuous value uniformly over `[low, high]`.
    pub fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> f64 {
        let v = self.suggest(name, Distribution::Uniform { low, high });
        coerce_float(&v)
    }

    /// Suggest a continuous value log-uniformly over `[low, high]`
    /// (both must be `> 0`). Good for scale parameters like learning rates.
    pub fn suggest_loguniform(&mut self, name: &str, low: f64, high: f64) -> f64 {
        let v = self.suggest(name, Distribution::LogUniform { low, high });
        coerce_float(&v)
    }

    /// Suggest an integer uniformly over the inclusive range `[low, high]`.
    pub fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> i64 {
        let v = self.suggest(name, Distribution::IntUniform { low, high });
        match v {
            Value::Int(x) => x,
            Value::Float(x) => x.round() as i64,
            Value::Categorical(_) => low,
        }
    }

    /// Suggest one of `choices`, returning the chosen label.
    pub fn suggest_categorical(&mut self, name: &str, choices: &[&str]) -> String {
        let dist = Distribution::Categorical {
            choices: choices.iter().map(|s| s.to_string()).collect(),
        };
        let v = self.suggest(name, dist);
        match v {
            Value::Categorical(s) => s,
            // Defensive: a sampler returning a numeric value for a categorical
            // is treated as an index into `choices`.
            Value::Int(i) => choices
                .get(i.max(0) as usize)
                .map(|s| s.to_string())
                .unwrap_or_default(),
            Value::Float(f) => choices
                .get(f as usize)
                .map(|s| s.to_string())
                .unwrap_or_default(),
        }
    }

    /// Report an intermediate objective value at `step` (e.g. per-epoch
    /// validation score). Consumed by pruners via [`Self::should_prune`].
    pub fn report(&mut self, step: usize, value: f64) {
        // Overwrite an existing report for the same step rather than duplicate.
        if let Some(slot) = self
            .trial
            .intermediate_values
            .iter_mut()
            .find(|(s, _)| *s == step)
        {
            slot.1 = value;
        } else {
            self.trial.intermediate_values.push((step, value));
        }
    }

    /// Ask the active pruner whether this trial should stop early, based on the
    /// intermediate values reported so far versus the rest of the study.
    pub fn should_prune(&self) -> bool {
        self.pruner.should_prune(self.study_state, &self.trial)
    }

    /// Read-only access to the study snapshot this trial sees.
    pub fn study_state(&self) -> &StudyState {
        self.study_state
    }

    /// Consume the context, yielding the trial it built up.
    pub(crate) fn into_trial(self) -> Trial {
        self.trial
    }
}

fn coerce_float(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        Value::Int(x) => *x as f64,
        Value::Categorical(_) => f64::NAN,
    }
}
