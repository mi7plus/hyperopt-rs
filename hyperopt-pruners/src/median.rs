use crate::{is_worse, median};
use hyperopt_core::{Pruner, StudyState, Trial};

/// Prunes a trial when its latest intermediate value is worse than the median
/// of other trials' values at the same step.
///
/// Mirrors Optuna's `MedianPruner`:
/// - No pruning until at least `n_startup_trials` trials have completed (so the
///   median is meaningful).
/// - No pruning before `n_warmup_steps` steps have elapsed within a trial (give
///   every trial a chance to get past a noisy start).
/// - At the current step, compare against the median of the values reported at
///   that same step by trials that reached it; prune if strictly worse.
#[derive(Debug, Clone)]
pub struct MedianPruner {
    n_startup_trials: usize,
    n_warmup_steps: usize,
    min_trials_at_step: usize,
}

impl Default for MedianPruner {
    fn default() -> Self {
        MedianPruner {
            n_startup_trials: 5,
            n_warmup_steps: 0,
            min_trials_at_step: 1,
        }
    }
}

impl MedianPruner {
    /// A median pruner with default gates (`n_startup_trials = 5`,
    /// `n_warmup_steps = 0`, `min_trials_at_step = 1`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of completed trials required before any pruning happens.
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Number of steps a trial must run before it becomes eligible for pruning.
    pub fn n_warmup_steps(mut self, n: usize) -> Self {
        self.n_warmup_steps = n;
        self
    }

    /// Minimum number of comparison values required at a step before the median
    /// is trusted enough to prune against.
    pub fn min_trials_at_step(mut self, n: usize) -> Self {
        self.min_trials_at_step = n;
        self
    }
}

impl Pruner for MedianPruner {
    fn should_prune(&self, study_state: &StudyState, trial: &Trial) -> bool {
        if study_state.n_completed() < self.n_startup_trials {
            return false;
        }
        let Some((step, value)) = trial.last_intermediate() else {
            return false;
        };
        if step < self.n_warmup_steps {
            return false;
        }
        let others = study_state.intermediate_values_at(step);
        if others.len() < self.min_trials_at_step {
            return false;
        }
        match median(&others) {
            Some(m) => is_worse(study_state.direction(), value, m),
            None => false,
        }
    }
}
