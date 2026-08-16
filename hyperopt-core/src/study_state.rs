use crate::{Direction, Trial, TrialState};

/// A read-only snapshot of a study's history, handed to samplers and pruners.
///
/// Both [`crate::Sampler::suggest`] and [`crate::Pruner::should_prune`] receive
/// a `StudyState` rather than a single trial so that adaptive samplers (TPE)
/// and cross-trial pruners (median, successive-halving) have everything they
/// need through one channel. Under parallel execution this snapshot is
/// deliberately *slightly stale* — see [`crate::Study::optimize_parallel`].
#[derive(Debug, Clone)]
pub struct StudyState {
    direction: Direction,
    trials: Vec<Trial>,
}

impl StudyState {
    /// Builds a snapshot from the study direction and its trials so far.
    pub fn new(direction: Direction, trials: Vec<Trial>) -> Self {
        StudyState { direction, trials }
    }

    /// The optimization direction of the owning study.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// All trials in the snapshot, in trial-number order.
    pub fn trials(&self) -> &[Trial] {
        &self.trials
    }

    /// Trials that completed successfully and carry a final value.
    pub fn completed_trials(&self) -> impl Iterator<Item = &Trial> {
        self.trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
    }

    /// Number of completed trials — used by samplers/pruners for warmup gates.
    pub fn n_completed(&self) -> usize {
        self.completed_trials().count()
    }

    /// Intermediate values reported at exactly `step` across all trials that
    /// reached it (completed or pruned). Used by [`MedianPruner`](crate).
    pub fn intermediate_values_at(&self, step: usize) -> Vec<f64> {
        self.trials
            .iter()
            .filter(|t| matches!(t.state, TrialState::Complete | TrialState::Pruned))
            .filter_map(|t| t.value_at_step(step))
            .collect()
    }

    /// Values of all trials that reached resource `>= step`, taken at the first
    /// step at or after `step`. Used by rung-based pruners.
    pub fn values_at_or_after(&self, step: usize) -> Vec<f64> {
        self.trials
            .iter()
            .filter_map(|t| t.value_at_or_after(step))
            .collect()
    }

    /// The best completed trial under the study direction, if any.
    pub fn best_trial(&self) -> Option<&Trial> {
        let mut best: Option<&Trial> = None;
        for t in self.completed_trials() {
            let v = t.value.unwrap();
            match best {
                None => best = Some(t),
                Some(b) => {
                    let bv = b.value.unwrap();
                    let better = match self.direction {
                        Direction::Minimize => v < bv,
                        Direction::Maximize => v > bv,
                    };
                    if better {
                        best = Some(t);
                    }
                }
            }
        }
        best
    }
}
