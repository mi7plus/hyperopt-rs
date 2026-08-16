use crate::{
    Direction, HyperoptError, ObjectiveError, ObjectiveResult, Pruner, Sampler, Storage,
    StudyMetadata, StudyState, Trial, TrialContext, TrialState,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;

/// A single optimization run: a direction, a sampler, a pruner, and a storage
/// backend, tied to a named study whose trials live in that storage.
///
/// Construct one with [`Study::new`], or use the ergonomic builder in the
/// `hyperopt` facade crate. Run trials with [`Study::optimize`] (sequential) or
/// [`Study::optimize_parallel`] (feature `parallel`).
pub struct Study {
    name: String,
    direction: Direction,
    sampler: Mutex<Box<dyn Sampler>>,
    pruner: Box<dyn Pruner>,
    storage: Box<dyn Storage>,
}

impl Study {
    /// Assemble a study from its parts. Persists study metadata immediately so
    /// the direction survives a reload. If the study already exists in
    /// `storage`, its recorded direction is authoritative and is adopted here
    /// (so a resumed study can't silently flip direction).
    pub fn new(
        name: impl Into<String>,
        direction: Direction,
        sampler: Box<dyn Sampler>,
        pruner: Box<dyn Pruner>,
        storage: Box<dyn Storage>,
    ) -> Result<Self, HyperoptError> {
        let name = name.into();
        let direction = match storage.load_study_metadata(&name)? {
            Some(meta) => meta.direction,
            None => {
                storage.save_study_metadata(&StudyMetadata {
                    study_name: name.clone(),
                    direction,
                })?;
                direction
            }
        };
        Ok(Study {
            name,
            direction,
            sampler: Mutex::new(sampler),
            pruner,
            storage,
        })
    }

    /// The study's name (its key in storage).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The optimization direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// A fresh snapshot of every trial recorded for this study.
    pub fn trials(&self) -> Result<Vec<Trial>, HyperoptError> {
        Ok(self.storage.load_trials(&self.name)?)
    }

    /// The best completed trial under the study direction, if any.
    pub fn best_trial(&self) -> Result<Option<Trial>, HyperoptError> {
        let trials = self.storage.load_trials(&self.name)?;
        let state = StudyState::new(self.direction, trials);
        Ok(state.best_trial().cloned())
    }

    /// The best objective value seen so far, if any trial has completed.
    pub fn best_value(&self) -> Result<Option<f64>, HyperoptError> {
        Ok(self.best_trial()?.and_then(|t| t.value))
    }

    /// Run `n_trials` sequentially. A trial that panics or returns
    /// [`ObjectiveError::Failed`] is marked `Failed` and the run continues; one
    /// that returns [`ObjectiveError::Pruned`] is marked `Pruned`.
    pub fn optimize<F>(&self, mut objective: F, n_trials: usize) -> Result<(), HyperoptError>
    where
        F: FnMut(&mut TrialContext) -> ObjectiveResult,
    {
        for _ in 0..n_trials {
            let existing = self.storage.load_trials(&self.name)?;
            let number = existing.len();
            let state = StudyState::new(self.direction, existing);
            let mut ctx =
                TrialContext::new(number, &self.sampler, &state, self.pruner.as_ref());

            let result = catch_unwind(AssertUnwindSafe(|| objective(&mut ctx)));
            let trial = finish_trial(ctx.into_trial(), result);
            self.storage.save_trial(&self.name, &trial)?;
        }
        Ok(())
    }

    /// Run `n_trials` across `n_workers` threads via `rayon`.
    ///
    /// **Stale-view semantics (by design, not a bug):** under parallelism
    /// several trials may be suggested before earlier ones finish and are
    /// saved, so a sampler necessarily works from a *partial, slightly stale*
    /// snapshot of study history. This matches Optuna's behaviour under
    /// parallel execution and is why a parallel run of a study can diverge
    /// somewhat from a sequential one — compare best-values, not exact
    /// trajectories. The sampler is shared behind a lock and the storage
    /// backend must be `Send + Sync`; both are audited for this.
    #[cfg(feature = "parallel")]
    pub fn optimize_parallel<F>(
        &self,
        objective: F,
        n_trials: usize,
        n_workers: usize,
    ) -> Result<(), HyperoptError>
    where
        F: Fn(&mut TrialContext) -> ObjectiveResult + Sync,
    {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let base = self.storage.load_trials(&self.name)?.len();
        let counter = AtomicUsize::new(base);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| HyperoptError::Storage(crate::StorageError::Backend(e.to_string())))?;

        pool.install(|| {
            (0..n_trials).into_par_iter().try_for_each(|_| {
                let number = counter.fetch_add(1, Ordering::SeqCst);
                // Each worker takes its own (possibly stale) snapshot.
                let existing = self.storage.load_trials(&self.name)?;
                let state = StudyState::new(self.direction, existing);
                let mut ctx =
                    TrialContext::new(number, &self.sampler, &state, self.pruner.as_ref());
                let result = catch_unwind(AssertUnwindSafe(|| objective(&mut ctx)));
                let trial = finish_trial(ctx.into_trial(), result);
                self.storage.save_trial(&self.name, &trial)?;
                Ok::<(), HyperoptError>(())
            })
        })
    }
}

/// Fold a trial's evaluation outcome (including a caught panic) into its final
/// state and value.
fn finish_trial(
    mut trial: Trial,
    result: std::thread::Result<ObjectiveResult>,
) -> Trial {
    match result {
        Ok(Ok(value)) => {
            trial.value = Some(value);
            trial.state = TrialState::Complete;
        }
        Ok(Err(ObjectiveError::Pruned)) => {
            trial.state = TrialState::Pruned;
        }
        Ok(Err(ObjectiveError::Failed(_))) => {
            trial.state = TrialState::Failed;
        }
        Err(_panic) => {
            // A single bad trial shouldn't abort a long-running study.
            trial.state = TrialState::Failed;
        }
    }
    trial
}
