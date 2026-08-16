use crate::{Distribution, StudyState, Trial, Value};

/// A pluggable search algorithm.
///
/// A sampler decides which concrete [`Value`] to return for each parameter a
/// trial requests. It receives the whole [`StudyState`] (read access to prior
/// trials) so adaptive samplers such as TPE have their history without a
/// separate channel, plus the current partially-built `trial`, the parameter
/// `name`, and its `distribution`.
///
/// Implementations must return a `Value` whose variant matches the
/// `distribution` (`Float` for `Uniform`/`LogUniform`, `Int` for `IntUniform`,
/// `Categorical` for `Categorical`).
///
/// `Send` is required so a sampler can be shared across worker threads (behind
/// a lock) under [`crate::Study::optimize_parallel`].
pub trait Sampler: Send {
    fn suggest(
        &mut self,
        study_state: &StudyState,
        trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Value;
}

/// A pluggable early-stopping policy.
///
/// Called from [`crate::TrialContext::should_prune`] inside the user's
/// objective, typically right after a `report(step, value)` call. The user's
/// loop is expected to break early when this returns `true`. Takes `&self` plus
/// the [`StudyState`] so cross-trial policies can compare against other trials'
/// intermediate histories.
pub trait Pruner: Send + Sync {
    fn should_prune(&self, study_state: &StudyState, trial: &Trial) -> bool;
}
