//! # hyperopt-pruners
//!
//! Early-stopping policies for `hyperopt-rs`. A [`Pruner`] inspects a trial's
//! intermediate reports against the rest of the study and decides whether
//! continuing is worth it. Pruners are queried from the user's objective via
//! [`hyperopt_core::TrialContext::should_prune`].
//!
//! - [`NopPruner`] — never prunes (the default; keeps the API shape consistent).
//! - [`MedianPruner`] — prune when a trial is worse than the median of others
//!   at the same step.
//! - [`SuccessiveHalvingPruner`] — ASHA-style rung promotion (the advanced option).

use hyperopt_core::{Direction, Pruner, StudyState, Trial};

mod median;
mod successive_halving;

pub use median::MedianPruner;
pub use successive_halving::SuccessiveHalvingPruner;

/// A no-op pruner: [`should_prune`](Pruner::should_prune) always returns
/// `false`. Use it when early-stopping isn't wanted but the objective still
/// calls `should_prune()` so the same code runs with and without pruning.
#[derive(Debug, Clone, Copy, Default)]
pub struct NopPruner;

impl NopPruner {
    pub fn new() -> Self {
        NopPruner
    }
}

impl Pruner for NopPruner {
    fn should_prune(&self, _study_state: &StudyState, _trial: &Trial) -> bool {
        false
    }
}

/// Returns `true` if `value` is *worse* than `reference` under `direction`
/// (used by pruners to decide whether a trial is lagging).
pub(crate) fn is_worse(direction: Direction, value: f64, reference: f64) -> bool {
    match direction {
        Direction::Minimize => value > reference,
        Direction::Maximize => value < reference,
    }
}

/// Median of a slice of finite values (average of the two middle elements for
/// even lengths). Returns `None` for an empty slice.
pub(crate) fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        Some(v[n / 2])
    } else {
        Some((v[n / 2 - 1] + v[n / 2]) / 2.0)
    }
}
