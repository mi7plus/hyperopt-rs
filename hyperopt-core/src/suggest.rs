use crate::TrialContext;

/// The portable objective interface: exactly the operations a user objective
/// performs on a trial (suggest parameters, report intermediate values, ask
/// whether to prune).
///
/// [`TrialContext`] implements it for local execution, and the distributed
/// worker's remote trial implements it too, so **the same objective closure can
/// run unchanged locally or against a coordinator** — just write it against
/// `&mut impl Suggest` instead of `&mut TrialContext`:
///
/// ```
/// use hyperopt_core::Suggest;
///
/// fn objective(trial: &mut impl Suggest) -> f64 {
///     let x = trial.suggest_float("x", -10.0, 10.0);
///     let y = trial.suggest_int("y", 0, 5) as f64;
///     (x - 2.0).powi(2) + y
/// }
/// ```
pub trait Suggest {
    /// Suggest a continuous value uniformly over `[low, high]`.
    fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> f64;
    /// Suggest a continuous value log-uniformly over `[low, high]`.
    fn suggest_loguniform(&mut self, name: &str, low: f64, high: f64) -> f64;
    /// Suggest an integer uniformly over the inclusive range `[low, high]`.
    fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> i64;
    /// Suggest one of `choices`, returning the chosen label.
    fn suggest_categorical(&mut self, name: &str, choices: &[&str]) -> String;
    /// Report an intermediate objective value at `step`, for pruners.
    fn report(&mut self, step: usize, value: f64);
    /// Ask whether this trial should stop early given what it has reported.
    ///
    /// Takes `&mut self` (unlike [`TrialContext::should_prune`], which only
    /// reads) so a *remote* implementation can perform the round-trip to its
    /// coordinator through the same connection; local implementations simply
    /// ignore the mutability.
    fn should_prune(&mut self) -> bool;
}

impl Suggest for TrialContext<'_> {
    fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> f64 {
        TrialContext::suggest_float(self, name, low, high)
    }
    fn suggest_loguniform(&mut self, name: &str, low: f64, high: f64) -> f64 {
        TrialContext::suggest_loguniform(self, name, low, high)
    }
    fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> i64 {
        TrialContext::suggest_int(self, name, low, high)
    }
    fn suggest_categorical(&mut self, name: &str, choices: &[&str]) -> String {
        TrialContext::suggest_categorical(self, name, choices)
    }
    fn report(&mut self, step: usize, value: f64) {
        TrialContext::report(self, step, value)
    }
    fn should_prune(&mut self) -> bool {
        TrialContext::should_prune(self)
    }
}
