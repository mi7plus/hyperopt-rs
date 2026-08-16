use crate::StorageError;
use std::fmt;

/// Signalled by a user objective to describe how a trial ended.
///
/// Returning `Ok(value)` completes the trial; returning one of these marks it
/// `Pruned` or `Failed` without aborting the whole study. A blanket `From`
/// makes `?` on any standard error turn into [`ObjectiveError::Failed`], while
/// [`ObjectiveError::pruned`] is used to bail out after `should_prune()`.
#[derive(Debug)]
pub enum ObjectiveError {
    /// The trial was stopped early by a pruner. Marked [`crate::TrialState::Pruned`].
    Pruned,
    /// The objective failed. Marked [`crate::TrialState::Failed`]; the study continues.
    Failed(Box<dyn std::error::Error + Send + Sync>),
}

impl ObjectiveError {
    /// Convenience constructor for the pruned case:
    /// `if ctx.should_prune() { return Err(ObjectiveError::pruned()); }`.
    pub fn pruned() -> Self {
        ObjectiveError::Pruned
    }
}

impl fmt::Display for ObjectiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectiveError::Pruned => write!(f, "trial pruned"),
            ObjectiveError::Failed(e) => write!(f, "objective failed: {e}"),
        }
    }
}

// NB: `ObjectiveError` deliberately does *not* implement `std::error::Error`.
// It is a control-flow signal (pruned vs. failed), and keeping it out of the
// `Error` hierarchy is what lets the blanket `From<E: Error>` below coexist
// with the standard `From<T> for T` — so `?` on any real error inside an
// objective converts cleanly into `Failed`.
impl<E> From<E> for ObjectiveError
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(e: E) -> Self {
        ObjectiveError::Failed(Box::new(e))
    }
}

/// The value an objective closure returns for one trial.
pub type ObjectiveResult = Result<f64, ObjectiveError>;

/// Errors raised by [`crate::Study`] operations (currently all storage-backed).
#[derive(Debug)]
pub enum HyperoptError {
    Storage(StorageError),
}

impl fmt::Display for HyperoptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HyperoptError::Storage(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HyperoptError {}

impl From<StorageError> for HyperoptError {
    fn from(e: StorageError) -> Self {
        HyperoptError::Storage(e)
    }
}
