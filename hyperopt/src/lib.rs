//! # hyperopt
//!
//! `hyperopt-rs` — an Optuna-shaped hyperparameter optimization framework for
//! Rust. This is the ergonomic facade crate: it re-exports the core types and
//! every sampler, pruner, and storage backend, and adds a [`StudyBuilder`] so a
//! study can be assembled in one fluent expression.
//!
//! ```no_run
//! use hyperopt::prelude::*;
//!
//! # fn main() -> Result<(), HyperoptError> {
//! let study = StudyBuilder::new("quadratic")
//!     .direction(Direction::Minimize)
//!     .sampler(TpeSampler::seeded(42))
//!     .build()?;
//!
//! study.optimize(|trial| {
//!     let x = trial.suggest_float("x", -10.0, 10.0);
//!     let y = trial.suggest_float("y", -10.0, 10.0);
//!     Ok((x - 2.0).powi(2) + (y + 3.0).powi(2))
//! }, 200)?;
//!
//! println!("best = {:?}", study.best_trial()?);
//! # Ok(()) }
//! ```

pub use hyperopt_core::{
    Direction, Distribution, HyperoptError, ObjectiveError, ObjectiveResult, ParamRecord, Pruner,
    Sampler, Storage, StorageError, Study, StudyMetadata, StudyState, Trial, TrialContext,
    TrialState, Value,
};
pub use hyperopt_pruners::{MedianPruner, NopPruner, SuccessiveHalvingPruner};
pub use hyperopt_samplers::{BoundHandling, CmaEsSampler, GridSampler, RandomSampler, TpeSampler};
pub use hyperopt_storage::InMemoryStorage;
#[cfg(feature = "sqlite")]
pub use hyperopt_storage::SqliteStorage;

/// Fluent builder for a [`Study`], with sensible defaults: a
/// [`RandomSampler`], a [`NopPruner`], an [`InMemoryStorage`], and
/// [`Direction::Minimize`]. Override any of them before calling
/// [`StudyBuilder::build`].
pub struct StudyBuilder {
    name: String,
    direction: Direction,
    sampler: Option<Box<dyn Sampler>>,
    pruner: Option<Box<dyn Pruner>>,
    storage: Option<Box<dyn Storage>>,
}

impl StudyBuilder {
    /// Start building a study with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        StudyBuilder {
            name: name.into(),
            direction: Direction::Minimize,
            sampler: None,
            pruner: None,
            storage: None,
        }
    }

    /// Set the optimization direction (default [`Direction::Minimize`]).
    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Set the search algorithm (default [`RandomSampler`]).
    pub fn sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Some(Box::new(sampler));
        self
    }

    /// Set the early-stopping policy (default [`NopPruner`]).
    pub fn pruner(mut self, pruner: impl Pruner + 'static) -> Self {
        self.pruner = Some(Box::new(pruner));
        self
    }

    /// Set the storage backend (default [`InMemoryStorage`]).
    pub fn storage(mut self, storage: impl Storage + 'static) -> Self {
        self.storage = Some(Box::new(storage));
        self
    }

    /// Assemble the [`Study`], applying defaults for anything not set.
    pub fn build(self) -> Result<Study, HyperoptError> {
        let sampler = self
            .sampler
            .unwrap_or_else(|| Box::new(RandomSampler::new()));
        let pruner = self.pruner.unwrap_or_else(|| Box::new(NopPruner::new()));
        let storage = self
            .storage
            .unwrap_or_else(|| Box::new(InMemoryStorage::new()));
        Study::new(self.name, self.direction, sampler, pruner, storage)
    }
}

/// Common imports for using the framework: the builder, the study/trial types,
/// direction, error/objective types, and all samplers and pruners.
pub mod prelude {
    pub use crate::StudyBuilder;
    pub use hyperopt_core::{
        Direction, Distribution, HyperoptError, ObjectiveError, ObjectiveResult, Pruner, Sampler,
        Storage, Study, Trial, TrialContext, TrialState, Value,
    };
    pub use hyperopt_pruners::{MedianPruner, NopPruner, SuccessiveHalvingPruner};
    pub use hyperopt_samplers::{
        BoundHandling, CmaEsSampler, GridSampler, RandomSampler, TpeSampler,
    };
    pub use hyperopt_storage::InMemoryStorage;
    #[cfg(feature = "sqlite")]
    pub use hyperopt_storage::SqliteStorage;
}
