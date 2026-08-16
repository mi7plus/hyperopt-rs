//! # hyperopt-core
//!
//! Core abstractions for `hyperopt-rs`, an Optuna-shaped hyperparameter
//! optimization framework for Rust. This crate defines the foundational types
//! and the three extension traits everything else plugs into:
//!
//! - [`Sampler`] — a pluggable search algorithm (random, grid, TPE, …).
//! - [`Pruner`] — a pluggable early-stopping policy.
//! - [`Storage`] — where trial history lives (in-memory, SQLite, …).
//!
//! Third parties can depend on just `hyperopt-core` to implement a new sampler
//! or pruner without pulling in SQLite or `rayon`.
//!
//! ## Define-by-run
//!
//! The search space is **not** declared up front. Instead the user's objective
//! closure receives a [`TrialContext`] and *calls* `suggest_*` methods on it;
//! each call records a [`Distribution`] and asks the active [`Sampler`] for a
//! [`Value`]. This allows conditional/dynamic search spaces — a later
//! suggestion can depend on an earlier one — which is the design this framework
//! is built around.
//!
//! ```no_run
//! # use hyperopt_core::*;
//! # fn run(study: &Study) -> Result<(), HyperoptError> {
//! study.optimize(|trial| {
//!     let x = trial.suggest_float("x", -10.0, 10.0);
//!     let y = trial.suggest_float("y", -10.0, 10.0);
//!     Ok((x - 2.0).powi(2) + (y + 3.0).powi(2)) // minimize
//! }, 100)?;
//! # Ok(()) }
//! ```

mod context;
mod distribution;
mod error;
mod storage;
mod study;
mod study_state;
mod traits;
mod trial;
mod value;

pub use context::TrialContext;
pub use distribution::Distribution;
pub use error::{HyperoptError, ObjectiveError, ObjectiveResult};
pub use storage::{Storage, StorageError, StudyMetadata};
pub use study::Study;
pub use study_state::StudyState;
pub use traits::{Pruner, Sampler};
pub use trial::{ParamRecord, Trial, TrialState};
pub use value::Value;

/// Optimization direction: whether the objective should be minimized or
/// maximized. Samplers and pruners read this from [`StudyState::direction`] and
/// adjust accordingly (e.g. TPE always minimizes internally, negating values
/// under `Maximize`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}

impl Direction {
    /// `true` if `a` is a better objective value than `b` under this direction.
    pub fn is_better(self, a: f64, b: f64) -> bool {
        match self {
            Direction::Minimize => a < b,
            Direction::Maximize => a > b,
        }
    }
}
