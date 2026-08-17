//! # hyperopt-samplers
//!
//! Pluggable [`Sampler`](hyperopt_core::Sampler) implementations for
//! `hyperopt-rs`. All three are interchangeable through the same
//! [`Study`](hyperopt_core::Study) API:
//!
//! - [`RandomSampler`] — independent random draws; the baseline.
//! - [`GridSampler`] — exhaustive enumeration over a caller-provided grid.
//! - [`TpeSampler`] — adaptive Tree-structured Parzen Estimator, wrapping the
//!   [`tpe`](https://docs.rs/tpe) crate.
//! - [`CmaEsSampler`] — Covariance Matrix Adaptation Evolution Strategy for
//!   continuous spaces, implemented from scratch.

mod cmaes;
mod grid;
mod random;
mod tpe_sampler;

pub use cmaes::{BoundHandling, CmaEsSampler};
pub use grid::GridSampler;
pub use random::RandomSampler;
pub use tpe_sampler::TpeSampler;
