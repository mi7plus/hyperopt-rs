//! # hyperopt-viz
//!
//! Optional analysis layer for `hyperopt-rs` studies — turning a completed
//! study into insight rather than just a best-trial number.
//!
//! - [`optimization_history_svg`] / [`best_so_far`] — the classic
//!   best-value-so-far vs. trial-number chart, as a one-call function.
//! - [`parameter_importance`] — a lightweight, explicitly-documented **proxy**
//!   for parameter importance (a decision-stump variance-explained score),
//!   standing in for full fANOVA which is out of scope for v1.
//!
//! Depends only on `plotters` (SVG backend), so it stays cheap to build.

mod history;
mod importance;

pub use history::{best_so_far, optimization_history_svg, VizError};
pub use importance::{parameter_importance, Importance};
