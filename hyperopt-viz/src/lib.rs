//! # hyperopt-viz
//!
//! Optional analysis layer for `hyperopt-rs` studies — turning a completed
//! study into insight rather than just a best-trial number.
//!
//! - [`optimization_history_svg`] / [`best_so_far`] — the classic
//!   best-value-so-far vs. trial-number chart, as a one-call function.
//! - [`parameter_importance`] — a lightweight, explicitly-documented **proxy**
//!   for parameter importance (a decision-stump variance-explained score). It is
//!   cheap and needs no model, and is the right default for a quick ranking.
//! - [`fanova_importance`] — the full **fANOVA** method (Hutter et al. 2014): a
//!   random forest of regression trees whose prediction function is decomposed
//!   by functional ANOVA to attribute variance to each parameter's main effect.
//!   Slower and model-based, but marginalizes over the other parameters rather
//!   than trusting a single univariate split.
//! - [`fanova_interactions`] — the second-order fANOVA terms: how much variance
//!   each *pair* of parameters explains together beyond their main effects.
//!
//! Depends only on `plotters` (SVG backend) and `rand`, so it stays cheap to
//! build.

mod fanova;
mod history;
mod importance;

pub use fanova::{
    fanova_importance, fanova_importance_with, fanova_interactions, fanova_interactions_with,
    FanovaOptions, Interaction,
};
pub use history::{best_so_far, optimization_history_svg, VizError};
pub use importance::{parameter_importance, Importance};
