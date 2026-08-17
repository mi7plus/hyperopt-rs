//! The line-delimited JSON protocol spoken between a [`Worker`](crate::Worker)
//! and a [`Coordinator`](crate::Coordinator).
//!
//! Each message is one `serde_json`-encoded [`Request`] or [`Response`] followed
//! by a newline. A worker holds one long-lived TCP connection and issues many
//! request/response round-trips over it (one `NewTrial`, then a `Suggest` per
//! parameter, optional `Report`/`ShouldPrune`, and a final `Finish`).

use hyperopt_core::{Distribution, Value};
use serde::{Deserialize, Serialize};

/// A request from a worker to the coordinator. Every variant names the `study`
/// so a stray connection to the wrong study is rejected rather than silently
/// mixing histories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    /// Present a shared-secret token. Sent first when the coordinator requires
    /// authentication; must succeed before any other request is served.
    Auth { token: String },
    /// Ask the coordinator to allocate the next trial number authoritatively.
    NewTrial { study: String },
    /// Ask the active sampler for one parameter's value, given study history.
    Suggest {
        study: String,
        number: usize,
        name: String,
        distribution: Distribution,
    },
    /// Record an intermediate value for pruning.
    Report {
        study: String,
        number: usize,
        step: usize,
        value: f64,
    },
    /// Ask the active pruner whether this trial should stop early.
    ShouldPrune { study: String, number: usize },
    /// Report the trial's terminal outcome.
    Finish {
        study: String,
        number: usize,
        outcome: Outcome,
    },
    /// Query the study's best objective value so far.
    BestValue { study: String },
}

/// How a trial ended, reported by the worker on `Finish`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Outcome {
    /// The objective returned this value.
    Complete(f64),
    /// A pruner stopped the trial early.
    Pruned,
    /// The objective errored or panicked.
    Failed,
}

/// A response from the coordinator to a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    /// The number allocated for a new trial.
    Trial { number: usize },
    /// A sampled parameter value.
    Value(Value),
    /// A generic acknowledgement (for `Report` / `Finish`).
    Ack,
    /// The pruner's verdict.
    Prune(bool),
    /// The best value so far, if any trial has completed.
    Best(Option<f64>),
    /// The request could not be served.
    Error(String),
}
