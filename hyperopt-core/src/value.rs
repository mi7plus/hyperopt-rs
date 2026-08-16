use serde::{Deserialize, Serialize};
use std::fmt;

/// A concrete hyperparameter value suggested for a trial.
///
/// For v1 the categorical variant is restricted to string/enum-like values
/// (`Categorical(String)`); arbitrary user types are intentionally out of scope
/// to keep the sampler/storage plumbing simple. Callers that need richer
/// categoricals should map their type to/from a stable string label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// A continuous value (from a `Uniform`/`LogUniform` distribution).
    Float(f64),
    /// A discrete integer value (from an `IntUniform` distribution).
    Int(i64),
    /// A categorical choice, identified by its label.
    Categorical(String),
}

impl Value {
    /// Returns the inner `f64` if this is a [`Value::Float`].
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(x) => Some(*x),
            _ => None,
        }
    }

    /// Returns the inner `i64` if this is a [`Value::Int`].
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(x) => Some(*x),
            _ => None,
        }
    }

    /// Returns the inner label if this is a [`Value::Categorical`].
    pub fn as_categorical(&self) -> Option<&str> {
        match self {
            Value::Categorical(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Best-effort numeric projection, used by adaptive samplers and the
    /// param-importance proxy. Categoricals have no meaningful scalar and
    /// return `None`.
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Value::Float(x) => Some(*x),
            Value::Int(x) => Some(*x as f64),
            Value::Categorical(_) => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(x) => write!(f, "{x}"),
            Value::Int(x) => write!(f, "{x}"),
            Value::Categorical(s) => write!(f, "{s}"),
        }
    }
}
