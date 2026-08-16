use crate::{Direction, Trial};
use std::fmt;

/// Persisted study-level metadata (everything about a study except its trials).
#[derive(Debug, Clone, PartialEq)]
pub struct StudyMetadata {
    pub study_name: String,
    pub direction: Direction,
}

/// Errors a [`Storage`] backend can raise.
#[derive(Debug)]
pub enum StorageError {
    /// A study with the requested name was not found.
    StudyNotFound(String),
    /// Serialization/deserialization of a trial failed.
    Serialization(String),
    /// The backing store (file, DB) is at an incompatible schema version.
    SchemaMismatch { found: i64, expected: i64 },
    /// Any backend-specific I/O or driver error.
    Backend(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::StudyNotFound(name) => write!(f, "study not found: {name}"),
            StorageError::Serialization(msg) => write!(f, "serialization error: {msg}"),
            StorageError::SchemaMismatch { found, expected } => write!(
                f,
                "storage schema version mismatch: found {found}, expected {expected}"
            ),
            StorageError::Backend(msg) => write!(f, "storage backend error: {msg}"),
        }
    }
}

impl std::error::Error for StorageError {}

/// Where a study's trial history lives.
///
/// Abstracting this keeps [`crate::Study`] agnostic to whether history is a
/// `Vec` in memory or rows in SQLite. Backends must be `Send + Sync` so a study
/// can be optimized in parallel; they are expected to use interior mutability
/// (all methods take `&self`) and upsert trials by `(study, trial number)`.
pub trait Storage: Send + Sync {
    /// Insert or update a trial for the given study.
    fn save_trial(&self, study_name: &str, trial: &Trial) -> Result<(), StorageError>;

    /// Load all trials for a study, in trial-number order. Returns an empty
    /// vec for a study that exists but has no trials yet.
    fn load_trials(&self, study_name: &str) -> Result<Vec<Trial>, StorageError>;

    /// Insert or update study-level metadata.
    fn save_study_metadata(&self, meta: &StudyMetadata) -> Result<(), StorageError>;

    /// Fetch study-level metadata, or `None` if the study is unknown.
    fn load_study_metadata(&self, study_name: &str)
        -> Result<Option<StudyMetadata>, StorageError>;
}
