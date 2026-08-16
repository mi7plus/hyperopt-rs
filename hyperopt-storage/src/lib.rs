//! # hyperopt-storage
//!
//! Storage backends for `hyperopt-rs`. A [`hyperopt_core::Storage`] backend
//! decides where a study's trial history lives.
//!
//! - [`InMemoryStorage`] — a `Vec`/map behind a lock; zero setup, no
//!   persistence across process restarts. The default for quick runs.
//! - [`SqliteStorage`] — trials persisted to a SQLite file (feature `sqlite`,
//!   on by default), so a study can be resumed after a restart and multiple
//!   local processes can share one study file.

mod memory;
pub use memory::InMemoryStorage;

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteStorage;
