//! # hyperopt-distributed
//!
//! Multi-machine distributed execution for `hyperopt-rs`: a **coordinator**
//! server owns one authoritative study (sampler + pruner + storage) and hands
//! out parameter suggestions, while **workers** on any number of machines run
//! the objective and round-trip each `suggest_*` / `report` / `should_prune`
//! call back to the coordinator.
//!
//! This is the true-distributed counterpart to `SqliteStorage`'s shared-file
//! model. Where shared storage lets several *local* processes cooperate but
//! leaves trial numbering and sampler state to race on the file, the
//! coordinator is the single source of truth: it assigns every trial number
//! atomically and runs the one sampler against the one history, so N machines
//! behave like one big [`Study::optimize_parallel`](hyperopt_core::Study) — with
//! the same intentional *slightly-stale-snapshot* semantics under concurrency.
//!
//! The transport is deliberately dependency-light: newline-delimited JSON over
//! `std::net` TCP, one thread per worker connection. No async runtime.
//!
//! ## Securing the transport
//!
//! Plain TCP is fine on a trusted network. For anything less, two opt-in layers
//! are available:
//!
//! - **Token auth** (always available, `std`-only): call
//!   [`Coordinator::require_token`] and have each worker call
//!   [`Worker::authenticate`] before optimizing. The token is compared in
//!   constant time and gates every other request.
//! - **TLS** (the `tls` feature, via `rustls` + `ring` — no OpenSSL): serve with
//!   [`Coordinator::listen_tls`] and connect with [`Worker::connect_tls`], which
//!   take DER certificate/key bytes so a self-signed coordinator needs no PEM
//!   plumbing. Combine both for authenticated *and* encrypted links.
//!
//! ## Sketch
//!
//! Coordinator (one process, anywhere reachable):
//!
//! ```no_run
//! use hyperopt_core::Direction;
//! use hyperopt_distributed::Coordinator;
//! use hyperopt_samplers::TpeSampler;
//! use hyperopt_pruners::NopPruner;
//! use hyperopt_storage::InMemoryStorage;
//!
//! # fn main() -> std::io::Result<()> {
//! let coord = Coordinator::new(
//!     "distributed-study",
//!     Direction::Minimize,
//!     Box::new(TpeSampler::seeded(42)),
//!     Box::new(NopPruner::new()),
//!     Box::new(InMemoryStorage::new()),
//! ).expect("build coordinator");
//! coord.serve("0.0.0.0:7777")?; // blocks, serving workers
//! # Ok(()) }
//! ```
//!
//! Worker (run on as many machines as you like, pointed at the coordinator):
//!
//! ```no_run
//! use hyperopt_distributed::{RemoteTrial, Worker};
//!
//! # fn main() -> std::io::Result<()> {
//! let mut worker = Worker::connect("coordinator-host:7777", "distributed-study")?;
//! worker.optimize(|trial: &mut RemoteTrial| {
//!     let x = trial.suggest_float("x", -10.0, 10.0);
//!     let y = trial.suggest_float("y", -10.0, 10.0);
//!     Ok((x - 2.0).powi(2) + (y + 3.0).powi(2))
//! }, 100)?;
//! println!("best so far: {:?}", worker.best_value()?);
//! # Ok(()) }
//! ```
//!
//! Because [`RemoteTrial`] implements [`hyperopt_core::Suggest`], the *same*
//! objective written against `&mut impl Suggest` runs unchanged locally (with a
//! [`TrialContext`](hyperopt_core::TrialContext)) or here on a worker.

mod coordinator;
mod protocol;
mod transport;
mod worker;

pub use coordinator::{Coordinator, Listening};
pub use protocol::{Outcome, Request, Response};
pub use worker::{RemoteTrial, Worker};
