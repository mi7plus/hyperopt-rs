//! The coordinator: a TCP server that owns the authoritative study — its
//! sampler, pruner, storage, and the trial-number counter — and hands out
//! parameter suggestions and pruning verdicts to remote workers.

use hyperopt_core::{
    Direction, HyperoptError, Pruner, Sampler, Storage, StorageError, StudyMetadata, StudyState,
    Trial, TrialState,
};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, ToSocketAddrs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::protocol::{Outcome, Request, Response};
use crate::transport::{constant_time_eq, Stream};

/// The authoritative owner of a distributed study.
///
/// A `Coordinator` holds the same parts a local [`Study`](hyperopt_core::Study)
/// does — a sampler behind a lock, a pruner, and a storage backend — plus an
/// atomic trial-number counter. Workers connect over TCP and drive the objective
/// on their own machines, round-tripping each `suggest_*` call back here so that
/// **every worker samples against one shared, authoritative history** and every
/// trial gets a unique number (the piece plain shared storage cannot guarantee
/// on its own).
///
/// Concurrency mirrors [`Study::optimize_parallel`](hyperopt_core::Study): the
/// sampler is serialized behind a `Mutex`, the storage backend is `Send + Sync`
/// with interior mutability, and workers necessarily see a *slightly stale*
/// snapshot of history — by design, exactly as under local parallelism.
pub struct Coordinator {
    name: String,
    direction: Direction,
    sampler: Mutex<Box<dyn Sampler>>,
    pruner: Box<dyn Pruner>,
    storage: Box<dyn Storage>,
    counter: AtomicUsize,
    /// Optional shared secret; when set, a worker must authenticate before any
    /// other request is served.
    token: Option<String>,
}

impl Coordinator {
    /// Assemble a coordinator from its parts, mirroring
    /// [`Study::new`](hyperopt_core::Study::new): study metadata is persisted (or
    /// an existing study's direction is adopted), and the trial counter is
    /// seeded past any trials already in storage so a resumed study keeps
    /// numbering where it left off.
    pub fn new(
        name: impl Into<String>,
        direction: Direction,
        sampler: Box<dyn Sampler>,
        pruner: Box<dyn Pruner>,
        storage: Box<dyn Storage>,
    ) -> Result<Self, HyperoptError> {
        let name = name.into();
        let direction = match storage.load_study_metadata(&name)? {
            Some(meta) => meta.direction,
            None => {
                storage.save_study_metadata(&StudyMetadata {
                    study_name: name.clone(),
                    direction,
                })?;
                direction
            }
        };
        let base = storage.load_trials(&name)?.len();
        Ok(Coordinator {
            name,
            direction,
            sampler: Mutex::new(sampler),
            pruner,
            storage,
            counter: AtomicUsize::new(base),
            token: None,
        })
    }

    /// Require workers to present this shared-secret token (via
    /// [`Worker::authenticate`](crate::Worker::authenticate)) before any request
    /// is served. Without this, the coordinator accepts any connection — fine on
    /// a trusted network, but pair it with a token (and/or TLS) otherwise.
    pub fn require_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Bind a TCP listener without yet accepting connections, so the caller can
    /// read the resolved [`local_addr`](Listening::local_addr) (useful when
    /// binding to port 0) before handing control to the blocking accept loop.
    pub fn listen(self, addr: impl ToSocketAddrs) -> std::io::Result<Listening> {
        let listener = TcpListener::bind(addr)?;
        Ok(Listening {
            listener,
            coord: Arc::new(self),
            transport: Transport::Plain,
        })
    }

    /// Like [`Coordinator::listen`], but wraps every connection in TLS using the
    /// given DER certificate chain and private key (PKCS#8/SEC1/PKCS#1). Requires
    /// the `tls` feature.
    #[cfg(feature = "tls")]
    pub fn listen_tls(
        self,
        addr: impl ToSocketAddrs,
        cert_chain_der: Vec<Vec<u8>>,
        private_key_der: Vec<u8>,
    ) -> std::io::Result<Listening> {
        let config = crate::transport::tls::server_config(cert_chain_der, private_key_der)?;
        let listener = TcpListener::bind(addr)?;
        Ok(Listening {
            listener,
            coord: Arc::new(self),
            transport: Transport::Tls(config),
        })
    }

    /// Bind and serve on `addr`, blocking forever (convenience over
    /// [`Coordinator::listen`] + [`Listening::run`]).
    pub fn serve(self, addr: impl ToSocketAddrs) -> std::io::Result<()> {
        self.listen(addr)?.run()
    }

    /// The study's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A fresh snapshot of every trial recorded for the study — safe to call
    /// while the coordinator is serving, to inspect progress.
    pub fn trials(&self) -> Result<Vec<Trial>, HyperoptError> {
        Ok(self.storage.load_trials(&self.name)?)
    }

    /// The best objective value seen so far, if any trial has completed.
    pub fn best_value(&self) -> Result<Option<f64>, HyperoptError> {
        let trials = self.storage.load_trials(&self.name)?;
        Ok(StudyState::new(self.direction, trials)
            .best_trial()
            .and_then(|t| t.value))
    }

    // --- request handling ---------------------------------------------------

    /// Handle one request, tracking whether this connection has authenticated.
    /// `Auth` is intercepted here; every other request is refused until the
    /// connection is authenticated (a no-op when no token is configured).
    fn handle_authed(&self, req: Request, authed: &mut bool) -> Response {
        if let Request::Auth { token } = &req {
            return match &self.token {
                Some(expected) => {
                    if constant_time_eq(token, expected) {
                        *authed = true;
                        Response::Ack
                    } else {
                        Response::Error("authentication failed".into())
                    }
                }
                // No token configured: authentication is a harmless no-op.
                None => {
                    *authed = true;
                    Response::Ack
                }
            };
        }
        if !*authed {
            return Response::Error("authentication required".into());
        }
        self.handle(req)
    }

    fn handle(&self, req: Request) -> Response {
        // Reject requests aimed at a different study than this coordinator owns.
        if req.study_name() != self.name {
            return Response::Error(format!(
                "unknown study {:?} (this coordinator serves {:?})",
                req.study_name(),
                self.name
            ));
        }
        let result = match req {
            // Auth is handled in `handle_authed` before we ever get here.
            Request::Auth { .. } => Ok(Response::Ack),
            Request::NewTrial { .. } => self.new_trial(),
            Request::Suggest {
                number,
                name,
                distribution,
                ..
            } => self.suggest(number, &name, distribution),
            Request::Report {
                number,
                step,
                value,
                ..
            } => self.report(number, step, value),
            Request::ShouldPrune { number, .. } => self.should_prune(number),
            Request::Finish {
                number, outcome, ..
            } => self.finish(number, outcome),
            Request::BestValue { .. } => self.best_value_response(),
        };
        result.unwrap_or_else(|e| Response::Error(e.to_string()))
    }

    fn new_trial(&self) -> Result<Response, StorageError> {
        let number = self.counter.fetch_add(1, Ordering::SeqCst);
        self.storage.save_trial(&self.name, &Trial::new(number))?;
        Ok(Response::Trial { number })
    }

    fn suggest(
        &self,
        number: usize,
        name: &str,
        distribution: hyperopt_core::Distribution,
    ) -> Result<Response, StorageError> {
        let (state, mut current) = self.partition(number)?;
        let value = {
            let mut sampler = self
                .sampler
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            sampler.suggest(&state, &current, name, &distribution)
        };
        current.record(name, distribution, value.clone());
        self.storage.save_trial(&self.name, &current)?;
        Ok(Response::Value(value))
    }

    fn report(&self, number: usize, step: usize, value: f64) -> Result<Response, StorageError> {
        let (_, mut current) = self.partition(number)?;
        if let Some(slot) = current
            .intermediate_values
            .iter_mut()
            .find(|(s, _)| *s == step)
        {
            slot.1 = value;
        } else {
            current.intermediate_values.push((step, value));
        }
        self.storage.save_trial(&self.name, &current)?;
        Ok(Response::Ack)
    }

    fn should_prune(&self, number: usize) -> Result<Response, StorageError> {
        let (state, current) = self.partition(number)?;
        Ok(Response::Prune(self.pruner.should_prune(&state, &current)))
    }

    fn finish(&self, number: usize, outcome: Outcome) -> Result<Response, StorageError> {
        let (_, mut current) = self.partition(number)?;
        match outcome {
            Outcome::Complete(v) => {
                current.value = Some(v);
                current.state = TrialState::Complete;
            }
            Outcome::Pruned => current.state = TrialState::Pruned,
            Outcome::Failed => current.state = TrialState::Failed,
        }
        self.storage.save_trial(&self.name, &current)?;
        Ok(Response::Ack)
    }

    fn best_value_response(&self) -> Result<Response, StorageError> {
        let trials = self.storage.load_trials(&self.name)?;
        let state = StudyState::new(self.direction, trials);
        Ok(Response::Best(state.best_trial().and_then(|t| t.value)))
    }

    /// Load the study, splitting off the in-flight trial `number` from the rest.
    /// The sampler/pruner see history *excluding* the current trial (matching
    /// the local [`TrialContext`](hyperopt_core::TrialContext) semantics), while
    /// the current trial — carrying the parameters suggested so far this trial —
    /// is returned separately.
    fn partition(&self, number: usize) -> Result<(StudyState, Trial), StorageError> {
        let all = self.storage.load_trials(&self.name)?;
        let mut current = None;
        let mut rest = Vec::with_capacity(all.len());
        for t in all {
            if t.number == number {
                current = Some(t);
            } else {
                rest.push(t);
            }
        }
        let current = current.unwrap_or_else(|| Trial::new(number));
        Ok((StudyState::new(self.direction, rest), current))
    }
}

/// How accepted connections are wrapped before the protocol runs over them.
enum Transport {
    /// Plain TCP.
    Plain,
    /// TLS via the `tls` feature.
    #[cfg(feature = "tls")]
    Tls(std::sync::Arc<rustls::ServerConfig>),
}

impl Transport {
    /// Wrap a freshly accepted TCP stream into a boxed connection stream.
    fn wrap(&self, tcp: std::net::TcpStream) -> std::io::Result<Stream> {
        match self {
            Transport::Plain => Ok(Box::new(tcp)),
            #[cfg(feature = "tls")]
            Transport::Tls(config) => {
                Ok(Box::new(crate::transport::tls::accept(config.clone(), tcp)?))
            }
        }
    }
}

/// A bound-but-not-yet-serving coordinator (see [`Coordinator::listen`]).
pub struct Listening {
    listener: TcpListener,
    coord: Arc<Coordinator>,
    transport: Transport,
}

impl Listening {
    /// The address the listener bound to (resolves port 0 to the real port).
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// A shared handle to the coordinator, so its study can be inspected
    /// (e.g. [`Coordinator::trials`], [`Coordinator::best_value`]) from another
    /// thread while [`Listening::run`] serves.
    pub fn coordinator(&self) -> Arc<Coordinator> {
        Arc::clone(&self.coord)
    }

    /// Serve forever: accept connections and handle each on its own thread.
    /// Returns only if accepting fails.
    pub fn run(self) -> std::io::Result<()> {
        for tcp in self.listener.incoming() {
            let tcp = tcp?;
            let stream = match self.transport.wrap(tcp) {
                Ok(s) => s,
                Err(e) => {
                    // A failed TLS handshake affects only that connection.
                    eprintln!("hyperopt-distributed: connection setup failed: {e}");
                    continue;
                }
            };
            let coord = Arc::clone(&self.coord);
            std::thread::spawn(move || {
                if let Err(e) = handle_connection(coord, stream) {
                    // A dropped/broken worker connection is normal; don't crash
                    // the whole coordinator over it.
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        eprintln!("hyperopt-distributed: worker connection ended: {e}");
                    }
                }
            });
        }
        Ok(())
    }
}

/// Serve one worker connection: read newline-delimited requests, answer each.
/// A single stream carries both directions (reads through the `BufReader`,
/// writes through its `get_mut`), so this works over plain TCP or TLS alike.
fn handle_connection(coord: Arc<Coordinator>, stream: Stream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream);
    let mut authed = coord.token.is_none();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break; // peer closed
        }
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(req) => coord.handle_authed(req, &mut authed),
            Err(e) => Response::Error(format!("bad request: {e}")),
        };
        let mut encoded = serde_json::to_string(&response).unwrap_or_else(|e| {
            serde_json::to_string(&Response::Error(e.to_string())).unwrap()
        });
        encoded.push('\n');
        reader.get_mut().write_all(encoded.as_bytes())?;
        reader.get_mut().flush()?;
    }
    Ok(())
}

impl Request {
    fn study_name(&self) -> &str {
        match self {
            Request::NewTrial { study }
            | Request::Suggest { study, .. }
            | Request::Report { study, .. }
            | Request::ShouldPrune { study, .. }
            | Request::Finish { study, .. }
            | Request::BestValue { study } => study,
            // Auth carries no study; it is handled before this check.
            Request::Auth { .. } => "",
        }
    }
}
