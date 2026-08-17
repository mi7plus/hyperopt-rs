//! The worker: connects to a [`Coordinator`](crate::Coordinator), runs the
//! user's objective locally, and round-trips each `suggest_*` / `report` /
//! `should_prune` call back to the coordinator so all workers share one history.

use hyperopt_core::{Distribution, ObjectiveError, ObjectiveResult, Suggest, Value};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::protocol::{Outcome, Request, Response};
use crate::transport::Stream;

/// A connection to a coordinator, bound to one study.
///
/// Create with [`Worker::connect`], then call [`Worker::optimize`] to pull and
/// run trials. Many workers (across many machines) can point at the same
/// coordinator concurrently; each runs its own objective and the coordinator
/// keeps their histories consistent.
pub struct Worker {
    conn: Conn,
}

impl Worker {
    /// Connect to the coordinator at `addr` for the study `study_name` over
    /// plain TCP. If the coordinator requires a token, call
    /// [`Worker::authenticate`] before [`Worker::optimize`].
    pub fn connect(addr: impl ToSocketAddrs, study_name: impl Into<String>) -> io::Result<Worker> {
        let tcp = TcpStream::connect(addr)?;
        Ok(Worker::from_stream(Box::new(tcp), study_name.into()))
    }

    /// Connect over TLS, trusting the given DER root certificate(s) and checking
    /// the coordinator's certificate against `server_name`. Requires the `tls`
    /// feature.
    #[cfg(feature = "tls")]
    pub fn connect_tls(
        addr: impl ToSocketAddrs,
        study_name: impl Into<String>,
        server_name: &str,
        root_cert_der: Vec<Vec<u8>>,
    ) -> io::Result<Worker> {
        let config = crate::transport::tls::client_config(root_cert_der)?;
        let tcp = TcpStream::connect(addr)?;
        let tls = crate::transport::tls::connect(config, server_name, tcp)?;
        Ok(Worker::from_stream(Box::new(tls), study_name.into()))
    }

    fn from_stream(stream: Stream, study: String) -> Worker {
        Worker {
            conn: Conn {
                stream: BufReader::new(stream),
                study,
            },
        }
    }

    /// Present a shared-secret token to a coordinator that requires one. Must be
    /// called before any trials are pulled; returns an error if the token is
    /// rejected.
    pub fn authenticate(&mut self, token: impl Into<String>) -> io::Result<()> {
        self.conn.authenticate(&token.into())
    }

    /// Pull and run up to `n_trials` trials from the coordinator.
    ///
    /// The objective is the ordinary `hyperopt` objective, written against
    /// [`RemoteTrial`] (or, generically, `&mut impl Suggest`). A trial whose
    /// objective panics or returns [`ObjectiveError::Failed`] is reported
    /// `Failed` and the worker continues; [`ObjectiveError::Pruned`] reports
    /// `Pruned`. A genuine network failure stops the worker and is returned.
    pub fn optimize<F>(&mut self, mut objective: F, n_trials: usize) -> io::Result<()>
    where
        F: FnMut(&mut RemoteTrial) -> ObjectiveResult,
    {
        for _ in 0..n_trials {
            let number = self.conn.new_trial()?;

            let mut trial = RemoteTrial {
                conn: &mut self.conn,
                number,
                net_error: None,
            };
            let result = catch_unwind(AssertUnwindSafe(|| objective(&mut trial)));
            let net_error = trial.net_error.take();
            drop(trial); // release the borrow on self.conn

            // A network error during the trial means the connection is unusable;
            // surface it rather than trying to report an outcome over it.
            if let Some(e) = net_error {
                return Err(e);
            }

            let outcome = match result {
                Ok(Ok(value)) => Outcome::Complete(value),
                Ok(Err(ObjectiveError::Pruned)) => Outcome::Pruned,
                Ok(Err(ObjectiveError::Failed(_))) => Outcome::Failed,
                Err(_panic) => Outcome::Failed,
            };
            self.conn.finish(number, outcome)?;
        }
        Ok(())
    }

    /// Query the study's best objective value so far.
    pub fn best_value(&mut self) -> io::Result<Option<f64>> {
        self.conn.best_value()
    }
}

/// A trial being evaluated on a worker. Its `suggest_*` / `report` /
/// `should_prune` calls transparently round-trip to the coordinator, so the
/// same objective body works locally (`&mut TrialContext`) or distributed
/// (`&mut RemoteTrial`) — write it against `&mut impl Suggest` to share one
/// closure across both.
pub struct RemoteTrial<'a> {
    conn: &'a mut Conn,
    number: usize,
    /// First network error seen; once set, calls short-circuit to defaults and
    /// [`Worker::optimize`] aborts the run after the objective returns.
    net_error: Option<io::Error>,
}

impl RemoteTrial<'_> {
    /// This trial's coordinator-assigned number.
    pub fn number(&self) -> usize {
        self.number
    }

    fn ask(&mut self, name: &str, distribution: Distribution) -> Value {
        if self.net_error.is_some() {
            return default_value(&distribution);
        }
        match self.conn.suggest(self.number, name, distribution.clone()) {
            Ok(v) => v,
            Err(e) => {
                self.net_error = Some(e);
                default_value(&distribution)
            }
        }
    }

    /// Suggest a continuous value uniformly over `[low, high]`.
    pub fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> f64 {
        coerce_float(&self.ask(name, Distribution::Uniform { low, high }))
    }

    /// Suggest a continuous value log-uniformly over `[low, high]`.
    pub fn suggest_loguniform(&mut self, name: &str, low: f64, high: f64) -> f64 {
        coerce_float(&self.ask(name, Distribution::LogUniform { low, high }))
    }

    /// Suggest an integer uniformly over the inclusive range `[low, high]`.
    pub fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> i64 {
        match self.ask(name, Distribution::IntUniform { low, high }) {
            Value::Int(x) => x,
            Value::Float(x) => x.round() as i64,
            Value::Categorical(_) => low,
        }
    }

    /// Suggest one of `choices`, returning the chosen label.
    pub fn suggest_categorical(&mut self, name: &str, choices: &[&str]) -> String {
        let dist = Distribution::Categorical {
            choices: choices.iter().map(|s| s.to_string()).collect(),
        };
        match self.ask(name, dist) {
            Value::Categorical(s) => s,
            Value::Int(i) => choices.get(i.max(0) as usize).map(|s| s.to_string()).unwrap_or_default(),
            Value::Float(f) => choices.get(f as usize).map(|s| s.to_string()).unwrap_or_default(),
        }
    }

    /// Report an intermediate objective value at `step`, for pruners.
    pub fn report(&mut self, step: usize, value: f64) {
        if self.net_error.is_some() {
            return;
        }
        if let Err(e) = self.conn.report(self.number, step, value) {
            self.net_error = Some(e);
        }
    }

    /// Ask the coordinator's pruner whether this trial should stop early.
    pub fn should_prune(&mut self) -> bool {
        if self.net_error.is_some() {
            return false;
        }
        match self.conn.should_prune(self.number) {
            Ok(prune) => prune,
            Err(e) => {
                self.net_error = Some(e);
                false
            }
        }
    }
}

impl Suggest for RemoteTrial<'_> {
    fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> f64 {
        RemoteTrial::suggest_float(self, name, low, high)
    }
    fn suggest_loguniform(&mut self, name: &str, low: f64, high: f64) -> f64 {
        RemoteTrial::suggest_loguniform(self, name, low, high)
    }
    fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> i64 {
        RemoteTrial::suggest_int(self, name, low, high)
    }
    fn suggest_categorical(&mut self, name: &str, choices: &[&str]) -> String {
        RemoteTrial::suggest_categorical(self, name, choices)
    }
    fn report(&mut self, step: usize, value: f64) {
        RemoteTrial::report(self, step, value)
    }
    fn should_prune(&mut self) -> bool {
        RemoteTrial::should_prune(self)
    }
}

/// One connection to the coordinator, with request/response framing. A single
/// stream carries both directions (plain TCP or TLS), read through the
/// `BufReader` and written through its `get_mut`.
struct Conn {
    stream: BufReader<Stream>,
    study: String,
}

impl Conn {
    /// Send one request and read exactly one response line.
    fn call(&mut self, request: &Request) -> io::Result<Response> {
        let mut line = serde_json::to_string(request).map_err(invalid_data)?;
        line.push('\n');
        self.stream.get_mut().write_all(line.as_bytes())?;
        self.stream.get_mut().flush()?;

        let mut buf = String::new();
        let n = self.stream.read_line(&mut buf)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "coordinator closed the connection",
            ));
        }
        serde_json::from_str::<Response>(buf.trim()).map_err(invalid_data)
    }

    fn authenticate(&mut self, token: &str) -> io::Result<()> {
        match self.call(&Request::Auth { token: token.to_string() })? {
            Response::Ack => Ok(()),
            other => Err(unexpected(other)),
        }
    }

    fn new_trial(&mut self) -> io::Result<usize> {
        match self.call(&Request::NewTrial { study: self.study.clone() })? {
            Response::Trial { number } => Ok(number),
            other => Err(unexpected(other)),
        }
    }

    fn suggest(&mut self, number: usize, name: &str, distribution: Distribution) -> io::Result<Value> {
        let req = Request::Suggest {
            study: self.study.clone(),
            number,
            name: name.to_string(),
            distribution,
        };
        match self.call(&req)? {
            Response::Value(v) => Ok(v),
            other => Err(unexpected(other)),
        }
    }

    fn report(&mut self, number: usize, step: usize, value: f64) -> io::Result<()> {
        let req = Request::Report { study: self.study.clone(), number, step, value };
        match self.call(&req)? {
            Response::Ack => Ok(()),
            other => Err(unexpected(other)),
        }
    }

    fn should_prune(&mut self, number: usize) -> io::Result<bool> {
        let req = Request::ShouldPrune { study: self.study.clone(), number };
        match self.call(&req)? {
            Response::Prune(p) => Ok(p),
            other => Err(unexpected(other)),
        }
    }

    fn finish(&mut self, number: usize, outcome: Outcome) -> io::Result<()> {
        let req = Request::Finish { study: self.study.clone(), number, outcome };
        match self.call(&req)? {
            Response::Ack => Ok(()),
            other => Err(unexpected(other)),
        }
    }

    fn best_value(&mut self) -> io::Result<Option<f64>> {
        match self.call(&Request::BestValue { study: self.study.clone() })? {
            Response::Best(v) => Ok(v),
            other => Err(unexpected(other)),
        }
    }
}

fn invalid_data(e: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

/// Turn an unexpected / error response into an `io::Error`.
fn unexpected(response: Response) -> io::Error {
    match response {
        Response::Error(msg) => io::Error::other(format!("coordinator error: {msg}")),
        other => io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unexpected response from coordinator: {other:?}"),
        ),
    }
}

fn coerce_float(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        Value::Int(x) => *x as f64,
        Value::Categorical(_) => f64::NAN,
    }
}

/// A within-range placeholder used only when a call has already failed and the
/// return value will be discarded once the worker aborts.
fn default_value(distribution: &Distribution) -> Value {
    match distribution {
        Distribution::Uniform { low, .. } | Distribution::LogUniform { low, .. } => {
            Value::Float(*low)
        }
        Distribution::IntUniform { low, .. } => Value::Int(*low),
        Distribution::Categorical { choices } => {
            Value::Categorical(choices.first().cloned().unwrap_or_default())
        }
    }
}
