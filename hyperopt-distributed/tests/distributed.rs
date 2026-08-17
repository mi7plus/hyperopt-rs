//! Distributed coordinator/worker: many workers on one coordinator share a
//! single authoritative history, every trial gets a unique number, the sampler
//! still converges, and the same objective body works locally or remotely.

use hyperopt_core::{Direction, Suggest, TrialState};
use hyperopt_distributed::{Coordinator, RemoteTrial, Worker};
use hyperopt_pruners::{MedianPruner, NopPruner};
use hyperopt_samplers::TpeSampler;
use hyperopt_storage::InMemoryStorage;
use std::thread;

/// A 2D bowl minimized at (2, -3). Written against `&mut impl Suggest` so the
/// exact same closure runs on a local study or a remote worker.
fn bowl(trial: &mut impl Suggest) -> hyperopt_core::ObjectiveResult {
    let x = trial.suggest_float("x", -10.0, 10.0);
    let y = trial.suggest_float("y", -10.0, 10.0);
    Ok((x - 2.0).powi(2) + (y + 3.0).powi(2))
}

fn build_coordinator(name: &str, pruner: Box<dyn hyperopt_core::Pruner>) -> Coordinator {
    Coordinator::new(
        name,
        Direction::Minimize,
        Box::new(TpeSampler::seeded(42)),
        pruner,
        Box::new(InMemoryStorage::new()),
    )
    .expect("build coordinator")
}

/// Bind a coordinator on an ephemeral port and serve it on a background thread.
fn serve(coord: Coordinator) -> (std::net::SocketAddr, std::sync::Arc<Coordinator>) {
    let listening = coord.listen("127.0.0.1:0").expect("bind");
    let addr = listening.local_addr().expect("addr");
    let handle = listening.coordinator();
    thread::spawn(move || {
        let _ = listening.run();
    });
    (addr, handle)
}

fn start_coordinator(
    name: &str,
    pruner: Box<dyn hyperopt_core::Pruner>,
) -> (std::net::SocketAddr, std::sync::Arc<Coordinator>) {
    serve(build_coordinator(name, pruner))
}

#[test]
fn many_workers_share_one_authoritative_history() {
    let (addr, coord) = start_coordinator("dist-share", Box::new(NopPruner::new()));

    // Four workers, 20 trials each => 80 trials total, run concurrently.
    let n_workers = 4;
    let per_worker = 20;
    let handles: Vec<_> = (0..n_workers)
        .map(|_| {
            let study = coord.name().to_string();
            thread::spawn(move || {
                let mut worker = Worker::connect(addr, study).expect("connect");
                worker.optimize(|t: &mut RemoteTrial| bowl(t), per_worker).expect("optimize");
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker thread");
    }

    let trials = coord.trials().expect("trials");
    assert_eq!(trials.len(), n_workers * per_worker, "every trial was recorded once");

    // Trial numbers are unique and contiguous 0..N — the coordinator assigned
    // them atomically despite concurrent workers (the race shared storage alone
    // would lose).
    let mut numbers: Vec<usize> = trials.iter().map(|t| t.number).collect();
    numbers.sort_unstable();
    numbers.dedup();
    assert_eq!(numbers.len(), n_workers * per_worker, "trial numbers are unique");
    assert_eq!(*numbers.last().unwrap(), n_workers * per_worker - 1, "numbers are contiguous");

    // All completed, and TPE over a shared history actually converged.
    assert!(trials.iter().all(|t| t.state == TrialState::Complete));
    let best = coord.best_value().expect("best").expect("some best");
    assert!(best < 1.0, "distributed TPE should converge on the bowl, got {best}");
}

#[test]
fn worker_reports_and_pruning_round_trip() {
    // A coordinator with a MedianPruner. The objective reports a decreasing
    // series and asks to be pruned — exercising Report + ShouldPrune RPCs.
    let (addr, coord) =
        start_coordinator("dist-prune", Box::new(MedianPruner::new().n_startup_trials(3)));

    let study = coord.name().to_string();
    let mut worker = Worker::connect(addr, study).expect("connect");
    let mut pruned_any = false;
    worker
        .optimize(
            |t: &mut RemoteTrial| {
                let x = t.suggest_float("x", -5.0, 5.0);
                for step in 0..10 {
                    // Deliberately bad trials plateau high so the pruner cuts them.
                    t.report(step, x.abs() + step as f64);
                    if t.should_prune() {
                        return Err(hyperopt_core::ObjectiveError::pruned());
                    }
                }
                Ok(x.abs())
            },
            30,
        )
        .expect("optimize");

    let trials = coord.trials().expect("trials");
    assert_eq!(trials.len(), 30);
    for t in &trials {
        if t.state == TrialState::Pruned {
            pruned_any = true;
        }
        // Every trial reached a terminal state.
        assert_ne!(t.state, TrialState::Running);
    }
    assert!(pruned_any, "the median pruner should have stopped at least one trial");
}

#[test]
fn token_auth_gates_access() {
    let (addr, coord) = serve(build_coordinator("dist-auth", Box::new(NopPruner::new())).require_token("s3cret"));

    // Without authenticating, the first real request is refused.
    {
        let mut worker = Worker::connect(addr, "dist-auth").expect("connect");
        let err = worker
            .optimize(|t: &mut RemoteTrial| Ok(t.suggest_float("x", 0.0, 1.0)), 1)
            .unwrap_err();
        assert!(
            err.to_string().contains("authentication required"),
            "unauthenticated worker should be refused, got: {err}"
        );
    }

    // A wrong token is rejected at the handshake.
    {
        let mut worker = Worker::connect(addr, "dist-auth").expect("connect");
        let err = worker.authenticate("wrong").unwrap_err();
        assert!(
            err.to_string().contains("authentication failed"),
            "wrong token should fail, got: {err}"
        );
    }

    // The correct token unlocks normal operation.
    {
        let mut worker = Worker::connect(addr, "dist-auth").expect("connect");
        worker.authenticate("s3cret").expect("auth");
        worker
            .optimize(|t: &mut RemoteTrial| bowl(t), 10)
            .expect("optimize after auth");
    }

    assert_eq!(coord.trials().unwrap().len(), 10, "only the authenticated worker's trials landed");
}

#[cfg(feature = "tls")]
#[test]
fn tls_round_trip_optimizes_over_an_encrypted_channel() {
    // Self-signed cert for "localhost"; the client trusts exactly this cert.
    let issued = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let cert_der = issued.cert.der().as_ref().to_vec();
    let key_der = issued.key_pair.serialize_der();

    let coord = build_coordinator("dist-tls", Box::new(NopPruner::new()));
    let listening = coord
        .listen_tls("127.0.0.1:0", vec![cert_der.clone()], key_der)
        .expect("bind tls");
    let addr = listening.local_addr().expect("addr");
    let handle = listening.coordinator();
    thread::spawn(move || {
        let _ = listening.run();
    });

    let mut worker =
        Worker::connect_tls(addr, "dist-tls", "localhost", vec![cert_der]).expect("connect tls");
    worker.optimize(|t: &mut RemoteTrial| bowl(t), 20).expect("optimize over tls");

    assert_eq!(handle.trials().unwrap().len(), 20);
    assert!(handle.best_value().unwrap().unwrap().is_finite());
}

#[test]
fn wrong_study_name_is_rejected() {
    let (addr, coord) = start_coordinator("dist-real", Box::new(NopPruner::new()));
    // Connect asking for a study the coordinator does not serve.
    let mut worker = Worker::connect(addr, "some-other-study").expect("connect");
    let err = worker
        .optimize(|t: &mut RemoteTrial| Ok(t.suggest_float("x", 0.0, 1.0)), 1)
        .unwrap_err();
    assert!(
        err.to_string().contains("unknown study"),
        "expected an unknown-study error, got: {err}"
    );
    // The real study got no trials from the misdirected worker.
    assert_eq!(coord.trials().unwrap().len(), 0);
}
