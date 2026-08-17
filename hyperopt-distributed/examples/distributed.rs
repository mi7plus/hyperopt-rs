//! Self-contained distributed demo: start a coordinator and several workers in
//! one process (the workers would normally live on other machines) and watch
//! them cooperatively optimize one shared study.
//!
//! Run with: `cargo run -p hyperopt-distributed --example distributed`

use hyperopt_core::{Direction, Suggest};
use hyperopt_distributed::{Coordinator, RemoteTrial, Worker};
use hyperopt_pruners::NopPruner;
use hyperopt_samplers::TpeSampler;
use hyperopt_storage::InMemoryStorage;
use std::thread;

/// 3D sphere centred at (2, -3, 1.5); global minimum 0. Written against
/// `&mut impl Suggest`, so it is byte-for-byte the same objective you'd hand to
/// a local `Study::optimize`.
fn sphere(trial: &mut impl Suggest) -> hyperopt_core::ObjectiveResult {
    let x = trial.suggest_float("x", -10.0, 10.0);
    let y = trial.suggest_float("y", -10.0, 10.0);
    let z = trial.suggest_float("z", -10.0, 10.0);
    Ok((x - 2.0).powi(2) + (y + 3.0).powi(2) + (z - 1.5).powi(2))
}

fn main() -> std::io::Result<()> {
    let study = "distributed-demo";

    // 1. Stand up the coordinator on an ephemeral local port.
    let coord = Coordinator::new(
        study,
        Direction::Minimize,
        Box::new(TpeSampler::seeded(42)),
        Box::new(NopPruner::new()),
        Box::new(InMemoryStorage::new()),
    )
    .expect("build coordinator");
    let listening = coord.listen("127.0.0.1:0")?;
    let addr = listening.local_addr()?;
    let handle = listening.coordinator();
    thread::spawn(move || {
        let _ = listening.run();
    });
    println!("coordinator serving study {study:?} on {addr}");

    // 2. Fan out workers (each of these would be a separate machine in a real
    //    deployment, all pointed at the coordinator's address).
    let n_workers = 4;
    let per_worker = 25;
    let workers: Vec<_> = (0..n_workers)
        .map(|w| {
            thread::spawn(move || {
                let mut worker = Worker::connect(addr, study).expect("connect");
                worker
                    .optimize(|t: &mut RemoteTrial| sphere(t), per_worker)
                    .expect("optimize");
                println!("  worker {w} finished {per_worker} trials");
            })
        })
        .collect();
    for w in workers {
        w.join().expect("worker thread");
    }

    // 3. Inspect the shared result.
    let trials = handle.trials().expect("trials");
    let best = handle.best_value().expect("best").expect("some best");
    println!(
        "ran {} trials across {n_workers} workers; best value = {best:.6}",
        trials.len()
    );
    Ok(())
}
