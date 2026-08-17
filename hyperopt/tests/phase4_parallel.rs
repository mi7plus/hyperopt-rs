//! Phase 4 definition-of-done: parallel execution produces valid results (no
//! data races, all trials recorded) and shows a real speedup over sequential
//! on an objective with an artificial per-trial cost. Only built with the
//! `parallel` feature: `cargo test -p hyperopt-rs --features parallel`.

#![cfg(feature = "parallel")]

use hyperopt_rs::prelude::*;
use std::time::{Duration, Instant};

/// Sphere with an artificial per-trial cost so the parallelism benefit is
/// measurable rather than dominated by per-trial overhead.
fn costly_sphere(trial: &mut TrialContext) -> ObjectiveResult {
    let x = trial.suggest_float("x", -10.0, 10.0);
    let y = trial.suggest_float("y", -10.0, 10.0);
    std::thread::sleep(Duration::from_millis(15));
    Ok((x - 2.0).powi(2) + (y + 3.0).powi(2))
}

#[test]
fn parallel_execution_is_race_free_and_complete() {
    let study = StudyBuilder::new("par")
        .direction(Direction::Minimize)
        .sampler(TpeSampler::seeded(5))
        .storage(InMemoryStorage::new())
        .build()
        .unwrap();

    let n = 64;
    study.optimize_parallel(costly_sphere, n, 8).unwrap();

    let trials = study.trials().unwrap();
    // Every trial was recorded exactly once, with a unique number.
    assert_eq!(trials.len(), n);
    let mut numbers: Vec<usize> = trials.iter().map(|t| t.number).collect();
    numbers.sort();
    numbers.dedup();
    assert_eq!(numbers.len(), n, "trial numbers must be unique (no lost updates)");

    // All completed with finite values, and a sensible best emerged.
    assert!(trials.iter().all(|t| t.state == TrialState::Complete));
    let best = study.best_value().unwrap().unwrap();
    assert!(best.is_finite() && best < 20.0, "best={best}");
}

#[test]
fn parallel_is_faster_than_sequential() {
    let n = 48;

    let seq = StudyBuilder::new("seq-bench")
        .sampler(RandomSampler::seeded(1))
        .storage(InMemoryStorage::new())
        .build()
        .unwrap();
    let t0 = Instant::now();
    seq.optimize(costly_sphere, n).unwrap();
    let seq_time = t0.elapsed();

    let par = StudyBuilder::new("par-bench")
        .sampler(RandomSampler::seeded(1))
        .storage(InMemoryStorage::new())
        .build()
        .unwrap();
    let t1 = Instant::now();
    par.optimize_parallel(costly_sphere, n, 4).unwrap();
    let par_time = t1.elapsed();

    let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64();
    println!(
        "sequential: {:?}, parallel(4): {:?}, speedup: {speedup:.2}x",
        seq_time, par_time
    );

    // 4 workers on a sleep-bound objective should be comfortably faster.
    // Assert a modest 2x to stay robust on shared CI machines.
    assert!(
        speedup > 2.0,
        "expected >2x speedup from 4 workers, got {speedup:.2}x"
    );
    assert_eq!(par.trials().unwrap().len(), n);
}
