//! Phase 2 definition-of-done (part 2): pruning demonstrably saves total
//! objective-function evaluations on a synthetic multi-step benchmark — the
//! saving is measured (step count with pruning on vs. off), not assumed.
//!
//! Both runs use the same seeded `RandomSampler`, so the sequence of `x` values
//! (and thus the trial trajectories) is identical between runs; the only
//! difference is whether the pruner stops bad trajectories early. That isolates
//! the effect being measured.

use hyperopt_rs::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

const N_TRIALS: usize = 40;
const N_STEPS: usize = 30;

/// Run the benchmark objective and return (total steps evaluated, best value).
fn run(pruner_on: bool) -> (usize, f64) {
    let steps = AtomicUsize::new(0);

    let mut builder = StudyBuilder::new(if pruner_on { "pruned" } else { "full" })
        .direction(Direction::Minimize)
        .sampler(RandomSampler::seeded(2024));
    if pruner_on {
        builder = builder.pruner(MedianPruner::new().n_startup_trials(5).n_warmup_steps(2));
    }
    let study = builder.build().unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0);
                // Final plateau loss depends on how good x is; every trajectory
                // starts 10 above its plateau and decays toward it.
                let plateau = (x - 2.0).powi(2);
                let start = plateau + 10.0;

                let mut last = start;
                for s in 0..N_STEPS {
                    steps.fetch_add(1, Ordering::Relaxed);
                    let loss = plateau + (start - plateau) * (-(s as f64) / 8.0).exp();
                    last = loss;
                    trial.report(s, loss);
                    if trial.should_prune() {
                        return Err(ObjectiveError::pruned());
                    }
                }
                Ok(last)
            },
            N_TRIALS,
        )
        .unwrap();

    (steps.load(Ordering::Relaxed), study.best_value().unwrap().unwrap())
}

#[test]
fn median_pruning_reduces_total_evaluations() {
    let (full_steps, full_best) = run(false);
    let (pruned_steps, pruned_best) = run(true);

    println!(
        "steps: full={full_steps} pruned={pruned_steps} \
         ({:.0}% saved); best: full={full_best:.4} pruned={pruned_best:.4}",
        100.0 * (1.0 - pruned_steps as f64 / full_steps as f64)
    );

    // Without pruning, every trial runs all steps.
    assert_eq!(full_steps, N_TRIALS * N_STEPS);

    // Pruning should cut a meaningful chunk of evaluations.
    assert!(
        pruned_steps < full_steps,
        "pruning did not save evaluations ({pruned_steps} vs {full_steps})"
    );
    assert!(
        (pruned_steps as f64) < 0.85 * full_steps as f64,
        "expected >15% evaluation savings, got {pruned_steps}/{full_steps}"
    );

    // And it should not meaningfully hurt the best value found (same sampler
    // sequence; good trials are never the ones pruned).
    assert!(
        pruned_best <= full_best + 1e-9,
        "pruning hurt the best value: pruned={pruned_best} full={full_best}"
    );
}
