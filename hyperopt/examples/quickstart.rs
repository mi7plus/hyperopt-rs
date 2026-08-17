//! End-to-end quickstart for `hyperopt-rs`.
//!
//! Run with:  `cargo run -p hyperopt-rs --example quickstart`
//!
//! Demonstrates define-by-run suggestions, a conditional search space, an
//! iterative objective with median pruning, and reading back the best trial.

use hyperopt_rs::prelude::*;

fn main() -> Result<(), HyperoptError> {
    let study = StudyBuilder::new("quickstart")
        .direction(Direction::Minimize)
        .sampler(TpeSampler::seeded(42))
        .pruner(MedianPruner::new().n_startup_trials(8).n_warmup_steps(2))
        .build()?;

    study.optimize(
        |trial| {
            // Define-by-run: a conditional search space. The model choice
            // decides which further hyperparameters exist for this trial.
            let model = trial.suggest_categorical("model", &["linear", "poly"]);
            let lr = trial.suggest_loguniform("lr", 1e-4, 1e-1);

            let degree = if model == "poly" {
                trial.suggest_int("degree", 2, 5)
            } else {
                1
            };

            // A synthetic "training loop" that reports intermediate loss so the
            // pruner can stop hopeless trials early.
            let target_lr = 1e-2_f64;
            let plateau = (lr.ln() - target_lr.ln()).powi(2) + (degree as f64 - 3.0).abs() * 0.5;
            let start = plateau + 5.0;

            let mut last = start;
            for epoch in 0..25 {
                let loss = plateau + (start - plateau) * (-(epoch as f64) / 6.0).exp();
                last = loss;
                trial.report(epoch, loss);
                if trial.should_prune() {
                    return Err(ObjectiveError::pruned());
                }
            }
            Ok(last)
        },
        120,
    )?;

    let trials = study.trials()?;
    let completed = trials.iter().filter(|t| t.state == TrialState::Complete).count();
    let pruned = trials.iter().filter(|t| t.state == TrialState::Pruned).count();

    println!("ran {} trials: {completed} complete, {pruned} pruned", trials.len());

    if let Some(best) = study.best_trial()? {
        println!("best value: {:.5}", best.value.unwrap());
        for p in &best.params {
            println!("  {} = {}", p.name, p.value);
        }
    }
    Ok(())
}
