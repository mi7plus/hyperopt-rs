//! Phase 1 definition-of-done: a user writes an objective with `suggest_*`
//! calls, runs `optimize` with `RandomSampler`, and gets a sensible best trial
//! on a toy problem (a 2D quadratic bowl with a known minimum).

use hyperopt_rs::prelude::*;

/// f(x, y) = (x - 2)^2 + (y + 3)^2, minimized at (2, -3) with value 0.
fn quadratic_bowl(trial: &mut TrialContext) -> ObjectiveResult {
    let x = trial.suggest_float("x", -10.0, 10.0);
    let y = trial.suggest_float("y", -10.0, 10.0);
    Ok((x - 2.0).powi(2) + (y + 3.0).powi(2))
}

#[test]
fn random_sampler_converges_on_quadratic_bowl() {
    let study = StudyBuilder::new("bowl")
        .direction(Direction::Minimize)
        .sampler(RandomSampler::seeded(7))
        .build()
        .unwrap();

    study.optimize(quadratic_bowl, 500).unwrap();

    let best = study.best_trial().unwrap().expect("a best trial exists");
    let best_value = best.value.unwrap();

    // 500 random samples over a 20x20 box get comfortably close to the minimum.
    assert!(
        best_value < 1.0,
        "expected best value < 1.0 after 500 random trials, got {best_value}"
    );

    // The best params should be near (2, -3).
    let x = best.param_value("x").unwrap().as_float().unwrap();
    let y = best.param_value("y").unwrap().as_float().unwrap();
    assert!((x - 2.0).abs() < 1.0, "x={x} not near 2");
    assert!((y + 3.0).abs() < 1.0, "y={y} not near -3");

    // All 500 trials recorded and completed.
    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 500);
    assert!(trials.iter().all(|t| t.state == TrialState::Complete));
}

#[test]
fn maximize_direction_flips_the_search() {
    // Maximize -(x-1)^2 => best near x = 1, best value near 0 (from below).
    let study = StudyBuilder::new("max")
        .direction(Direction::Maximize)
        .sampler(RandomSampler::seeded(3))
        .build()
        .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0);
                Ok(-(x - 1.0).powi(2))
            },
            300,
        )
        .unwrap();

    let best = study.best_value().unwrap().unwrap();
    assert!(best > -0.2, "maximized value should be near 0, got {best}");
}

#[test]
fn failed_trials_do_not_abort_the_study() {
    let study = StudyBuilder::new("robust")
        .sampler(RandomSampler::seeded(1))
        .build()
        .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0);
                if trial.number() % 3 == 0 {
                    // Simulate a bad trial: panic on every third.
                    panic!("intentional failure");
                }
                Ok(x)
            },
            30,
        )
        .unwrap();

    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 30);
    let failed = trials.iter().filter(|t| t.state == TrialState::Failed).count();
    let complete = trials.iter().filter(|t| t.state == TrialState::Complete).count();
    assert_eq!(failed, 10, "every third trial should have failed");
    assert_eq!(complete, 20);
    // A completed best trial still exists despite the failures.
    assert!(study.best_trial().unwrap().is_some());
}
