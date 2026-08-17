//! CMA-ES sampler: it plugs into the same `Study` API as the other samplers,
//! converges on a smooth continuous objective faster than random search, is
//! reproducible under a fixed seed, and coexists with categorical parameters
//! (which fall back to independent sampling).

use hyperopt::prelude::*;

/// 3D sphere centred at (2, -3, 1.5); minimum value 0.
fn sphere(trial: &mut TrialContext) -> ObjectiveResult {
    let x = trial.suggest_float("x", -10.0, 10.0);
    let y = trial.suggest_float("y", -10.0, 10.0);
    let z = trial.suggest_float("z", -10.0, 10.0);
    Ok((x - 2.0).powi(2) + (y + 3.0).powi(2) + (z - 1.5).powi(2))
}

fn best_with<S: Sampler + 'static>(sampler: S, n_trials: usize, name: &str) -> f64 {
    let study = StudyBuilder::new(name)
        .direction(Direction::Minimize)
        .sampler(sampler)
        .build()
        .unwrap();
    study.optimize(sphere, n_trials).unwrap();
    study.best_value().unwrap().unwrap()
}

#[test]
fn cmaes_shares_the_same_study_api() {
    let best = best_with(CmaEsSampler::seeded(1), 80, "cmaes-api");
    assert!(best.is_finite());
    // 80 trials of CMA-ES on a smooth 3D bowl should get comfortably close.
    assert!(best < 1.0, "CMA-ES should approach the optimum, got {best}");
}

#[test]
fn cmaes_beats_random_on_average() {
    let seeds = [11u64, 22, 33, 44, 55, 66, 77, 88];
    let n_trials = 80;

    let mut random_sum = 0.0;
    let mut cmaes_sum = 0.0;
    for &s in &seeds {
        random_sum += best_with(RandomSampler::seeded(s), n_trials, &format!("rnd-{s}"));
        cmaes_sum += best_with(CmaEsSampler::seeded(s), n_trials, &format!("cma-{s}"));
    }
    let random_mean = random_sum / seeds.len() as f64;
    let cmaes_mean = cmaes_sum / seeds.len() as f64;

    println!("mean best over {} seeds: random={random_mean:.4}, cmaes={cmaes_mean:.4}", seeds.len());
    assert!(
        cmaes_mean < random_mean,
        "expected CMA-ES to beat random on average: cmaes={cmaes_mean:.4} random={random_mean:.4}"
    );
}

#[test]
fn cmaes_is_reproducible_under_a_fixed_seed() {
    let a = best_with(CmaEsSampler::seeded(7), 60, "cma-repro-a");
    let b = best_with(CmaEsSampler::seeded(7), 60, "cma-repro-b");
    assert_eq!(a, b, "same seed must give the same best value");
}

#[test]
fn cmaes_reflection_reaches_a_bound_adjacent_optimum() {
    // The optimum sits exactly on the box corner: x = +10 (upper bound),
    // y = -10 (lower bound). Reflection (the default) keeps sampling smooth
    // across those bounds instead of piling mass on them.
    let study = StudyBuilder::new("cma-bounds")
        .direction(Direction::Minimize)
        .sampler(CmaEsSampler::seeded(5))
        .build()
        .unwrap();
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0);
                let y = trial.suggest_float("y", -10.0, 10.0);
                Ok((x - 10.0).powi(2) + (y + 10.0).powi(2))
            },
            120,
        )
        .unwrap();
    let best = study.best_value().unwrap().unwrap();
    assert!(best < 0.5, "reflection should reach the boundary optimum, got {best}");
}

#[test]
fn cmaes_clamp_mode_still_optimizes() {
    // The alternative bound handling is selectable and still converges.
    let study = StudyBuilder::new("cma-clamp")
        .direction(Direction::Minimize)
        .sampler(CmaEsSampler::seeded(9).bound_handling(BoundHandling::Clamp))
        .build()
        .unwrap();
    study.optimize(sphere, 80).unwrap();
    assert!(study.best_value().unwrap().unwrap() < 1.0);
}

#[test]
fn cmaes_coexists_with_categorical_parameters() {
    // `mode` is categorical (outside CMA-ES's space; sampled independently),
    // while `x` is continuous and CMA-ES-optimized. The objective is minimized
    // at mode == "b" and x == 2.
    let study = StudyBuilder::new("cma-mixed")
        .direction(Direction::Minimize)
        .sampler(CmaEsSampler::seeded(3))
        .build()
        .unwrap();
    study
        .optimize(
            |trial| {
                let mode = trial.suggest_categorical("mode", &["a", "b", "c"]);
                let x = trial.suggest_float("x", -10.0, 10.0);
                let penalty = if mode == "b" { 0.0 } else { 5.0 };
                Ok((x - 2.0).powi(2) + penalty)
            },
            120,
        )
        .unwrap();

    let best = study.best_trial().unwrap().unwrap();
    // The continuous part should be well-optimized regardless of the category.
    let x = best.param_value("x").unwrap().as_float().unwrap();
    assert!((x - 2.0).abs() < 1.5, "CMA-ES should tune x near 2, got {x}");
    assert!(study.best_value().unwrap().unwrap() < 1.0);
}
