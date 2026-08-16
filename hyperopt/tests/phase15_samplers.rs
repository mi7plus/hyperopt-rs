//! Phase 1.5 definition-of-done: all three samplers implement the same
//! `Sampler` API and are interchangeable through `Study`; and TPE converges to
//! a better best-value on average than random given enough trials.

use hyperopt::prelude::*;

/// 3D sphere centred at (2, -3, 1.5); minimum value 0.
fn sphere(trial: &mut TrialContext) -> ObjectiveResult {
    let x = trial.suggest_float("x", -10.0, 10.0);
    let y = trial.suggest_float("y", -10.0, 10.0);
    let z = trial.suggest_float("z", -10.0, 10.0);
    Ok((x - 2.0).powi(2) + (y + 3.0).powi(2) + (z - 1.5).powi(2))
}

fn best_with<S: Sampler + 'static>(sampler: S, n_trials: usize, seed: u64) -> f64 {
    let study = StudyBuilder::new(format!("cmp-{seed}"))
        .direction(Direction::Minimize)
        .sampler(sampler)
        .build()
        .unwrap();
    study.optimize(sphere, n_trials).unwrap();
    study.best_value().unwrap().unwrap()
}

#[test]
fn all_three_samplers_share_the_same_study_api() {
    // Random and TPE run on the continuous sphere via one API.
    let r = best_with(RandomSampler::seeded(1), 60, 1);
    let t = best_with(TpeSampler::seeded(1), 60, 2);
    assert!(r.is_finite() && t.is_finite());

    // Grid runs on the same API too (on a discrete grid over the same params).
    let grid = GridSampler::new()
        .add_float_grid("x", &[-4.0, 0.0, 2.0, 6.0])
        .add_float_grid("y", &[-6.0, -3.0, 0.0, 4.0])
        .add_float_grid("z", &[-2.0, 1.5, 5.0]);
    let n = grid.grid_size();
    let study = StudyBuilder::new("grid-api").sampler(grid).build().unwrap();
    study.optimize(sphere, n).unwrap();
    // The grid contains the exact optimum (2, -3, 1.5) => best value 0.
    assert_eq!(study.best_value().unwrap().unwrap(), 0.0);
}

#[test]
fn grid_sampler_enumerates_the_whole_grid_once() {
    let grid = GridSampler::new()
        .add_int_grid("a", &[1, 2, 3])
        .add_categorical_grid("b", &["p", "q"]);
    let size = grid.grid_size();
    assert_eq!(size, 6);

    let study = StudyBuilder::new("grid-enum").sampler(grid).build().unwrap();
    study
        .optimize(
            |trial| {
                let a = trial.suggest_int("a", 1, 3);
                let b = trial.suggest_categorical("b", &["p", "q"]);
                // Distinct value per grid point.
                Ok(a as f64 + if b == "q" { 0.5 } else { 0.0 })
            },
            size,
        )
        .unwrap();

    // Every one of the 6 grid combinations should appear exactly once.
    let trials = study.trials().unwrap();
    let mut combos: Vec<(i64, String)> = trials
        .iter()
        .map(|t| {
            (
                t.param_value("a").unwrap().as_int().unwrap(),
                t.param_value("b").unwrap().as_categorical().unwrap().to_string(),
            )
        })
        .collect();
    combos.sort();
    combos.dedup();
    assert_eq!(combos.len(), 6, "grid should cover all 6 unique combinations");
}

#[test]
fn tpe_beats_random_on_average() {
    // Average best-value across several seeds; TPE's adaptive search should
    // come out ahead of pure random given a decent budget.
    let seeds = [11u64, 22, 33, 44, 55, 66, 77, 88];
    let n_trials = 80;

    let mut random_sum = 0.0;
    let mut tpe_sum = 0.0;
    for &s in &seeds {
        random_sum += best_with(RandomSampler::seeded(s), n_trials, s);
        tpe_sum += best_with(TpeSampler::seeded(s), n_trials, s);
    }
    let random_mean = random_sum / seeds.len() as f64;
    let tpe_mean = tpe_sum / seeds.len() as f64;

    println!("mean best over {} seeds: random={random_mean:.4}, tpe={tpe_mean:.4}", seeds.len());
    assert!(
        tpe_mean < random_mean,
        "expected TPE to beat random on average: tpe={tpe_mean:.4} random={random_mean:.4}"
    );
}
