//! Phase 5 definition-of-done: optimization-history and parameter-importance
//! outputs render correctly on a completed study, and the importance proxy
//! ranks a genuinely-important parameter above an irrelevant one.

use hyperopt_core::{Direction, Distribution, Trial, TrialState, Value};
use hyperopt_viz::{
    best_so_far, fanova_importance, fanova_interactions, optimization_history_svg,
    parameter_importance,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Build completed trials whose objective depends only on `x` (important) and
/// not at all on `noise` (irrelevant): value = (x - 2)^2.
fn synthetic_trials(n: usize) -> Vec<Trial> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|i| {
            let x: f64 = rng.random_range(-5.0..5.0);
            let noise: f64 = rng.random_range(-5.0..5.0);
            let value = (x - 2.0).powi(2);
            let mut t = Trial::new(i);
            t.record("x", Distribution::Uniform { low: -5.0, high: 5.0 }, Value::Float(x));
            t.record(
                "noise",
                Distribution::Uniform { low: -5.0, high: 5.0 },
                Value::Float(noise),
            );
            t.value = Some(value);
            t.state = TrialState::Complete;
            t
        })
        .collect()
}

#[test]
fn importance_ranks_the_influential_parameter_first() {
    let trials = synthetic_trials(120);
    let imps = parameter_importance(&trials);

    assert_eq!(imps.len(), 2);
    // The important parameter ranks first and explains far more variance.
    assert_eq!(imps[0].name, "x", "x should be ranked most important");

    let x = imps.iter().find(|i| i.name == "x").unwrap();
    let noise = imps.iter().find(|i| i.name == "noise").unwrap();
    assert!(
        x.variance_explained > 0.3,
        "x should explain real variance, got {}",
        x.variance_explained
    );
    assert!(
        x.variance_explained > noise.variance_explained * 2.0,
        "x ({}) should dominate noise ({})",
        x.variance_explained,
        noise.variance_explained
    );

    // Normalized importances sum to ~1.
    let sum: f64 = imps.iter().map(|i| i.normalized).sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

#[test]
fn fanova_ranks_the_influential_parameter_first() {
    let trials = synthetic_trials(120);
    let imps = fanova_importance(&trials);

    assert_eq!(imps.len(), 2);
    assert_eq!(imps[0].name, "x", "x should be ranked most important");

    let x = imps.iter().find(|i| i.name == "x").unwrap();
    let noise = imps.iter().find(|i| i.name == "noise").unwrap();
    assert!(
        x.variance_explained > 0.3,
        "x should explain real variance, got {}",
        x.variance_explained
    );
    assert!(
        x.variance_explained > noise.variance_explained * 3.0,
        "x ({}) should dominate noise ({})",
        x.variance_explained,
        noise.variance_explained
    );

    // Main effects need not sum to 1 (interactions absorb the rest), but the
    // normalized shares always do when any variance is explained.
    let sum: f64 = imps.iter().map(|i| i.normalized).sum();
    assert!((sum - 1.0).abs() < 1e-9, "normalized shares sum to 1, got {sum}");
}

/// fANOVA marginalizes over the other parameter, so it should not be fooled
/// into crediting an irrelevant parameter even when the objective is a clean
/// additive function of the important one.
#[test]
fn fanova_additive_objective_isolates_the_real_driver() {
    let mut rng = StdRng::seed_from_u64(7);
    let trials: Vec<Trial> = (0..150)
        .map(|i| {
            let a: f64 = rng.random_range(0.0..10.0);
            let b: f64 = rng.random_range(0.0..10.0);
            // Depends on `a` linearly; `b` is pure noise in the objective.
            let value = 3.0 * a + rng.random_range(-0.01..0.01);
            let mut t = Trial::new(i);
            t.record("a", Distribution::Uniform { low: 0.0, high: 10.0 }, Value::Float(a));
            t.record("b", Distribution::Uniform { low: 0.0, high: 10.0 }, Value::Float(b));
            t.value = Some(value);
            t.state = TrialState::Complete;
            t
        })
        .collect();

    let imps = fanova_importance(&trials);
    let a = imps.iter().find(|i| i.name == "a").unwrap();
    let b = imps.iter().find(|i| i.name == "b").unwrap();
    assert!(a.normalized > 0.9, "a should carry ~all importance, got {}", a.normalized);
    assert!(b.normalized < 0.1, "b is irrelevant, got {}", b.normalized);
}

/// A categorical whose level determines the objective should dominate a
/// numeric distractor.
#[test]
fn fanova_handles_categorical_parameters() {
    let mut rng = StdRng::seed_from_u64(11);
    let levels = ["low", "mid", "high"];
    let trials: Vec<Trial> = (0..150)
        .map(|i| {
            let k = rng.random_range(0..levels.len());
            let noise: f64 = rng.random_range(-5.0..5.0);
            let value = (k as f64) * 10.0; // objective set purely by the category
            let mut t = Trial::new(i);
            t.record(
                "mode",
                Distribution::Categorical {
                    choices: levels.iter().map(|s| s.to_string()).collect(),
                },
                Value::Categorical(levels[k].to_string()),
            );
            t.record("noise", Distribution::Uniform { low: -5.0, high: 5.0 }, Value::Float(noise));
            t.value = Some(value);
            t.state = TrialState::Complete;
            t
        })
        .collect();

    let imps = fanova_importance(&trials);
    assert_eq!(imps[0].name, "mode", "the deciding category should rank first");
    let mode = imps.iter().find(|i| i.name == "mode").unwrap();
    assert!(mode.normalized > 0.8, "mode should carry most importance, got {}", mode.normalized);
}

/// Build trials whose objective is a *pure interaction* `y = x1 * x2` over a
/// symmetric domain (so each parameter's main effect is ~0) plus an irrelevant
/// `z`. Only the (x1, x2) pair should carry importance.
fn interaction_trials(n: usize) -> Vec<Trial> {
    let mut rng = StdRng::seed_from_u64(2024);
    (0..n)
        .map(|i| {
            let x1: f64 = rng.random_range(-5.0..5.0);
            let x2: f64 = rng.random_range(-5.0..5.0);
            let z: f64 = rng.random_range(-5.0..5.0);
            let mut t = Trial::new(i);
            let dist = Distribution::Uniform { low: -5.0, high: 5.0 };
            t.record("x1", dist.clone(), Value::Float(x1));
            t.record("x2", dist.clone(), Value::Float(x2));
            t.record("z", dist, Value::Float(z));
            t.value = Some(x1 * x2);
            t.state = TrialState::Complete;
            t
        })
        .collect()
}

#[test]
fn fanova_interactions_detect_a_pure_interaction() {
    let trials = interaction_trials(250);

    // Main effects are all weak: neither x1, x2, nor z explains much alone.
    let mains = fanova_importance(&trials);
    for m in &mains {
        assert!(
            m.variance_explained < 0.35,
            "{} should have a weak main effect on a pure interaction, got {}",
            m.name,
            m.variance_explained
        );
    }

    // The interaction between x1 and x2 dominates and ranks first.
    let inter = fanova_interactions(&trials);
    assert_eq!(inter.len(), 3, "three parameters => three pairs");
    let top = &inter[0];
    let pair = {
        let mut p = [top.first.as_str(), top.second.as_str()];
        p.sort_unstable();
        p
    };
    assert_eq!(pair, ["x1", "x2"], "the interacting pair should rank first");
    assert!(
        top.variance_explained > 0.3,
        "x1*x2 interaction should explain real variance, got {}",
        top.variance_explained
    );
    assert!(top.normalized > 0.7, "the pair should carry most interaction mass, got {}", top.normalized);

    let sum: f64 = inter.iter().map(|i| i.normalized).sum();
    assert!((sum - 1.0).abs() < 1e-9, "normalized interaction shares sum to 1, got {sum}");
}

#[test]
fn fanova_additive_objective_has_negligible_interaction() {
    // synthetic_trials => value = (x - 2)^2, additive in x with irrelevant noise.
    let trials = synthetic_trials(150);
    let inter = fanova_interactions(&trials);
    assert_eq!(inter.len(), 1, "two parameters => one pair");
    assert!(
        inter[0].variance_explained < 0.1,
        "an additive objective has almost no x/noise interaction, got {}",
        inter[0].variance_explained
    );
}

#[test]
fn fanova_empty_or_degenerate_is_safe() {
    // No trials at all.
    assert!(fanova_importance(&[]).is_empty());

    // One completed trial (not enough to fit anything).
    let mut t = Trial::new(0);
    t.record("x", Distribution::Uniform { low: 0.0, high: 1.0 }, Value::Float(0.5));
    t.value = Some(1.0);
    t.state = TrialState::Complete;
    assert!(fanova_importance(std::slice::from_ref(&t)).is_empty());
}

#[test]
fn best_so_far_is_monotonic_under_minimize() {
    let trials = synthetic_trials(60);
    let series = best_so_far(&trials, Direction::Minimize);
    assert_eq!(series.len(), 60);
    for w in series.windows(2) {
        assert!(
            w[1].1 <= w[0].1,
            "best-so-far must be non-increasing under Minimize"
        );
    }
}

#[test]
fn optimization_history_renders_an_svg() {
    let trials = synthetic_trials(50);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.svg");

    optimization_history_svg(&trials, Direction::Minimize, &path, "Optimization History")
        .unwrap();

    let svg = std::fs::read_to_string(&path).unwrap();
    assert!(svg.contains("<svg"), "output should be an SVG document");
    assert!(svg.contains("Best value so far"), "axis label should be present");
    assert!(svg.len() > 500, "SVG looks too small to be a real chart");
}
