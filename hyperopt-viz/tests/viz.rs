//! Phase 5 definition-of-done: optimization-history and parameter-importance
//! outputs render correctly on a completed study, and the importance proxy
//! ranks a genuinely-important parameter above an irrelevant one.

use hyperopt_core::{Direction, Distribution, Trial, TrialState, Value};
use hyperopt_viz::{best_so_far, optimization_history_svg, parameter_importance};
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
