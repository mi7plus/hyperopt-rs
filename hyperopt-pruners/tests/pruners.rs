//! Phase 2 definition-of-done (part 1): a bad-trajectory trial is pruned by
//! `MedianPruner` while a good-trajectory trial is not. Uses hand-built study
//! state so the behaviour is deterministic and not sampler-dependent.

use hyperopt_core::{Direction, Pruner, StudyState, Trial, TrialState};
use hyperopt_pruners::{MedianPruner, NopPruner, SuccessiveHalvingPruner};

/// A completed trial that reported `value` at `step`.
fn completed_at(number: usize, step: usize, value: f64) -> Trial {
    let mut t = Trial::new(number);
    t.intermediate_values.push((step, value));
    t.value = Some(value);
    t.state = TrialState::Complete;
    t
}

/// A running trial that has just reported `value` at `step`.
fn running_at(step: usize, value: f64) -> Trial {
    let mut t = Trial::new(100);
    t.intermediate_values.push((step, value));
    t.state = TrialState::Running;
    t
}

fn history() -> Vec<Trial> {
    // Six completed trials, each reporting at step 5. Median at step 5 ~ 0.35.
    vec![
        completed_at(0, 5, 0.10),
        completed_at(1, 5, 0.20),
        completed_at(2, 5, 0.30),
        completed_at(3, 5, 0.40),
        completed_at(4, 5, 0.50),
        completed_at(5, 5, 0.60),
    ]
}

#[test]
fn median_pruner_prunes_bad_trajectory_keeps_good() {
    let state = StudyState::new(Direction::Minimize, history());
    let pruner = MedianPruner::new(); // n_startup_trials = 5

    let bad = running_at(5, 5.0); // far worse than the median
    let good = running_at(5, 0.05); // better than everyone

    assert!(pruner.should_prune(&state, &bad), "bad trajectory should prune");
    assert!(
        !pruner.should_prune(&state, &good),
        "good trajectory should not prune"
    );
}

#[test]
fn median_pruner_respects_startup_and_warmup_gates() {
    // Too few completed trials => never prune, even a terrible one.
    let few = StudyState::new(Direction::Minimize, vec![completed_at(0, 5, 0.1)]);
    let bad = running_at(5, 9.0);
    assert!(!MedianPruner::new().should_prune(&few, &bad));

    // Warmup: a high warmup-steps gate suppresses pruning at an early step.
    let state = StudyState::new(Direction::Minimize, history());
    let warmed = MedianPruner::new().n_warmup_steps(10);
    assert!(!warmed.should_prune(&state, &running_at(5, 5.0)));
}

#[test]
fn median_pruner_honours_direction() {
    // Under Maximize, a *low* value is the bad one.
    let state = StudyState::new(Direction::Maximize, history());
    let pruner = MedianPruner::new();
    assert!(pruner.should_prune(&state, &running_at(5, -1.0)));
    assert!(!pruner.should_prune(&state, &running_at(5, 9.0)));
}

#[test]
fn nop_pruner_never_prunes() {
    let state = StudyState::new(Direction::Minimize, history());
    let awful = running_at(5, 1e9);
    assert!(!NopPruner::new().should_prune(&state, &awful));
}

#[test]
fn successive_halving_promotes_top_fraction() {
    // Four peers reach rung resource 4 (min_resource=1, eta=2 => rungs 1,2,4).
    let peers = vec![
        completed_at(0, 4, 0.10),
        completed_at(1, 4, 0.20),
        completed_at(2, 4, 0.30),
        completed_at(3, 4, 0.40),
    ];
    let state = StudyState::new(Direction::Minimize, peers);
    let pruner = SuccessiveHalvingPruner::new()
        .min_resource(1)
        .reduction_factor(2);

    // Bottom of the pack at the rung => pruned.
    assert!(pruner.should_prune(&state, &running_at(4, 5.0)));
    // Best at the rung => promoted (kept).
    assert!(!pruner.should_prune(&state, &running_at(4, 0.05)));
    // Below the first rung => never pruned yet.
    assert!(!pruner.should_prune(&state, &running_at(0, 5.0)));
}
