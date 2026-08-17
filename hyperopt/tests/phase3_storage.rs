//! Phase 3 definition-of-done: a study optimized partway, persisted to
//! `SqliteStorage`, then reloaded via a fresh `Study` on the same file,
//! continues suggesting trials that account for the already-completed history.
//! We prove the history is really loaded and used by showing that TPE's
//! post-reload suggestion differs from a cold start with the same seed.

use hyperopt_rs::prelude::*;

fn sphere(trial: &mut TrialContext) -> ObjectiveResult {
    let x = trial.suggest_float("x", -10.0, 10.0);
    let y = trial.suggest_float("y", -10.0, 10.0);
    Ok((x - 2.0).powi(2) + (y + 3.0).powi(2))
}

#[test]
fn sqlite_study_resumes_across_a_fresh_open() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("study.db");

    // --- Session 1: optimize partway and let the Study/storage drop. ---
    let n_first = 30;
    {
        let storage = SqliteStorage::open(&db).unwrap();
        let study = StudyBuilder::new("resumable")
            .direction(Direction::Minimize)
            .sampler(TpeSampler::seeded(123))
            .storage(storage)
            .build()
            .unwrap();
        study.optimize(sphere, n_first).unwrap();
        assert_eq!(study.trials().unwrap().len(), n_first);
    }

    // --- Session 2: fresh open of the same file continues where it left off. ---
    let storage = SqliteStorage::open(&db).unwrap();
    let resumed = StudyBuilder::new("resumable")
        .direction(Direction::Minimize)
        .sampler(TpeSampler::seeded(123))
        .storage(storage)
        .build()
        .unwrap();

    // History survived the reopen, with faithful params + numbering.
    let loaded = resumed.trials().unwrap();
    assert_eq!(loaded.len(), n_first);
    assert!(loaded.iter().all(|t| t.param_value("x").is_some()));
    assert_eq!(loaded.last().unwrap().number, n_first - 1);

    // Run one more trial; its number continues the sequence.
    resumed.optimize(sphere, 1).unwrap();
    let after = resumed.trials().unwrap();
    assert_eq!(after.len(), n_first + 1);
    let warm_trial = after.last().unwrap();
    assert_eq!(warm_trial.number, n_first);
    let warm_x = warm_trial.param_value("x").unwrap().as_float().unwrap();

    // --- Cold start: same sampler seed, but no history. ---
    let cold = StudyBuilder::new("cold")
        .direction(Direction::Minimize)
        .sampler(TpeSampler::seeded(123))
        .build()
        .unwrap();
    cold.optimize(sphere, 1).unwrap();
    let cold_x = cold
        .trials()
        .unwrap()
        .last()
        .unwrap()
        .param_value("x")
        .unwrap()
        .as_float()
        .unwrap();

    // With 30 prior trials loaded, TPE is past its warmup and samples from a
    // model; the cold start (no history) is a random warmup draw. Same seed,
    // different code path => different suggestion. This only holds if the
    // reloaded history was actually used.
    assert!(
        (warm_x - cold_x).abs() > 1e-9,
        "resumed suggestion ({warm_x}) matched cold start ({cold_x}); \
         history was not used after reload"
    );
}

#[test]
fn in_memory_and_sqlite_agree_on_a_run() {
    let run = |study: Study| {
        study.optimize(sphere, 40).unwrap();
        study.trials().unwrap().len()
    };

    let mem = StudyBuilder::new("mem")
        .sampler(RandomSampler::seeded(9))
        .storage(InMemoryStorage::new())
        .build()
        .unwrap();
    let sql = StudyBuilder::new("sql")
        .sampler(RandomSampler::seeded(9))
        .storage(SqliteStorage::open_in_memory().unwrap())
        .build()
        .unwrap();

    assert_eq!(run(mem), 40);
    assert_eq!(run(sql), 40);
}
