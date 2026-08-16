//! Phase 3: schema versioning guards against silently misreading a study file
//! written by an incompatible future version.

#![cfg(feature = "sqlite")]

use hyperopt_core::{Direction, Storage, StorageError, StudyMetadata, Trial, TrialState};
use hyperopt_storage::SqliteStorage;

#[test]
fn rejects_incompatible_schema_version() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("bad.db");

    // Create a valid store, then bump the recorded schema version past ours.
    {
        let _ = SqliteStorage::open(&db).unwrap();
    }
    {
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute("UPDATE hyperopt_meta SET schema_version = 999 WHERE id = 1", [])
            .unwrap();
    }

    match SqliteStorage::open(&db) {
        Err(StorageError::SchemaMismatch { found, expected }) => {
            assert_eq!(found, 999);
            assert_eq!(expected, 1);
        }
        Err(other) => panic!("expected SchemaMismatch, got {other:?}"),
        Ok(_) => panic!("expected SchemaMismatch, but open succeeded"),
    }
}

#[test]
fn trials_round_trip_faithfully() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage
        .save_study_metadata(&StudyMetadata {
            study_name: "s".into(),
            direction: Direction::Maximize,
        })
        .unwrap();

    let mut t = Trial::new(0);
    t.record(
        "lr",
        hyperopt_core::Distribution::LogUniform { low: 1e-4, high: 1e-1 },
        hyperopt_core::Value::Float(0.01),
    );
    t.record(
        "opt",
        hyperopt_core::Distribution::Categorical {
            choices: vec!["adam".into(), "sgd".into()],
        },
        hyperopt_core::Value::Categorical("adam".into()),
    );
    t.intermediate_values.push((0, 0.5));
    t.intermediate_values.push((1, 0.7));
    t.value = Some(0.7);
    t.state = TrialState::Complete;

    storage.save_trial("s", &t).unwrap();

    let loaded = storage.load_trials("s").unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0], t, "trial should round-trip byte-for-byte");

    let meta = storage.load_study_metadata("s").unwrap().unwrap();
    assert_eq!(meta.direction, Direction::Maximize);
}
