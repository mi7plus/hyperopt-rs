use hyperopt_core::{Direction, Storage, StorageError, StudyMetadata, Trial};
use rusqlite::{Connection, OptionalExtension};
use std::path::Path;
use std::sync::Mutex;

/// On-disk storage format version. Bumped only on a breaking change to the
/// SQLite schema; [`SqliteStorage::open`] refuses to read a file written by a
/// newer, incompatible version rather than silently misreading it.
const SCHEMA_VERSION: i64 = 1;

/// SQLite-backed storage: trials are persisted to a file so studies survive
/// process restarts and can be resumed. Mirrors Optuna's RDB storage pattern.
///
/// Each trial is stored as a JSON document keyed by `(study_name, number)`,
/// which keeps the schema stable while faithfully round-tripping the
/// define-by-run parameter set and every intermediate report.
///
/// The connection is guarded by a `Mutex`, so the backend is `Send + Sync` and
/// usable under parallel execution. Several independent processes can also open
/// the same file (SQLite handles the file-level locking) for lightweight
/// multi-process coordination — a practical middle ground short of full
/// distributed execution.
pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

impl SqliteStorage {
    /// Open (creating if absent) a study database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let conn = Connection::open(path).map_err(backend)?;
        Self::from_connection(conn)
    }

    /// Open an anonymous in-memory SQLite database (useful for tests).
    pub fn open_in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory().map_err(backend)?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self, StorageError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS hyperopt_meta (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 schema_version INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS studies (
                 name TEXT PRIMARY KEY,
                 direction TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS trials (
                 study_name TEXT NOT NULL,
                 number INTEGER NOT NULL,
                 data TEXT NOT NULL,
                 PRIMARY KEY (study_name, number)
             );",
        )
        .map_err(backend)?;

        // Establish or verify the schema version.
        let existing: Option<i64> = conn
            .query_row(
                "SELECT schema_version FROM hyperopt_meta WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(backend)?;
        match existing {
            None => {
                conn.execute(
                    "INSERT INTO hyperopt_meta (id, schema_version) VALUES (1, ?1)",
                    [SCHEMA_VERSION],
                )
                .map_err(backend)?;
            }
            Some(v) if v != SCHEMA_VERSION => {
                return Err(StorageError::SchemaMismatch {
                    found: v,
                    expected: SCHEMA_VERSION,
                });
            }
            Some(_) => {}
        }

        Ok(SqliteStorage {
            conn: Mutex::new(conn),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Storage for SqliteStorage {
    fn save_trial(&self, study_name: &str, trial: &Trial) -> Result<(), StorageError> {
        let data = serde_json::to_string(trial)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        let conn = self.lock();
        conn.execute(
            "INSERT INTO trials (study_name, number, data) VALUES (?1, ?2, ?3)
             ON CONFLICT(study_name, number) DO UPDATE SET data = excluded.data",
            rusqlite::params![study_name, trial.number as i64, data],
        )
        .map_err(backend)?;
        Ok(())
    }

    fn load_trials(&self, study_name: &str) -> Result<Vec<Trial>, StorageError> {
        let conn = self.lock();
        let mut stmt = conn
            .prepare("SELECT data FROM trials WHERE study_name = ?1 ORDER BY number ASC")
            .map_err(backend)?;
        let rows = stmt
            .query_map([study_name], |row| row.get::<_, String>(0))
            .map_err(backend)?;
        let mut trials = Vec::new();
        for row in rows {
            let data = row.map_err(backend)?;
            let trial: Trial = serde_json::from_str(&data)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            trials.push(trial);
        }
        Ok(trials)
    }

    fn save_study_metadata(&self, meta: &StudyMetadata) -> Result<(), StorageError> {
        let conn = self.lock();
        conn.execute(
            "INSERT INTO studies (name, direction) VALUES (?1, ?2)
             ON CONFLICT(name) DO UPDATE SET direction = excluded.direction",
            rusqlite::params![meta.study_name, direction_to_str(meta.direction)],
        )
        .map_err(backend)?;
        Ok(())
    }

    fn load_study_metadata(
        &self,
        study_name: &str,
    ) -> Result<Option<StudyMetadata>, StorageError> {
        let conn = self.lock();
        let row: Option<String> = conn
            .query_row(
                "SELECT direction FROM studies WHERE name = ?1",
                [study_name],
                |row| row.get(0),
            )
            .optional()
            .map_err(backend)?;
        match row {
            Some(dir) => Ok(Some(StudyMetadata {
                study_name: study_name.to_string(),
                direction: direction_from_str(&dir)?,
            })),
            None => Ok(None),
        }
    }
}

fn backend(e: rusqlite::Error) -> StorageError {
    StorageError::Backend(e.to_string())
}

fn direction_to_str(d: Direction) -> &'static str {
    match d {
        Direction::Minimize => "minimize",
        Direction::Maximize => "maximize",
    }
}

fn direction_from_str(s: &str) -> Result<Direction, StorageError> {
    match s {
        "minimize" => Ok(Direction::Minimize),
        "maximize" => Ok(Direction::Maximize),
        other => Err(StorageError::Backend(format!("unknown direction: {other}"))),
    }
}
