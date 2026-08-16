use hyperopt_core::{Storage, StorageError, StudyMetadata, Trial};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

/// In-memory storage: trials kept in a `BTreeMap` per study (ordered by trial
/// number) behind a `Mutex`. Fast and dependency-free, but everything is lost
/// when the process exits. This is the Phase 1–2 default.
#[derive(Default)]
pub struct InMemoryStorage {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    trials: HashMap<String, BTreeMap<usize, Trial>>,
    meta: HashMap<String, StudyMetadata>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Storage for InMemoryStorage {
    fn save_trial(&self, study_name: &str, trial: &Trial) -> Result<(), StorageError> {
        let mut inner = self.lock();
        inner
            .trials
            .entry(study_name.to_string())
            .or_default()
            .insert(trial.number, trial.clone());
        Ok(())
    }

    fn load_trials(&self, study_name: &str) -> Result<Vec<Trial>, StorageError> {
        let inner = self.lock();
        Ok(inner
            .trials
            .get(study_name)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default())
    }

    fn save_study_metadata(&self, meta: &StudyMetadata) -> Result<(), StorageError> {
        let mut inner = self.lock();
        inner.meta.insert(meta.study_name.clone(), meta.clone());
        Ok(())
    }

    fn load_study_metadata(
        &self,
        study_name: &str,
    ) -> Result<Option<StudyMetadata>, StorageError> {
        let inner = self.lock();
        Ok(inner.meta.get(study_name).cloned())
    }
}
