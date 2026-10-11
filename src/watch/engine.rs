use super::manifest::Manifest;
use super::{BranchMarker, BranchTransition, ReconcileRequest};
use std::path::Path;

pub trait WatchIndexBackend: Send + Sync + 'static {
    fn apply_incremental(&self, request: ReconcileRequest) -> Result<(), String>;
    fn begin_rebuild(&self, transition: &BranchTransition) -> Result<(), String>;
    fn reconcile(&self, request: ReconcileRequest) -> Result<(), String>;
    fn complete_reconciliation(&self, request: &ReconcileRequest) -> Result<(), String>;
    fn current_manifest(&self, root: &Path, max_file_bytes: u64) -> Result<Manifest, String>;
    /// Persist pending or interrupted watcher work before its loop stops.
    fn queue_shutdown_reconciliation(&self, _workspace_id: &str) -> Result<(), String> {
        Err("watch backend does not support durable shutdown reconciliation".into())
    }

    fn prepare_startup_branch(
        &self,
        _current: Option<&BranchMarker>,
    ) -> Result<Option<BranchTransition>, String> {
        Ok(None)
    }

    fn branch_observation_failed(&self) -> Result<(), String> {
        Ok(())
    }
    fn startup_reconcile_required(&self, _manifest: &Manifest) -> Result<bool, String> {
        Ok(true)
    }
}
