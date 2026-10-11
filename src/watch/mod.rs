//! Filesystem watching and reconciliation scheduling.
//!
//! Ordinary path events use capability reads and targeted persistence. Full
//! manifest reconciliation is reserved for startup, explicit full requests,
//! periodic mismatch repair, branch/ignore transitions, and watcher failures.
//! A backend must not publish a reconciliation as current until its indexing
//! transaction has committed; branch transitions make old results non-current
//! before their full reconciliation starts.

mod branch;
mod engine;
mod manifest;
mod queue;
mod service;
pub(crate) use manifest::{build_manifest, build_paths_manifest};
pub use manifest::Manifest;

pub use branch::{BranchMarker, BranchTransition, read_branch_marker};
pub use engine::WatchIndexBackend;
pub use service::{WatchStopOutcome, WorkspaceWatcher};


use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use thiserror::Error;

/// The hard upper bound on coalesced relative paths held by one watcher.
pub const MAX_QUEUED_PATHS: usize = 10_000;

/// Maximum time `WorkspaceWatcher::stop` waits for synchronous backend work.
pub const WATCHER_STOP_TIMEOUT: Duration = Duration::from_secs(2);

/// Watcher timing and scanner settings.
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Quiet time required before a path batch is reconciled.
    pub debounce: Duration,
    /// Maximum coalesced paths; values above [`MAX_QUEUED_PATHS`] are rejected.
    pub queue_capacity: usize,
    /// Interval between full manifest/hash comparisons.
    pub periodic_reconcile_interval: Duration,
    /// Poll interval for Git HEAD and its referenced ref.
    pub branch_poll_interval: Duration,
    /// Delay before retrying failed reconciliation work.
    pub retry_interval: Duration,
    /// Maximum file size considered by the manifest scanner.
    pub max_file_bytes: u64,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            debounce: Duration::from_millis(300),
            queue_capacity: MAX_QUEUED_PATHS,
            periodic_reconcile_interval: Duration::from_secs(5 * 60),
            branch_poll_interval: Duration::from_secs(1),
            retry_interval: Duration::from_secs(1),
            max_file_bytes: 1024 * 1024,
        }
    }
}

impl WatchConfig {
    pub(crate) fn validate(&self) -> Result<(), WatchError> {
        if self.queue_capacity == 0 || self.queue_capacity > MAX_QUEUED_PATHS {
            return Err(WatchError::InvalidConfig(format!(
                "queue_capacity must be in 1..={MAX_QUEUED_PATHS}"
            )));
        }
        if self.periodic_reconcile_interval.is_zero()
            || self.branch_poll_interval.is_zero()
            || self.retry_interval.is_zero()
            || self.max_file_bytes == 0
        {
            return Err(WatchError::InvalidConfig(
                "periodic, branch-poll, and retry intervals and max_file_bytes must be nonzero"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Why the watcher requested an index reconciliation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconcileReason {
    Startup,
    PathChanges,
    TreeStructureChanged,
    IgnoreRulesChanged,
    QueueOverflow,
    NotifyError,
    UnknownNotifyEvent,
    PeriodicManifestMismatch,
    BranchTransition,
    BranchObservationFailed,
    BackendFailure,
}

/// Reconciliation request passed to the persistence adapter.
#[derive(Debug, Clone)]
pub struct ReconcileRequest {
    pub workspace_id: String,
    pub root: PathBuf,
    pub reason: ReconcileReason,
    /// Whether the backend must compare the complete workspace manifest.
    pub full_scan: bool,
    /// Sorted, deduplicated relative paths for targeted refreshes or a full pass.
    pub changed_paths: Vec<PathBuf>,
    /// Present only while rebuilding after a Git HEAD/ref transition.
    pub branch_transition: Option<BranchTransition>,
}

/// Local watcher health. It reports dirty state until persistence confirms that
/// reconciliation completed; it is not a substitute for the database status.
#[derive(Debug, Clone, Default)]
pub struct WatchStatus {
    pub dirty: bool,
    /// True after stop has been requested but the worker has not exited yet.
    pub stopping: bool,
    /// True if a bounded stop wait has expired; `stopping` reports the current state.
    pub stop_timed_out: bool,
    pub queued_paths: usize,
    pub full_reconcile_count: u64,
    pub last_error: Option<String>,
    /// Exact persistence failure from the worker's unconditional shutdown enqueue.
    pub shutdown_reconciliation_error: Option<String>,
    pub last_reconciled_at: Option<SystemTime>,
}

/// Errors while configuring or operating the watcher service.
#[derive(Debug, Error)]
pub enum WatchError {
    #[error("watcher I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("notify watcher error: {0}")]
    Notify(String),
    #[error("watch backend error: {0}")]
    Backend(String),
    #[error("invalid watcher configuration: {0}")]
    InvalidConfig(String),
    #[error("a watcher worker is already tracked for this workspace root")]
    WorkerAlreadyTracked,
    #[error("the process watcher-worker limit ({0}) has been reached")]
    WorkerLimit(usize),

}
#[cfg(test)]
mod tests;
