use super::*;
use super::service::diff_manifests;
use super::manifest::Manifest;
use super::queue::{CoalescingQueue, QueueInsert};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[test]
fn paths_coalesce_and_overflow_is_reported() {
    let now = Instant::now();
    let mut queue = CoalescingQueue::default();
    assert_eq!(queue.insert("a.rs".into(), now, 1), QueueInsert::Added);
    assert_eq!(queue.insert("a.rs".into(), now + Duration::from_millis(1), 1), QueueInsert::Coalesced);
    assert_eq!(queue.insert("b.rs".into(), now, 1), QueueInsert::Overflow);
    assert!(queue.is_dirty());
    assert_eq!(queue.drain_due(now + Duration::from_millis(400), Duration::from_millis(300)), vec![PathBuf::from("a.rs")]);
}
#[test]
fn missing_git_marker_is_a_stable_non_git_observation() {
    let temp = TempDir::new().unwrap();

    assert_eq!(read_branch_marker(temp.path()).unwrap(), None);
    std::fs::write(temp.path().join(".git"), "malformed").unwrap();
    assert_eq!(
        read_branch_marker(temp.path()).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData,
    );
}

#[test]
fn periodic_manifest_comparison_detects_dropped_add_modify_delete() {
    let before = Manifest::from([(PathBuf::from("old.rs"), "old".into()), (PathBuf::from("same.rs"), "same".into())]);
    let after = Manifest::from([(PathBuf::from("new.rs"), "new".into()), (PathBuf::from("same.rs"), "changed".into())]);
    let mut changed = diff_manifests(Some(&before), &after);
    changed.sort();
    assert_eq!(changed, vec![PathBuf::from("new.rs"), PathBuf::from("old.rs"), PathBuf::from("same.rs")]);
}

struct RecordingBackend {
    events: Mutex<Vec<String>>,
    paths: Mutex<Vec<Vec<PathBuf>>>,
    reconciles: AtomicUsize,
    fail: bool,
}

impl WatchIndexBackend for RecordingBackend {
    fn apply_incremental(&self, request: ReconcileRequest) -> Result<(), String> {
        self.events.lock().push(format!("incremental:{:?}", request.reason));
        self.paths.lock().push(request.changed_paths);
        if self.fail { Err("incremental failed".into()) } else { Ok(()) }
    }

    fn begin_rebuild(&self, _: &BranchTransition) -> Result<(), String> {
        self.events.lock().push("begin".into());
        if self.fail { Err("begin failed".into()) } else { Ok(()) }
    }

    fn reconcile(&self, request: ReconcileRequest) -> Result<(), String> {
        self.reconciles.fetch_add(1, Ordering::SeqCst);
        self.events.lock().push(format!(
            "reconcile:{:?}:{}",
            request.reason,
            request.branch_transition.is_some(),
        ));
        if self.fail { Err("reconcile failed".into()) } else { Ok(()) }
    }

    fn complete_reconciliation(&self, _: &ReconcileRequest) -> Result<(), String> {
        self.events.lock().push("complete".into());
        if self.fail { Err("completion failed".into()) } else { Ok(()) }
    }

    fn current_manifest(&self, _: &std::path::Path, _: u64) -> Result<Manifest, String> {
        Ok(Manifest::new())
    }

    fn queue_shutdown_reconciliation(&self, _: &str) -> Result<(), String> {
        self.events.lock().push("shutdown-reconcile".into());
        Ok(())
    }
}
struct BlockingReconcileBackend {
    started: std::sync::mpsc::SyncSender<()>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
    queued_shutdown_reconciles: AtomicUsize,
}

impl WatchIndexBackend for BlockingReconcileBackend {
    fn apply_incremental(&self, _: ReconcileRequest) -> Result<(), String> { Ok(()) }
    fn begin_rebuild(&self, _: &BranchTransition) -> Result<(), String> { Ok(()) }
    fn reconcile(&self, _: ReconcileRequest) -> Result<(), String> {
        self.started.send(()).map_err(|error| error.to_string())?;
        self.release.lock().recv().map_err(|error| error.to_string())?;
        Err("injected reconciliation failure".into())
    }
    fn complete_reconciliation(&self, _: &ReconcileRequest) -> Result<(), String> { Ok(()) }
    fn current_manifest(&self, _: &std::path::Path, _: u64) -> Result<Manifest, String> {
        Ok(Manifest::new())
    }
    fn queue_shutdown_reconciliation(&self, _: &str) -> Result<(), String> {
        self.queued_shutdown_reconciles.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
struct ShutdownQueueFailureBackend;

impl WatchIndexBackend for ShutdownQueueFailureBackend {
    fn apply_incremental(&self, _: ReconcileRequest) -> Result<(), String> { Ok(()) }
    fn begin_rebuild(&self, _: &BranchTransition) -> Result<(), String> { Ok(()) }
    fn reconcile(&self, _: ReconcileRequest) -> Result<(), String> { Ok(()) }
    fn complete_reconciliation(&self, _: &ReconcileRequest) -> Result<(), String> { Ok(()) }
    fn current_manifest(&self, _: &std::path::Path, _: u64) -> Result<Manifest, String> {
        Ok(Manifest::new())
    }
    fn queue_shutdown_reconciliation(&self, _: &str) -> Result<(), String> {
        Err("injected durable queue failure".into())
    }
}

#[test]
fn clean_stop_reports_shutdown_reconciliation_queue_failure() {
    let temp = TempDir::new().unwrap();
    let mut watcher = WorkspaceWatcher::start(
        "queue-failure",
        temp.path(),
        Arc::new(ShutdownQueueFailureBackend),
        test_config(),
    ).unwrap();

    assert_eq!(
        watcher.stop(),
        WatchStopOutcome::ReconciliationQueueFailed("injected durable queue failure".into()),
    );
    assert_eq!(
        watcher.status().shutdown_reconciliation_error.as_deref(),
        Some("injected durable queue failure"),
    );
    assert!(watcher.status().last_error.as_deref().is_some_and(|error| {
        error.contains("could not queue full reconciliation: injected durable queue failure")
    }));
}

#[test]
fn stop_during_blocked_backend_is_bounded_and_keeps_reconciliation_queued() {
    let temp = TempDir::new().unwrap();
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let backend = Arc::new(BlockingReconcileBackend {
        started: started_tx,
        release: Mutex::new(release_rx),
        queued_shutdown_reconciles: AtomicUsize::new(0),
    });
    let mut watcher = WorkspaceWatcher::start("blocked", temp.path(), backend.clone(), test_config()).unwrap();
    started_rx.recv_timeout(Duration::from_secs(3)).unwrap();

    let started_at = Instant::now();
    let outcome = watcher.stop_until(Instant::now() + Duration::from_millis(100));
    assert_eq!(outcome, WatchStopOutcome::TimedOut);
    assert!(started_at.elapsed() < Duration::from_secs(1));
    assert!(watcher.status().stop_timed_out);
    assert!(matches!(
        WorkspaceWatcher::start("blocked", temp.path(), backend.clone(), test_config()),
        Err(WatchError::WorkerAlreadyTracked)
    ));

    release_tx.send(()).unwrap();
    assert!(wait_until(|| backend.queued_shutdown_reconciles.load(Ordering::SeqCst) == 1));
    assert_eq!(
        watcher.stop_until(Instant::now() + Duration::from_secs(1)),
        WatchStopOutcome::Stopped
    );
}
#[test]
fn directory_create_and_move_schedule_full_reconciliation() {
    let temp = TempDir::new().unwrap();
    let created_directory = temp.path().join("created");
    std::fs::create_dir(&created_directory).unwrap();
    let status = std::sync::Mutex::new(WatchStatus::default());
    let mut queue = CoalescingQueue::default();
    let mut force_full = false;
    let mut reason = ReconcileReason::Startup;

    let create = notify::Event::new(notify::EventKind::Create(
        notify::event::CreateKind::Folder,
    )).add_path(created_directory.clone());
    super::service::handle_event(
        create, temp.path(), &mut queue, 8, &status, &mut force_full, &mut reason,
    );
    assert!(force_full);
    assert_eq!(reason, ReconcileReason::TreeStructureChanged);
    assert_eq!(queue.len(), 0);

    force_full = false;
    let moved_directory = temp.path().join("moved");
    std::fs::create_dir(&moved_directory).unwrap();
    let rename = notify::Event::new(notify::EventKind::Modify(
        notify::event::ModifyKind::Name(notify::event::RenameMode::Both),
    )).add_path(temp.path().join("old")).add_path(moved_directory);
    super::service::handle_event(
        rename, temp.path(), &mut queue, 8, &status, &mut force_full, &mut reason,
    );
    assert!(force_full);
    assert_eq!(reason, ReconcileReason::TreeStructureChanged);
    assert_eq!(queue.len(), 0);
}

#[test]
fn watched_root_registration_event_does_not_request_reconciliation() {
    let temp = TempDir::new().unwrap();
    let status = std::sync::Mutex::new(WatchStatus::default());
    let mut queue = CoalescingQueue::default();
    let mut force_full = false;
    let mut reason = ReconcileReason::Startup;
    let root_event = notify::Event::new(notify::EventKind::Create(
        notify::event::CreateKind::Folder,
    )).add_path(temp.path().to_path_buf());

    super::service::handle_event(
        root_event, temp.path(), &mut queue, 8, &status, &mut force_full, &mut reason,
    );

    assert!(!force_full);
    assert_eq!(queue.len(), 0);
}



fn wait_until(mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if predicate() { return true; }
        std::thread::sleep(Duration::from_millis(10));
    }
    predicate()
}

fn test_config() -> WatchConfig {
    WatchConfig { debounce: Duration::from_millis(20), queue_capacity: 8, periodic_reconcile_interval: Duration::from_secs(10), branch_poll_interval: Duration::from_millis(20), retry_interval: Duration::from_millis(20), max_file_bytes: 1024 * 1024 }
}

#[test]
fn branch_transition_begins_generation_before_reconcile() {
    let temp = TempDir::new().unwrap();
    std::process::Command::new("git").args(["init", "--initial-branch=main"]).arg(temp.path()).output().unwrap();
    let git_dir = temp.path().join(".git");
    std::fs::create_dir_all(git_dir.join("refs/heads")).unwrap();
    std::fs::write(git_dir.join("refs/heads/main"), format!("{}\n", "a".repeat(40))).unwrap();
    let backend = Arc::new(RecordingBackend { events: Mutex::new(Vec::new()), paths: Mutex::new(Vec::new()), reconciles: AtomicUsize::new(0), fail: false });
    let mut watcher = WorkspaceWatcher::start("test", temp.path(), backend.clone(), test_config()).unwrap();
    assert!(wait_until(|| backend.reconciles.load(Ordering::SeqCst) >= 1));
    backend.events.lock().clear();
    std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/other\n").unwrap();
    std::fs::write(git_dir.join("refs/heads/other"), format!("{}\n", "b".repeat(40))).unwrap();
    assert!(wait_until(|| {
        let events = backend.events.lock();
        events.iter().any(|event| event == "begin")
            && events.iter().any(|event| event == "reconcile:BranchTransition:true")
    }));

    let events = backend.events.lock().clone();
    let begin = events.iter().position(|event| event == "begin").unwrap();
    let reconcile = events.iter().position(|event| event == "reconcile:BranchTransition:true").unwrap();
    assert!(begin < reconcile);
    watcher.stop();
}

#[test]
fn stopping_clean_watcher_queues_durable_full_reconciliation() {
    let temp = TempDir::new().unwrap();
    let backend = Arc::new(RecordingBackend {
        events: Mutex::new(Vec::new()),
        paths: Mutex::new(Vec::new()),
        reconciles: AtomicUsize::new(0),
        fail: false,
    });
    let mut watcher =
        WorkspaceWatcher::start("clean-stop", temp.path(), backend.clone(), test_config())
            .unwrap();
    assert!(wait_until(|| backend.reconciles.load(Ordering::SeqCst) == 1));
    assert!(wait_until(|| {
        backend.events.lock().iter().any(|event| event == "complete")
    }));
    assert!(wait_until(|| !watcher.status().dirty));
    backend.events.lock().clear();


    assert_eq!(watcher.stop(), WatchStopOutcome::Stopped);
    assert_eq!(
        backend.events.lock().as_slice(),
        &[String::from("shutdown-reconcile")],
        "a clean local queue cannot prove the notify callback is fenced",
    );
    assert!(watcher.status().dirty);
    assert!(watcher
        .status()
        .last_error
        .is_some_and(|message| message.contains("durable full reconciliation queued")));
}

#[test]
fn ordinary_file_change_reaches_incremental_backend() {
    let temp = TempDir::new().unwrap();
    std::process::Command::new("git").arg("init").arg(temp.path()).output().unwrap();
    let backend = Arc::new(RecordingBackend { events: Mutex::new(Vec::new()), paths: Mutex::new(Vec::new()), reconciles: AtomicUsize::new(0), fail: false });
    let mut watcher = WorkspaceWatcher::start("test", temp.path(), backend.clone(), test_config()).unwrap();
    assert!(wait_until(|| backend.reconciles.load(Ordering::SeqCst) >= 1));
    backend.events.lock().clear();
    std::fs::write(temp.path().join("created.rs"), "fn created() {}").unwrap();
    assert!(wait_until(|| backend.events.lock().iter().any(|event| event == "incremental:PathChanges")));
    assert_eq!(
        backend.paths.lock().last().cloned().unwrap(),
        vec![PathBuf::from("created.rs")],
    );
    assert_eq!(backend.reconciles.load(Ordering::SeqCst), 1);
    watcher.stop();
}
#[test]
fn non_git_branch_polling_does_not_schedule_repeated_full_reconciles() {
    let temp = TempDir::new().unwrap();
    let backend = Arc::new(RecordingBackend {
        events: Mutex::new(Vec::new()),
        paths: Mutex::new(Vec::new()),
        reconciles: AtomicUsize::new(0),
        fail: false,
    });
    let mut watcher = WorkspaceWatcher::start("non-git", temp.path(), backend.clone(), test_config()).unwrap();
    assert!(wait_until(|| backend.reconciles.load(Ordering::SeqCst) == 1));
    std::thread::sleep(Duration::from_millis(120));

    assert_eq!(backend.reconciles.load(Ordering::SeqCst), 1);
    assert_eq!(watcher.status().full_reconcile_count, 1);
    assert!(!watcher.status().last_error.is_some_and(|error| {
        error.contains("Git") || error.contains("branch")
    }));
    watcher.stop();
}


struct DroppedEventBackend {
    calls: AtomicUsize,
    reconciles: AtomicUsize,
    events: Mutex<Vec<String>>,
}
impl WatchIndexBackend for DroppedEventBackend {
    fn apply_incremental(&self, _: ReconcileRequest) -> Result<(), String> { Ok(()) }
    fn begin_rebuild(&self, _: &BranchTransition) -> Result<(), String> { Ok(()) }
    fn reconcile(&self, request: ReconcileRequest) -> Result<(), String> {
        self.reconciles.fetch_add(1, Ordering::SeqCst);
        self.events.lock().push(format!("{:?}", request.reason));
        Ok(())
    }
    fn complete_reconciliation(&self, _: &ReconcileRequest) -> Result<(), String> { Ok(()) }
    fn current_manifest(&self, _: &std::path::Path, _: u64) -> Result<Manifest, String> {
        if self.calls.fetch_add(1, Ordering::SeqCst) < 2 { Ok(Manifest::new()) }
        else { Ok(Manifest::from([(PathBuf::from("missed.rs"), "hash".into())])) }
    }
}

#[test]
fn periodic_manifest_repairs_a_dropped_notification() {
    let temp = TempDir::new().unwrap();
    std::process::Command::new("git").arg("init").arg(temp.path()).output().unwrap();
    let backend = Arc::new(DroppedEventBackend { calls: AtomicUsize::new(0), reconciles: AtomicUsize::new(0), events: Mutex::new(Vec::new()) });
    let config = WatchConfig { periodic_reconcile_interval: Duration::from_millis(50), branch_poll_interval: Duration::from_secs(5), debounce: Duration::from_millis(10), ..test_config() };
    let mut watcher = WorkspaceWatcher::start("test", temp.path(), backend.clone(), config).unwrap();
    assert!(wait_until(|| backend.events.lock().iter().any(|event| event == "PeriodicManifestMismatch")));
    assert!(backend.reconciles.load(Ordering::SeqCst) >= 2);
    watcher.stop();
}
#[test]
fn callback_failure_keeps_watcher_dirty() {
    let temp = TempDir::new().unwrap();
    std::process::Command::new("git").arg("init").arg(temp.path()).output().unwrap();
    let backend = Arc::new(RecordingBackend { events: Mutex::new(Vec::new()), paths: Mutex::new(Vec::new()), reconciles: AtomicUsize::new(0), fail: true });
    let mut watcher = WorkspaceWatcher::start("test", temp.path(), backend.clone(), test_config()).unwrap();
    assert!(wait_until(|| backend.reconciles.load(Ordering::SeqCst) >= 1));
    assert!(watcher.status().dirty);
    watcher.stop();
}
