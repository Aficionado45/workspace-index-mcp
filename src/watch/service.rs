use super::branch::{read_branch_marker, BranchTransition};
use super::engine::WatchIndexBackend;
use super::manifest::Manifest;
use super::queue::{CoalescingQueue, QueueInsert};
use super::{
    ReconcileReason, ReconcileRequest, WATCHER_STOP_TIMEOUT, WatchConfig, WatchError, WatchStatus,
};
use notify::event::{CreateKind, ModifyKind, RemoveKind};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::Mutex as RegistryMutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, LazyLock, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

const MAX_TRACKED_WATCHERS: usize = 128;


static WATCHER_WORKERS: LazyLock<RegistryMutex<HashMap<PathBuf, TrackedWorker>>> =
    LazyLock::new(|| RegistryMutex::new(HashMap::new()));

enum TrackedWorker {
    Starting,
    Running(JoinHandle<()>),
}

pub struct WorkspaceWatcher {
    status: Arc<Mutex<WatchStatus>>,
    stop: Arc<AtomicBool>,
    root: PathBuf,
    finished: mpsc::Receiver<()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchStopOutcome {
    /// The worker exited and its unconditional reconciliation enqueue succeeded.
    Stopped,
    /// The worker exited, but durable reconciliation could not be enqueued.
    ReconciliationQueueFailed(String),
    /// The wait bound expired; a synchronous backend call may continue on its tracked worker.
    TimedOut,
}

impl WorkspaceWatcher {
    pub fn start<B: WatchIndexBackend + ?Sized>(
        workspace_id: impl Into<String>,
        root: &Path,
        backend: Arc<B>,
        config: WatchConfig,
    ) -> Result<Self, WatchError> {
        config.validate()?;
        let root = root.canonicalize()?;
        if !root.is_dir() {
            return Err(WatchError::InvalidConfig("workspace root is not a directory".into()));
        }
        let reservation = reserve_worker(&root)?;
        let workspace_id = workspace_id.into();
        let startup_observation = match read_branch_marker(&root) {
            Ok(current) => backend.prepare_startup_branch(current.as_ref()).map(|_| ()),
            Err(error) => Err(error.to_string()),
        };
        if startup_observation.is_err() {
            backend.branch_observation_failed().map_err(WatchError::Backend)?;
        }
        let status = Arc::new(Mutex::new(WatchStatus { dirty: true, ..WatchStatus::default() }));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_status = Arc::clone(&status);
        let thread_stop = Arc::clone(&stop);
        let (event_tx, event_rx) = mpsc::sync_channel::<notify::Result<Event>>(2048);
        let dropped_event = Arc::new(AtomicBool::new(false));
        let dropped_flag = Arc::clone(&dropped_event);
        let mut watcher: RecommendedWatcher = notify::recommended_watcher(move |result| {
            if event_tx.try_send(result).is_err() {
                dropped_flag.store(true, Ordering::Release);
            }
        }).map_err(|error| WatchError::Notify(error.to_string()))?;
        watcher.watch(&root, RecursiveMode::Recursive)
            .map_err(|error| WatchError::Notify(error.to_string()))?;

        let (finished_tx, finished) = mpsc::sync_channel(1);
        let worker_root = root.clone();
        let worker_name = format!("workspace-index-watch-{workspace_id}");
        let worker = thread::Builder::new().name(worker_name).spawn(move || {
            run_loop(WatcherLoop {
                workspace_id,
                root: worker_root,
                backend,
                config,
                _watcher: watcher,
                event_rx,
                dropped: dropped_event,
                status: thread_status,
                stop: thread_stop,
            });
            let _ = finished_tx.send(());
        })?;
        reservation.install(worker);
        Ok(Self { status, stop, root, finished })
    }

    pub fn status(&self) -> WatchStatus {
        reap_finished_workers();
        self.status.lock().map(|status| status.clone()).unwrap_or_default()
    }

    /// Requests shutdown and waits no longer than the process-wide stop bound.
    pub fn stop(&mut self) -> WatchStopOutcome {
        self.stop_until(Instant::now() + WATCHER_STOP_TIMEOUT)
    }

    pub(crate) fn stop_until(&mut self, deadline: Instant) -> WatchStopOutcome {
        let wait_bound = deadline.saturating_duration_since(Instant::now());
        self.stop.store(true, Ordering::Release);
        if let Ok(mut status) = self.status.lock() {
            status.stopping = true;
        }
        let mut notified = false;
        loop {
            reap_finished_workers();
            if !worker_is_tracked(&self.root) {
                if let Ok(mut status) = self.status.lock() {
                    status.stopping = false;
                    return status.shutdown_reconciliation_error.clone().map_or(
                        WatchStopOutcome::Stopped,
                        WatchStopOutcome::ReconciliationQueueFailed,
                    );
                }
                return WatchStopOutcome::ReconciliationQueueFailed(
                    "watcher status lock poisoned before shutdown persistence could be confirmed".into(),
                );
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            if notified {
                thread::yield_now();
            } else {
                match self.finished.recv_timeout(remaining.min(Duration::from_millis(10))) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => notified = true,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
        }
        if let Ok(mut status) = self.status.lock() {
            status.stop_timed_out = true;
            status.dirty = true;
            status.last_error = Some(format!(
                "watcher stop exceeded its {:?} wait bound; worker remains tracked",
                wait_bound
            ));
        }
        WatchStopOutcome::TimedOut
    }
}

impl Drop for WorkspaceWatcher {
    fn drop(&mut self) {
        if self.stop.load(Ordering::Acquire) {
            reap_finished_workers();
        } else {
            match self.stop() {
                WatchStopOutcome::Stopped => {}
                WatchStopOutcome::ReconciliationQueueFailed(error) => eprintln!(
                    "workspace-index: watcher dropped without durable shutdown reconciliation: {error}"
                ),
                WatchStopOutcome::TimedOut => eprintln!(
                    "workspace-index: watcher dropped after the {:?} stop bound; its worker remains tracked",
                    WATCHER_STOP_TIMEOUT
                ),
            }
        }
    }
}

fn watcher_workers() -> &'static RegistryMutex<HashMap<PathBuf, TrackedWorker>> {
    &WATCHER_WORKERS
}

struct WorkerReservation {
    root: PathBuf,
    installed: bool,
}

impl WorkerReservation {
    fn install(mut self, worker: JoinHandle<()>) {
        let mut workers = watcher_workers().lock();
        if let Some(slot) = workers.get_mut(&self.root) {
            *slot = TrackedWorker::Running(worker);
        } else {
            workers.insert(self.root.clone(), TrackedWorker::Running(worker));
        }
        self.installed = true;
    }
}

impl Drop for WorkerReservation {
    fn drop(&mut self) {
        if !self.installed {
            watcher_workers().lock().remove(&self.root);
        }
    }
}

fn reserve_worker(root: &Path) -> Result<WorkerReservation, WatchError> {
    reap_finished_workers();
    let mut workers = watcher_workers().lock();
    if workers.contains_key(root) {
        return Err(WatchError::WorkerAlreadyTracked);
    }
    if workers.len() >= MAX_TRACKED_WATCHERS {
        return Err(WatchError::WorkerLimit(MAX_TRACKED_WATCHERS));
    }
    let root = root.to_path_buf();
    workers.insert(root.clone(), TrackedWorker::Starting);
    Ok(WorkerReservation { root, installed: false })
}

fn reap_finished_workers() {
    let handles = {
        let mut workers = watcher_workers().lock();
        let finished = workers.iter().filter_map(|(root, worker)| match worker {
            TrackedWorker::Running(handle) if handle.is_finished() => Some(root.clone()),
            TrackedWorker::Starting | TrackedWorker::Running(_) => None,
        }).collect::<Vec<_>>();
        finished.into_iter().filter_map(|root| match workers.remove(&root) {
            Some(TrackedWorker::Running(handle)) => Some(handle),
            Some(TrackedWorker::Starting) | None => None,
        }).collect::<Vec<_>>()
    };
    for handle in handles {
        if handle.join().is_err() {
            eprintln!("workspace-index: tracked watcher worker panicked");
        }
    }
}

fn worker_is_tracked(root: &Path) -> bool {
    watcher_workers().lock().contains_key(root)
}

struct WatcherLoop<B: ?Sized> {
    workspace_id: String,
    root: PathBuf,
    backend: Arc<B>,
    config: WatchConfig,
    _watcher: RecommendedWatcher,
    event_rx: mpsc::Receiver<notify::Result<Event>>,
    dropped: Arc<AtomicBool>,
    status: Arc<Mutex<WatchStatus>>,
    stop: Arc<AtomicBool>,
}

fn run_loop<B: WatchIndexBackend + ?Sized>(state: WatcherLoop<B>) {
    let WatcherLoop {
        workspace_id,
        root,
        backend,
        config,
        _watcher,
        event_rx,
        dropped,
        status,
        stop,
    } = state;
    let mut queue = CoalescingQueue::default();
    let mut force_full = false;
    let mut reason = ReconcileReason::Startup;
    let mut observed_branch = None;
    let mut pending_transition = None;
    let mut rebuild_started = false;
    match read_branch_marker(&root) {
        Ok(current) => {
            observed_branch = Some(current.clone());
            match backend.prepare_startup_branch(current.as_ref()) {
                Ok(Some(transition)) => {
                    pending_transition = Some(transition);
                    rebuild_started = true;
                    force_full = true;
                    reason = ReconcileReason::BranchTransition;
                }
                Ok(None) => {}
                Err(error) => {
                    mark_dirty(&status, error);
                    if let Err(queue_error) = backend.branch_observation_failed() {
                        mark_dirty(&status, queue_error);
                    }
                    force_full = true;
                    reason = ReconcileReason::BackendFailure;
                }
            }
        }
        Err(error) => {
            mark_dirty(&status, error.to_string());
            if let Err(queue_error) = backend.branch_observation_failed() {
                mark_dirty(&status, queue_error);
            }
            force_full = true;
            reason = ReconcileReason::BranchObservationFailed;
        }
    }
    let mut manifest = match backend.current_manifest(&root, config.max_file_bytes) {
        Ok(manifest) => Some(manifest),
        Err(error) => {
            mark_dirty(&status, error);
            force_full = true;
            reason = ReconcileReason::BackendFailure;
            None
        }
    };
    if let Some(current_manifest) = manifest.as_ref() {
        match backend.startup_reconcile_required(current_manifest) {
            Ok(required) => force_full |= required,
            Err(error) => {
                mark_dirty(&status, error);
                force_full = true;
                reason = ReconcileReason::BackendFailure;
            }
        }
    }
    if !force_full {
        mark_clean(&status);
    }
    let mut next_periodic = Instant::now() + config.periodic_reconcile_interval;
    let mut next_branch_check = Instant::now() + config.branch_poll_interval;
    let mut retry_at = Instant::now();

    while !stop.load(Ordering::Acquire) {
        let now = Instant::now();
        let timeout = [
            next_periodic.saturating_duration_since(now),
            next_branch_check.saturating_duration_since(now),
            retry_at.saturating_duration_since(now),
            config.debounce,
        ]
        .into_iter()
        .min()
        .unwrap_or(Duration::from_millis(50))
        .min(Duration::from_millis(100));
        match event_rx.recv_timeout(timeout) {
            Ok(Ok(event)) => handle_event(
                event,
                &root,
                &mut queue,
                config.queue_capacity,
                &status,
                &mut force_full,
                &mut reason,
            ),
            Ok(Err(error)) => {
                mark_dirty(&status, error.to_string());
                force_full = true;
                reason = ReconcileReason::NotifyError;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                mark_dirty(&status, "notify event channel closed".into());
                force_full = true;
                reason = ReconcileReason::NotifyError;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        if dropped.swap(false, Ordering::AcqRel) {
            mark_dirty(&status, "notify event queue overflow".into());
            force_full = true;
            reason = ReconcileReason::NotifyError;
        }

        let now = Instant::now();
        if now >= next_branch_check {
            next_branch_check = now + config.branch_poll_interval;
            match read_branch_marker(&root) {
                Ok(current) => {
                    match observed_branch.replace(current.clone()) {
                        Some(previous) if previous != current => {
                            pending_transition = Some(BranchTransition {
                                before: previous,
                                after: current,
                            });
                            rebuild_started = false;
                            force_full = true;
                            reason = ReconcileReason::BranchTransition;
                        }
                        None => match backend.prepare_startup_branch(current.as_ref()) {
                            Ok(Some(transition)) => {
                                pending_transition = Some(transition);
                                rebuild_started = true;
                                force_full = true;
                                reason = ReconcileReason::BranchTransition;
                            }
                            Ok(None) => {}
                            Err(error) => {
                                mark_dirty(&status, error);
                                force_full = true;
                                reason = ReconcileReason::BackendFailure;
                                retry_at = now + config.retry_interval;
                            }
                        },
                        Some(_) => {}
                    }
                }
                Err(error) => {
                    mark_dirty(&status, error.to_string());
                    if let Err(queue_error) = backend.branch_observation_failed() {
                        mark_dirty(&status, queue_error);
                    }
                    force_full = true;
                    reason = ReconcileReason::BranchObservationFailed;
                    retry_at = now + config.retry_interval;
                }
            }
        }
        if now >= next_periodic {
            next_periodic = now + config.periodic_reconcile_interval;
            match backend.current_manifest(&root, config.max_file_bytes) {
                Ok(current) => {
                    let changed = diff_manifests(manifest.as_ref(), &current);
                    if !changed.is_empty() {
                        force_full = true;
                        reason = ReconcileReason::PeriodicManifestMismatch;
                    }
                    manifest = Some(current);
                }
                Err(error) => {
                    mark_dirty(&status, error);
                    force_full = true;
                    reason = ReconcileReason::BackendFailure;
                }
            }
        }
        let mut due = queue.drain_due(now, config.debounce);
        due.sort();
        due.dedup();
        if let Ok(mut current_status) = status.lock() {
            current_status.queued_paths = queue.len();
        }
        if queue.is_dirty() {
            force_full = true;
            reason = ReconcileReason::QueueOverflow;
        }
        if !due.is_empty() && !force_full {
            let request = ReconcileRequest {
                workspace_id: workspace_id.clone(),
                root: root.clone(),
                reason: ReconcileReason::PathChanges,
                full_scan: false,
                changed_paths: due.clone(),
                branch_transition: None,
            };
            if let Err(error) = backend
                .apply_incremental(request.clone())
                .and_then(|_| backend.complete_reconciliation(&request))
            {
                mark_dirty(&status, error);
                force_full = true;
                reason = ReconcileReason::BackendFailure;
                retry_at = now + config.retry_interval;
            } else {
                mark_clean(&status);
            }
        }
        if let Some(transition) = &pending_transition
            && !rebuild_started
            && now >= retry_at
        {
            match backend.begin_rebuild(transition) {
                Ok(()) => rebuild_started = true,
                Err(error) => {
                    mark_dirty(&status, error);
                    retry_at = now + config.retry_interval;
                }
            }
        }
        if force_full && now >= retry_at && pending_transition.as_ref().is_none_or(|_| rebuild_started) {
            let request = ReconcileRequest {
                workspace_id: workspace_id.clone(),
                root: root.clone(),
                reason,
                full_scan: true,
                changed_paths: due.clone(),
                branch_transition: pending_transition.clone(),
            };
            let result = backend
                .reconcile(request.clone())
                .and_then(|_| backend.complete_reconciliation(&request));
            match result {
                Ok(()) => {
                    force_full = false;
                    queue.clear();
                    pending_transition = None;
                    rebuild_started = false;
                    match backend.current_manifest(&root, config.max_file_bytes) {
                        Ok(current) => {
                            manifest = Some(current);
                            mark_clean(&status);
                        }
                        Err(error) => {
                            mark_dirty(&status, error);
                            force_full = true;
                            retry_at = now + config.retry_interval;
                        }
                    }
                }
                Err(error) => {
                    mark_dirty(&status, error);
                    retry_at = now + config.retry_interval;
                }
            }
            if let Ok(mut state) = status.lock() {
                state.full_reconcile_count += 1;
            }
        }
        if !due.is_empty() && force_full {
            for path in due {
                let _ = queue.insert(path, now, config.queue_capacity);
            }
        }
    }
    // A local channel drain cannot fence a notify callback already in flight.
    // Always queue a durable full reconcile before releasing the backend.
    match backend.queue_shutdown_reconciliation(&workspace_id) {
        Ok(()) => {
            if let Ok(mut current_status) = status.lock() {
                current_status.shutdown_reconciliation_error = None;
            }
            mark_dirty(
                &status,
                "watcher stopped with a durable full reconciliation queued".into(),
            );
        }
        Err(error) => {
            if let Ok(mut current_status) = status.lock() {
                current_status.shutdown_reconciliation_error = Some(error.clone());
            }
            mark_dirty(
                &status,
                format!("watcher stopped; could not queue full reconciliation: {error}"),
            );
        }
    }
    if let Ok(mut current_status) = status.lock() {
        current_status.stopping = false;
    }
}

pub(super) fn handle_event(
    event: Event,
    root: &Path,
    queue: &mut CoalescingQueue,
    capacity: usize,
    status: &Mutex<WatchStatus>,
    force_full: &mut bool,
    reason: &mut ReconcileReason,
) {
    if matches!(event.kind, EventKind::Any | EventKind::Other) {
        mark_dirty(status, "unknown notify event".into());
        *force_full = true;
        *reason = ReconcileReason::UnknownNotifyEvent;
        return;
    }
    let tree_change = is_tree_change(event.kind);
    for event_path in event.paths {
        let relative = if event_path.is_absolute() {
            match event_path.strip_prefix(root) {
                Ok(path) => path.to_path_buf(),
                Err(_) => continue,
            }
        } else {
            event_path
        };
        // Recursive watcher registration can report the already-existing root
        // directory. Its existence does not imply a workspace tree change.
        if relative.as_os_str().is_empty() || relative == Path::new(".") {
            continue;
        }
        if relative.components().any(|component| matches!(
            component,
            std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_)
        )) {
            continue;
        }
        if relative == Path::new(".git") || relative.starts_with(".git") {
            continue;
        }
        if tree_change || path_is_directory(root, &relative) {
            mark_dirty(status, "directory or rename event requires a full reconciliation".into());
            *force_full = true;
            *reason = ReconcileReason::TreeStructureChanged;
            continue;
        }
        let is_ignore = relative.file_name().is_some_and(|name| {
            name == ".gitignore" || name == ".workspace-indexignore" || name == ".augmentignore"
        });
        match queue.insert(relative, Instant::now(), capacity) {
            QueueInsert::Overflow => {
                mark_dirty(status, "watcher path queue is full".into());
                *force_full = true;
                *reason = ReconcileReason::QueueOverflow;
            }
            _ if is_ignore => {
                *force_full = true;
                *reason = ReconcileReason::IgnoreRulesChanged;
            }
            _ => {}
        }
    }
}

fn is_tree_change(kind: EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(CreateKind::Any | CreateKind::Folder | CreateKind::Other)
            | EventKind::Remove(RemoveKind::Any | RemoveKind::Folder | RemoveKind::Other)
            | EventKind::Modify(ModifyKind::Name(_))
    )
}

fn path_is_directory(root: &Path, relative: &Path) -> bool {
    std::fs::symlink_metadata(root.join(relative))
        .is_ok_and(|metadata| metadata.file_type().is_dir())
}
pub(super) fn diff_manifests(previous: Option<&Manifest>, current: &Manifest) -> Vec<PathBuf> {
    let Some(previous) = previous else { return current.keys().cloned().collect(); };
    previous.iter().filter(|(path, hash)| current.get(*path) != Some(*hash)).map(|(path, _)| path.clone()).chain(current.iter().filter(|(path, hash)| previous.get(*path) != Some(*hash)).map(|(path, _)| path.clone())).collect::<std::collections::BTreeSet<_>>().into_iter().collect()
}
fn mark_dirty(status: &Mutex<WatchStatus>, error: String) { if let Ok(mut value) = status.lock() { value.dirty = true; value.last_error = Some(error); } }
fn mark_clean(status: &Mutex<WatchStatus>) { if let Ok(mut value) = status.lock() { value.dirty = false; value.last_error = None; value.last_reconciled_at = Some(SystemTime::now()); value.queued_paths = 0; } }
