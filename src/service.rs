const PROFILE_FINGERPRINT_VERSION: &str = "profile-v1";

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::changeset::{
    ChangeOperation, ChangesetError, ChangesetOutcome, ChangesetService, PreparedChangeSet,
};
use crate::config::{Config, EmbeddingConfig, EmbeddingMetric, EmbeddingProviderKind, Paths};
use crate::index::adapter::EmbeddingProvider;
use crate::index::embedding::{LocalEmbedder, RemoteEmbedder};
use crate::index::pipeline::IndexReport;
use crate::index::profiles::ProfileManager;
use crate::index::reconcile::{
    existing_snapshot_for_profiles, persist_workspace, prepare_workspace_profiles, ProfileProvider,
    WorkspacePreparation,
};
use crate::index::sequencer::{Sequencer, VectorWatermark};
use crate::model::{ErrorCode, IndexState, Page, ToolResult, Warning, WorkspaceMode};
use crate::retrieval::search::{
    CodeSearchRequest, SearchResult, SearchResultCache, SearchService, SearchServiceOptions,
    SearchExecution, SemanticSnapshotCache, SymbolSearchRequest, SymbolSearchResult,
};
use crate::security::{WorkspaceFs, file_read};
use crate::state::schema;
use crate::state::{OwnerLock, RegistryOptions, StateError, WorkspaceRecord, WorkspaceRegistry};
use crate::watch::{BranchTransition, ReconcileRequest, WatchIndexBackend};

type SemanticQueryMap = HashMap<String, (String, Arc<Vec<f32>>)>;

type GraphHmac = Hmac<Sha256>;

const LEGACY_FASTEMBED_PROFILE_ID: &str = "local-fastembed";
const SEARCH_REFRESH_TIMEOUT_SECONDS: u64 = 10;
const MAX_TRACKED_INDEX_WORKERS: usize = 1;

#[derive(Default)]
struct IndexWorkerRegistry {
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl IndexWorkerRegistry {
    fn run_until_deadline<T, F>(&self, deadline: Instant, task: F) -> Option<Result<T, ServiceError>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, ServiceError> + Send + 'static,
    {
        if deadline.saturating_duration_since(Instant::now()).is_zero() {
            return None;
        }
        self.reap_finished();
        let receiver = self.spawn(task)?;
        let timeout = deadline.saturating_duration_since(Instant::now());
        if timeout.is_zero() {
            return None;
        }
        let result = match receiver.recv_timeout(timeout) {
            Ok(result) => Some(result),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => Some(Err(ServiceError::new(
                ErrorCode::Conflict,
                "index refresh worker stopped before returning a result",
                true,
            ))),
        };
        self.reap_finished();
        result
    }

    fn spawn<T, F>(&self, task: F) -> Option<mpsc::Receiver<Result<T, ServiceError>>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, ServiceError> + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut workers = self.workers.lock();
        if workers.len() >= MAX_TRACKED_INDEX_WORKERS {
            return None;
        }
        let worker = thread::Builder::new()
            .name("workspace-index-refresh-retry".into())
            .spawn(move || {
                let _ = sender.send(task());
            })
            .ok()?;
        workers.push(worker);
        Some(receiver)
    }

    fn reap_finished(&self) {
        let finished = {
            let mut workers = self.workers.lock();
            let mut finished = Vec::new();
            let mut active = Vec::with_capacity(workers.len());
            for worker in workers.drain(..) {
                if worker.is_finished() {
                    finished.push(worker);
                } else {
                    active.push(worker);
                }
            }
            *workers = active;
            finished
        };
        for worker in finished {
            if worker.join().is_err() {
                eprintln!("workspace-index: tracked refresh/retry worker panicked");
            }
        }
    }

    #[cfg(test)]
    fn tracked_workers(&self) -> usize {
        self.reap_finished();
        self.workers.lock().len()
    }
}

const MAX_SYNCHRONOUS_STALE_PATHS: usize = 3;
/// Bounds search text to the smaller configured source-file and MCP payload limits.
fn max_search_query_bytes(config: &Config) -> usize {
    let max_tool_bytes = u64::try_from(config.limits.max_tool_bytes).unwrap_or(u64::MAX);
    let max_bytes = config.limits.max_file_bytes.min(max_tool_bytes);
    usize::try_from(max_bytes).unwrap_or(usize::MAX)
}
fn stale_search_worker_warning() -> Warning {
    Warning {
        code: "STALE_REFRESH_PENDING".into(),
        message: format!(
            "Stale refresh/retry used only the remaining {SEARCH_REFRESH_TIMEOUT_SECONDS}-second search budget. A timed-out synchronous provider/filesystem call may keep running; at most {MAX_TRACKED_INDEX_WORKERS} refresh/retry worker is tracked per service, and further workers are denied until it exits. Stale paths remain in durable index jobs."
        ),
        retryable: false,
    }
}
fn stale_search_worker_failure_warning() -> Warning {
    Warning {
        code: "INDEX_DEGRADED".into(),
        message: "Stale-candidate refresh or retry failed; remaining paths stay in durable index jobs for later reconciliation.".into(),
        retryable: false,
    }
}
#[derive(Clone, Debug)]
struct ProfileIdentity {
    profile_id: String,
    provider_kind: String,
    endpoint: Option<String>,
    model: String,
    revision: Option<String>,
    dimension: usize,
    metric: String,
    api_key_env: Option<String>,
    query_template_version: String,
    document_template_version: String,
    batch_size: usize,
    timeout_seconds: u64,
    retry_delays_ms: Vec<u64>,
}

struct IndexGenerationStart {
    generation: u64,
    claimed_jobs: Vec<(i64, String)>,
}

struct SemanticQueryVectors {
    queries: SemanticQueryMap,
    unavailable: bool,
}

struct ProviderSlot {
    fingerprint: String,
    dimension: usize,
    provider: Mutex<Box<dyn EmbeddingProvider + Send>>,
    admission: Arc<Mutex<()>>,
}

impl EmbeddingProvider for ProviderSlot {
    fn fingerprint(&self) -> String { self.fingerprint.clone() }

    fn dimension(&self) -> usize { self.dimension }

    fn batch_size(&self) -> usize { self.provider.lock().batch_size() }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let _admission = self.admission.lock();
        self.provider.lock().embed_batch(texts)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphCursorPayload {
    metadata: crate::index::graphify::GraphSnapshotMetadata,
    graph_fingerprint: String,
    limit: usize,
    offset: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphSnapshotKey {
    metadata: crate::index::graphify::GraphSnapshotMetadata,
    fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WorkspaceRegistrationIdentity {
    workspace_id: String,
    root_path: PathBuf,
    root_device: u64,
    root_inode: u64,
    registration_epoch: u64,
}

impl WorkspaceRegistrationIdentity {
    fn graph_metadata(&self, visible_generation: u64) -> crate::index::graphify::GraphSnapshotMetadata {
        crate::index::graphify::GraphSnapshotMetadata {
            workspace_id: self.workspace_id.clone(),
            registration_epoch: self.registration_epoch,
            root_device: self.root_device,
            root_inode: self.root_inode,
            visible_generation,
        }
    }
}
#[derive(Debug)]
struct WorkspaceGraphStateRow {
    root_path: String,
    device: i64,
    inode: i64,
    registration_epoch: i64,
    visible_generation: Option<i64>,
    branch_transition: Option<bool>,
}

#[derive(Debug)]
struct GraphPublicationRow {
    root_path: String,
    device: i64,
    inode: i64,
    registration_epoch: i64,
    visible_generation: i64,
    branch_transition: bool,
}

struct GraphSnapshot {
    metadata: crate::index::graphify::GraphSnapshotMetadata,
    fingerprint: String,
    nodes: Vec<crate::index::graphify::GraphNode>,
    edges: Vec<crate::index::graphify::GraphEdge>,
    content_hashes: BTreeMap<String, String>,
    content_bytes: usize,
}

#[derive(Default)]
struct GraphSnapshotCache {
    snapshots: HashMap<GraphSnapshotKey, Arc<GraphSnapshot>>,
    content_bytes: usize,
}



#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct ServiceError {
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
}

impl ServiceError {
    fn new(code: ErrorCode, message: impl Into<String>, retryable: bool) -> Self {
        Self { code, message: message.into(), retryable }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct WorkspaceIndexStatus {
    pub workspace_id: String,
    pub state: IndexState,
    pub visible_generation: u64,
    pub building_generation: u64,
    pub dirty: bool,
    pub health: String,
    pub registration_epoch: u64,
    pub mode_epoch: u64,
    pub manifest_version: String,
    pub graph_generation: Option<u64>,
    pub file_count: u64,
    pub chunk_count: u64,
    pub queued_jobs: u64,
    pub running_jobs: u64,
    pub active_embedding_profile: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceManifestEntry {
    pub path: String,
    pub blake3: String,
    pub generation: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceManifest {
    pub workspace_id: String,
    pub visible_generation: u64,
    pub manifest_version: String,
    pub entries: Vec<WorkspaceManifestEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceGraphPage {
    pub workspace_id: String,
    pub visible_generation: u64,
    pub nodes: Vec<crate::index::graphify::GraphNode>,
    pub edges: Vec<crate::index::graphify::GraphEdge>,
    pub truncated: bool,
    pub nodes_truncated: bool,
    pub edges_truncated: bool,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    pub ok: bool,
    pub schema_version: i32,
    pub workspaces: i64,
    pub payload_cleanup_pending: usize,
    pub payload_cleanup_unresolved: usize,
}
pub struct WorkspaceService {
    paths: Paths,
    config: Config,
    private_data_root: PathBuf,
    connection: Arc<Mutex<Connection>>,
    cursor_secret: Vec<u8>,
    profile_identity: ProfileIdentity,
    provider_slots: Arc<Mutex<HashMap<String, Arc<ProviderSlot>>>>,
    provider_gate: Arc<Mutex<()>>,
    root_locks: Arc<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>>,
    graph_snapshots: Arc<Mutex<GraphSnapshotCache>>,
    graphify_failures: Arc<Mutex<HashSet<crate::index::graphify::GraphSnapshotMetadata>>>,
    search_result_sets: Arc<SearchResultCache>,
    semantic_snapshots: Arc<SemanticSnapshotCache>,
    index_workers: Arc<IndexWorkerRegistry>,
    _owner_lock: Arc<OwnerLock>,
}
impl Clone for WorkspaceService {
    fn clone(&self) -> Self {
        Self {
            paths: self.paths.clone(),
            config: self.config.clone(),
            private_data_root: self.private_data_root.clone(),
            connection: Arc::clone(&self.connection),
            cursor_secret: self.cursor_secret.clone(),
            profile_identity: self.profile_identity.clone(),
            provider_slots: Arc::clone(&self.provider_slots),
            provider_gate: Arc::clone(&self.provider_gate),
            root_locks: Arc::clone(&self.root_locks),
            graph_snapshots: Arc::clone(&self.graph_snapshots),
            graphify_failures: Arc::clone(&self.graphify_failures),
            search_result_sets: Arc::clone(&self.search_result_sets),
            semantic_snapshots: Arc::clone(&self.semantic_snapshots),
            index_workers: Arc::clone(&self.index_workers),
            _owner_lock: Arc::clone(&self._owner_lock),
        }
    }
}


impl WorkspaceService {
    pub fn open(paths: Paths, config: Config) -> Result<Self, ServiceError> {
        Self::open_inner(paths, config, None)
    }

    fn open_inner(
        paths: Paths,
        config: Config,
        injected_embedder: Option<Box<dyn EmbeddingProvider + Send>>,
    ) -> Result<Self, ServiceError> {
        validate_config(&config)?;
        paths.create_private_dirs().map_err(|_| {
            ServiceError::new(ErrorCode::Conflict, "workspace state directories are unavailable", false)
        })?;
        let private_data_root = paths.data_dir.canonicalize().map_err(|_| {
            ServiceError::new(ErrorCode::Conflict, "workspace state directory is unavailable", false)
        })?;
        let owner_lock = OwnerLock::acquire(&paths.data_dir).map_err(map_state_error)?;
        let mut connection = Connection::open(&paths.database).map_err(|_| {
            ServiceError::new(ErrorCode::Conflict, "workspace index database is unavailable", true)
        })?;
        
        schema::initialize(&mut connection).map_err(|_| {
            ServiceError::new(ErrorCode::Conflict, "workspace index database could not be initialized", false)
        })?;
        let connection = Arc::new(Mutex::new(connection));
        ChangesetService::with_limits(
            &connection,
            &paths.pending_dir,
            config.limits.changeset_expiry_seconds,
            config.limits.max_file_bytes,
        )
        .map_err(map_changeset_error)?;
        let cursor_secret = load_cursor_secret(&paths.data_dir.join("cursor.key")).map_err(|_| {
            ServiceError::new(ErrorCode::Conflict, "workspace cursor secret is unavailable", false)
        })?;
        let provider_gate = Arc::new(Mutex::new(()));
        let (profile_identity, provider, adapter_fingerprint, dimension) = match injected_embedder {
            Some(provider) => {
                let fingerprint = provider.fingerprint();
                let dimension = provider.dimension();
                let identity = profile_identity(&config.embedding, &fingerprint, dimension, None)?;
                (identity, provider, fingerprint, dimension)
            }
            None => build_configured_provider(&config.embedding)?,
        };
        let slot = Arc::new(ProviderSlot {
            fingerprint: adapter_fingerprint,
            dimension,
            provider: Mutex::new(provider),
            admission: Arc::clone(&provider_gate),
        });
        let mut provider_slots = HashMap::new();
        provider_slots.insert(profile_identity.profile_id.clone(), slot);
        let service = Self {
            paths,
            config,
            private_data_root,
            connection,
            cursor_secret,
            profile_identity,
            provider_slots: Arc::new(Mutex::new(provider_slots)),
            provider_gate,
            root_locks: Arc::new(Mutex::new(HashMap::new())),
            graph_snapshots: Arc::new(Mutex::new(GraphSnapshotCache::default())),
            graphify_failures: Arc::new(Mutex::new(HashSet::new())),
            search_result_sets: Arc::new(SearchResultCache::default()),
            semantic_snapshots: Arc::new(SemanticSnapshotCache::default()),
            index_workers: Arc::new(IndexWorkerRegistry::default()),
            _owner_lock: Arc::new(owner_lock),
        };
        service.initialize_profiles_and_recover()?;
        service.detect_offline_branch_transitions()?;
        Ok(service)
    }

    #[cfg(test)]
    pub(crate) fn open_with_provider(
        paths: Paths,
        config: Config,
        provider: Box<dyn EmbeddingProvider + Send>,
    ) -> Result<Self, ServiceError> {
        Self::open_inner(paths, config, Some(provider))
    }

    pub fn config(&self) -> &Config { &self.config }

    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceRecord>, ServiceError> {
        
        
        let mut guard = self.connection.lock();
        let registry = WorkspaceRegistry::from_connection(
            &mut guard,
            RegistryOptions { allow_non_git_roots: self.config.allow_non_git_roots },
        );
        registry.list().map_err(map_state_error)
    }

    pub fn register_workspace(&self, id: &str, root: &Path) -> Result<WorkspaceRecord, ServiceError> {
        
        let mut guard = self.connection.lock();
        let mut registry = WorkspaceRegistry::from_connection(
            &mut guard,
            RegistryOptions { allow_non_git_roots: self.config.allow_non_git_roots },
        );
        registry.register(id, root).map_err(map_state_error)
    }

    pub fn set_workspace_mode(&self, id: &str, mode: WorkspaceMode) -> Result<WorkspaceRecord, ServiceError> {
        
        let mut guard = self.connection.lock();
        let mut registry = WorkspaceRegistry::from_connection(
            &mut guard,
            RegistryOptions { allow_non_git_roots: self.config.allow_non_git_roots },
        );
        registry.set_mode(id, mode).map_err(map_state_error)
    }

    pub fn unregister_workspace(&self, id: &str) -> Result<(), ServiceError> {
        let registration = self.workspace_registration_identity(id)?;
        let root_lock = self.root_lock(&registration.root_path);
        let _root_guard = root_lock.lock();
        if self.workspace_registration_identity(id)? != registration {
            return Err(ServiceError::new(ErrorCode::Conflict, "workspace registration changed during unregister", true));
        }
        {
            let mut guard = self.connection.lock();
            let mut registry = WorkspaceRegistry::from_connection(
                &mut guard,
                RegistryOptions { allow_non_git_roots: self.config.allow_non_git_roots },
            );
            registry.unregister_for_root(id, &registration.root_path).map_err(map_state_error)?;
        }
        self.invalidate_graph_registration(id, registration.registration_epoch);
        crate::index::graphify::remove_registration_graphs(
            &self.private_data_root,
            id,
            registration.registration_epoch,
        );
        match ChangesetService::for_service(
            &self.connection,
            &self.paths.pending_dir,
            self.config.limits.changeset_expiry_seconds,
            self.config.limits.max_file_bytes,
        ).and_then(|service| service.retry_payload_cleanup()) {
            Ok(0) => {}
            Ok(_) | Err(_) => eprintln!("workspace-index: unregister committed; private payload cleanup remains durably queued"),
        }
        Ok(())
    }
    fn workspace_registration_identity(
        &self,
        workspace_id: &str,
    ) -> Result<WorkspaceRegistrationIdentity, ServiceError> {
        let registration = self.connection.lock().query_row(
            "SELECT root_path, device_id, inode, registration_epoch FROM workspaces WHERE id=?1",
            [workspace_id],
            |row| Ok(WorkspaceRegistrationIdentity {
                workspace_id: workspace_id.to_owned(),
                root_path: PathBuf::from(row.get::<_, String>(0)?),
                root_device: row.get::<_, i64>(1)? as u64,
                root_inode: row.get::<_, i64>(2)? as u64,
                registration_epoch: row.get::<_, i64>(3)?.max(0) as u64,
            }),
        ).optional().map_err(map_database_error)?.ok_or_else(|| {
            ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
        })?;
        Ok(registration)
    }

    pub fn resolve_workspace(&self, requested: Option<&str>) -> Result<String, ServiceError> {
        let connection = self.connection.lock();
        if let Some(id) = requested {
            let exists: bool = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM workspaces WHERE id=?1)", [id], |row| row.get(0),
            ).map_err(map_database_error)?;
            return exists.then(|| id.to_owned()).ok_or_else(|| {
                ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
            });
        }
        let mut statement = connection.prepare("SELECT id FROM workspaces ORDER BY id LIMIT 2")
            .map_err(map_database_error)?;
        let ids = statement.query_map([], |row| row.get::<_, String>(0))
            .map_err(map_database_error)?.collect::<Result<Vec<_>, _>>()
            .map_err(|_| ServiceError::new(ErrorCode::Conflict, "workspace lookup failed", false))?;
        match ids.as_slice() {
            [id] => Ok(id.clone()),
            [] => Err(ServiceError::new(ErrorCode::WorkspaceRequired, "no workspace is registered", false)),
            _ => Err(ServiceError::new(ErrorCode::WorkspaceRequired, "workspace_id is required when multiple workspaces are registered", false)),
        }
    }

    pub fn validate_workspace_ids(&self, mut ids: Vec<String>) -> Result<Vec<String>, ServiceError> {
        if ids.is_empty() { ids.push(self.resolve_workspace(None)?); }
        ids.sort();
        ids.dedup();
        let connection = self.connection.lock();
        for id in &ids {
            let exists: bool = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM workspaces WHERE id=?1)", [id], |row| row.get(0),
            ).map_err(map_database_error)?;
            if !exists {
                return Err(ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false));
            }
        }
        Ok(ids)
    }

    pub fn workspace_root(&self, workspace_id: &str) -> Result<PathBuf, ServiceError> {
        self.connection.lock().query_row(
            "SELECT root_path FROM workspaces WHERE id=?1", [workspace_id], |row| row.get::<_, String>(0),
        ).map(PathBuf::from).map_err(|error| {
            if error == rusqlite::Error::QueryReturnedNoRows {
                ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
            } else { map_database_error(error) }
        })
    }

    fn workspace_capability(&self, workspace_id: &str) -> Result<WorkspaceFs, ServiceError> {
        let identity = self.connection.lock().query_row(
            "SELECT root_path, device_id, inode FROM workspaces WHERE id=?1",
            [workspace_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
        ).optional().map_err(map_database_error)?.ok_or_else(|| {
            ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
        })?;
        WorkspaceFs::open_registered(Path::new(&identity.0), identity.1 as u64, identity.2 as u64)
            .map_err(|error| match error {
                crate::security::workspace_fs::WorkspaceFsError::Unsupported =>
                    ServiceError::new(ErrorCode::Unsupported, "safe workspace filesystem capability is unavailable", false),
                crate::security::workspace_fs::WorkspaceFsError::Conflict =>
                    ServiceError::new(ErrorCode::Conflict, "registered workspace root identity changed", false),
                _ => ServiceError::new(ErrorCode::Conflict, "workspace filesystem capability is unavailable", false),
            })
    }

    fn search_capabilities(
        &self,
        workspace_ids: &[String],
    ) -> Result<HashMap<String, WorkspaceFs>, ServiceError> {
        let mut capabilities = HashMap::with_capacity(workspace_ids.len());
        for workspace_id in workspace_ids {
            match self.workspace_capability(workspace_id) {
                Ok(workspace) => { capabilities.insert(workspace_id.clone(), workspace); }
                Err(error) if error.code == ErrorCode::WorkspaceNotFound => {}
                Err(error) => return Err(error),
            }
        }
        Ok(capabilities)
    }

    fn queue_full_reconciliation(&self, workspace_id: &str) -> Result<(), ServiceError> {
        
        
        let mut guard = self.connection.lock();
        let transaction = guard.transaction().map_err(map_database_error)?;
        transaction.execute(
            "INSERT INTO workspace_index_state
                (workspace_id, generation, updated_at, building_generation, visible_generation, dirty, health)
             VALUES (?1, 0, ?2, 0, 0, 1, 'reconcile_required')
             ON CONFLICT(workspace_id) DO UPDATE SET
                dirty=1, health='reconcile_required', updated_at=excluded.updated_at",
            params![workspace_id, unix_seconds() as i64],
        ).map_err(map_database_error)?;
        transaction.execute(
            "INSERT OR IGNORE INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
             VALUES (?1, '', 0, 'queued', ?2)",
            params![workspace_id, unix_seconds() as i64],
        ).map_err(map_database_error)?;
        transaction.commit().map_err(map_database_error)
    }
    fn full_reconcile_pending(&self, workspace_id: &str) -> Result<bool, ServiceError> {
        self.connection.lock().query_row(
            "SELECT EXISTS (
                SELECT 1 FROM index_jobs
                WHERE workspace_id=?1 AND relative_path='' AND state='queued'
             )",
            [workspace_id],
            |row| row.get(0),
        ).map_err(map_database_error)
    }

    fn watcher_manifest(
        &self,
        workspace_id: &str,
        expected_root: &Path,
        max_file_bytes: u64,
    ) -> Result<crate::watch::Manifest, ServiceError> {
        let workspace = match self.workspace_capability(workspace_id) {
            Ok(workspace) => workspace,
            Err(error) => {
                self.queue_full_reconciliation(workspace_id)?;
                return Err(error);
            }
        };
        if workspace.root_path() != expected_root {
            self.queue_full_reconciliation(workspace_id)?;
            return Err(ServiceError::new(ErrorCode::Conflict, "registered workspace root changed", true));
        }
        match crate::watch::build_manifest(&workspace, &self.private_data_root, max_file_bytes) {
            Ok(manifest) => Ok(manifest),
            Err(_) => {
                self.queue_full_reconciliation(workspace_id)?;
                Err(ServiceError::new(ErrorCode::Conflict, "workspace manifest could not be read safely", true))
            }
        }
    }

    pub fn index_status(&self, workspace_id: &str) -> Result<WorkspaceIndexStatus, ServiceError> {
        let connection = self.connection.lock();
        let epochs: Option<(i64, i64, i64, i64)> = connection.query_row(
            "SELECT registration_epoch, mode_epoch, device_id, inode
             FROM workspaces WHERE id=?1",
            [workspace_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        ).optional().map_err(map_database_error)?;
        let (registration_epoch, mode_epoch, root_device, root_inode) = epochs.ok_or_else(|| {
            ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
        })?;
        let state_row: Option<(i64, i64, i64, String, Option<String>)> = connection.query_row(
            "SELECT visible_generation, dirty, building_generation, health, active_profile_id
             FROM workspace_index_state WHERE workspace_id=?1",
            [workspace_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        ).optional().map_err(map_database_error)?;
        let (visible, dirty, building, health, state_profile) =
            state_row.unwrap_or((0, 0, 0, "unknown".to_owned(), None));
        let visible_generation = visible.max(0) as u64;
        let entries = read_manifest_entries(&connection, workspace_id)?;
        let manifest_version = manifest_version(&entries, visible_generation);
        let graph_metadata = crate::index::graphify::GraphSnapshotMetadata {
            workspace_id: workspace_id.to_owned(),
            registration_epoch: registration_epoch.max(0) as u64,
            root_device: root_device as u64,
            root_inode: root_inode as u64,
            visible_generation,
        };
        let graph_generation = if self.graphify_failures.lock().contains(&graph_metadata) {
            None
        } else {
            crate::index::graphify::graph_snapshot_path(&self.private_data_root, &graph_metadata)
                .and_then(|path| read_graph_snapshot(&self.graph_snapshots, &graph_metadata, &path).ok())
                .map(|snapshot| snapshot.metadata.visible_generation)
        };
        let file_count = count_for_workspace(&connection, "files", workspace_id)?;
        let chunk_count = count_for_workspace(&connection, "chunks", workspace_id)?;
        let queued_jobs = connection.query_row(
            "SELECT count(*) FROM index_jobs WHERE workspace_id=?1 AND state='queued'", [workspace_id],
            |row| row.get::<_, i64>(0),
        ).map_err(map_database_error)?.max(0) as u64;
        let running_jobs = connection.query_row(
            "SELECT count(*) FROM index_jobs WHERE workspace_id=?1 AND state='running'", [workspace_id],
            |row| row.get::<_, i64>(0),
        ).map_err(map_database_error)?.max(0) as u64;
        let active_profile = state_profile;
        let profile_degraded = match &active_profile {
            Some(profile) => connection.query_row(
                "SELECT backfill_required != 0 OR applied_generation != ?3
                 FROM profile_applied_generations WHERE workspace_id=?1 AND profile_id=?2",
                params![workspace_id, profile, visible_generation as i64],
                |row| row.get::<_, bool>(0),
            ).optional().map_err(map_database_error)?.unwrap_or(true),
            None => true,
        };
        let state = if dirty != 0 || building.max(0) as u64 > visible_generation { IndexState::Reconciling }
            else if visible_generation == 0 { IndexState::Unindexed }
            else if queued_jobs > 0 || running_jobs > 0 || profile_degraded || health != "ready" { IndexState::Degraded }
            else { IndexState::Ready };
        Ok(WorkspaceIndexStatus {
            workspace_id: workspace_id.to_owned(), state, visible_generation,
            building_generation: building.max(0) as u64, dirty: dirty != 0, health,
            registration_epoch: registration_epoch.max(0) as u64,
            mode_epoch: mode_epoch.max(0) as u64, manifest_version, graph_generation,
            file_count, chunk_count, queued_jobs, running_jobs, active_embedding_profile: active_profile,
        })
    }

    pub fn workspace_manifest(&self, workspace_id: &str) -> Result<WorkspaceManifest, ServiceError> {
        let connection = self.connection.lock();
        let generation = connection.query_row(
            "SELECT visible_generation FROM workspace_index_state WHERE workspace_id=?1",
            [workspace_id],
            |row| row.get::<_, i64>(0),
        ).optional().map_err(map_database_error)?.ok_or_else(|| {
            ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
        })?.max(0) as u64;
        let entries = read_manifest_entries(&connection, workspace_id)?;
        Ok(WorkspaceManifest {
            workspace_id: workspace_id.to_owned(),
            visible_generation: generation,
            manifest_version: manifest_version(&entries, generation),
            entries,
        })
    }

    pub fn doctor_report(&self) -> Result<DoctorReport, ServiceError> {
        schema::initialize(&mut self.connection.lock()).map_err(|_| {
            ServiceError::new(ErrorCode::Conflict, "workspace index database could not be checked", false)
        })?;
        let payload_cleanup_pending = match ChangesetService::for_service(
            &self.connection,
            &self.paths.pending_dir,
            self.config.limits.changeset_expiry_seconds,
            self.config.limits.max_file_bytes,
        ) {
            Ok(service) => {
                service.enqueue_terminal_payloads().map_err(map_changeset_error)?;
                service.expire_prepared_payloads().map_err(map_changeset_error)?;
                service.retry_payload_cleanup().map_err(map_changeset_error)?
            }
            Err(_) => mark_pending_cleanup_directory_unavailable(&self.connection)?,
        };
        let workspaces = self.connection.lock().query_row("SELECT count(*) FROM workspaces", [], |row| row.get(0))
            .map_err(map_database_error)?;
        let payload_cleanup_unresolved: i64 = self.connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup WHERE last_error IS NOT NULL",
            [],
            |row| row.get(0),
        ).map_err(map_database_error)?;
        Ok(DoctorReport {
            ok: payload_cleanup_pending == 0 && payload_cleanup_unresolved == 0,
            schema_version: schema::CURRENT_SCHEMA_VERSION,
            workspaces,
            payload_cleanup_pending,
            payload_cleanup_unresolved: usize::try_from(payload_cleanup_unresolved)
                .map_err(|_| ServiceError::new(ErrorCode::Conflict, "cleanup failure count is invalid", false))?,
        })
    }

    pub fn index_generation(&self, workspace_id: &str) -> Result<u64, ServiceError> {
        self.connection.lock().query_row(
            "SELECT visible_generation FROM workspace_index_state WHERE workspace_id=?1", [workspace_id],
            |row| row.get::<_, i64>(0),
        ).optional().map(|generation| generation.unwrap_or(0).max(0) as u64).map_err(map_database_error)
    }
    fn graph_state(
        &self,
        workspace_id: &str,
    ) -> Result<(WorkspaceRegistrationIdentity, u64, bool), ServiceError> {
        let row: Option<WorkspaceGraphStateRow> = self.connection.lock()
            .query_row(
                "SELECT w.root_path, w.device_id, w.inode, w.registration_epoch,
                        s.visible_generation, s.branch_transition
                 FROM workspaces w
                 LEFT JOIN workspace_index_state s ON s.workspace_id=w.id
                 WHERE w.id=?1",
                [workspace_id],
                |row| Ok(WorkspaceGraphStateRow {
                    root_path: row.get(0)?,
                    device: row.get(1)?,
                    inode: row.get(2)?,
                    registration_epoch: row.get(3)?,
                    visible_generation: row.get(4)?,
                    branch_transition: row.get(5)?,
                }),
            )
            .optional()
            .map_err(map_database_error)?;
        let row = row.ok_or_else(|| {
            ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
        })?;
        Ok((
            WorkspaceRegistrationIdentity {
                workspace_id: workspace_id.to_owned(),
                root_path: PathBuf::from(row.root_path),
                root_device: row.device as u64,
                root_inode: row.inode as u64,
                registration_epoch: row.registration_epoch.max(0) as u64,
            },
            row.visible_generation.unwrap_or_default().max(0) as u64,
            row.branch_transition.unwrap_or(false),
        ))
    }

    fn graph_publication_is_current(
        &self,
        registration: &WorkspaceRegistrationIdentity,
        generation: u64,
    ) -> Result<bool, ServiceError> {
        let current: Option<GraphPublicationRow> = self.connection.lock()
            .query_row(
                "SELECT w.root_path, w.device_id, w.inode, w.registration_epoch,
                        s.visible_generation, s.branch_transition
                 FROM workspaces w
                 JOIN workspace_index_state s ON s.workspace_id=w.id
                 WHERE w.id=?1",
                [&registration.workspace_id],
                |row| Ok(GraphPublicationRow {
                    root_path: row.get(0)?,
                    device: row.get(1)?,
                    inode: row.get(2)?,
                    registration_epoch: row.get(3)?,
                    visible_generation: row.get(4)?,
                    branch_transition: row.get(5)?,
                }),
            )
            .optional()
            .map_err(map_database_error)?;
        Ok(current.is_some_and(|current| {
            Path::new(&current.root_path) == registration.root_path
                && current.device as u64 == registration.root_device
                && current.inode as u64 == registration.root_inode
                && current.registration_epoch.max(0) as u64 == registration.registration_epoch
                && current.visible_generation.max(0) as u64 == generation
                && !current.branch_transition
        }))
    }

    fn invalidate_graph_registration(&self, workspace_id: &str, registration_epoch: u64) {
        let mut cache = self.graph_snapshots.lock();
        let obsolete = cache.snapshots.keys()
            .filter(|key| key.metadata.workspace_id == workspace_id
                && key.metadata.registration_epoch == registration_epoch)
            .cloned()
            .collect::<Vec<_>>();
        for key in obsolete {
            if let Some(snapshot) = cache.snapshots.remove(&key) {
                cache.content_bytes = cache.content_bytes.saturating_sub(snapshot.content_bytes);
            }
        }
        self.graphify_failures.lock().retain(|metadata| {
            metadata.workspace_id != workspace_id || metadata.registration_epoch != registration_epoch
        });
    }

    fn invalidate_graph_snapshot(&self, metadata: &crate::index::graphify::GraphSnapshotMetadata) {
        let mut cache = self.graph_snapshots.lock();
        let obsolete = cache.snapshots.keys()
            .filter(|key| key.metadata == *metadata)
            .cloned()
            .collect::<Vec<_>>();
        for key in obsolete {
            if let Some(snapshot) = cache.snapshots.remove(&key) {
                cache.content_bytes = cache.content_bytes.saturating_sub(snapshot.content_bytes);
            }
        }
    }

    fn graph_unavailable_warning(&self) -> Warning {
        Warning {
            code: "GRAPH_UNAVAILABLE".into(),
            message: "Graphify was requested but unavailable; no current graph is published.".into(),
            retryable: false,
        }
    }

    fn mark_graph_failure(&self, metadata: &crate::index::graphify::GraphSnapshotMetadata) {
        let same_registration = |candidate: &crate::index::graphify::GraphSnapshotMetadata| {
            candidate.workspace_id == metadata.workspace_id
                && candidate.registration_epoch == metadata.registration_epoch
                && candidate.root_device == metadata.root_device
                && candidate.root_inode == metadata.root_inode
        };
        let mut failures = self.graphify_failures.lock();
        let newest = failures.iter().filter(|candidate| same_registration(candidate))
            .map(|candidate| candidate.visible_generation).max();
        if newest.is_some_and(|generation| generation > metadata.visible_generation) {
            return;
        }
        failures.retain(|candidate| {
            !same_registration(candidate) || candidate.visible_generation >= metadata.visible_generation
        });
        failures.insert(metadata.clone());
    }

    fn record_graph_failure(&self, metadata: &crate::index::graphify::GraphSnapshotMetadata) {
        self.mark_graph_failure(metadata);
        self.invalidate_graph_snapshot(metadata);
        crate::index::graphify::remove_graph_snapshot(&self.private_data_root, metadata);
    }
    fn finish_graph_extraction(
        &self,
        root_lock: &Mutex<()>,
        registration: &WorkspaceRegistrationIdentity,
        generation: u64,
        extraction: Result<crate::index::graphify::ExtractedGraph, String>,
    ) -> bool {
        let _publication_guard = root_lock.lock();
        if !self.graph_publication_is_current(registration, generation).unwrap_or(false) {
            return false;
        }
        let metadata = registration.graph_metadata(generation);
        self.invalidate_graph_snapshot(&metadata);
        match extraction.and_then(|graph| {
            crate::index::graphify::publish_graph(&self.private_data_root, &metadata, graph)
        }) {
            Ok(()) => {
                self.graphify_failures.lock().retain(|previous| {
                    previous.workspace_id != metadata.workspace_id
                        || previous.registration_epoch != metadata.registration_epoch
                });
            }
            Err(_) => {
                self.record_graph_failure(&metadata);
                eprintln!("GRAPH_UNAVAILABLE: optional graph extraction did not publish a current snapshot");
            }
        }
        true
    }
    pub fn graphify_warning(&self, workspace_id: &str) -> Option<Warning> {
        let (registration, generation, _) = self.graph_state(workspace_id).ok()?;
        let metadata = registration.graph_metadata(generation);
        self.graphify_failures.lock().contains(&metadata)
            .then(|| self.graph_unavailable_warning())
    }

    pub fn reconcile(&self, workspace_id: &str, full: bool, graphify: bool) -> Result<IndexReport, ServiceError> {
        self.reconcile_with_paths(workspace_id, full, &[], graphify)
    }

    pub fn refresh_paths(&self, workspace_id: &str, paths: &[String]) -> Result<IndexReport, ServiceError> {
        self.reconcile_with_paths(workspace_id, false, paths, false)
    }

    fn reconcile_with_paths(
        &self,
        workspace_id: &str,
        force_all: bool,
        force_paths: &[String],
        force_graphify: bool,
    ) -> Result<IndexReport, ServiceError> {
        let registration = self.workspace_registration_identity(workspace_id)?;
        let root = registration.root_path.clone();
        let root_lock = self.root_lock(&root);
        let root_guard = root_lock.lock();
        if self.workspace_registration_identity(workspace_id)? != registration {
            return Err(ServiceError::new(ErrorCode::Conflict, "workspace registration changed during reconciliation", true));
        }
        let workspace = match self.workspace_capability(workspace_id) {
            Ok(workspace) if workspace.root_path() == root => workspace,
            Ok(_) => return Err(ServiceError::new(ErrorCode::Conflict, "registered workspace root changed", true)),
            Err(error) => {
                self.queue_full_reconciliation(workspace_id)?;
                return Err(error);
            }
        };
        let mut paths = force_paths
            .iter()
            .filter(|path| !path.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        paths.sort();
        paths.dedup();
        let full_sentinel = force_paths.iter().any(String::is_empty);
        let active_profile_id = self.connection.lock().query_row(
            "SELECT active_profile_id FROM workspace_index_state WHERE workspace_id=?1",
            [workspace_id],
            |row| row.get::<_, Option<String>>(0),
        ).optional().map_err(map_database_error)?.flatten();
        let profile_id = self.profile_identity.profile_id.clone();
        let mut profile_slots = Vec::new();
        let mut old_profile_unavailable = false;
        if let Some(active) = active_profile_id.as_ref()
            && active != &profile_id
        {
            match self.provider_slot_for_profile(active)? {
                Some(slot) => profile_slots.push((active.clone(), slot)),
                None => old_profile_unavailable = true,
            }
        }
        let desired_slot = self.provider_slot_for_profile(&profile_id)?.ok_or_else(|| {
            ServiceError::new(ErrorCode::EmbeddingUnavailable, "configured embedding provider is unavailable", false)
        })?;
        profile_slots.push((profile_id.clone(), desired_slot));
        let profile_ids = profile_slots.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
        let profile_providers = profile_slots.iter().map(|(id, slot)| ProfileProvider {
            profile_id: id,
            provider: slot.as_ref(),
        }).collect::<Vec<_>>();
        let ignore_rules_changed = paths.iter().any(|path| is_ignore_control_path(Path::new(path)));
        let requires_profile_backfill = self.profiles_require_backfill(workspace_id, &profile_ids)?;
        let force_reindex = force_all || requires_profile_backfill;
        let mut full_scan = force_reindex
            || paths.is_empty()
            || full_sentinel
            || ignore_rules_changed
            || self.full_reconcile_pending(workspace_id)?
            || self.branch_transition_pending(workspace_id)?;
        let (before_branch, before_manifest, branch_transition) = match self.reconciliation_snapshot(
            workspace_id,
            &workspace,
            (!full_scan).then_some(paths.as_slice()),
        ) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.queue_full_reconciliation(workspace_id)?;
                return Err(error);
            }
        };
        full_scan |= branch_transition;
        let job_paths = if full_scan { &[] } else { paths.as_slice() };
        let generation_start = self.begin_index_generation(workspace_id, job_paths)?;
        let generation = generation_start.generation;
        let claimed_jobs = generation_start.claimed_jobs;
        let snapshot = {
            let connection = self.connection.lock();
            existing_snapshot_for_profiles(&connection, workspace_id, &profile_ids)
        };
        let snapshot = match snapshot {
            Ok(snapshot) => snapshot,
            Err(_) => {
                self.release_reconcile_jobs(workspace_id, &claimed_jobs)?;
                self.queue_full_reconciliation(workspace_id)?;
                return Err(ServiceError::new(ErrorCode::Conflict, "workspace index metadata could not be read", true));
            }
        };
        let prepare_paths = if full_scan { &[] } else { paths.as_slice() };
        let plan = match prepare_workspace_profiles(WorkspacePreparation {
            workspace_id,
            workspace: &workspace,
            excluded_data_root: &self.private_data_root,
            profiles: &profile_providers,
            generation,
            max_file_bytes: self.config.limits.max_file_bytes,
            force_all: force_reindex,
            force_paths: prepare_paths,
            snapshot,
        }) {
            Ok(plan) => plan,
            Err(_) => {
                self.release_reconcile_jobs(workspace_id, &claimed_jobs)?;
                self.queue_full_reconciliation(workspace_id)?;
                return Err(ServiceError::new(ErrorCode::Conflict, "workspace reconciliation failed", true));
            }
        };
        let (after_branch, after_manifest, after_transition) = match self.reconciliation_snapshot(
            workspace_id,
            &workspace,
            (!full_scan).then_some(paths.as_slice()),
        ) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.release_reconcile_jobs(workspace_id, &claimed_jobs)?;
                self.queue_full_reconciliation(workspace_id)?;
                return Err(error);
            }
        };
        if before_branch != after_branch || before_manifest != after_manifest
            || (!full_scan && after_transition)
        {
            self.release_reconcile_jobs(workspace_id, &claimed_jobs)?;
            if before_branch != after_branch {
                self.queue_branch_reconciliation(workspace_id, true)?;
            }
            self.queue_full_reconciliation(workspace_id)?;
            return Err(ServiceError::new(
                ErrorCode::Conflict,
                "workspace changed during reconciliation; a complete retry was queued",
                true,
            ));
        }
        let desired_failed = plan.degraded_profile_ids.contains(&profile_id);
        let mut report = plan.report.clone();
        report.degraded_semantic = desired_failed || old_profile_unavailable;
        let persist_result = (|| {
            
            
        let mut guard = self.connection.lock();
        let transaction = guard.transaction().map_err(map_database_error)?;
            persist_workspace(&transaction, &plan).map_err(|_| {
                ServiceError::new(ErrorCode::Conflict, "workspace index metadata could not be saved", true)
            })?;
            match after_branch.as_ref() {
                Some(marker) => {
                    let git_dir = marker.git_dir.to_string_lossy().into_owned();
                    transaction.execute(
                        "UPDATE workspaces
                         SET git_dir=?1, git_head_ref=?2, git_head_commit=?3, updated_at=?4
                         WHERE id=?5",
                        params![
                            git_dir,
                            marker.head_ref.as_deref(),
                            marker.commit.as_deref(),
                            unix_seconds() as i64,
                            workspace_id,
                        ],
                    ).map_err(map_database_error)?;
                }
                None => {
                    transaction.execute(
                        "UPDATE workspaces
                         SET git_dir=NULL, git_head_ref=NULL, git_head_commit=NULL, updated_at=?1
                         WHERE id=?2",
                        params![unix_seconds() as i64, workspace_id],
                    ).map_err(map_database_error)?;
                }
            }
            transaction.execute(
                "UPDATE workspace_index_state
                 SET branch_transition=0
                 WHERE workspace_id=?1 AND visible_generation=?2 AND building_generation=?2",
                params![workspace_id, generation as i64],
            ).map_err(map_database_error)?;
            for indexed_profile in &profile_ids {
                if !plan.degraded_profile_ids.contains(indexed_profile) {
                    transaction.execute(
                        "UPDATE profile_applied_generations SET backfill_required=0
                         WHERE workspace_id=?1 AND profile_id=?2",
                        params![workspace_id, indexed_profile],
                    ).map_err(map_database_error)?;
                }
            }
            for (job_id, path) in &claimed_jobs {
                let absent_in_both_manifests =
                    !before_manifest.contains_key(Path::new(path))
                        && !after_manifest.contains_key(Path::new(path));
                let state = if path.is_empty()
                    || plan.covered_paths.contains(path)
                    || absent_in_both_manifests
                {
                    "done"
                } else {
                    "queued"
                };
                transition_index_job(&transaction, *job_id, workspace_id, "running", state)?;
            }
            transaction.commit().map_err(map_database_error)
        })();
        if let Err(error) = persist_result {
            self.release_reconcile_jobs(workspace_id, &claimed_jobs)?;
            self.queue_full_reconciliation(workspace_id)?;
            return Err(error);
        }
        let mut desired_sync_ok = false;
        let mut maintenance_failed = false;
        for sync_profile_id in &profile_ids {
            match self.sync_profile_snapshot(workspace_id, sync_profile_id) {
                Ok(watermark) => {
                    if sync_profile_id == &profile_id {
                        if self.try_promote_profile(workspace_id, sync_profile_id, &watermark).is_err() {
                            maintenance_failed = true;
                        }
                        desired_sync_ok = match self.profile_is_active_and_current(workspace_id, sync_profile_id) {
                            Ok(current) => current,
                            Err(_) => {
                                maintenance_failed = true;
                                false
                            }
                        };
                    }
                }
                Err(_) => {
                    report.degraded_semantic = true;
                    maintenance_failed |= self.record_profile_sync_failure(workspace_id, sync_profile_id).is_err();
                }
            }
        }
        if !desired_sync_ok {
            report.degraded_semantic = true;
            if self.queue_profile_backfill(workspace_id, &profile_id).is_err() {
                maintenance_failed = true;
                let _ = self.queue_full_reconciliation(workspace_id);
            }
        }
        let active_current = match self.active_profile_is_current(workspace_id) {
            Ok(current) => current,
            Err(_) => {
                maintenance_failed = true;
                false
            }
        };
        if self.set_workspace_health(workspace_id, if active_current { "ready" } else { "degraded" }).is_err() {
            maintenance_failed = true;
            report.degraded_semantic = true;
        }
        report.degraded_semantic |= !active_current;
        if self.retire_unreferenced_profiles().is_err() {
            maintenance_failed = true;
            report.degraded_semantic = true;
        }
        if maintenance_failed {
            report.degraded_semantic = true;
            let _ = self.queue_full_reconciliation(workspace_id);
            eprintln!("INDEX_DEGRADED: post-commit semantic maintenance needs recovery");
        }
        if force_graphify {
            drop(root_guard);
            let extraction = if self.config.graphify.enabled {
                let allowlist = {
                    let connection = self.connection.lock();
                    crate::index::graphify::load_graph_allowlist(&connection, workspace_id, generation)
                };
                allowlist.and_then(|allowlist| {
                    crate::index::graphify::extract_graph(
                        crate::index::graphify::GraphExtractionRequest {
                            config: &self.config.graphify,
                            workspace_id,
                            workspace: &workspace,
                            data_dir: &self.private_data_root,
                            allowlist: &allowlist,
                            max_file_bytes: self.config.limits.max_file_bytes,
                        },
                    )
                })
            } else {
                Err("GRAPH_UNAVAILABLE".into())
            };
            if extraction.is_err() {
                self.mark_graph_failure(&registration.graph_metadata(generation));
            }
            let _ = self.finish_graph_extraction(
                root_lock.as_ref(),
                &registration,
                generation,
                extraction,
            );
        }
        Ok(report)
    }

    pub fn read_graph(
        &self,
        workspace_id: &str,
        limit: usize,
        cursor: Option<&str>,
    ) -> Result<(WorkspaceGraphPage, Vec<Warning>), ServiceError> {
        if !(1..=100).contains(&limit) {
            return Err(ServiceError::new(ErrorCode::InvalidArgument, "graph page limit must be between 1 and 100", false));
        }
        let (registration, generation, branch_transition) = match self.graph_state(workspace_id) {
            Ok(state) => state,
            Err(error) if cursor.is_some() && error.code == ErrorCode::WorkspaceNotFound => {
                return Err(stale_graph_cursor());
            }
            Err(error) => return Err(error),
        };
        if branch_transition {
            if cursor.is_some() {
                return Err(stale_graph_cursor());
            }
            return Ok((empty_graph_page(workspace_id, generation), vec![Warning {
                code: "RECONCILING".into(),
                message: "Workspace branch transition is in progress; graph results are withheld.".into(),
                retryable: false,
            }]));
        }
        let metadata = registration.graph_metadata(generation);
        if self.graphify_failures.lock().contains(&metadata) {
            return Ok((empty_graph_page(workspace_id, generation), vec![self.graph_unavailable_warning()]));
        }
        let decoded_payload = cursor.map(|value| decode_graph_cursor(&self.cursor_secret, value)).transpose()?;
        if decoded_payload.as_ref().is_some_and(|payload| {
            payload.metadata != metadata || payload.limit != limit
        }) {
            return Err(stale_graph_cursor());
        }
        let workspace = self.workspace_capability(workspace_id)?;
        if workspace.root_path() != registration.root_path {
            return self.graph_read_unavailable(workspace_id, generation, cursor);
        }
        let Some(path) = crate::index::graphify::graph_snapshot_path(&self.private_data_root, &metadata) else {
            return self.graph_read_unavailable(workspace_id, generation, cursor);
        };
        let graph = match read_graph_snapshot(&self.graph_snapshots, &metadata, &path) {
            Ok(graph) => graph,
            Err(_) if cursor.is_some() => return Err(stale_graph_cursor()),
            Err(_) => return Ok((empty_graph_page(workspace_id, generation), vec![self.graph_unavailable_warning()])),
        };
        if graph.metadata != metadata
            || !graph_sources_match(&workspace, &graph.content_hashes, self.config.limits.max_file_bytes)
        {
            self.invalidate_graph_snapshot(&metadata);
            return self.graph_read_unavailable(workspace_id, generation, cursor);
        }
        if self.graph_state(workspace_id)? != (registration.clone(), generation, false) {
            return Err(stale_graph_cursor());
        }
        let payload = match decoded_payload {
            Some(payload) if payload.graph_fingerprint == graph.fingerprint => payload,
            Some(_) => return Err(stale_graph_cursor()),
            None => GraphCursorPayload {
                metadata: metadata.clone(),
                graph_fingerprint: graph.fingerprint.clone(),
                limit,
                offset: 0,
            },
        };
        if payload.offset > graph.nodes.len() {
            return Err(stale_graph_cursor());
        }
        let warnings = Vec::new();
        let mut node_count = payload.offset.saturating_add(limit).min(graph.nodes.len()) - payload.offset;
        let mut page = graph_page_candidate(
            workspace_id, generation, &graph, &payload, node_count,
            limit.saturating_mul(MAX_GRAPH_PAGE_EDGES_PER_NODE), &self.cursor_secret,
        )?;
        while !graph_page_fits(&page, &warnings, self.config.limits.max_tool_bytes) {
            if page.edges.pop().is_some() {
                page = graph_page_candidate_with_counts(
                    workspace_id, generation, &graph, &payload, node_count,
                    page.edges.len(), &self.cursor_secret,
                )?;
                continue;
            }
            if node_count == 0 {
                return Err(ServiceError::new(
                    ErrorCode::InvalidArgument,
                    "A graph page record exceeds the configured serialized response byte limit",
                    false,
                ));
            }
            node_count -= 1;
            page = graph_page_candidate(
                workspace_id, generation, &graph, &payload, node_count,
                limit.saturating_mul(MAX_GRAPH_PAGE_EDGES_PER_NODE), &self.cursor_secret,
            )?;
        }
        if self.graph_state(workspace_id)? != (registration, generation, false) {
            return Err(stale_graph_cursor());
        }
        if self.graphify_failures.lock().contains(&metadata) {
            return Ok((empty_graph_page(workspace_id, generation), vec![self.graph_unavailable_warning()]));
        }
        Ok((page, warnings))
    }

    fn graph_read_unavailable(
        &self,
        workspace_id: &str,
        generation: u64,
        cursor: Option<&str>,
    ) -> Result<(WorkspaceGraphPage, Vec<Warning>), ServiceError> {
        if cursor.is_some() {
            return Err(stale_graph_cursor());
        }
        Ok((empty_graph_page(workspace_id, generation), vec![self.graph_unavailable_warning()]))
    }

    fn reconciliation_snapshot(
        &self,
        workspace_id: &str,
        workspace: &WorkspaceFs,
        paths: Option<&[String]>,
    ) -> Result<(Option<crate::watch::BranchMarker>, crate::watch::Manifest, bool), ServiceError> {
        let branch = match crate::watch::read_branch_marker(workspace.root_path()) {
            Ok(Some(branch)) => Some(branch),
            Ok(None) if self.config.allow_non_git_roots => None,
            Ok(None) | Err(_) => {
                self.queue_branch_reconciliation(workspace_id, true)?;
                return Err(ServiceError::new(
                    ErrorCode::Conflict,
                    "workspace Git marker could not be read safely",
                    true,
                ));
            }
        };
        let branch_transition = self.ensure_branch_matches_index(
            workspace_id,
            branch.as_ref(),
            true,
        )?;
        let manifest = match if let Some(paths) = paths.filter(|_| !branch_transition) {
            crate::watch::build_paths_manifest(
                workspace,
                paths,
                self.config.limits.max_file_bytes,
            )
        } else {
            crate::watch::build_manifest(
                workspace,
                &self.private_data_root,
                self.config.limits.max_file_bytes,
            )
        } {
            Ok(manifest) => manifest,
            Err(_) => {
                self.queue_full_reconciliation(workspace_id)?;
                return Err(ServiceError::new(
                    ErrorCode::Conflict,
                    "workspace manifest could not be read safely",
                    true,
                ));
            }
        };
        Ok((branch, manifest, branch_transition))
    }

    fn stored_branch_state(
        &self,
        workspace_id: &str,
    ) -> Result<(Option<crate::watch::BranchMarker>, u64, bool), ServiceError> {
        let state = self.connection.lock().query_row(
            "SELECT w.git_dir,w.git_head_ref,w.git_head_commit,
                    COALESCE(s.visible_generation,0),COALESCE(s.branch_transition,0)
             FROM workspaces w LEFT JOIN workspace_index_state s ON s.workspace_id=w.id
             WHERE w.id=?1",
            [workspace_id],
            |row| Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, bool>(4)?,
            )),
        ).optional().map_err(map_database_error)?.ok_or_else(|| {
            ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
        })?;
        let marker = match state.0 {
            Some(git_dir) => Some(crate::watch::BranchMarker {
                git_dir: PathBuf::from(git_dir),
                head_ref: state.1,
                commit: state.2,
            }),
            None if state.1.is_none() && state.2.is_none() => None,
            None => return Err(ServiceError::new(
                ErrorCode::Conflict,
                "persisted workspace Git marker is inconsistent",
                false,
            )),
        };
        Ok((marker, state.3.max(0) as u64, state.4))
    }

    fn branch_transition_pending(&self, workspace_id: &str) -> Result<bool, ServiceError> {
        self.stored_branch_state(workspace_id).map(|(_, _, pending)| pending)
    }

    fn ensure_branch_matches_index(
        &self,
        workspace_id: &str,
        current: Option<&crate::watch::BranchMarker>,
        root_locked: bool,
    ) -> Result<bool, ServiceError> {
        let (previous, visible_generation, transition) = self.stored_branch_state(workspace_id)?;
        if transition {
            return Ok(true);
        }
        if visible_generation > 0 && previous.as_ref() != current {
            self.queue_branch_reconciliation(workspace_id, root_locked)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn queue_branch_reconciliation(
        &self,
        workspace_id: &str,
        root_locked: bool,
    ) -> Result<(), ServiceError> {
        if !self.branch_transition_pending(workspace_id)? {
            if root_locked {
                self.begin_rebuild_locked(workspace_id)?;
            } else {
                self.begin_rebuild(workspace_id)?;
            }
        }
        self.queue_full_reconciliation(workspace_id)
    }
    fn prepare_watch_startup_branch(
        &self,
        workspace_id: &str,
        current: Option<&crate::watch::BranchMarker>,
    ) -> Result<Option<BranchTransition>, ServiceError> {
        if current.is_none() && !self.config.allow_non_git_roots {
            self.queue_branch_reconciliation(workspace_id, false)?;
            return Err(ServiceError::new(
                ErrorCode::Conflict,
                "workspace Git marker could not be read safely",
                true,
            ));
        }
        let (previous, visible_generation, transition) = self.stored_branch_state(workspace_id)?;
        if transition {
            self.queue_full_reconciliation(workspace_id)?;
            return Ok(Some(BranchTransition {
                before: previous,
                after: current.cloned(),
            }));
        }
        if visible_generation > 0 && previous.as_ref() != current {
            self.queue_branch_reconciliation(workspace_id, false)?;
            return Ok(Some(BranchTransition {
                before: previous,
                after: current.cloned(),
            }));
        }
        Ok(None)
    }

    fn branch_observation_failed(&self, workspace_id: &str) -> Result<(), ServiceError> {
        self.queue_branch_reconciliation(workspace_id, false)
    }
    fn startup_reconcile_required(
        &self,
        workspace_id: &str,
        current_manifest: &crate::watch::Manifest,
    ) -> Result<bool, ServiceError> {
        let state = self.connection.lock().query_row(
            "SELECT COALESCE(s.dirty,1),COALESCE(s.visible_generation,0),
                    COALESCE(s.branch_transition,0),
                    EXISTS (SELECT 1 FROM index_jobs j
                            WHERE j.workspace_id=w.id AND j.state IN ('queued','running'))
             FROM workspaces w LEFT JOIN workspace_index_state s ON s.workspace_id=w.id
             WHERE w.id=?1",
            [workspace_id],
            |row| Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, bool>(2)?,
                row.get::<_, bool>(3)?,
            )),
        ).optional().map_err(map_database_error)?.ok_or_else(|| {
            ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false)
        })?;
        if state.0 != 0 || state.1 == 0 || state.2 || state.3 {
            return Ok(true);
        }
        let indexed = self.workspace_manifest(workspace_id)?.entries.into_iter()
            .map(|entry| (PathBuf::from(entry.path), entry.blake3))
            .collect::<std::collections::BTreeMap<_, _>>();
        let current = current_manifest.iter()
            .filter(|(_, hash)| !hash.starts_with("excluded:") && !hash.starts_with("control:"))
            .map(|(path, hash)| (path.to_path_buf(), hash.to_owned()))
            .collect::<std::collections::BTreeMap<_, _>>();
        Ok(indexed != current)
    }

    fn release_reconcile_jobs(
        &self,
        workspace_id: &str,
        jobs: &[(i64, String)],
    ) -> Result<(), ServiceError> {
        if jobs.is_empty() {
            return Ok(());
        }
        
        
        let mut guard = self.connection.lock();
        let transaction = guard.transaction().map_err(map_database_error)?;
        for (job_id, _) in jobs {
            transition_index_job(&transaction, *job_id, workspace_id, "running", "queued")?;
        }
        transaction.commit().map_err(map_database_error)
    }

    fn begin_index_generation(
        &self,
        workspace_id: &str,
        job_paths: &[String],
    ) -> Result<IndexGenerationStart, ServiceError> {
        let profile_id = self.profile_identity.profile_id.clone();
        let mut guard = self.connection.lock();
        let transaction = guard.transaction().map_err(map_database_error)?;
        ensure_profile_identity(&transaction, &self.profile_identity)?;
        let interrupted_jobs = {
            let mut statement = transaction.prepare(
                "SELECT id FROM index_jobs WHERE workspace_id=?1 AND state='running' ORDER BY id",
            ).map_err(map_database_error)?;
            statement.query_map([workspace_id], |row| row.get::<_, i64>(0))
                .map_err(map_database_error)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(map_database_error)?
        };
        for job_id in interrupted_jobs {
            transition_index_job(&transaction, job_id, workspace_id, "running", "queued")?;
        }
        let current = transaction.query_row(
            "SELECT MAX(generation, building_generation, visible_generation)
             FROM workspace_index_state WHERE workspace_id=?1",
            [workspace_id], |row| row.get::<_, i64>(0),
        ).optional().map_err(map_database_error)?.unwrap_or(0).max(0);
        let generation = current.checked_add(1).ok_or_else(|| {
            ServiceError::new(ErrorCode::Conflict, "workspace generation is out of range", false)
        })?;
        transaction.execute(
            "INSERT INTO workspace_index_state (
                workspace_id, generation, updated_at, building_generation, visible_generation,
                dirty, health, desired_profile_id, active_profile_id
             ) VALUES (?1, ?2, ?3, ?2, 0, 1, 'reconciling', ?4, NULL)
             ON CONFLICT(workspace_id) DO UPDATE SET
                generation=MAX(workspace_index_state.generation, excluded.generation),
                building_generation=MAX(workspace_index_state.building_generation, excluded.building_generation),
                dirty=1, health='reconciling',
                desired_profile_id=excluded.desired_profile_id,
                updated_at=excluded.updated_at",
            params![workspace_id, generation, unix_seconds() as i64, profile_id],
        ).map_err(map_database_error)?;
        for path in job_paths {
            if path.is_empty() {
                continue;
            }
            transaction.execute(
                "INSERT OR IGNORE INTO index_jobs
                    (workspace_id, relative_path, priority, state, created_at)
                 VALUES (?1, ?2, 0, 'queued', ?3)",
                params![workspace_id, path, unix_seconds() as i64],
            ).map_err(map_database_error)?;
        }
        let path_filter = if job_paths.is_empty() {
            String::new()
        } else {
            format!(" AND relative_path IN ({})", job_paths.iter().map(|_| "?").collect::<Vec<_>>().join(","))
        };
        let sql = format!(
            "SELECT id, relative_path FROM index_jobs
             WHERE workspace_id=?1 AND state='queued'{path_filter} ORDER BY priority, id"
        );
        let claimed_jobs = {
            let mut statement = transaction.prepare(&sql).map_err(map_database_error)?;
            let mut binds: Vec<&dyn rusqlite::ToSql> = vec![&workspace_id as &dyn rusqlite::ToSql];
            binds.extend(job_paths.iter().map(|path| path as &dyn rusqlite::ToSql));
            statement.query_map(rusqlite::params_from_iter(binds), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            }).map_err(map_database_error)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(map_database_error)?
        };
        for (job_id, _) in &claimed_jobs {
            transition_index_job(&transaction, *job_id, workspace_id, "queued", "running")?;
        }
        transaction.commit().map_err(map_database_error)?;
        Ok(IndexGenerationStart {
            generation: generation as u64,
            claimed_jobs,
        })
    }

    fn semantic_query_vectors(
        &self,
        request: &CodeSearchRequest,
    ) -> Result<SemanticQueryVectors, ServiceError> {
        let contexts = {
            let connection = self.connection.lock();
            let rows = request.workspace_ids.iter().map(|workspace_id| {
                connection.query_row(
                    "SELECT s.active_profile_id, s.visible_generation, s.branch_transition,
                            g.applied_generation, g.backfill_required, p.dimension
                     FROM workspace_index_state s
                     LEFT JOIN profile_applied_generations g
                       ON g.workspace_id=s.workspace_id AND g.profile_id=s.active_profile_id
                     LEFT JOIN embedding_profiles p ON p.id=s.active_profile_id
                     WHERE s.workspace_id=?1",
                    [workspace_id],
                    |row| Ok((
                        workspace_id.clone(),
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, bool>(2)?,
                        row.get::<_, Option<i64>>(3)?,
                        row.get::<_, Option<bool>>(4)?,
                        row.get::<_, Option<i64>>(5)?,
                    )),
                ).optional().map_err(map_database_error)
            }).collect::<Result<Vec<_>, ServiceError>>()?;
            rows.into_iter().flatten().collect::<Vec<_>>()
        };
        let mut queries = HashMap::new();
        let mut profile_vectors = HashMap::new();
        let explicit_vector = match (&request.query_vector, &request.profile_id) {
            (Some(vector), Some(profile_id)) => Some((profile_id.clone(), Arc::new(vector.clone()))),
            _ => None,
        };
        let mut unavailable = false;
        if request.query.is_empty() && explicit_vector.is_none() {
            return Ok(SemanticQueryVectors { queries, unavailable });
        }
        for (workspace_id, profile, visible, branch_transition, applied, backfill, dimension) in contexts {
            if branch_transition {
                continue;
            }
            let Some(profile_id) = profile else { continue };
            if visible <= 0 || applied != Some(visible) || backfill != Some(false) {
                continue;
            }
            let Some(dimension) = dimension.and_then(|value| usize::try_from(value).ok()) else {
                unavailable = true;
                continue;
            };
            let vector = if let Some((explicit_profile, vector)) = &explicit_vector {
                if explicit_profile != &profile_id {
                    continue;
                }
                if vector.len() != dimension || vector.iter().any(|value| !value.is_finite()) {
                    unavailable = true;
                    continue;
                }
                Arc::clone(vector)
            } else if let Some(vector) = profile_vectors.get(&profile_id).cloned() {
                vector
            } else {
                let Some(provider) = self.provider_slot_for_profile(&profile_id)? else {
                    unavailable = true;
                    continue;
                };
                let text = [request.query.clone()];
                let Ok(mut vectors) = provider.embed_batch(&text) else {
                    unavailable = true;
                    continue;
                };
                if vectors.len() != 1 || vectors[0].len() != dimension
                    || vectors[0].iter().any(|value| !value.is_finite())
                {
                    unavailable = true;
                    continue;
                }
                let vector = Arc::new(vectors.remove(0));
                profile_vectors.insert(profile_id.clone(), Arc::clone(&vector));
                vector
            };
            queries.insert(workspace_id, (profile_id, vector));
        }
        Ok(SemanticQueryVectors { queries, unavailable })
    }

    fn independent_connection(&self) -> Result<Connection, ServiceError> {
        let connection = Connection::open(&self.paths.database).map_err(map_database_error)?;
        connection.busy_timeout(std::time::Duration::from_secs(5)).map_err(map_database_error)?;
        connection.pragma_update(None, "foreign_keys", "ON").map_err(map_database_error)?;
        connection.pragma_update(None, "synchronous", "FULL").map_err(map_database_error)?;
        Ok(connection)
    }

    pub fn code_search(
        &self,
        mut request: CodeSearchRequest,
    ) -> Result<ToolResult<Page<SearchResult>>, ServiceError> {
        let query_limit = max_search_query_bytes(&self.config);
        if request.query.len() > query_limit {
            return Err(ServiceError::new(
                ErrorCode::InvalidArgument,
                format!("search query exceeds the configured {query_limit}-byte limit"),
                false,
            ));
        }
        let deadline = Instant::now() + Duration::from_secs(SEARCH_REFRESH_TIMEOUT_SECONDS);
        request.limit = request.limit.clamp(1, self.config.limits.max_results.clamp(1, 100));
        let cursor_requested = request.cursor.is_some();
        let SemanticQueryVectors { queries: semantic_queries, unavailable: embedding_unavailable } =
            self.semantic_query_vectors(&request)?;
        let mut execution = self.code_search_once(request.clone(), semantic_queries.clone())?;
        let stale_paths_found = !execution.stale_paths.is_empty();
        let mut worker_pending = false;
        let mut worker_failed = false;
        if execution.result.is_success() && stale_paths_found {
            match self.refresh_stale_paths_until(&execution.stale_paths, deadline) {
                Some(Ok(())) => {}
                Some(Err(_)) => worker_failed = true,
                None => worker_pending = true,
            }
            match self.code_search_retry(request, semantic_queries, deadline) {
                Some(Ok(retried)) => execution = retried,
                Some(Err(_)) => worker_failed = true,
                None => worker_pending = true,
            }
        }
        let mut result = self.finish_search_execution(execution);
        if stale_paths_found && !cursor_requested
            && result.error().is_some_and(|error| error.code == ErrorCode::StaleCursor)
        {
            result = ToolResult::success(Page {
                items: Vec::new(),
                truncated: true,
                next_cursor: None,
            }).with_warning(Warning {
                code: "INDEX_DEGRADED".into(),
                message: "Stale candidates were withheld while index freshness changed.".into(),
                retryable: false,
            });
        }
        if worker_pending {
            result = result.with_warning(stale_search_worker_warning());
        }
        if worker_failed {
            result = result.with_warning(stale_search_worker_failure_warning());
        }
        Ok(if embedding_unavailable {
            result.with_warning(Warning {
                code: "EMBEDDING_UNAVAILABLE".into(),
                message: "Semantic query embedding unavailable; lexical and symbol results are returned.".into(),
                retryable: false,
            })
        } else {
            result
        })
    }

    fn code_search_once(
        &self,
        request: CodeSearchRequest,
        semantic_queries: SemanticQueryMap,
    ) -> Result<SearchExecution<SearchResult>, ServiceError> {
        let capabilities = self.search_capabilities(&request.workspace_ids)?;
        let capture = {
            let mut connection = self.independent_connection()?;
            let transaction = connection.transaction().map_err(map_database_error)?;
            let capture = SearchService::with_capabilities(
                &transaction,
                &self.cursor_secret,
                SearchServiceOptions {
                    capabilities: HashMap::new(),
                    max_file_bytes: self.config.limits.max_file_bytes,
                    max_results: self.config.limits.max_results,
                    max_snippet_bytes: self.config.limits.max_snippet_bytes,
                    max_tool_bytes: self.config.limits.max_tool_bytes,
                    vectors_dir: self.paths.vectors_dir.clone(),
                },
            )
            .with_shared_caches(
                Arc::clone(&self.search_result_sets),
                Arc::clone(&self.semantic_snapshots),
            )
            .with_semantic_queries(semantic_queries)
            .capture_code_search(request);
            transaction.commit().map_err(map_database_error)?;
            capture
        };
        self.validate_and_enqueue(capture.execute(&capabilities))
    }

    pub fn symbol_search(
        &self,
        mut request: SymbolSearchRequest,
    ) -> Result<ToolResult<Page<SymbolSearchResult>>, ServiceError> {
        let deadline = Instant::now() + Duration::from_secs(SEARCH_REFRESH_TIMEOUT_SECONDS);
        request.limit = request.limit.clamp(1, self.config.limits.max_results.clamp(1, 100));
        let cursor_requested = request.cursor.is_some();
        let mut execution = self.symbol_search_once(request.clone())?;
        let stale_paths_found = !execution.stale_paths.is_empty();
        let mut worker_pending = false;
        let mut worker_failed = false;
        if execution.result.is_success() && stale_paths_found {
            match self.refresh_stale_paths_until(&execution.stale_paths, deadline) {
                Some(Ok(())) => {}
                Some(Err(_)) => worker_failed = true,
                None => worker_pending = true,
            }
            match self.symbol_search_retry(request, deadline) {
                Some(Ok(retried)) => execution = retried,
                Some(Err(_)) => worker_failed = true,
                None => worker_pending = true,
            }
        }
        let mut result = self.finish_search_execution(execution);
        if stale_paths_found && !cursor_requested
            && result.error().is_some_and(|error| error.code == ErrorCode::StaleCursor)
        {
            result = ToolResult::success(Page {
                items: Vec::new(),
                truncated: true,
                next_cursor: None,
            }).with_warning(Warning {
                code: "INDEX_DEGRADED".into(),
                message: "Stale candidates were withheld while index freshness changed.".into(),
                retryable: false,
            });
        }
        if worker_pending {
            result = result.with_warning(stale_search_worker_warning());
        }
        if worker_failed {
            result = result.with_warning(stale_search_worker_failure_warning());
        }
        Ok(result)
    }


    fn symbol_search_once(
        &self,
        request: SymbolSearchRequest,
    ) -> Result<SearchExecution<SymbolSearchResult>, ServiceError> {
        let capabilities = self.search_capabilities(&request.workspace_ids)?;
        let capture = {
            let mut connection = self.independent_connection()?;
            let transaction = connection.transaction().map_err(map_database_error)?;
            let capture = SearchService::with_capabilities(
                &transaction,
                &self.cursor_secret,
                SearchServiceOptions {
                    capabilities: HashMap::new(),
                    max_file_bytes: self.config.limits.max_file_bytes,
                    max_results: self.config.limits.max_results,
                    max_snippet_bytes: self.config.limits.max_snippet_bytes,
                    max_tool_bytes: self.config.limits.max_tool_bytes,
                    vectors_dir: self.paths.vectors_dir.clone(),
                },
            )
            .with_shared_caches(
                Arc::clone(&self.search_result_sets),
                Arc::clone(&self.semantic_snapshots),
            )
            .capture_symbol_search(request);
            transaction.commit().map_err(map_database_error)?;
            capture
        };
        self.validate_and_enqueue(capture.execute(&capabilities))
    }

    fn validate_and_enqueue<T>(
        &self,
        mut execution: SearchExecution<T>,
    ) -> Result<SearchExecution<T>, ServiceError> {
        let mut connection = self.connection.lock();
        let current = execution.still_current(&connection);
        enqueue_refresh_paths(&mut connection, &execution.stale_paths)?;
        if !current && execution.result.is_success() {
            execution.result = ToolResult::failure(
                ErrorCode::StaleCursor,
                "Workspace generation changed while validating search citations",
                false,
            );
        }
        Ok(execution)
    }

    fn finish_search_execution<T>(&self, mut execution: SearchExecution<T>) -> ToolResult<Page<T>> {
        if execution.result.is_success()
            && !execution.still_current(&self.connection.lock())
        {
            execution.result = ToolResult::failure(
                ErrorCode::StaleCursor,
                "Workspace generation changed while validating search citations",
                false,
            );
        }
        execution.result
    }

    fn refresh_stale_paths_until(
        &self,
        stale_paths: &[(String, String)],
        deadline: Instant,
    ) -> Option<Result<(), ServiceError>> {
        let mut paths_by_workspace = std::collections::BTreeMap::<String, Vec<String>>::new();
        if deadline.saturating_duration_since(Instant::now()).is_zero() {
            return None;
        }
        for (workspace_id, path) in stale_paths.iter().take(MAX_SYNCHRONOUS_STALE_PATHS) {
            paths_by_workspace.entry(workspace_id.clone()).or_default().push(path.clone());
        }
        let service = self.clone();
        self.index_workers.run_until_deadline(deadline, move || {
            paths_by_workspace.into_iter().try_for_each(|(workspace_id, paths)| {
                service.refresh_paths(&workspace_id, &paths).map(|_| ())
            })
        })
    }

    fn code_search_retry(
        &self,
        request: CodeSearchRequest,
        semantic_queries: HashMap<String, (String, Arc<Vec<f32>>)>,
        deadline: Instant,
    ) -> Option<Result<SearchExecution<SearchResult>, ServiceError>> {
        self.index_workers.run_until_deadline(deadline, {
            let service = self.clone();
            move || service.code_search_once(request, semantic_queries)
        })
    }

    fn symbol_search_retry(
        &self,
        request: SymbolSearchRequest,
        deadline: Instant,
    ) -> Option<Result<SearchExecution<SymbolSearchResult>, ServiceError>> {
        self.index_workers.run_until_deadline(deadline, {
            let service = self.clone();
            move || service.symbol_search_once(request)
        })
    }

    pub fn read_file(
        &self,
        workspace_id: &str,
        path: &Path,
        start_line: Option<usize>,
        end_line: Option<usize>,
        expected_sha256: Option<&str>,
    ) -> Result<crate::security::FileRead, ServiceError> {
        let workspace = self.workspace_capability(workspace_id)?;
        file_read(
            &workspace, path, start_line, end_line, expected_sha256,
            self.config.limits.max_file_bytes, self.config.limits.max_tool_bytes,
        ).map_err(|error| ServiceError::new(error.code, error.message, false))
    }

    pub fn changeset_prepare(
        &self,
        workspace_id: &str,
        operations: Vec<ChangeOperation>,
    ) -> Result<PreparedChangeSet, ServiceError> {
        let root = self.workspace_root(workspace_id)?;
        let root_lock = self.root_lock(&root);
        let _root_guard = root_lock.lock();
        
        ChangesetService::for_service(
            &self.connection, &self.paths.pending_dir,
            self.config.limits.changeset_expiry_seconds, self.config.limits.max_file_bytes,
        ).and_then(|service| service.prepare(workspace_id, operations)).map_err(map_changeset_error)
    }

    pub fn changeset_apply(self: &Arc<Self>, operation_id: &str, digest: &str) -> Result<ChangesetOutcome, ServiceError> {
        let root = self.changeset_root(operation_id)?;
        let root_lock = self.root_lock(&root);
        let _root_guard = root_lock.lock();
        
        let mut outcome = ChangesetService::for_service(
            &self.connection, &self.paths.pending_dir,
            self.config.limits.changeset_expiry_seconds, self.config.limits.max_file_bytes,
        ).and_then(|service| service.apply(operation_id, digest)).map_err(map_changeset_error)?;

        if outcome.source_status == "applied" || outcome.source_status == "partial" {
            let mut paths = Vec::new();
            let workspace_id = {
                let guard = self.connection.lock();
                guard.query_row("SELECT workspace_id FROM pending_changesets WHERE operation_id = ?1", [operation_id], |row| row.get::<_, String>(0)).unwrap_or_default()
            };
            {
                let guard = self.connection.lock();
                if let Ok(mut stmt) = guard.prepare("SELECT relative_path, destination_path FROM changeset_files WHERE operation_id = ?1 AND effect_state = 'applied'")
                    && let Ok(mut rows) = stmt.query([operation_id])
                {
                    while let Ok(Some(row)) = rows.next() {
                        if let Ok(p) = row.get::<_, String>(0) { paths.push(p); }
                        if let Ok(Some(d)) = row.get::<_, Option<String>>(1) { paths.push(d); }
                    }
                }
            }
            if !paths.is_empty() && !workspace_id.is_empty() {
                drop(_root_guard);
                let jobs = paths.iter()
                    .map(|path| (workspace_id.clone(), path.clone()))
                    .collect::<Vec<_>>();
                let durable_jobs = {
                    let mut connection = self.connection.lock();
                    enqueue_refresh_paths(&mut connection, &jobs).is_ok()
                };
                let service = self.clone();
                let refresh = self.index_workers.run_until_deadline(
                    Instant::now() + Duration::from_secs(SEARCH_REFRESH_TIMEOUT_SECONDS),
                    move || service.refresh_paths(&workspace_id, &paths).map(|_| ()),
                );
                if matches!(refresh, Some(Ok(()))) {
                    outcome.index_status = "ready".into();
                } else {
                    outcome.index_status = "degraded".into();
                    let message = if durable_jobs {
                        "Source changes remain applied; the index refresh is still running or failed. Persisted index jobs will be retried on restart."
                    } else {
                        "Source changes remain applied; the index refresh did not complete and durable retry admission could not be confirmed. Do not retry the source apply."
                    };
                    outcome.warnings.push(crate::model::Warning {
                        code: "INDEX_DEGRADED".into(),
                        message: message.into(),
                        retryable: false,
                    });
                }
            }
        }
        Ok(outcome)
    }

    pub fn changeset_status(&self, operation_id: &str) -> Result<ChangesetOutcome, ServiceError> {
        
        ChangesetService::for_service(
            &self.connection, &self.paths.pending_dir,
            self.config.limits.changeset_expiry_seconds, self.config.limits.max_file_bytes,
        ).and_then(|service| service.status(operation_id)).map_err(map_changeset_error)
    }

    pub fn changeset_discard(&self, operation_id: &str, digest: &str) -> Result<ChangesetOutcome, ServiceError> {
        
        ChangesetService::for_service(
            &self.connection, &self.paths.pending_dir,
            self.config.limits.changeset_expiry_seconds, self.config.limits.max_file_bytes,
        ).and_then(|service| service.discard(operation_id, digest)).map_err(map_changeset_error)
    }

    pub fn watch_backend(self: &Arc<Self>, workspace_id: &str) -> Arc<dyn WatchIndexBackend> {
        Arc::new(WorkspaceWatchBackend {
            service: Arc::clone(self),
            workspace_id: workspace_id.to_owned(),
        })
    }

    fn begin_rebuild(&self, workspace_id: &str) -> Result<(), ServiceError> {
        let root = self.workspace_root(workspace_id)?;
        let root_lock = self.root_lock(&root);
        let _root_guard = root_lock.lock();
        self.begin_rebuild_locked(workspace_id)
    }

    fn begin_rebuild_locked(&self, workspace_id: &str) -> Result<(), ServiceError> {
        let mut guard = self.connection.lock();
        let transaction = guard.transaction().map_err(map_database_error)?;
        let (generation, visible): (i64, i64) = transaction.query_row(
            "SELECT MAX(generation, building_generation, visible_generation), visible_generation
             FROM workspace_index_state WHERE workspace_id=?1",
            [workspace_id], |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional().map_err(map_database_error)?.unwrap_or((0, 0));
        let building = generation.checked_add(1).ok_or_else(|| {
            ServiceError::new(ErrorCode::Conflict, "workspace generation is out of range", false)
        })?;
        transaction.execute(
            "INSERT INTO workspace_index_state (
                workspace_id, generation, updated_at, building_generation, visible_generation, dirty,
                health, branch_transition
             ) VALUES (?1, ?2, ?3, ?2, ?4, 1, 'reconciling', 1)
             ON CONFLICT(workspace_id) DO UPDATE SET
                generation=MAX(workspace_index_state.generation, excluded.generation),
                building_generation=MAX(workspace_index_state.building_generation, excluded.building_generation),
                dirty=1, health='reconciling', branch_transition=1, updated_at=excluded.updated_at",
            params![workspace_id, building, unix_seconds() as i64, visible],
        ).map_err(map_database_error)?;
        transaction.commit().map_err(map_database_error)?;
        Ok(())
    }

    fn provider_slot_for_profile(
        &self,
        profile_id: &str,
    ) -> Result<Option<Arc<ProviderSlot>>, ServiceError> {
        if let Some(slot) = self.provider_slots.lock().get(profile_id).cloned() {
            return Ok(Some(slot));
        }
        let metadata = {
            let connection = self.connection.lock();
            connection.query_row(
                "SELECT provider_kind, endpoint, model, revision, dimension, metric,
                        api_key_env, query_template_version, document_template_version,
                        batch_size, timeout_seconds, retry_delays_ms
                 FROM embedding_profiles WHERE id=?1",
                [profile_id],
                |row| Ok((
                    row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?, row.get::<_, Option<String>>(3)?,
                    row.get::<_, i64>(4)?, row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?, row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?, row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?, row.get::<_, String>(11)?,
                )),
            ).optional().map_err(map_database_error)?
        };
        let Some((
            provider_kind, endpoint, model, revision, dimension, metric, api_key_env,
            query_template, document_template, batch_size, timeout_seconds, retry_delays,
        )) = metadata else {
            return Ok(None);
        };
        let (Ok(dimension), Ok(batch_size), Ok(timeout_seconds), Ok(retry_delays_ms)) = (
            usize::try_from(dimension),
            usize::try_from(batch_size),
            u64::try_from(timeout_seconds),
            serde_json::from_str::<Vec<u64>>(&retry_delays),
        ) else {
            return Ok(None);
        };
        let Some(metric) = parse_embedding_metric(&metric) else { return Ok(None) };
        let legacy_fastembed = profile_id == LEGACY_FASTEMBED_PROFILE_ID
            && provider_kind == EmbeddingProviderKind::FastEmbed.as_str()
            && endpoint.is_none()
            && model == "BAAI/bge-small-en-v1.5"
            && revision.is_none()
            && dimension == 384
            && metric == EmbeddingMetric::Cosine
            && api_key_env.is_none()
            && query_template == "query-v1"
            && document_template == "document-v1";
        if provider_kind != EmbeddingProviderKind::FastEmbed.as_str()
            && !self.config.embedding.allow_remote
        {
            return Ok(None);
        }
        let mut config = self.config.embedding.clone();
        config.provider = provider_kind;
        config.endpoint = endpoint;
        config.model = model;
        config.revision = revision;
        config.dimension = Some(dimension);
        config.metric = metric;
        config.api_key_env = api_key_env;
        config.query_template_version = query_template;
        config.document_template_version = document_template;
        config.batch_size = batch_size;
        config.timeout_seconds = timeout_seconds;
        config.retry_delays_ms = retry_delays_ms;
        let (identity, provider, fingerprint, dimension) = match build_configured_provider(&config) {
            Ok(built) => built,
            Err(_) => return Ok(None),
        };
        if identity.profile_id != profile_id && !legacy_fastembed {
            return Ok(None);
        }
        let slot = Arc::new(ProviderSlot {
            fingerprint,
            dimension,
            provider: Mutex::new(provider),
            admission: Arc::clone(&self.provider_gate),
        });
        let mut slots = self.provider_slots.lock();
        Ok(Some(Arc::clone(slots.entry(profile_id.to_owned()).or_insert(slot))))
    }

    fn profiles_require_backfill(
        &self,
        workspace_id: &str,
        profile_ids: &[String],
    ) -> Result<bool, ServiceError> {
        let connection = self.connection.lock();
        for profile_id in profile_ids {
            let required = connection.query_row(
                "SELECT backfill_required FROM profile_applied_generations
                 WHERE workspace_id=?1 AND profile_id=?2",
                params![workspace_id, profile_id],
                |row| row.get::<_, bool>(0),
            ).optional().map_err(map_database_error)?;
            if required.unwrap_or(false) {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn initialize_profiles_and_recover(&self) -> Result<(), ServiceError> {
        let current_profile = self.profile_identity.profile_id.clone();
        {
            let mut connection = self.connection.lock();
            let transaction = connection.transaction().map_err(map_database_error)?;
            transaction.execute(
                "DELETE FROM index_jobs
                 WHERE state='running' AND EXISTS (
                    SELECT 1 FROM index_jobs queued
                    WHERE queued.workspace_id=index_jobs.workspace_id
                      AND queued.relative_path=index_jobs.relative_path
                      AND queued.state='queued'
                 )",
                [],
            ).map_err(map_database_error)?;
            transaction.execute(
                "UPDATE index_jobs SET state='queued' WHERE state='running'",
                [],
            ).map_err(map_database_error)?;
            ensure_profile_identity(&transaction, &self.profile_identity)?;
            let workspaces = {
                let mut statement = transaction.prepare("SELECT id FROM workspaces ORDER BY id")
                    .map_err(map_database_error)?;
                statement.query_map([], |row| row.get::<_, String>(0))
                    .map_err(map_database_error)?
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(map_database_error)?
            };
            for workspace_id in workspaces {
                let state: Option<(Option<String>, Option<String>, i64, i64)> = transaction.query_row(
                    "SELECT desired_profile_id, active_profile_id, visible_generation, dirty
                     FROM workspace_index_state WHERE workspace_id=?1",
                    [&workspace_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                ).optional().map_err(map_database_error)?;
                let needs_reconcile = match state {
                    Some((desired, active, visible, dirty)) =>
                        desired.as_deref() != Some(current_profile.as_str())
                            || active.as_deref() != Some(current_profile.as_str())
                            || dirty != 0 || visible == 0,
                    None => true,
                };
                transaction.execute(
                    "INSERT INTO workspace_index_state (
                        workspace_id, generation, updated_at, building_generation,
                        visible_generation, dirty, health, desired_profile_id, active_profile_id
                     ) VALUES (?1, 0, ?2, 0, 0, 1, 'reconcile_required', ?3, NULL)
                     ON CONFLICT(workspace_id) DO UPDATE SET
                        desired_profile_id=excluded.desired_profile_id",
                    params![workspace_id, unix_seconds() as i64, current_profile],
                ).map_err(map_database_error)?;
                if needs_reconcile {
                    transaction.execute(
                        "UPDATE workspace_index_state
                         SET dirty=1, health='reconcile_required', updated_at=?1
                         WHERE workspace_id=?2",
                        params![unix_seconds() as i64, workspace_id],
                    ).map_err(map_database_error)?;
                    queue_full_reconcile_in_transaction(&transaction, &workspace_id)?;
                }
            }
            transaction.commit().map_err(map_database_error)?;
        }
        self.recover_vector_snapshots()
    }
    fn detect_offline_branch_transitions(&self) -> Result<(), ServiceError> {
        let workspaces = {
            let connection = self.connection.lock();
            let mut statement = connection.prepare("SELECT id FROM workspaces ORDER BY id")
                .map_err(map_database_error)?;
            statement.query_map([], |row| row.get::<_, String>(0))
                .map_err(map_database_error)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(map_database_error)?
        };
        for workspace_id in workspaces {
            let workspace = self.workspace_capability(&workspace_id)?;
            let current = match crate::watch::read_branch_marker(workspace.root_path()) {
                Ok(Some(marker)) => Some(marker),
                Ok(None) if self.config.allow_non_git_roots => None,
                Ok(None) | Err(_) => {
                    self.queue_branch_reconciliation(&workspace_id, false)?;
                    continue;
                }
            };
            let (previous, visible, transition) = self.stored_branch_state(&workspace_id)?;
            if transition {
                self.queue_full_reconciliation(&workspace_id)?;
            } else if visible > 0 && previous != current {
                self.queue_branch_reconciliation(&workspace_id, false)?;
            }
        }
        Ok(())
    }


    fn recover_vector_snapshots(&self) -> Result<(), ServiceError> {
        let profiles = {
            let connection = self.connection.lock();
            let mut statement = connection.prepare(
                "SELECT workspace_id, active_profile_id FROM workspace_index_state
                   WHERE active_profile_id IS NOT NULL
                 UNION SELECT workspace_id, desired_profile_id FROM workspace_index_state
                   WHERE desired_profile_id IS NOT NULL
                 UNION SELECT v.workspace_id, v.profile_id FROM vector_ops v
                   JOIN embedding_profiles p ON p.id=v.profile_id
                   WHERE p.state!='retired'
                 ORDER BY 1, 2",
            ).map_err(map_database_error)?;
            statement.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
                .map_err(map_database_error)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(map_database_error)?
        };
        for (workspace_id, profile_id) in profiles {
            match self.sync_profile_snapshot(&workspace_id, &profile_id) {
                Ok(watermark) => {
                    if self.workspace_desired_profile(&workspace_id)?.as_deref() == Some(profile_id.as_str()) {
                        let _ = self.try_promote_profile(&workspace_id, &profile_id, &watermark);
                    }
                }
                Err(_) => self.record_profile_sync_failure(&workspace_id, &profile_id)?,
            }
        }
        self.retire_unreferenced_profiles()
    }
    fn sync_profile_snapshot(
        &self,
        workspace_id: &str,
        profile_id: &str,
    ) -> Result<VectorWatermark, ServiceError> {
        let (dimension, metric) = self.connection.lock().query_row(
            "SELECT dimension, metric FROM embedding_profiles WHERE id=?1",
            [profile_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        ).optional().map_err(map_database_error)?.ok_or_else(|| {
            ServiceError::new(ErrorCode::EmbeddingUnavailable, "embedding profile is unavailable", false)
        })?;
        let dimension = usize::try_from(dimension).map_err(|_| {
            ServiceError::new(ErrorCode::EmbeddingUnavailable, "embedding profile dimension is invalid", false)
        })?;
        let sequencer = Sequencer::new(
            workspace_id.to_owned(), profile_id.to_owned(),
            self.paths.vectors_dir.clone(), dimension, metric,
        );
        let mut connection = self.independent_connection()?;
        sequencer.sync(&mut connection).map_err(|_| {
            ServiceError::new(ErrorCode::EmbeddingUnavailable, "semantic snapshot synchronization failed", true)
        })
    }

    fn try_promote_profile(
        &self,
        workspace_id: &str,
        profile_id: &str,
        watermark: &VectorWatermark,
    ) -> Result<bool, ServiceError> {
        ProfileManager::new(workspace_id.to_owned())
            .try_promote(&mut self.connection.lock(), profile_id, watermark)
            .map_err(|_| ServiceError::new(
                ErrorCode::EmbeddingUnavailable,
                "embedding profile promotion could not be verified",
                true,
            ))
    }

    fn workspace_desired_profile(&self, workspace_id: &str) -> Result<Option<String>, ServiceError> {
        self.connection.lock().query_row(
            "SELECT desired_profile_id FROM workspace_index_state WHERE workspace_id=?1",
            [workspace_id],
            |row| row.get(0),
        ).optional().map(|value| value.flatten()).map_err(map_database_error)
    }

    fn profile_is_active_and_current(
        &self,
        workspace_id: &str,
        profile_id: &str,
    ) -> Result<bool, ServiceError> {
        let state = self.connection.lock().query_row(
            "SELECT s.active_profile_id, s.visible_generation,
                    g.applied_generation, g.backfill_required
             FROM workspace_index_state s
             LEFT JOIN profile_applied_generations g
               ON g.workspace_id=s.workspace_id AND g.profile_id=?2
             WHERE s.workspace_id=?1",
            params![workspace_id, profile_id],
            |row| Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, Option<bool>>(3)?,
            )),
        ).optional().map_err(map_database_error)?;
        Ok(state.is_some_and(|(active, visible, applied, backfill)| {
            active.as_deref() == Some(profile_id)
                && applied == Some(visible)
                && backfill == Some(false)
        }))
    }

    fn active_profile_is_current(&self, workspace_id: &str) -> Result<bool, ServiceError> {
        let state = self.connection.lock().query_row(
            "SELECT active_profile_id, visible_generation
             FROM workspace_index_state WHERE workspace_id=?1",
            [workspace_id],
            |row| Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?)),
        ).optional().map_err(map_database_error)?;
        match state {
            Some((Some(profile_id), _)) => self.profile_is_active_and_current(workspace_id, &profile_id),
            Some((None, visible)) => Ok(visible == 0),
            None => Ok(false),
        }
    }

    fn record_profile_sync_failure(
        &self,
        workspace_id: &str,
        _profile_id: &str,
    ) -> Result<(), ServiceError> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction().map_err(map_database_error)?;
        transaction.execute(
            "UPDATE workspace_index_state SET health='degraded', updated_at=?1
             WHERE workspace_id=?2",
            params![unix_seconds() as i64, workspace_id],
        ).map_err(map_database_error)?;
        queue_full_reconcile_in_transaction(&transaction, workspace_id)?;
        transaction.commit().map_err(map_database_error)
    }

    fn queue_profile_backfill(
        &self,
        workspace_id: &str,
        profile_id: &str,
    ) -> Result<(), ServiceError> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction().map_err(map_database_error)?;
        let now = unix_seconds() as i64;
        transaction.execute(
            "INSERT OR IGNORE INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
             SELECT f.workspace_id, f.relative_path, 0, 'queued', ?1
             FROM files f JOIN workspace_index_state s ON s.workspace_id=f.workspace_id
             WHERE f.workspace_id=?2 AND f.visible_generation=s.visible_generation
               AND EXISTS (
                 SELECT 1 FROM chunks c
                 WHERE c.workspace_id=f.workspace_id AND c.relative_path=f.relative_path
                   AND NOT EXISTS (
                     SELECT 1 FROM chunk_embeddings e
                     WHERE e.workspace_id=c.workspace_id AND e.relative_path=c.relative_path
                       AND e.chunk_index=c.chunk_index AND e.profile_id=?3
                   )
               )",
            params![now, workspace_id, profile_id],
        ).map_err(map_database_error)?;
        let queued = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM index_jobs
             WHERE workspace_id=?1 AND state='queued')",
            [workspace_id],
            |row| row.get::<_, bool>(0),
        ).map_err(map_database_error)?;
        if !queued {
            queue_full_reconcile_in_transaction(&transaction, workspace_id)?;
        }
        transaction.execute(
            "UPDATE workspace_index_state SET health='degraded', updated_at=?1
             WHERE workspace_id=?2",
            params![now, workspace_id],
        ).map_err(map_database_error)?;
        transaction.commit().map_err(map_database_error)
    }

    fn set_workspace_health(&self, workspace_id: &str, health: &str) -> Result<(), ServiceError> {
        self.connection.lock().execute(
            "UPDATE workspace_index_state SET health=?1, updated_at=?2 WHERE workspace_id=?3",
            params![health, unix_seconds() as i64, workspace_id],
        ).map(|_| ()).map_err(map_database_error)
    }

    fn retire_unreferenced_profiles(&self) -> Result<(), ServiceError> {
        self.connection.lock().execute(
            "UPDATE embedding_profiles SET state='retired'
             WHERE state='active'
               AND NOT EXISTS (
                 SELECT 1 FROM workspace_index_state s
                 WHERE s.active_profile_id=embedding_profiles.id
                    OR s.desired_profile_id=embedding_profiles.id
               )",
            [],
        ).map(|_| ()).map_err(map_database_error)
    }

    fn changeset_root(&self, operation_id: &str) -> Result<PathBuf, ServiceError> {
        let connection = self.connection.lock();
        match connection.query_row(
            "SELECT w.root_path FROM pending_changesets p
             JOIN workspaces w ON w.id=p.workspace_id WHERE p.operation_id=?1",
            [operation_id],
            |row| row.get::<_, String>(0),
        ) {
            Ok(path) => Ok(PathBuf::from(path)),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                let archived = connection.query_row(
                    "SELECT EXISTS(SELECT 1 FROM unregistered_changesets WHERE operation_id=?1)",
                    [operation_id],
                    |row| row.get::<_, bool>(0),
                ).map_err(map_database_error)?;
                let (code, message) = if archived {
                    (ErrorCode::Conflict, "archived changeset cannot be applied")
                } else {
                    (ErrorCode::WorkspaceNotFound, "changeset not found")
                };
                Err(ServiceError::new(code, message, false))
            }
            Err(error) => Err(map_database_error(error)),
        }
    }

    fn root_lock(&self, root: &Path) -> Arc<Mutex<()>> {
        let mut locks = self.root_locks.lock();
        Arc::clone(locks.entry(root.to_path_buf()).or_insert_with(|| Arc::new(Mutex::new(()))))
    }
}

struct WorkspaceWatchBackend {
    service: Arc<WorkspaceService>,
    workspace_id: String,
}

impl WorkspaceWatchBackend {
    fn service(&self) -> &WorkspaceService {
        self.service.as_ref()
    }

    fn check_request(&self, request: &ReconcileRequest) -> Result<(), String> {
        if request.workspace_id == self.workspace_id { Ok(()) }
        else { Err("watcher workspace identity changed".into()) }
    }
}

impl WatchIndexBackend for WorkspaceWatchBackend {
    fn apply_incremental(&self, request: ReconcileRequest) -> Result<(), String> {
        self.check_request(&request)?;
        let service = self.service();
        let paths = match watcher_path_strings(&request.changed_paths) {
            Ok(paths) => paths,
            Err(error) => {
                service.queue_full_reconciliation(&self.workspace_id)
                    .map_err(|queue_error| queue_error.to_string())?;
                return Err(error);
            }
        };
        match service.refresh_paths(&self.workspace_id, &paths) {
            Ok(_) => Ok(()),
            Err(error) => {
                service.queue_full_reconciliation(&self.workspace_id)
                    .map_err(|queue_error| queue_error.to_string())?;
                Err(error.to_string())
            }
        }
    }

    fn begin_rebuild(&self, _transition: &BranchTransition) -> Result<(), String> {
        self.service().begin_rebuild(&self.workspace_id).map_err(|error| error.to_string())
    }

    fn reconcile(&self, request: ReconcileRequest) -> Result<(), String> {
        self.check_request(&request)?;
        let service = self.service();
        let result = if request.full_scan {
            service.reconcile(&self.workspace_id, true, false)
        } else {
            let paths = match watcher_path_strings(&request.changed_paths) {
                Ok(paths) => paths,
                Err(error) => {
                    service.queue_full_reconciliation(&self.workspace_id)
                        .map_err(|queue_error| queue_error.to_string())?;
                    return Err(error);
                }
            };
            service.refresh_paths(&self.workspace_id, &paths)
        };
        match result {
            Ok(_) => Ok(()),
            Err(error) => {
                service.queue_full_reconciliation(&self.workspace_id)
                    .map_err(|queue_error| queue_error.to_string())?;
                Err(error.to_string())
            }
        }
    }

    fn complete_reconciliation(&self, _request: &ReconcileRequest) -> Result<(), String> { Ok(()) }

    fn current_manifest(&self, root: &Path, max_file_bytes: u64) -> Result<crate::watch::Manifest, String> {
        self.service().watcher_manifest(&self.workspace_id, root, max_file_bytes)
            .map_err(|error| error.to_string())
    }

    fn queue_shutdown_reconciliation(&self, workspace_id: &str) -> Result<(), String> {
        if workspace_id != self.workspace_id {
            return Err("watcher workspace identity changed".into());
        }
        self.service().queue_full_reconciliation(workspace_id)
            .map_err(|error| error.to_string())
    }

    fn prepare_startup_branch(
        &self,
        current: Option<&crate::watch::BranchMarker>,
    ) -> Result<Option<BranchTransition>, String> {
        self.service().prepare_watch_startup_branch(&self.workspace_id, current)
            .map_err(|error| error.to_string())
    }

    fn branch_observation_failed(&self) -> Result<(), String> {
        self.service().branch_observation_failed(&self.workspace_id)
            .map_err(|error| error.to_string())
    }

    fn startup_reconcile_required(&self, manifest: &crate::watch::Manifest) -> Result<bool, String> {
        self.service().startup_reconcile_required(&self.workspace_id, manifest)
            .map_err(|error| error.to_string())
    }
}

fn watcher_path_strings(paths: &[PathBuf]) -> Result<Vec<String>, String> {
    paths.iter().map(|path| {
        path.to_str().map(str::to_owned)
            .ok_or_else(|| "watcher path is not valid UTF-8".to_owned())
    }).collect()
}

fn validate_config(config: &Config) -> Result<(), ServiceError> {
    config.validate().map_err(|error| {
        ServiceError::new(ErrorCode::InvalidArgument, error.to_string(), false)
    })?;
    config.embedding.provider_kind().map_err(|error| {
        ServiceError::new(ErrorCode::InvalidArgument, error.to_string(), false)
    })?;
    config.embedding.supported_dimension().map_err(|error| {
        ServiceError::new(ErrorCode::InvalidArgument, error.to_string(), false)
    })?;
    let limits = &config.limits;
    if limits.max_file_bytes == 0 || limits.max_file_bytes > i64::MAX as u64
        || limits.max_results == 0 || limits.max_results > 100
        || limits.max_snippet_bytes == 0 || limits.max_snippet_bytes > limits.max_tool_bytes
        || limits.max_tool_bytes == 0 || limits.changeset_expiry_seconds == 0
        || limits.changeset_expiry_seconds > (i64::MAX as u64).saturating_sub(unix_seconds())
        || limits.watcher_queue_paths == 0 || limits.watcher_queue_paths > crate::watch::MAX_QUEUED_PATHS
        || config.reconciliation_interval_seconds == 0 {
        return Err(ServiceError::new(ErrorCode::InvalidArgument, "configured limits are invalid", false));
    }
    Ok(())
}

fn build_configured_provider(
    config: &EmbeddingConfig,
) -> Result<(ProfileIdentity, Box<dyn EmbeddingProvider + Send>, String, usize), ServiceError> {
    config.validate().map_err(|_| {
        ServiceError::new(ErrorCode::InvalidArgument, "embedding configuration is invalid", false)
    })?;
    let provider_kind = config.provider_kind().map_err(|_| {
        ServiceError::new(ErrorCode::InvalidArgument, "embedding provider is unsupported", false)
    })?;
    if provider_kind == EmbeddingProviderKind::FastEmbed && config.revision.is_some() {
        return Err(ServiceError::new(
            ErrorCode::InvalidArgument,
            "FastEmbed model revisions are not configurable in this build",
            false,
        ));
    }
    let expected_dimension = config.supported_dimension().map_err(|_| {
        ServiceError::new(ErrorCode::InvalidArgument, "embedding model dimension is unsupported", false)
    })?;
    let (provider, adapter_fingerprint, dimension, endpoint): (
        Box<dyn EmbeddingProvider + Send>, String, usize, Option<String>,
    ) = match provider_kind {
        EmbeddingProviderKind::FastEmbed => {
            let cache = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".cache"))
                .join("workspace-index-mcp/models");
            let provider = LocalEmbedder::with_batch_size(cache, config.batch_size).map_err(|_| {
                ServiceError::new(ErrorCode::EmbeddingUnavailable, "configured local embedding model could not be initialized", false)
            })?;
            let fingerprint = provider.fingerprint();
            let dimension = provider.dimension();
            (Box::new(provider), fingerprint, dimension, None)
        }
        EmbeddingProviderKind::Ollama | EmbeddingProviderKind::OpenAiCompatible => {
            if !config.allow_remote {
                return Err(ServiceError::new(ErrorCode::InvalidArgument, "remote embedding requires allow_remote = true", false));
            }
            let provider = RemoteEmbedder::from_config(config).map_err(|_| {
                ServiceError::new(ErrorCode::InvalidArgument, "configured remote embedding credentials or endpoint are unavailable", false)
            })?;
            let fingerprint = provider.fingerprint();
            let dimension = provider.dimension();
            let endpoint = Some(provider.normalized_endpoint().to_owned());
            (Box::new(provider), fingerprint, dimension, endpoint)
        }
    };
    if dimension != expected_dimension {
        return Err(ServiceError::new(ErrorCode::InvalidArgument, "embedding provider dimension does not match configuration", false));
    }
    let identity = profile_identity(config, &adapter_fingerprint, dimension, endpoint)?;
    Ok((identity, provider, adapter_fingerprint, dimension))
}

fn profile_identity(
    config: &EmbeddingConfig,
    adapter_fingerprint: &str,
    dimension: usize,
    endpoint: Option<String>,
) -> Result<ProfileIdentity, ServiceError> {
    let provider_kind = config.provider_kind().map_err(|_| {
        ServiceError::new(ErrorCode::InvalidArgument, "embedding provider is unsupported", false)
    })?;
    let model = config.model.trim().to_owned();
    let revision = config.revision.as_ref().map(|revision| revision.trim().to_owned());
    let hnsw_metric = match config.metric {
        EmbeddingMetric::Cosine => "cosine",
        EmbeddingMetric::L2 => "l2",
        EmbeddingMetric::InnerProduct => "ip",
    }.to_owned();
    let dimension_text = dimension.to_string();
    let batch_size_text = config.batch_size.to_string();
    let timeout_text = config.timeout_seconds.to_string();
    let mut digest = Sha256::new();
    for field in [
        PROFILE_FINGERPRINT_VERSION,
        provider_kind.as_str(),
        endpoint.as_deref().unwrap_or(""),
        model.as_str(),
        revision.as_deref().unwrap_or(""),
        dimension_text.as_str(),
        config.metric.as_str(),
        config.query_template_version.as_str(),
        config.document_template_version.as_str(),
        config.api_key_env.as_deref().unwrap_or(""),
        batch_size_text.as_str(),
        timeout_text.as_str(),
        adapter_fingerprint,
    ] {
        digest_field(&mut digest, field);
    }
    for delay in &config.retry_delays_ms {
        digest.update(delay.to_be_bytes());
    }
    let fingerprint = format!("{}:{:x}", PROFILE_FINGERPRINT_VERSION, digest.finalize());
    let hash = fingerprint.rsplit(':').next().unwrap_or_default();
    Ok(ProfileIdentity {
        profile_id: format!("profile-{hash}"),
        provider_kind: provider_kind.as_str().to_owned(),
        endpoint,
        model,
        revision,
        dimension,
        metric: hnsw_metric,
        api_key_env: config.api_key_env.clone(),
        query_template_version: config.query_template_version.clone(),
        document_template_version: config.document_template_version.clone(),
        batch_size: config.batch_size,
        timeout_seconds: config.timeout_seconds,
        retry_delays_ms: config.retry_delays_ms.clone(),
    })
}

fn digest_field(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value.as_bytes());
}

fn ensure_profile_identity(
    connection: &Connection,
    identity: &ProfileIdentity,
) -> Result<(), ServiceError> {
    let retries = serde_json::to_string(&identity.retry_delays_ms).map_err(|_| {
        ServiceError::new(ErrorCode::Conflict, "embedding profile configuration could not be saved", false)
    })?;
    connection.execute(
        "INSERT OR IGNORE INTO embedding_profiles (
            id, provider_kind, endpoint, model, revision, dimension, metric, state, created_at,
            api_key_env, query_template_version, document_template_version,
            batch_size, timeout_seconds, retry_delays_ms
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'building', ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            identity.profile_id, identity.provider_kind, identity.endpoint, identity.model,
            identity.revision, identity.dimension as i64, identity.metric, unix_seconds() as i64,
            identity.api_key_env, identity.query_template_version, identity.document_template_version,
            identity.batch_size as i64, identity.timeout_seconds as i64, retries,
        ],
    ).map_err(map_database_error)?;
    let stored = connection.query_row(
        "SELECT provider_kind, endpoint, model, revision, dimension, metric,
                api_key_env, query_template_version, document_template_version,
                batch_size, timeout_seconds, retry_delays_ms
         FROM embedding_profiles WHERE id=?1",
        [&identity.profile_id],
        |row| Ok((
            row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?,
            row.get::<_, String>(2)?, row.get::<_, Option<String>>(3)?,
            row.get::<_, i64>(4)?, row.get::<_, String>(5)?,
            row.get::<_, Option<String>>(6)?, row.get::<_, String>(7)?,
            row.get::<_, String>(8)?, row.get::<_, i64>(9)?,
            row.get::<_, i64>(10)?, row.get::<_, String>(11)?,
        )),
    ).map_err(map_database_error)?;
    let expected = (
        identity.provider_kind.clone(), identity.endpoint.clone(), identity.model.clone(),
        identity.revision.clone(), identity.dimension as i64, identity.metric.clone(),
        identity.api_key_env.clone(), identity.query_template_version.clone(),
        identity.document_template_version.clone(), identity.batch_size as i64,
        identity.timeout_seconds as i64, retries,
    );
    if stored != expected {
        return Err(ServiceError::new(ErrorCode::Conflict, "embedding profile identity conflicts with persisted metadata", false));
    }
    connection.execute(
        "UPDATE embedding_profiles SET state='building'
         WHERE id=?1 AND state IN ('failed','retired')
           AND NOT EXISTS (
             SELECT 1 FROM workspace_index_state
             WHERE active_profile_id=?1 OR desired_profile_id=?1
           )",
        [&identity.profile_id],
    ).map_err(map_database_error)?;
    Ok(())
}

fn queue_full_reconcile_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    workspace_id: &str,
) -> Result<(), ServiceError> {
    transaction.execute(
        "INSERT OR IGNORE INTO index_jobs
            (workspace_id, relative_path, priority, state, created_at)
         VALUES (?1, '', 0, 'queued', ?2)",
        params![workspace_id, unix_seconds() as i64],
    ).map_err(map_database_error)?;
    Ok(())
}

fn transition_index_job(
    transaction: &rusqlite::Transaction<'_>,
    job_id: i64,
    workspace_id: &str,
    from: &str,
    to: &str,
) -> Result<(), ServiceError> {
    let changed = transaction.execute(
        "UPDATE OR IGNORE index_jobs SET state=?1
         WHERE id=?2 AND workspace_id=?3 AND state=?4",
        params![to, job_id, workspace_id, from],
    ).map_err(map_database_error)?;
    if changed == 0 {
        transaction.execute(
            "DELETE FROM index_jobs WHERE id=?1 AND workspace_id=?2 AND state=?3",
            params![job_id, workspace_id, from],
        ).map_err(map_database_error)?;
    }
    Ok(())
}

fn mark_pending_cleanup_directory_unavailable(
    connection: &Mutex<Connection>,
) -> Result<usize, ServiceError> {
    let mut guard = connection.lock();
    let transaction = guard.transaction().map_err(map_database_error)?;
    let terminal_ids = {
        let mut statement = transaction.prepare(
            "SELECT operation_id FROM pending_changesets
             WHERE state IN ('Applied','Partial','EffectUnknown','Expired','Discarded')",
        ).map_err(map_database_error)?;
        statement.query_map([], |row| row.get::<_, String>(0))
            .map_err(map_database_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_database_error)?
    };
    let queued_at = unix_seconds() as i64;
    for operation_id in terminal_ids {
        transaction.execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES (?1, 'terminal', ?2) ON CONFLICT(operation_id) DO NOTHING",
            params![operation_id, queued_at],
        ).map_err(map_database_error)?;
    }
    transaction.execute(
        "UPDATE changeset_payload_cleanup
         SET attempts=attempts+1, last_error='pending_directory_unavailable'",
        [],
    ).map_err(map_database_error)?;
    let pending = transaction.query_row(
        "SELECT count(*) FROM changeset_payload_cleanup",
        [],
        |row| row.get::<_, i64>(0),
    ).map_err(map_database_error)?;
    transaction.commit().map_err(map_database_error)?;
    usize::try_from(pending)
        .map_err(|_| ServiceError::new(ErrorCode::Conflict, "cleanup queue count is invalid", false))
}

fn parse_embedding_metric(value: &str) -> Option<EmbeddingMetric> {
    match value {
        "cosine" => Some(EmbeddingMetric::Cosine),
        "l2" | "l2sq" => Some(EmbeddingMetric::L2),
        "ip" | "inner_product" => Some(EmbeddingMetric::InnerProduct),
        _ => None,
    }
}
fn count_for_workspace(connection: &Connection, table: &str, workspace_id: &str) -> Result<u64, ServiceError> {
    let query = match table {
        "files" => "SELECT count(*) FROM files WHERE workspace_id=?1",
        "chunks" => "SELECT count(*) FROM chunks WHERE workspace_id=?1",
        _ => return Err(ServiceError::new(ErrorCode::InvalidArgument, "unsupported status query", false)),
    };
    connection.query_row(query, [workspace_id], |row| row.get::<_, i64>(0))
        .map(|count| count.max(0) as u64).map_err(map_database_error)
}

const MAX_GRAPH_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_GRAPH_NODES: usize = 100_000;
const MAX_GRAPH_EDGES: usize = 500_000;
const MAX_GRAPH_PAGE_EDGES_PER_NODE: usize = 5;
const MAX_CACHED_GRAPH_BYTES: usize = MAX_GRAPH_FILE_BYTES as usize;
const MAX_CACHED_GRAPH_SNAPSHOTS: usize = 128;

fn read_manifest_entries(
    connection: &Connection,
    workspace_id: &str,
) -> Result<Vec<WorkspaceManifestEntry>, ServiceError> {
    let mut statement = connection.prepare(
        "SELECT relative_path, content_hash, file_generation FROM files
         WHERE workspace_id=?1 ORDER BY relative_path",
    ).map_err(map_database_error)?;
    statement.query_map([workspace_id], |row| Ok(WorkspaceManifestEntry {
        path: row.get(0)?, blake3: row.get(1)?, generation: row.get(2)?,
    })).map_err(map_database_error)?.collect::<Result<Vec<_>, _>>()
        .map_err(|_| ServiceError::new(ErrorCode::Conflict, "workspace manifest could not be read", false))
}

fn manifest_version(entries: &[WorkspaceManifestEntry], generation: u64) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&generation.to_le_bytes());
    for entry in entries {
        hasher.update(&(entry.path.len() as u64).to_le_bytes());
        hasher.update(entry.path.as_bytes());
        hasher.update(entry.blake3.as_bytes());
        hasher.update(&entry.generation.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn empty_graph_page(workspace_id: &str, generation: u64) -> WorkspaceGraphPage {
    WorkspaceGraphPage {
        workspace_id: workspace_id.to_owned(),
        visible_generation: generation,
        nodes: Vec::new(),
        edges: Vec::new(),
        truncated: false,
        nodes_truncated: false,
        edges_truncated: false,
        next_cursor: None,
    }
}

fn graph_page_candidate(
    workspace_id: &str,
    generation: u64,
    graph: &GraphSnapshot,
    payload: &GraphCursorPayload,
    node_count: usize,
    edge_limit: usize,
    secret: &[u8],
) -> Result<WorkspaceGraphPage, ServiceError> {
    graph_page_candidate_with_counts(
        workspace_id, generation, graph, payload, node_count,
        edge_limit, secret,
    )
}

fn graph_page_candidate_with_counts(
    workspace_id: &str,
    generation: u64,
    graph: &GraphSnapshot,
    payload: &GraphCursorPayload,
    node_count: usize,
    edge_count: usize,
    secret: &[u8],
) -> Result<WorkspaceGraphPage, ServiceError> {
    let node_end = payload.offset.saturating_add(node_count).min(graph.nodes.len());
    let nodes = graph.nodes[payload.offset..node_end].to_vec();
    let ids = nodes.iter().map(|node| node.id.as_str()).collect::<HashSet<_>>();
    let page_edges = graph.edges.iter()
        .filter(|edge| ids.contains(edge.source.as_str()) && ids.contains(edge.target.as_str()))
        .take(edge_count)
        .cloned()
        .collect::<Vec<_>>();
    let nodes_truncated = node_end < graph.nodes.len();
    let page_edge_count = graph.edges.iter().filter(|edge| {
        ids.contains(edge.source.as_str()) && ids.contains(edge.target.as_str())
    }).count();
    let edges_truncated = page_edges.len() < page_edge_count;
    let next_cursor = if nodes_truncated {
        let mut next = payload.clone();
        next.offset = node_end;
        Some(encode_graph_cursor(secret, &next)?)
    } else {
        None
    };
    Ok(WorkspaceGraphPage {
        workspace_id: workspace_id.to_owned(),
        visible_generation: generation,
        nodes,
        edges: page_edges,
        truncated: nodes_truncated || edges_truncated,
        nodes_truncated,
        edges_truncated,
        next_cursor,
    })
}

fn graph_page_fits(page: &WorkspaceGraphPage, warnings: &[Warning], max_bytes: usize) -> bool {
    let result = ToolResult::Success {
        data: page,
        warnings: warnings.to_vec(),
        meta: crate::model::Meta { request_id: String::new() },
    };
    serde_json::to_vec(&result).is_ok_and(|serialized| serialized.len() <= max_bytes)
}

fn stale_graph_cursor() -> ServiceError {
    ServiceError::new(ErrorCode::StaleCursor, "Graph cursor is invalid or stale", false)
}

fn encode_graph_cursor(secret: &[u8], payload: &GraphCursorPayload) -> Result<String, ServiceError> {
    let bytes = serde_json::to_vec(payload).map_err(|_| {
        ServiceError::new(ErrorCode::Conflict, "Graph cursor could not be encoded", false)
    })?;
    let mut mac = GraphHmac::new_from_slice(secret).map_err(|_| {
        ServiceError::new(ErrorCode::Conflict, "Graph cursor secret is invalid", false)
    })?;
    mac.update(&bytes);
    Ok(format!("{}.{}", encode_cursor_hex(&bytes), encode_cursor_hex(&mac.finalize().into_bytes())))
}

fn decode_graph_cursor(secret: &[u8], cursor: &str) -> Result<GraphCursorPayload, ServiceError> {
    if cursor.len() > 8192 { return Err(stale_graph_cursor()); }
    let (payload_hex, signature_hex) = cursor.split_once('.').ok_or_else(stale_graph_cursor)?;
    if signature_hex.contains('.') { return Err(stale_graph_cursor()); }
    let bytes = decode_cursor_hex(payload_hex).ok_or_else(stale_graph_cursor)?;
    let signature = decode_cursor_hex(signature_hex).ok_or_else(stale_graph_cursor)?;
    let mut mac = GraphHmac::new_from_slice(secret).map_err(|_| stale_graph_cursor())?;
    mac.update(&bytes);
    mac.verify_slice(&signature).map_err(|_| stale_graph_cursor())?;
    serde_json::from_slice(&bytes).map_err(|_| stale_graph_cursor())
}

fn encode_cursor_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn decode_cursor_hex(value: &str) -> Option<Vec<u8>> {
    fn digit(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().as_chunks::<2>().0 {
        output.push((digit(pair[0])? << 4) | digit(pair[1])?);
    }
    value.len().is_multiple_of(2).then_some(output)
}

fn read_graph_snapshot(
    cache: &Mutex<GraphSnapshotCache>,
    metadata: &crate::index::graphify::GraphSnapshotMetadata,
    path: &Path,
) -> Result<Arc<GraphSnapshot>, String> {
    let bytes = read_graph_bytes(path)?;
    let fingerprint = blake3::hash(&bytes).to_hex().to_string();
    let key = GraphSnapshotKey { metadata: metadata.clone(), fingerprint: fingerprint.clone() };
    let mut cache = cache.lock();
    if let Some(snapshot) = cache.snapshots.get(&key) {
        return Ok(Arc::clone(snapshot));
    }
    let published: crate::index::graphify::PublishedGraphData =
        serde_json::from_slice(&bytes).map_err(|_| "Graphify output is malformed")?;
    let crate::index::graphify::PublishedGraphData {
        metadata: stored_metadata,
        nodes,
        edges,
        content_hashes,
    } = published;
    let graph = crate::index::graphify::GraphData { nodes, edges };
    if stored_metadata != *metadata
        || graph.nodes.len() > MAX_GRAPH_NODES
        || graph.edges.len() > MAX_GRAPH_EDGES
        || !crate::index::graphify::validate_graph_data(&graph)
        || !crate::index::graphify::validate_content_hashes(&content_hashes)
        || graph.nodes.iter().any(|node| !content_hashes.contains_key(&node.relative_path))
    {
        return Err("Graphify output exceeds resource limits or contains invalid identity, hashes, or records".into());
    }
    let crate::index::graphify::GraphData { mut nodes, mut edges } = graph;
    nodes.sort_by(|left, right| left.id.cmp(&right.id));
    edges.sort_by(|left, right| left.source.cmp(&right.source)
        .then_with(|| left.target.cmp(&right.target))
        .then_with(|| left.kind.cmp(&right.kind)));
    let snapshot = Arc::new(GraphSnapshot {
        metadata: metadata.clone(),
        fingerprint,
        nodes,
        edges,
        content_hashes,
        content_bytes: bytes.len(),
    });
    let obsolete = cache.snapshots.keys()
        .filter(|candidate| candidate.metadata == *metadata)
        .cloned()
        .collect::<Vec<_>>();
    for candidate in obsolete {
        if let Some(removed) = cache.snapshots.remove(&candidate) {
            cache.content_bytes = cache.content_bytes.saturating_sub(removed.content_bytes);
        }
    }
    while cache.snapshots.len() >= MAX_CACHED_GRAPH_SNAPSHOTS
        || cache.content_bytes.saturating_add(snapshot.content_bytes) > MAX_CACHED_GRAPH_BYTES
    {
        let Some(candidate) = cache.snapshots.keys().next().cloned() else { break; };
        if let Some(removed) = cache.snapshots.remove(&candidate) {
            cache.content_bytes = cache.content_bytes.saturating_sub(removed.content_bytes);
        }
    }
    cache.content_bytes = cache.content_bytes.saturating_add(snapshot.content_bytes);
    cache.snapshots.insert(key, Arc::clone(&snapshot));
    Ok(snapshot)
}

fn graph_sources_match(
    workspace: &WorkspaceFs,
    content_hashes: &BTreeMap<String, String>,
    max_file_bytes: u64,
) -> bool {
    content_hashes.iter().all(|(relative_path, expected_hash)| {
        workspace.read(Path::new(relative_path), max_file_bytes).is_ok_and(|bytes| {
            blake3::hash(&bytes).to_hex().as_str() == expected_hash
        })
    })
}


fn read_graph_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound { "Graphify output is missing" }
        else { "Graphify output could not be read" }
    })?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_GRAPH_FILE_BYTES {
        return Err("Graphify output is invalid or too large".into());
    }
    let bytes = fs::read(path).map_err(|_| "Graphify output could not be read")?;
    if bytes.len() as u64 > MAX_GRAPH_FILE_BYTES {
        return Err("Graphify output is invalid or too large".into());
    }
    Ok(bytes)
}


fn enqueue_refresh_paths(
    connection: &mut Connection,
    paths: &[(String, String)],
) -> Result<(), ServiceError> {
    if paths.is_empty() {
        return Ok(());
    }
    
         
        let transaction = connection.transaction().map_err(map_database_error)?;
    let queued_at = unix_seconds() as i64;
    let mut affected_workspaces = std::collections::BTreeSet::new();
    for (workspace_id, relative_path) in paths {
        affected_workspaces.insert(workspace_id);
        transaction.execute(
            "INSERT OR IGNORE INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
             VALUES (?1, ?2, 0, 'queued', ?3)",
            params![workspace_id, relative_path, queued_at],
        ).map_err(map_database_error)?;
    }
    for workspace_id in affected_workspaces {
        transaction.execute(
            "UPDATE workspace_index_state SET dirty=1, health='reconcile_required', updated_at=?1
             WHERE workspace_id=?2",
            params![queued_at, workspace_id],
        ).map_err(map_database_error)?;
    }
    transaction.commit().map_err(map_database_error)
}

fn map_state_error(error: StateError) -> ServiceError {
    match error {
        StateError::Busy => ServiceError::new(ErrorCode::Busy, "workspace state is busy", true),
        StateError::InvalidWorkspaceId(_) => ServiceError::new(ErrorCode::InvalidArgument, "workspace_id is invalid", false),
        StateError::InvalidWorkspaceRoot(_) | StateError::NotGitWorktree(_) =>
            ServiceError::new(ErrorCode::InvalidArgument, "workspace root is invalid or is not an allowed Git worktree", false),
        StateError::WorkspaceNotFound(_) => ServiceError::new(ErrorCode::WorkspaceNotFound, "workspace not found", false),
        StateError::WorkspaceOverlap(_) => ServiceError::new(ErrorCode::Conflict, "workspace root conflicts with a registered workspace", false),
        StateError::InvalidVectorOperation =>
            ServiceError::new(ErrorCode::InvalidArgument, "vector operation is invalid", false),
        StateError::UnsupportedSchema(_) => ServiceError::new(ErrorCode::Conflict, "workspace database schema is unsupported", false),
        StateError::Unsupported => ServiceError::new(ErrorCode::Unsupported, "safe filesystem identity or mutation containment is unavailable", false),
        StateError::Database(_) | StateError::Io(_) => map_database_error_message(),
    }
}

fn map_database_error(error: rusqlite::Error) -> ServiceError {
    if matches!(&error, rusqlite::Error::SqliteFailure(failure, _) if failure.code == rusqlite::ErrorCode::DatabaseBusy || failure.code == rusqlite::ErrorCode::DatabaseLocked) {
        ServiceError::new(ErrorCode::Busy, "workspace state is busy", true)
    } else { map_database_error_message() }
}

fn map_database_error_message() -> ServiceError {
    ServiceError::new(ErrorCode::Conflict, "workspace state operation failed", true)
}

fn map_changeset_error(error: ChangesetError) -> ServiceError {
    let message = match error.code {
        ErrorCode::InvalidArgument => "changeset request is invalid",
        ErrorCode::WorkspaceRequired => "workspace_id is required",
        ErrorCode::WorkspaceNotFound => "workspace or source file was not found",
        ErrorCode::ReadOnly => "workspace is read-only",
        ErrorCode::PathEscape => "changeset path is invalid",
        ErrorCode::SymlinkRefused => "changeset path crosses a symlink or leaves the workspace",
        ErrorCode::Conflict => "changeset preconditions or workspace identity changed",
        ErrorCode::Expired => "changeset has expired",
        ErrorCode::Busy => "changeset is busy or has already been claimed",
        ErrorCode::StaleCursor => "changeset state is stale",
        ErrorCode::EmbeddingUnavailable => "embedding provider is unavailable",
        ErrorCode::Unsupported => "source mutation is disabled because safe containment cannot be proven",
    };
    ServiceError::new(error.code, message, error.code == ErrorCode::Busy)
}

fn load_cursor_secret(path: &Path) -> std::io::Result<Vec<u8>> {
    if let Ok(secret) = std::fs::read(path) {
        return validate_cursor_secret(secret);
    }
    let mut secret = Vec::with_capacity(32);
    secret.extend_from_slice(uuid::Uuid::new_v4().as_bytes());
    secret.extend_from_slice(uuid::Uuid::new_v4().as_bytes());
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    { use std::os::unix::fs::OpenOptionsExt; options.mode(0o600); }
    match options.open(path) {
        Ok(mut file) => { file.write_all(&secret)?; file.sync_all()?; Ok(secret) }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => validate_cursor_secret(std::fs::read(path)?),
        Err(error) => Err(error),
    }
}

fn validate_cursor_secret(secret: Vec<u8>) -> std::io::Result<Vec<u8>> {
    if secret.len() == 32 { Ok(secret) }
    else { Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid cursor secret length")) }
}

fn unix_seconds() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
fn is_ignore_control_path(path: &Path) -> bool {
    path.file_name().is_some_and(|name| {
        name == ".gitignore" || name == ".workspace-indexignore" || name == ".augmentignore"
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    struct DeterministicProvider;
    impl EmbeddingProvider for DeterministicProvider {
        fn fingerprint(&self) -> String { "test-deterministic".into() }
        fn dimension(&self) -> usize { 2 }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            Ok(texts.iter().map(|text| vec![text.len() as f32, 1.0]).collect())
        }
    }
    struct PausableProvider {
        pause_next: Arc<std::sync::atomic::AtomicBool>,
        started: mpsc::SyncSender<()>,
        release: Arc<Mutex<mpsc::Receiver<()>>>,
    }
    impl EmbeddingProvider for PausableProvider {
        fn fingerprint(&self) -> String { "test-pausable".into() }
        fn dimension(&self) -> usize { 2 }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            if self.pause_next.swap(false, std::sync::atomic::Ordering::SeqCst) {
                self.started.send(()).map_err(|error| error.to_string())?;
                self.release.lock().recv().map_err(|error| error.to_string())?;
            }
            Ok(vec![vec![1.0, 0.0]; texts.len()])
        }
    }
    struct InvalidResponseProvider { response: Vec<Vec<f32>> }
    impl EmbeddingProvider for InvalidResponseProvider {
        fn fingerprint(&self) -> String { "test-replacement-provider".into() }
        fn dimension(&self) -> usize { 2 }
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, String> { Ok(self.response.clone()) }
    }

    struct MutatingProvider {
        path: std::path::PathBuf,
        changed: std::sync::atomic::AtomicBool,
    }
    impl EmbeddingProvider for MutatingProvider {
        fn fingerprint(&self) -> String { "mutating-test-provider".into() }
        fn dimension(&self) -> usize { 2 }
        fn embed_batch(&self, batch: &[String]) -> Result<Vec<Vec<f32>>, String> {
            if !self.changed.swap(true, std::sync::atomic::Ordering::SeqCst) {
                std::fs::write(&self.path, "fn changed_during_scan() {}")
                    .map_err(|error| error.to_string())?;
            }
            Ok(vec![vec![1.0, 0.0]; batch.len()])
        }
    }



    struct RecordingProvider { texts: Arc<Mutex<Vec<String>>> }
    impl EmbeddingProvider for RecordingProvider {
        fn fingerprint(&self) -> String { "recording-test-provider".into() }
        fn dimension(&self) -> usize { 2 }
        fn embed_batch(&self, batch: &[String]) -> Result<Vec<Vec<f32>>, String> {
            self.texts.lock().extend(batch.iter().cloned());
            Ok(vec![vec![1.0, 0.0]; batch.len()])
        }
    }
    struct AlternateProvider;
    impl EmbeddingProvider for AlternateProvider {
        fn fingerprint(&self) -> String { "test-replacement-provider".into() }
        fn dimension(&self) -> usize { 2 }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            Ok(texts.iter().map(|text| vec![2.0, text.len() as f32 + 1.0]).collect())
        }
    }

    fn semantic_request(workspace_ids: &[&str], query: &str) -> CodeSearchRequest {
        CodeSearchRequest {
            query: query.to_owned(),
            query_vector: None,
            workspace_ids: workspace_ids.iter().map(|id| (*id).to_owned()).collect(),
            profile_id: None,
            languages: Vec::new(),
            path_globs: Vec::new(),
            limit: 10,
            cursor: None,
        }
    }

    fn install_test_provider(
        service: &WorkspaceService,
        profile_id: &str,
        provider: Box<dyn EmbeddingProvider + Send>,
    ) {
        let slot = Arc::new(ProviderSlot {
            fingerprint: provider.fingerprint(),
            dimension: provider.dimension(),
            provider: Mutex::new(provider),
            admission: Arc::clone(&service.provider_gate),
        });
        service.provider_slots.lock().insert(profile_id.to_owned(), slot);
    }

    #[test]
    fn timed_out_refresh_workers_remain_tracked_and_bound_repeated_admission() {
        let workers = IndexWorkerRegistry::default();
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let (finished_tx, finished_rx) = mpsc::sync_channel(1);
        let release_rx = Arc::new(Mutex::new(release_rx));

        let result = workers.run_until_deadline(
            Instant::now() + Duration::from_millis(50),
            {
                let release_rx = Arc::clone(&release_rx);
                move || {
                    started_tx.send(()).map_err(|error| ServiceError::new(
                        ErrorCode::Conflict, error.to_string(), false,
                    ))?;
                    release_rx.lock().recv().map_err(|error| ServiceError::new(
                        ErrorCode::Conflict, error.to_string(), false,
                    ))?;
                    finished_tx.send(()).map_err(|error| ServiceError::new(
                        ErrorCode::Conflict, error.to_string(), false,
                    ))?;
                    Ok(1)
                }
            },
        );
        assert!(result.is_none());
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(workers.tracked_workers(), 1);

        let second_started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let second_result = workers.run_until_deadline(
            Instant::now() + Duration::from_secs(1),
            {
                let second_started = Arc::clone(&second_started);
                move || {
                    second_started.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(2)
                }
            },
        );
        assert!(second_result.is_none());
        assert_eq!(second_started.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(workers.tracked_workers(), 1);

        release_tx.send(()).unwrap();
        finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while workers.tracked_workers() != 0 && Instant::now() < deadline {
            workers.reap_finished();
            std::thread::yield_now();
        }
        assert_eq!(workers.tracked_workers(), 0);
        assert!(matches!(
            workers.run_until_deadline(Instant::now() + Duration::from_secs(1), || Ok(3)),
            Some(Ok(3))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn supported_source_symlink_is_excluded_without_becoming_a_deletion() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        let outside = temp.path().join("outside.rs");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn indexed() {}").expect("source file");
        std::fs::write(&outside, "fn outside() {}").expect("outside file");
        service.register_workspace("scan", &root).expect("workspace registration");
        service.reconcile("scan", false, false).expect("initial index");
        service.connection.lock().execute(
            "INSERT INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
             VALUES ('scan', 'pending.rs', 0, 'queued', 0)",
            [],
        ).expect("queued reconciliation job");

        std::fs::remove_file(root.join("source.rs")).expect("remove source");
        symlink(&outside, root.join("source.rs")).expect("replace source with symlink");
        let report = service.reconcile("scan", false, false).expect("symlink is excluded safely");
        assert_eq!(report.deleted, 0);
        assert_eq!(report.excluded, 1);

        let connection = service.connection.lock();
        let source_rows: i64 = connection.query_row(
            "SELECT count(*) FROM files WHERE workspace_id='scan' AND relative_path='source.rs'",
            [],
            |row| row.get(0),
        ).expect("excluded source row");
        let queued_jobs: i64 = connection.query_row(
            "SELECT count(*) FROM index_jobs WHERE workspace_id='scan' AND state='queued'",
            [],
            |row| row.get(0),
        ).expect("queued reconciliation jobs");
        assert_eq!(source_rows, 0);
        assert_eq!(queued_jobs, 0);
    }

    #[cfg(unix)]
    #[test]
    fn watcher_manifest_records_supported_symlink_as_excluded() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().expect("temporary directory");
        let service = Arc::new(WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens"));
        let root = temp.path().join("workspace");
        let outside = temp.path().join("outside.rs");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn indexed() {}").expect("source file");
        std::fs::write(&outside, "fn outside() {}").expect("outside file");
        service.register_workspace("watched", &root).expect("workspace registration");
        service.reconcile("watched", false, false).expect("initial index");
        let backend = service.watch_backend("watched");
        let expected_hash = blake3::hash(b"fn indexed() {}").to_hex().to_string();
        let manifest = backend.current_manifest(&root, 1024 * 1024).expect("capability manifest");
        assert_eq!(manifest.get(Path::new("source.rs")).map(String::as_str), Some(expected_hash.as_str()));

        std::fs::remove_file(root.join("source.rs")).expect("remove source");
        symlink(&outside, root.join("source.rs")).expect("replace source with symlink");
        let manifest = backend.current_manifest(&root, 1024 * 1024).expect("symlink is excluded");
        assert_eq!(manifest.get(Path::new("source.rs")).map(String::as_str), Some("excluded:symlink"));
    }


    #[test]
    fn scanner_setup_failure_keeps_published_rows_and_queued_jobs() {
        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace-state");
        std::fs::create_dir_all(&root).expect("state directory");
        let service = WorkspaceService::open_with_provider(
            Paths::resolve(None, Some(&root)).expect("workspace-local state paths"),
            test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        service.register_workspace("scan", &root).expect("workspace registration");
        {
            let connection = service.connection.lock();
            connection.execute(
                "INSERT INTO files
                    (workspace_id, relative_path, content_hash, size_bytes, visible_generation,
                     file_generation, parser_version, chunker_version)
                 VALUES ('scan', 'retained.rs', 'previous-hash', 10, 1, 1, 1, 1)",
                [],
            ).expect("prior indexed row");
            connection.execute(
                "INSERT INTO workspace_index_state
                    (workspace_id, generation, updated_at, building_generation, visible_generation, dirty, health)
                 VALUES ('scan', 1, 0, 1, 1, 0, 'ready')",
                [],
            ).expect("published generation");
            connection.execute(
                "INSERT INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
                 VALUES ('scan', 'pending.rs', 0, 'queued', 0)",
                [],
            ).expect("queued job");
        }

        assert!(service.reconcile("scan", false, false).is_err());
        let connection = service.connection.lock();
        let (files, visible_generation): (i64, i64) = connection.query_row(
            "SELECT (SELECT count(*) FROM files WHERE workspace_id='scan'),
                    visible_generation FROM workspace_index_state WHERE workspace_id='scan'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).expect("unchanged published state");
        let queued: i64 = connection.query_row(
            "SELECT count(*) FROM index_jobs WHERE workspace_id='scan' AND state='queued'",
            [], |row| row.get(0),
        ).expect("queued work remains");
        assert_eq!((files, visible_generation), (1, 1));
        assert!(queued >= 2);
    }

    #[test]
    fn configured_file_size_limit_excludes_oversized_sources() {
        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("large.rs"), format!("// {}\n", "x".repeat(128)))
            .expect("oversized source");
        let config = Config {
            allow_non_git_roots: true,
            limits: crate::config::Limits { max_file_bytes: 32, ..crate::config::Limits::default() },
            ..Config::default()
        };
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), config, Box::new(DeterministicProvider),
        ).expect("service opens");
        service.register_workspace("limited", &root).expect("workspace registration");
        let report = service.reconcile("limited", false, false).expect("bounded reconciliation");
        assert_eq!(report.updated, 0);
        assert_eq!(service.index_status("limited").expect("index status").file_count, 0);
    }

    #[test]
    fn representation_version_changes_rebuild_unchanged_content() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("same.rs"), "fn stable() {}").expect("source file");
        service.register_workspace("versions", &root).expect("workspace registration");
        assert_eq!(service.reconcile("versions", false, false).expect("initial reconciliation").updated, 1);
        service.connection.lock().execute(
            "UPDATE files SET parser_version=1, chunker_version=1
             WHERE workspace_id='versions' AND relative_path='same.rs'",
            [],
        ).expect("simulate old representation");

        let report = service.reconcile("versions", false, false).expect("rebuild old representation");
        assert_eq!(report.updated, 1);
        assert_eq!(report.unchanged, 0);
    }

    #[test]
    fn reconciliation_does_not_publish_a_manifest_changed_during_embedding() {
        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let source = root.join("source.rs");
        std::fs::write(&source, "fn original() {}").expect("source file");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp),
            test_config(),
            Box::new(MutatingProvider {
                path: source,
                changed: std::sync::atomic::AtomicBool::new(false),
            }),
        ).expect("service opens");
        service.register_workspace("unstable", &root).expect("workspace registration");

        assert!(service.reconcile("unstable", false, false).is_err());
        let connection = service.connection.lock();
        let (files, visible_generation, queued): (i64, i64, i64) = connection.query_row(
            "SELECT (SELECT count(*) FROM files WHERE workspace_id='unstable'),
                    visible_generation,
                    (SELECT count(*) FROM index_jobs WHERE workspace_id='unstable' AND state='queued')
             FROM workspace_index_state WHERE workspace_id='unstable'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).expect("unstable reconciliation state");
        assert_eq!((files, visible_generation), (0, 0));
        assert!(queued > 0);
    }

    #[test]
    fn invalid_embedding_batches_keep_lexical_index_and_degrade_semantics() {
        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn lexical_only() {}").expect("source file");

        for (index, response) in [
            Vec::new(),
            vec![vec![1.0]],
            vec![vec![1.0, f32::NAN]],
            vec![vec![1.0, f32::INFINITY]],
        ].into_iter().enumerate() {
            let data = temp.path().join(format!("data-{index}"));
            let paths = Paths::resolve(None, Some(&data)).expect("temporary paths");
            let service = WorkspaceService::open_with_provider(
                paths, test_config(), Box::new(InvalidResponseProvider { response }),
            ).expect("service opens");
            service.register_workspace("scan", &root).expect("workspace registration");
            let report = service.reconcile("scan", false, false).expect("lexical reconciliation");
            assert!(report.degraded_semantic);

            let connection = service.connection.lock();
            let lexical_chunks: i64 = connection.query_row(
                "SELECT count(*) FROM chunks WHERE workspace_id='scan'", [], |row| row.get(0),
            ).expect("lexical chunks");
            let embeddings: i64 = connection.query_row(
                "SELECT count(*) FROM chunk_embeddings WHERE workspace_id='scan'", [], |row| row.get(0),
            ).expect("semantic vectors");
            let vector_ops: i64 = connection.query_row(
                "SELECT count(*) FROM vector_ops WHERE workspace_id='scan'", [], |row| row.get(0),
            ).expect("vector operations");
            assert!(lexical_chunks > 0);
            assert_eq!(embeddings, 0);
            assert_eq!(vector_ops, 0);
            drop(connection);
            let search = service.code_search(semantic_request(&["scan"], "lexical_only"))
                .expect("lexical search survives semantic failure");
            assert!(!search.data().expect("successful lexical result").items.is_empty());
            assert!(search.warnings().iter().any(|warning| warning.code == "INDEX_DEGRADED"));
            assert_eq!(service.index_status("scan").expect("index status").state, IndexState::Degraded);
        }
    }

    #[test]
    fn ast_chunks_preserve_adjacent_functions_oversized_class_and_source_ranges() {
        let temp = TempDir::new().expect("temporary directory");
        let embedded_texts = Arc::new(Mutex::new(Vec::new()));
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(),
            Box::new(RecordingProvider { texts: Arc::clone(&embedded_texts) }),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let mut source = String::from(
            "/** retained function docs */\nfunction adjacentOne() {\n  return 1;\n}\n\
             function adjacentTwo() {\n  return 2;\n}\n\
             class Large {\n  small() {\n    return 3;\n  }\n  huge() {\n",
        );
        for index in 0..165 {
            source.push_str(&format!(
                "    const value{index} = \"abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz\";\n"
            ));
        }
        source.push_str("  }\n}\n");
        std::fs::write(root.join("fixture.ts"), &source).expect("TypeScript fixture");
        service.register_workspace("ast", &root).expect("workspace registration");
        service.reconcile("ast", false, false).expect("AST reconciliation");

        let connection = service.connection.lock();
        let chunks = connection.prepare(
            "SELECT start_line, end_line, content FROM chunks
             WHERE workspace_id='ast' AND relative_path='fixture.ts' ORDER BY chunk_index",
        ).expect("chunk query").query_map([], |row| {
            let start_line = usize::try_from(row.get::<_, i64>(0)?).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Integer,
                    Box::new(error),
                )
            })?;
            let end_line = usize::try_from(row.get::<_, i64>(1)?).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Integer,
                    Box::new(error),
                )
            })?;
            Ok((start_line, end_line, row.get::<_, String>(2)?))
        }).expect("chunk rows").collect::<Result<Vec<_>, _>>().expect("chunk records");
        let first = chunks.iter().find(|(_, _, content)| content.contains("function adjacentOne")).expect("first function region");
        assert!(!first.2.contains("function adjacentTwo"));
        let second = chunks.iter().find(|(_, _, content)| content.contains("function adjacentTwo")).expect("second function region");
        assert!(!second.2.contains("function adjacentOne"));
        assert!(chunks.iter().any(|(_, _, content)| content.contains("small()")));
        assert!(chunks.iter().any(|(_, _, content)| content.contains("value164")));
        let embedded_texts = embedded_texts.lock();
        assert!(embedded_texts.iter().all(|text| text.chars().count().div_ceil(4) <= 1_200));
        assert!(embedded_texts.iter().any(|text| {
            text.contains("path: fixture.ts")
                && text.contains("language: typescript")
                && text.contains("parent: Large")
                && text.contains("signature:")
        }));
        assert!(embedded_texts.iter().any(|text| text.contains("retained function docs")));
        for (start_line, end_line, content) in &chunks {
            let has_matching_source_range = source.match_indices(content).any(|(byte_start, matched)| {
                let byte_end = byte_start + matched.len();
                let actual_start = source[..byte_start].bytes().filter(|byte| *byte == b'\n').count() + 1;
                let actual_end = source[..byte_end.saturating_sub(1)].bytes().filter(|byte| *byte == b'\n').count() + 1;
                (*start_line, *end_line) == (actual_start, actual_end)
            });
            assert!(has_matching_source_range, "chunk lines must identify an exact source slice: {content:?}");
        }
    }

    #[test]
    fn data_directory_inside_workspace_is_not_indexed() {
        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        let data = root.join(".private-index");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let paths = Paths::resolve(None, Some(&data)).expect("workspace-local data paths");
        let service = WorkspaceService::open_with_provider(
            paths, test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        std::fs::write(root.join("source.rs"), "fn visible() {}").expect("source file");
        std::fs::write(data.join("private.rs"), "fn hidden_state() {}").expect("private data");
        service.register_workspace("data-root", &root).expect("workspace registration");
        service.reconcile("data-root", false, false).expect("workspace reconciliation");
        let connection = service.connection.lock();
        let indexed: i64 = connection.query_row(
            "SELECT count(*) FROM files WHERE workspace_id='data-root'", [], |row| row.get(0),
        ).expect("indexed file count");
        assert_eq!(indexed, 1);
        let private: i64 = connection.query_row(
            "SELECT count(*) FROM files WHERE workspace_id='data-root' AND relative_path LIKE '.private-index/%'",
            [], |row| row.get(0),
        ).expect("private file count");
        assert_eq!(private, 0);
    }
    #[test]
    fn targeted_refresh_updates_changed_paths_without_reembedding_unchanged_files() {
        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("changed.rs"), "fn old_target_content() {}").expect("changed source");
        std::fs::write(root.join("steady.rs"), "fn steady_target_content() {}").expect("steady source");
        let embedded = Arc::new(Mutex::new(Vec::new()));
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp),
            test_config(),
            Box::new(RecordingProvider { texts: Arc::clone(&embedded) }),
        ).expect("service opens");
        service.register_workspace("targeted", &root).expect("workspace registration");
        service.reconcile("targeted", true, false).expect("initial index");

        embedded.lock().clear();
        std::fs::write(root.join("changed.rs"), "fn new_target_content() {}").expect("modify source");
        let report = service.refresh_paths("targeted", &["changed.rs".into()])
            .expect("targeted modification refresh");
        assert_eq!(report.updated, 1);
        let embedded_texts = embedded.lock().clone();
        assert!(embedded_texts.iter().any(|text| text.contains("new_target_content")));
        assert!(!embedded_texts.iter().any(|text| text.contains("steady_target_content")));

        std::fs::write(root.join("created.rs"), "fn created_target_content() {}").expect("create source");
        assert_eq!(
            service.refresh_paths("targeted", &["created.rs".into()]).expect("create refresh").updated,
            1,
        );
        std::fs::rename(root.join("changed.rs"), root.join("renamed.rs")).expect("rename source");
        let rename = service.refresh_paths("targeted", &["changed.rs".into(), "renamed.rs".into()])
            .expect("rename refresh");
        assert_eq!(rename.updated, 1);
        assert_eq!(rename.deleted, 1);
        std::fs::remove_file(root.join("created.rs")).expect("delete source");
        assert_eq!(
            service.refresh_paths("targeted", &["created.rs".into()]).expect("delete refresh").deleted,
            1,
        );

        let connection = service.connection.lock();
        let paths = {
            let mut statement = connection.prepare(
                "SELECT relative_path FROM files WHERE workspace_id='targeted' ORDER BY relative_path",
            ).expect("indexed paths query");
            statement.query_map([], |row| row.get::<_, String>(0))
                .expect("indexed paths").collect::<std::result::Result<Vec<_>, _>>()
                .expect("indexed paths rows")
        };
        assert_eq!(paths, vec!["renamed.rs".to_owned(), "steady.rs".to_owned()]);
    }
    #[test]
    fn indexed_non_git_workspace_stays_current_across_branch_polls() {
        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn non_git_stable() {}").expect("source file");
        let service = Arc::new(WorkspaceService::open_with_provider(
            test_paths(&temp),
            test_config(),
            Box::new(DeterministicProvider),
        ).expect("service opens"));
        service.register_workspace("non-git", &root).expect("workspace registration");
        service.reconcile("non-git", true, false).expect("initial index");
        assert_eq!(crate::watch::read_branch_marker(&root).unwrap(), None);
        let generation = service.index_generation("non-git").expect("visible generation");
        let config = crate::watch::WatchConfig {
            branch_poll_interval: std::time::Duration::from_millis(20),
            periodic_reconcile_interval: std::time::Duration::from_secs(10),
            ..crate::watch::WatchConfig::default()
        };
        let mut watcher = crate::watch::WorkspaceWatcher::start(
            "non-git",
            &root,
            service.watch_backend("non-git"),
            config,
        ).expect("workspace watcher starts");
        std::thread::sleep(std::time::Duration::from_millis(120));

        assert_eq!(watcher.status().full_reconcile_count, 0);
        assert_eq!(service.index_generation("non-git").expect("stable generation"), generation);
        watcher.stop();
    }
    #[test]
    fn timed_out_watcher_retains_service_until_shutdown_reconciliation_is_queued() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let root = temp.path().join("watch-retention");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn source() {}\n").expect("source file");
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let pause_next = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let provider = PausableProvider {
            pause_next: Arc::clone(&pause_next),
            started: started_tx,
            release: Arc::new(Mutex::new(release_rx)),
        };
        let service = Arc::new(WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(provider),
        ).expect("service opens"));
        service.register_workspace("watch-retention", &root).expect("workspace registration");
        service.reconcile("watch-retention", true, false).expect("initial index");

        let config = crate::watch::WatchConfig {
            debounce: Duration::from_millis(20),
            branch_poll_interval: Duration::from_secs(10),
            periodic_reconcile_interval: Duration::from_secs(10),
            ..crate::watch::WatchConfig::default()
        };
        let mut watcher = crate::watch::WorkspaceWatcher::start(
            "watch-retention",
            &root,
            service.watch_backend("watch-retention"),
            config,
        ).expect("watcher starts");
        let startup_deadline = Instant::now() + Duration::from_secs(2);
        while watcher.status().dirty && Instant::now() < startup_deadline {
            std::thread::yield_now();
        }
        assert!(!watcher.status().dirty, "watcher startup settled");

        pause_next.store(true, std::sync::atomic::Ordering::SeqCst);
        std::fs::write(root.join("first.rs"), "fn first_change() {}\n").expect("first change");
        started_rx.recv_timeout(Duration::from_secs(3)).expect("incremental worker entered provider");
        assert_eq!(
            watcher.stop_until(Instant::now() + Duration::from_millis(25)),
            crate::watch::WatchStopOutcome::TimedOut,
        );
        std::fs::write(root.join("second.rs"), "fn pending_change() {}\n").expect("pending change");
        drop(service);
        release_tx.send(()).expect("release provider");

        let connection = Connection::open(&paths.database).expect("read durable state");
        let deadline = Instant::now() + Duration::from_secs(3);
        let queued = loop {
            let queued: bool = connection.query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM index_jobs
                    WHERE workspace_id='watch-retention' AND relative_path='' AND state='queued'
                 )",
                [],
                |row| row.get(0),
            ).expect("queued reconciliation query");
            if queued || Instant::now() >= deadline {
                break queued;
            }
            std::thread::yield_now();
        };
        assert!(queued, "timed-out worker durably queued shutdown reconciliation");
        assert_eq!(
            watcher.stop_until(Instant::now() + Duration::from_secs(2)),
            crate::watch::WatchStopOutcome::Stopped,
        );
    }


    fn test_paths(temp: &TempDir) -> Paths {
        Paths::resolve(None, Some(&temp.path().join("data"))).expect("temporary paths")
    }

    fn test_config() -> Config {
        Config { allow_non_git_roots: true, ..Config::default() }
    }
    fn write_test_graph_snapshot(
        service: &WorkspaceService,
        workspace_id: &str,
        generation: u64,
        node_id: &str,
        source: &[u8],
    ) -> (crate::index::graphify::GraphSnapshotMetadata, PathBuf) {
        let (registration, visible_generation, _) = service.graph_state(workspace_id)
            .expect("workspace graph state");
        assert_eq!(visible_generation, generation);
        let metadata = registration.graph_metadata(generation);
        let path = crate::index::graphify::graph_snapshot_path(&service.private_data_root, &metadata)
            .expect("valid test graph identity");
        let published = crate::index::graphify::PublishedGraphData {
            metadata: metadata.clone(),
            nodes: vec![crate::index::graphify::GraphNode {
                id: node_id.to_owned(),
                label: None,
                kind: None,
                relative_path: "source.rs".into(),
                range: None,
                evidence: None,
            }],
            edges: Vec::new(),
            content_hashes: BTreeMap::from([(
                "source.rs".into(),
                blake3::hash(source).to_hex().to_string(),
            )]),
        };
        std::fs::create_dir_all(path.parent().expect("graph directory"))
            .expect("create graph directory");
        std::fs::write(&path, serde_json::to_vec(&published).expect("serialize graph"))
            .expect("write graph");
        (metadata, path)
    }

    #[test]
    fn unsupported_provider_fails_before_opening_a_service() {
        let temp = TempDir::new().expect("temporary directory");
        let mut config = test_config();
        config.embedding.provider = "unavailable".into();
        let error = WorkspaceService::open(test_paths(&temp), config).err().expect("unsupported provider");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(error.message.contains("provider"));
    }

    #[test]
    fn unsupported_model_fails_before_opening_a_service() {
        let temp = TempDir::new().expect("temporary directory");
        let mut config = test_config();
        config.embedding.model = "unknown-model".into();
        let error = WorkspaceService::open(test_paths(&temp), config).err().expect("unsupported model");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(error.message.contains("model"));
    }
    #[test]
    fn oversized_search_query_is_rejected_before_provider_embedding() {
        let temp = TempDir::new().expect("temporary directory");
        let mut config = test_config();
        config.limits.max_file_bytes = 64;
        config.limits.max_snippet_bytes = 64;
        let observed = Arc::new(Mutex::new(Vec::new()));
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp),
            config,
            Box::new(RecordingProvider { texts: Arc::clone(&observed) }),
        ).expect("service opens");

        let error = service.code_search(semantic_request(&[], &"q".repeat(65)))
            .expect_err("query above the source-file byte bound");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(error.message.contains("64-byte limit"));
        assert!(observed.lock().is_empty(), "oversized query must not reach the provider");
    }


    #[test]
    fn service_owns_exclusive_profile_lock_and_reconciles_registered_roots() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(paths.clone(), test_config(), Box::new(DeterministicProvider))
            .expect("service opens");
        let second = WorkspaceService::open_with_provider(paths, test_config(), Box::new(DeterministicProvider));
        assert_eq!(second.err().expect("exclusive owner lock").code, ErrorCode::Busy);

        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "service-owned indexing").expect("workspace file");


        service.register_workspace("sample", &root).expect("workspace registration");
        let report = service.reconcile("sample", false, false).expect("reconciliation");
        assert_eq!(report.updated, 1);
        assert_eq!(service.index_status("sample").expect("workspace status").visible_generation, 1);
    }
    #[test]
    fn reconciliation_backfills_migrated_profile_before_semantic_promotion() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "legacy snapshot").expect("workspace file");
        service.register_workspace("legacy", &root).expect("workspace registration");
        service.reconcile("legacy", false, false).expect("initial reconciliation");
        let profile_id = service.profile_identity.profile_id.clone();

        {
            let connection = service.connection.lock();
            connection.execute(
                "UPDATE workspace_index_state
                 SET generation=9, building_generation=9, visible_generation=4, dirty=1,
                     health='reconcile_required', desired_profile_id=?1, active_profile_id=?1
                 WHERE workspace_id='legacy'",
                [&profile_id],
            ).expect("simulate migrated v3 generation metadata");
            connection.execute(
                "UPDATE files SET visible_generation=4, file_generation=4 WHERE workspace_id='legacy'",
                [],
            ).expect("align imported file snapshot");
            connection.execute(
                "UPDATE chunk_embeddings SET generation=4 WHERE workspace_id='legacy'",
                [],
            ).expect("align imported embedding rows");
            connection.execute(
                "INSERT INTO profile_applied_generations
                    (workspace_id, profile_id, applied_generation, backfill_required, updated_at)
                 VALUES ('legacy', ?1, 0, 1, 0)
                 ON CONFLICT(workspace_id, profile_id) DO UPDATE SET
                    applied_generation=0, backfill_required=1",
                [&profile_id],
            ).expect("seed migration backfill state");
        }

        service.reconcile("legacy", false, false).expect("reconcile migrated workspace");
        assert_eq!(service.index_generation("legacy").expect("visible generation"), 10);
        let status = service.index_status("legacy").expect("workspace status");
        assert_eq!(status.visible_generation, 10);
        assert_eq!(status.state, IndexState::Ready);
        assert_eq!(status.active_embedding_profile.as_deref(), Some(profile_id.as_str()));
        let connection = service.connection.lock();
        let state: (i64, i64, i64, i64, String, Option<String>, Option<String>) = connection.query_row(
            "SELECT generation, building_generation, visible_generation, dirty, health,
                    desired_profile_id, active_profile_id
             FROM workspace_index_state WHERE workspace_id='legacy'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?)),
        ).expect("v4 state");
        assert_eq!(state, (
            10, 10, 10, 0, "ready".into(),
            Some(profile_id.clone()), Some(profile_id.clone()),
        ));
        let watermark: (i64, i64) = connection.query_row(
            "SELECT applied_generation, backfill_required
             FROM profile_applied_generations WHERE workspace_id='legacy' AND profile_id=?1",
            [&profile_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).expect("profile watermark");
        assert_eq!(watermark, (10, 0));
    }
    #[test]
    fn registered_root_replacement_blocks_reads_and_changeset_effects() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        let displaced = temp.path().join("displaced-workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.txt"), "original").expect("source file");
        service.register_workspace("rooted", &root).expect("workspace registration");
        assert_eq!(
            service.set_workspace_mode("rooted", WorkspaceMode::ReadWrite)
                .expect_err("unsafe mutation backend").code,
            ErrorCode::Unsupported,
        );
        {
            let connection = service.connection.lock();
            connection.execute(
                "UPDATE workspaces SET mode='read_write' WHERE id='rooted'",
                [],
            ).expect("simulate legacy persisted write mode");
        }
        assert_eq!(service.list_workspaces().unwrap()[0].mode, WorkspaceMode::ReadOnly);

        let prepared = service.changeset_prepare("rooted", vec![ChangeOperation::Replace {
            path: "source.txt".into(),
            expected_sha256: test_sha256(b"original"),
            replacements: vec![crate::changeset::Replacement {
                old_text: "original".into(), new_text: "changed".into(), replace_all: false,
            }],
        }]).expect("changeset prepare");

        std::fs::rename(&root, &displaced).expect("displace registered root");
        std::fs::create_dir(&root).expect("replacement root");
        std::fs::write(root.join("source.txt"), "original").expect("replacement sentinel");
        let service = std::sync::Arc::new(service);
        assert_eq!(
            service.read_file("rooted", Path::new("source.txt"), None, None, None)
                .expect_err("root identity mismatch").code,
            ErrorCode::Conflict,
        );
        assert_eq!(
            service.changeset_apply(&prepared.operation_id, &prepared.digest)
                .expect_err("root identity mismatch").code,
            ErrorCode::Conflict,
        );
        assert_eq!(test_sha256(&std::fs::read(root.join("source.txt")).unwrap()), test_sha256(b"original"));
        assert_eq!(std::fs::read_to_string(displaced.join("source.txt")).unwrap(), "original");

        let state: String = service.connection.lock().query_row(
            "SELECT state FROM pending_changesets WHERE operation_id=?1",
            [&prepared.operation_id],
            |row| row.get(0),
        ).expect("unclaimed changeset state");
        assert_eq!(state, "Prepared");
    }

    fn test_sha256(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(bytes))
    }
    #[test]
    fn code_search_refreshes_stale_candidates_and_retries_once() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let file_path = root.join("main.rs");
        std::fs::write(&file_path, "fn old_content() {}").expect("source file");
        service.register_workspace("ws1", &root).expect("workspace registration");
        service.reconcile("ws1", true, false).expect("initial reconciliation");
        
        // Change the file on disk to make it stale
        std::fs::write(&file_path, "fn new_content() {}").expect("source file");
        
        let req = CodeSearchRequest {
            query: "old_content".to_string(),
            query_vector: None,
            workspace_ids: vec!["ws1".to_string()],
            profile_id: None,
            languages: vec![],
            path_globs: vec![],
            limit: 10,
            cursor: None,
        };
        
        let initial_generation = service.index_generation("ws1").expect("initial generation");
        let result = service.code_search(req).expect("code search");
        let data = result.data().unwrap();
        assert!(!data.items.is_empty());
        assert!(data.items.iter().all(|item| {
            item.relative_path == "main.rs"
                && item.snippet.contains("new_content")
                && !item.snippet.contains("old_content")
                && item.sha256 == test_sha256(b"fn new_content() {}")
        }));
        assert!(!result.warnings().iter().any(|warning| warning.code == "INDEX_DEGRADED"));
        assert!(service.index_generation("ws1").expect("refreshed generation") > initial_generation);
        assert_eq!(
            service.read_file("ws1", Path::new("main.rs"), None, None, None)
                .expect("fresh source read").text,
            "fn new_content() {}",
        );
        let connection = service.connection.lock();
        let completed_jobs: i64 = connection.query_row(
            "SELECT count(*) FROM index_jobs WHERE workspace_id='ws1' AND state='done' AND relative_path='main.rs'",
            [],
            |row| row.get(0),
        ).expect("completed refresh jobs");
        let queued_jobs: i64 = connection.query_row(
            "SELECT count(*) FROM index_jobs WHERE workspace_id='ws1' AND state='queued' AND relative_path='main.rs'",
            [],
            |row| row.get(0),
        ).expect("queued refresh jobs");
        assert_eq!((completed_jobs, queued_jobs), (1, 0));
    }
    #[test]
    fn symbol_search_refreshes_stale_candidates_and_retries_once() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("symbol-stale-workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let file_path = root.join("source.rs");
        std::fs::write(&file_path, "fn stale_symbol() {}\n").expect("source file");
        service.register_workspace("symbol-stale", &root).expect("workspace registration");
        service.reconcile("symbol-stale", true, false).expect("initial reconciliation");
        let initial_generation = service.index_generation("symbol-stale").expect("initial generation");
        std::fs::write(&file_path, "fn changed_symbol() {}\n").expect("replace source");

        let result = service.symbol_search(SymbolSearchRequest {
            query: "stale_symbol".into(),
            workspace_ids: vec!["symbol-stale".into()],
            direction: None,
            limit: 10,
            cursor: None,
        }).expect("symbol search");
        assert!(result.data().expect("successful result").items.is_empty());
        assert!(!result.warnings().iter().any(|warning| warning.code == "INDEX_DEGRADED"));
        assert!(service.index_generation("symbol-stale").expect("refreshed generation") > initial_generation);
    }

    #[test]
    fn search_citation_sha256_round_trips_through_file_read() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("citation-workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let source = "fn citation_target() {}\n";
        std::fs::write(root.join("source.rs"), source).expect("source file");
        service.register_workspace("citation", &root).expect("workspace registration");
        service.reconcile("citation", true, false).expect("initial reconciliation");

        let result = service.code_search(CodeSearchRequest {
            query: "citation_target".into(),
            query_vector: None,
            workspace_ids: vec!["citation".into()],
            profile_id: None,
            languages: Vec::new(),
            path_globs: Vec::new(),
            limit: 10,
            cursor: None,
        }).expect("code search");
        let citation = &result.data().expect("search result").items[0];
        assert_eq!(citation.content_hash_algorithm, "blake3");
        assert_eq!(citation.content_hash, blake3::hash(source.as_bytes()).to_hex().to_string());
        let read = service.read_file(
            "citation", Path::new("source.rs"), None, None, Some(&citation.sha256),
        ).expect("citation SHA-256 accepted by file_read");
        assert_eq!(read.sha256, citation.sha256);
        assert_eq!(read.text, source);
    }


    #[test]
    fn graph_cursor_pages_nodes_and_edges_and_rejects_tampering_stale_generations_and_limits() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let source = "fn indexed() {}";
        std::fs::write(root.join("source.rs"), source).expect("source file");
        service.register_workspace("graph", &root).expect("workspace registration");
        service.reconcile("graph", true, false).expect("initial index");
        let generation = service.index_generation("graph").expect("visible generation");
        let registration = service.workspace_registration_identity("graph").expect("registration");
        let metadata = registration.graph_metadata(generation);
        let nodes = ["a", "b", "c", "d"].into_iter().map(|id| {
            crate::index::graphify::GraphNode {
                id: id.into(), label: Some(id.into()), kind: Some("function".into()),
                relative_path: "source.rs".into(), range: None, evidence: None,
            }
        }).collect::<Vec<_>>();
        let edges = (0..12).map(|_| crate::index::graphify::GraphEdge {
            source: "a".into(), target: "b".into(), kind: Some("calls".into()),
        }).chain(std::iter::once(crate::index::graphify::GraphEdge {
            source: "c".into(), target: "d".into(), kind: Some("calls".into()),
        })).collect::<Vec<_>>();
        let graph = crate::index::graphify::PublishedGraphData {
            metadata: metadata.clone(),
            nodes,
            edges,
            content_hashes: BTreeMap::from([(
                "source.rs".into(),
                blake3::hash(source.as_bytes()).to_hex().to_string(),
            )]),
        };
        let graph_path = crate::index::graphify::graph_snapshot_path(
            &service.private_data_root, &metadata,
        ).expect("valid graph path");
        std::fs::create_dir_all(graph_path.parent().expect("graph directory"))
            .expect("create graph directory");
        std::fs::write(&graph_path, serde_json::to_vec(&graph).unwrap())
            .expect("graph file");

        let (first, warnings) = service.read_graph("graph", 2, None).expect("first graph page");
        assert!(warnings.is_empty());
        assert_eq!(first.nodes.iter().map(|node| node.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(first.edges.len(), 10);
        assert!(first.nodes_truncated);
        assert!(first.edges_truncated);
        assert!(first.truncated);
        let cursor = first.next_cursor.expect("graph continuation");

        let (second, _) = service.read_graph("graph", 2, Some(&cursor)).expect("second node page");
        assert_eq!(second.nodes.iter().map(|node| node.id.as_str()).collect::<Vec<_>>(), vec!["c", "d"]);
        assert_eq!(second.edges.len(), 1);
        assert!(!second.truncated);
        assert!(second.next_cursor.is_none());
        let mut tampered = cursor.clone();
        let last = tampered.len() - 1;
        let replacement = if tampered.ends_with('0') { "1" } else { "0" };
        tampered.replace_range(last.., replacement);
        assert_eq!(service.read_graph("graph", 2, Some(&tampered)).unwrap_err().code, ErrorCode::StaleCursor);
        assert_eq!(service.read_graph("graph", 1, Some(&cursor)).unwrap_err().code, ErrorCode::StaleCursor);
        assert_eq!(service.read_graph("other", 2, Some(&cursor)).unwrap_err().code, ErrorCode::StaleCursor);
        assert_eq!(service.read_graph("graph", 2, Some("%not-a-cursor")).unwrap_err().code, ErrorCode::StaleCursor);
        assert_eq!(service.read_graph("graph", 2, Some("")).unwrap_err().code, ErrorCode::StaleCursor);

        service.connection.lock().execute(
            "UPDATE workspace_index_state SET building_generation=building_generation+1, visible_generation=visible_generation+1 WHERE workspace_id='graph'",
            [],
        ).expect("advance visible generation");
        assert_eq!(service.read_graph("graph", 2, Some(&cursor)).unwrap_err().code, ErrorCode::StaleCursor);
    }
    #[test]
    fn graph_reads_withhold_the_previous_generation_during_branch_transition() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let source = "fn indexed() {}";
        std::fs::write(root.join("source.rs"), source).expect("source file");
        service.register_workspace("graph-transition", &root).expect("workspace registration");
        service.reconcile("graph-transition", true, false).expect("initial index");
        let generation = service.index_generation("graph-transition").expect("visible generation");
        let registration = service.workspace_registration_identity("graph-transition")
            .expect("registration");
        let metadata = registration.graph_metadata(generation);
        let graph = crate::index::graphify::PublishedGraphData {
            metadata: metadata.clone(),
            nodes: ["a", "b"].into_iter().map(|id| crate::index::graphify::GraphNode {
                id: id.into(), label: Some(id.into()), kind: Some("function".into()),
                relative_path: "source.rs".into(), range: None, evidence: None,
            }).collect(),
            edges: Vec::new(),
            content_hashes: BTreeMap::from([(
                "source.rs".into(),
                blake3::hash(source.as_bytes()).to_hex().to_string(),
            )]),
        };
        let graph_path = crate::index::graphify::graph_snapshot_path(
            &service.private_data_root, &metadata,
        ).expect("valid graph path");
        std::fs::create_dir_all(graph_path.parent().expect("graph directory"))
            .expect("create graph directory");
        std::fs::write(&graph_path, serde_json::to_vec(&graph).unwrap())
            .expect("graph file");
        let (first, _) = service.read_graph("graph-transition", 1, None)
            .expect("current graph page");
        let cursor = first.next_cursor.expect("graph cursor");

        service.connection.lock().execute(
            "UPDATE workspace_index_state SET branch_transition=1 WHERE workspace_id='graph-transition'",
            [],
        ).expect("begin branch transition");
        let (withheld, warnings) = service.read_graph("graph-transition", 1, None)
            .expect("transition returns a reconciling page");
        assert!(withheld.nodes.is_empty());
        assert!(withheld.edges.is_empty());
        assert!(withheld.next_cursor.is_none());
        assert!(warnings.iter().any(|warning| warning.code == "RECONCILING"));
        assert_eq!(
            service.read_graph("graph-transition", 1, Some(&cursor)).unwrap_err().code,
            ErrorCode::StaleCursor,
        );
    }
    #[test]
    fn startup_marks_offline_branch_change_before_serving_search() {
        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("git-workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn offline_branch_marker() {}\n")
            .expect("source file");
        assert!(std::process::Command::new("git")
            .args(["init", "--initial-branch=main"])
            .arg(&root)
            .output().expect("initialize Git").status.success());
        assert!(std::process::Command::new("git")
            .args(["-C"]).arg(&root).args(["add", "source.rs"])
            .output().expect("stage source").status.success());
        assert!(std::process::Command::new("git")
            .args(["-C"]).arg(&root)
            .args(["-c", "user.name=Workspace Test", "-c", "user.email=test@example.invalid",
                   "commit", "-m", "initial"])
            .output().expect("commit source").status.success());

        let paths = test_paths(&temp);
        let initial = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("initial service");
        initial.register_workspace("offline", &root).expect("workspace registration");
        initial.reconcile("offline", true, false).expect("initial generation");
        drop(initial);

        assert!(std::process::Command::new("git")
            .args(["-C"]).arg(&root).args(["checkout", "-b", "offline-transition"])
            .output().expect("switch branch while stopped").status.success());
        let reopened = WorkspaceService::open_with_provider(
            paths, test_config(), Box::new(DeterministicProvider),
        ).expect("service reopens");
        let transition: bool = reopened.connection.lock().query_row(
            "SELECT branch_transition FROM workspace_index_state WHERE workspace_id='offline'",
            [],
            |row| row.get(0),
        ).expect("persisted branch transition");
        assert!(transition);
        let result = reopened.code_search(CodeSearchRequest {
            query: "offline_branch_marker".into(),
            query_vector: None,
            workspace_ids: vec!["offline".into()],
            profile_id: None,
            languages: Vec::new(),
            path_globs: Vec::new(),
            limit: 10,
            cursor: None,
        }).expect("search while startup reconciliation is pending");
        assert!(result.data().expect("successful reconciling result").items.is_empty());
        assert!(result.warnings().iter().any(|warning| warning.code == "RECONCILING"));
        let reconciled = reopened.reconcile("offline", false, false)
            .expect("stable branch generation publishes");
        assert!(reconciled.unchanged > 0);
        assert!(
            !reopened.connection.lock().query_row(
                "SELECT branch_transition FROM workspace_index_state WHERE workspace_id='offline'",
                [],
                |row| row.get::<_, bool>(0),
            ).expect("published branch transition"),
        );
        let current = reopened.code_search(semantic_request(&["offline"], "offline_branch_marker"))
            .expect("search after transition clears");
        assert_eq!(current.data().expect("current branch result").items.len(), 1);
    }

    #[test]
    fn graph_snapshot_cache_reuses_identical_content_and_replaces_changed_content() {
        let temp = TempDir::new().expect("temporary directory");
        let path = temp.path().join("graph.json");
        let metadata = crate::index::graphify::GraphSnapshotMetadata {
            workspace_id: "ws".into(),
            registration_epoch: 3,
            root_device: 10,
            root_inode: 20,
            visible_generation: 4,
        };
        let content_hash = blake3::hash(b"source").to_hex().to_string();
        let fixture = |ids: [&str; 2]| crate::index::graphify::PublishedGraphData {
            metadata: metadata.clone(),
            nodes: ids.into_iter().map(|id| crate::index::graphify::GraphNode {
                id: id.into(),
                label: None,
                kind: None,
                relative_path: "x".into(),
                range: None,
                evidence: None,
            }).collect(),
            edges: Vec::new(),
            content_hashes: BTreeMap::from([("x".into(), content_hash.clone())]),
        };
        std::fs::write(&path, serde_json::to_vec(&fixture(["b", "a"])).unwrap())
            .expect("initial graph");
        let cache = Mutex::new(GraphSnapshotCache::default());
        let first = read_graph_snapshot(&cache, &metadata, &path).expect("first snapshot");
        let repeated = read_graph_snapshot(&cache, &metadata, &path).expect("cached snapshot");
        assert!(Arc::ptr_eq(&first, &repeated));
        assert_eq!(first.nodes[0].id, "a");
        assert_eq!(first.content_hashes.get("x"), Some(&content_hash));

        std::fs::write(&path, serde_json::to_vec(&fixture(["c", "d"])).unwrap())
            .expect("replacement graph");
        let replaced = read_graph_snapshot(&cache, &metadata, &path).expect("replacement snapshot");
        assert!(!Arc::ptr_eq(&first, &replaced));
        assert_eq!(replaced.nodes[0].id, "c");
        assert_eq!(cache.lock().snapshots.len(), 1);
    }
    #[test]
    fn stale_graph_generation_cannot_publish_or_prune_a_newer_snapshot() {
        let temp = TempDir::new().expect("temporary directory");
        let service = Arc::new(WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens"));
        let root = temp.path().join("graph-race");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let source = root.join("source.rs");
        std::fs::write(&source, "fn old_graph() {}\n").expect("initial source");
        service.register_workspace("graph-race", &root).expect("workspace registration");
        service.reconcile("graph-race", true, false).expect("initial index");
        let old_generation = service.index_generation("graph-race").expect("old generation");
        let registration = service.workspace_registration_identity("graph-race")
            .expect("captured registration");
        let old_metadata = registration.graph_metadata(old_generation);
        let root_lock = service.root_lock(&registration.root_path);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let worker_service = Arc::clone(&service);
        let worker_registration = registration.clone();
        let delayed_extraction = crate::index::graphify::ExtractedGraph {
            data: crate::index::graphify::GraphData {
                nodes: vec![crate::index::graphify::GraphNode {
                    id: "old-node".into(),
                    label: None,
                    kind: None,
                    relative_path: "source.rs".into(),
                    range: None,
                    evidence: None,
                }],
                edges: Vec::new(),
            },
            content_hashes: BTreeMap::from([(
                "source.rs".into(),
                blake3::hash(b"fn old_graph() {}\n").to_hex().to_string(),
            )]),
        };
        let worker = std::thread::spawn(move || {
            release_rx.recv().expect("delayed extraction released");
            let result = worker_service.finish_graph_extraction(
                root_lock.as_ref(),
                &worker_registration,
                old_generation,
                Ok(delayed_extraction),
            );
            result_tx.send(result).expect("publication result sent");
        });

        std::fs::write(&source, "fn new_graph() {}\n").expect("new source");
        service.reconcile("graph-race", true, false).expect("new generation");
        let new_generation = service.index_generation("graph-race").expect("new generation");
        assert!(new_generation > old_generation);
        let new_metadata = registration.graph_metadata(new_generation);
        let current_graph = crate::index::graphify::graph_snapshot_path(
            &service.private_data_root, &new_metadata,
        ).expect("current graph path");
        let old_graph = crate::index::graphify::graph_snapshot_path(
            &service.private_data_root, &old_metadata,
        ).expect("old graph path");
        std::fs::create_dir_all(current_graph.parent().expect("current graph directory"))
            .expect("create current graph directory");
        std::fs::write(&current_graph, b"newer snapshot").expect("current graph snapshot");

        release_tx.send(()).expect("release delayed extraction");
        assert!(!result_rx.recv_timeout(Duration::from_secs(1)).expect("publication finished"));
        worker.join().expect("graph worker");
        assert!(!old_graph.exists());
        assert_eq!(std::fs::read(current_graph).expect("newer snapshot remains"), b"newer snapshot");
    }
    #[test]
    fn current_graphify_failure_hides_existing_files_and_cached_snapshots() {
        let temp = TempDir::new().expect("temporary directory");
        let mut config = test_config();
        config.graphify.enabled = false;
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), config, Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("failed-graph");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let source = b"fn current_source() {}\n";
        std::fs::write(root.join("source.rs"), source).expect("source file");
        service.register_workspace("failed-graph", &root).expect("workspace registration");
        service.reconcile("failed-graph", true, false).expect("initial index");
        service.reconcile("failed-graph", true, true).expect("failed optional graphify remains successful");
        let generation = service.index_generation("failed-graph").expect("visible generation");
        let (metadata, path) = write_test_graph_snapshot(
            &service, "failed-graph", generation, "stale-node", source,
        );
        let cached = read_graph_snapshot(&service.graph_snapshots, &metadata, &path)
            .expect("manually restored stale graph is cached");
        assert_eq!(cached.nodes[0].id, "stale-node");

        let (page, warnings) = service.read_graph("failed-graph", 10, None).expect("failed graph page");

        assert!(path.is_file());
        assert_eq!(service.graph_snapshots.lock().snapshots.len(), 1);
        assert!(page.nodes.is_empty());
        assert_eq!(warnings[0].code, "GRAPH_UNAVAILABLE");
        assert!(service.index_status("failed-graph").expect("index status").graph_generation.is_none());
    }

    #[test]
    fn graph_reads_revalidate_cached_source_hashes_before_returning_nodes() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("graph-freshness");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let source = b"fn original_source() {}\n";
        let source_path = root.join("source.rs");
        std::fs::write(&source_path, source).expect("source file");
        service.register_workspace("graph-freshness", &root).expect("workspace registration");
        service.reconcile("graph-freshness", true, false).expect("initial index");
        let generation = service.index_generation("graph-freshness").expect("visible generation");
        let (metadata, _) = write_test_graph_snapshot(
            &service, "graph-freshness", generation, "fresh-node", source,
        );
        let (page, warnings) = service.read_graph("graph-freshness", 10, None)
            .expect("current graph page");
        assert!(warnings.is_empty());
        assert_eq!(page.nodes[0].id, "fresh-node");
        let source_hash = blake3::hash(source).to_hex().to_string();
        assert_eq!(
            service.graph_snapshots.lock().snapshots.values().next()
                .and_then(|snapshot| snapshot.content_hashes.get("source.rs")),
            Some(&source_hash),
        );

        std::fs::write(&source_path, "fn modified_source() {}\n").expect("modify source");
        let (page, warnings) = service.read_graph("graph-freshness", 10, None)
            .expect("stale graph page is withheld");
        assert!(page.nodes.is_empty());
        assert_eq!(warnings[0].code, "GRAPH_UNAVAILABLE");
        assert!(!service.graph_snapshots.lock().snapshots.keys()
            .any(|key| key.metadata == metadata));
    }
    #[test]
    fn reconciliation_syncs_durable_hnsw_and_noop_keeps_semantic_search_current() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn indexed_symbol() {}").expect("source file");
        service.register_workspace("semantic", &root).expect("workspace registration");

        let first = service.reconcile("semantic", false, false).expect("initial reconciliation");
        assert!(!first.degraded_semantic);
        let status = service.index_status("semantic").expect("workspace status");
        let profile_id = status.active_embedding_profile.expect("active profile");
        let base = service.paths.vectors_dir.join(format!("semantic_{profile_id}"));
        let manifest: crate::index::vector::CacheManifest = serde_json::from_slice(
            &std::fs::read(base.with_extension("json")).expect("durable HNSW manifest"),
        ).expect("valid HNSW manifest");
        assert_eq!(manifest.generation, status.visible_generation);
        let index = crate::index::vector::VectorIndex::load(&base, &manifest)
            .expect("durable HNSW snapshot");
        assert_eq!(index.size(), 1);

        let search = service.code_search(semantic_request(&["semantic"], "vectoronlyprobe"))
            .expect("semantic search");
        let items = &search.data().expect("successful search").items;
        assert_eq!(items.len(), 1);
        assert!(items[0].component_ranks.contains_key("semantic"));

        let no_op = service.reconcile("semantic", false, false).expect("no-op reconciliation");
        assert_eq!(no_op.unchanged, 1);
        assert!(!no_op.degraded_semantic);
        let current = service.index_status("semantic").expect("current status");
        let watermark: (i64, bool) = service.connection.lock().query_row(
            "SELECT applied_generation, backfill_required FROM profile_applied_generations
             WHERE workspace_id='semantic' AND profile_id=?1",
            [&profile_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).expect("applied HNSW watermark");
        assert_eq!(watermark, (current.visible_generation as i64, false));
        let no_op_search = service.code_search(semantic_request(&["semantic"], "vectoronlyprobe"))
            .expect("semantic search after no-op");
        assert!(no_op_search.data().expect("successful search").items[0]
            .component_ranks.contains_key("semantic"));
    }

    #[test]
    fn startup_rebuilds_deleted_and_corrupt_hnsw_snapshots() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn indexed_symbol() {}").expect("source file");
        service.register_workspace("restart", &root).expect("workspace registration");
        service.reconcile("restart", false, false).expect("initial reconciliation");
        let profile_id = service.index_status("restart").expect("status")
            .active_embedding_profile.expect("active profile");
        let base = paths.vectors_dir.join(format!("restart_{profile_id}"));
        drop(service);

        std::fs::remove_file(base.with_extension("usearch")).expect("delete vector cache");
        std::fs::remove_file(base.with_extension("json")).expect("delete cache manifest");
        let rebuilt = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service rebuilds deleted cache");
        let manifest: crate::index::vector::CacheManifest = serde_json::from_slice(
            &std::fs::read(base.with_extension("json")).expect("rebuilt manifest"),
        ).expect("valid rebuilt manifest");
        assert_eq!(
            crate::index::vector::VectorIndex::load(&base, &manifest)
                .expect("rebuilt HNSW snapshot").size(),
            1,
        );
        drop(rebuilt);

        std::fs::write(base.with_extension("usearch"), b"corrupt cache")
            .expect("corrupt vector cache");
        let repaired = WorkspaceService::open_with_provider(
            paths, test_config(), Box::new(DeterministicProvider),
        ).expect("service rebuilds corrupt cache");
        let manifest: crate::index::vector::CacheManifest = serde_json::from_slice(
            &std::fs::read(base.with_extension("json")).expect("repaired manifest"),
        ).expect("valid repaired manifest");
        assert_eq!(
            crate::index::vector::VectorIndex::load(&base, &manifest)
                .expect("repaired HNSW snapshot").size(),
            1,
        );
        assert_eq!(repaired.index_status("restart").expect("status").state, IndexState::Ready);
    }

    #[test]
    fn remote_provider_configuration_is_constructed_with_normalized_endpoint_and_dimension() {
        let temp = TempDir::new().expect("temporary directory");
        let mut config = test_config();
        config.embedding.provider = "ollama".into();
        config.embedding.model = "nomic-embed-text".into();
        config.embedding.allow_remote = true;
        config.embedding.endpoint = Some("http://LOCALHOST:11434".into());
        let service = WorkspaceService::open(test_paths(&temp), config)
            .expect("remote adapter is constructed without making a request");
        assert_eq!(service.profile_identity.dimension, 768);
        assert_eq!(
            service.profile_identity.endpoint.as_deref(),
            Some("http://localhost:11434/api/embed"),
        );
        let persisted: (String, Option<String>, String, i64, String) = service.connection.lock().query_row(
            "SELECT provider_kind, endpoint, model, dimension, metric
             FROM embedding_profiles WHERE id=?1",
            [&service.profile_identity.profile_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        ).expect("persisted immutable provider profile");
        assert_eq!(persisted, (
            "ollama".into(), Some("http://localhost:11434/api/embed".into()),
            "nomic-embed-text".into(), 768, "cosine".into(),
        ));
        drop(service);

        let mut missing_key = test_config();
        missing_key.embedding.provider = "ollama".into();
        missing_key.embedding.model = "nomic-embed-text".into();
        missing_key.embedding.allow_remote = true;
        missing_key.embedding.endpoint = Some("http://localhost:11434".into());
        let key_name = format!("WIM_MISSING_KEY_{}", uuid::Uuid::new_v4().simple());
        missing_key.embedding.api_key_env = Some(key_name.clone());
        let error = WorkspaceService::open(test_paths(&temp), missing_key)
            .err().expect("configured missing credential fails startup");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(!error.message.contains(&key_name));
    }

    #[test]
    fn profile_switch_keeps_verified_old_profile_active_until_new_watermark_is_current() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn stable_entry_point() {}").expect("source file");
        let initial = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("initial service");
        initial.register_workspace("switch", &root).expect("workspace registration");
        initial.reconcile("switch", false, false).expect("initial reconciliation");
        let old_profile = initial.index_status("switch").expect("initial status")
            .active_embedding_profile.expect("old active profile");
        drop(initial);

        let mut changed_config = test_config();
        changed_config.embedding.query_template_version = "query-v2".into();
        let incomplete = WorkspaceService::open_with_provider(
            paths.clone(),
            changed_config.clone(),
            Box::new(InvalidResponseProvider { response: vec![vec![f32::NAN, 1.0]] }),
        ).expect("service opens with new profile");
        let incomplete_profile = incomplete.profile_identity.profile_id.clone();
        assert_ne!(incomplete_profile, old_profile);
        assert_eq!(
            incomplete.index_status("switch").expect("status").active_embedding_profile.as_deref(),
            Some(old_profile.as_str()),
        );
        install_test_provider(&incomplete, &old_profile, Box::new(DeterministicProvider));
        let before = incomplete.code_search(semantic_request(&["switch"], "profilemigrationprobe"))
            .expect("old profile remains searchable during migration");
        assert!(before.data().expect("successful search").items[0]
            .component_ranks.contains_key("semantic"));

        let failed = incomplete.reconcile("switch", false, false).expect("lexical migration result");
        assert!(failed.degraded_semantic);
        let state: (Option<String>, Option<String>, bool) = incomplete.connection.lock().query_row(
            "SELECT active_profile_id, desired_profile_id, branch_transition FROM workspace_index_state
             WHERE workspace_id='switch'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).expect("migration profiles");
        assert_eq!(state.0.as_deref(), Some(old_profile.as_str()));
        assert_eq!(state.1.as_deref(), Some(incomplete_profile.as_str()));
        assert!(!state.2, "profile backfill is not a branch transition");
        assert!(incomplete.profile_is_active_and_current("switch", &old_profile)
            .expect("old profile watermark"));
        let incomplete_generation = incomplete.index_generation("switch").expect("failed profile generation");
        let incomplete_profile_coverage: i64 = incomplete.connection.lock().query_row(
            "SELECT count(*) FROM chunk_embeddings
             WHERE workspace_id='switch' AND profile_id=?1 AND generation=?2",
            params![incomplete_profile, incomplete_generation as i64],
            |row| row.get(0),
        ).expect("failed profile has no current-generation embeddings");
        assert_eq!(incomplete_profile_coverage, 0);
        let incomplete_watermark: Option<(i64, bool)> = incomplete.connection.lock().query_row(
            "SELECT applied_generation, backfill_required FROM profile_applied_generations
             WHERE workspace_id='switch' AND profile_id=?1",
            [&incomplete_profile],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional().expect("failed profile watermark");
        assert_ne!(incomplete_watermark, Some((incomplete_generation as i64, false)));
        assert!(incomplete.index_status("switch").expect("failed profile status")
            .active_embedding_profile.as_deref() == Some(old_profile.as_str()));
        let after = incomplete.code_search(semantic_request(&["switch"], "profilemigrationprobe"))
            .expect("old profile remains searchable while new coverage is incomplete");
        assert!(after.data().expect("successful search").items[0]
            .component_ranks.contains_key("semantic"));
        let queued_retry: bool = incomplete.connection.lock().query_row(
            "SELECT EXISTS(SELECT 1 FROM index_jobs
             WHERE workspace_id='switch' AND relative_path='source.rs' AND state='queued')",
            [],
            |row| row.get(0),
        ).expect("profile backfill job");
        assert!(queued_retry);
        incomplete.connection.lock().execute(
            "INSERT INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
             VALUES ('switch', 'source.rs', 0, 'running', ?1)",
            [unix_seconds() as i64],
        ).expect("simulate interrupted same-path job beside its queued retry");
        drop(incomplete);

        let promoted = WorkspaceService::open_with_provider(
            paths, changed_config, Box::new(AlternateProvider),
        ).expect("service opens with complete replacement provider");
        let new_profile = promoted.profile_identity.profile_id.clone();
        assert_ne!(new_profile, old_profile);
        assert_eq!(new_profile, incomplete_profile);
        install_test_provider(&promoted, &old_profile, Box::new(DeterministicProvider));
        let complete = promoted.reconcile("switch", false, false).expect("complete profile backfill");
        assert!(!complete.degraded_semantic);
        assert_eq!(
            promoted.index_status("switch").expect("promoted status")
                .active_embedding_profile.as_deref(),
            Some(new_profile.as_str()),
        );
        assert!(promoted.profile_is_active_and_current("switch", &new_profile)
            .expect("replacement profile watermark"));
        let visible_generation = promoted.index_generation("switch").expect("promoted generation");
        let (chunk_count, embedded_count): (i64, i64) = promoted.connection.lock().query_row(
            "SELECT
                (SELECT count(*) FROM chunks WHERE workspace_id='switch'),
                (SELECT count(*) FROM chunk_embeddings
                 WHERE workspace_id='switch' AND profile_id=?1 AND generation=?2)",
            params![new_profile, visible_generation as i64],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).expect("replacement profile vector coverage");
        assert!(chunk_count > 0);
        assert_eq!(embedded_count, chunk_count);
        let watermark: (i64, bool) = promoted.connection.lock().query_row(
            "SELECT applied_generation, backfill_required
             FROM profile_applied_generations WHERE workspace_id='switch' AND profile_id=?1",
            [&new_profile],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).expect("replacement profile watermark");
        assert_eq!(watermark, (visible_generation as i64, false));
        let snapshot_base = promoted.paths.vectors_dir.join(format!("switch_{new_profile}"));
        let manifest: crate::index::vector::CacheManifest = serde_json::from_slice(
            &std::fs::read(snapshot_base.with_extension("json")).expect("replacement HNSW manifest"),
        ).expect("valid replacement HNSW manifest");
        assert_eq!(manifest.generation, visible_generation);
        assert_eq!(manifest.count, chunk_count as u64);
        assert_eq!(
            crate::index::vector::VectorIndex::load(&snapshot_base, &manifest)
                .expect("replacement HNSW snapshot").size(),
            chunk_count as usize,
        );
        let unfinished_jobs: i64 = promoted.connection.lock().query_row(
            "SELECT count(*) FROM index_jobs
             WHERE workspace_id='switch' AND state IN ('queued', 'running')",
            [],
            |row| row.get(0),
        ).expect("pending reconciliation jobs");
        assert_eq!(unfinished_jobs, 0);
    }

    #[test]
    fn cross_workspace_semantic_search_uses_each_workspace_active_profile() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        for (workspace_id, source) in [
            ("ws-a", "fn first_workspace_symbol() {}"),
            ("ws-b", "fn second_workspace_symbol() {}"),
        ] {
            let root = temp.path().join(workspace_id);
            std::fs::create_dir_all(&root).expect("workspace directory");
            std::fs::write(root.join("main.rs"), source).expect("workspace source");
            service.register_workspace(workspace_id, &root).expect("workspace registration");
            service.reconcile(workspace_id, false, false).expect("workspace reconciliation");
        }
        let other_identity = profile_identity(
            &service.config.embedding, "other-workspace-provider", 2, None,
        ).expect("other immutable profile identity");
        let generation = service.index_generation("ws-b").expect("second workspace generation");
        let mut embedding = Vec::new();
        embedding.extend_from_slice(&1.0_f32.to_le_bytes());
        embedding.extend_from_slice(&0.0_f32.to_le_bytes());
        let numeric_id = {
            let connection = service.connection.lock();
            ensure_profile_identity(&connection, &other_identity).expect("persist other profile");
            connection.execute(
                "UPDATE embedding_profiles SET state='active' WHERE id=?1",
                [&other_identity.profile_id],
            ).expect("activate second workspace profile");
            connection.execute(
                "INSERT INTO chunk_embeddings
                    (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding)
                 VALUES ('ws-b', 'main.rs', 0, ?1, ?2, 'main.rs#0', ?3)",
                params![other_identity.profile_id, generation as i64, embedding],
            ).expect("second-profile vector");
            let id: i64 = connection.query_row(
                "SELECT numeric_vector_id FROM chunk_embeddings
                 WHERE workspace_id='ws-b' AND profile_id=?1",
                [&other_identity.profile_id],
                |row| row.get(0),
            ).expect("stable numeric vector id");
            connection.execute(
                "INSERT INTO profile_applied_generations
                    (workspace_id, profile_id, applied_generation, backfill_required, updated_at)
                 VALUES ('ws-b', ?1, ?2, 0, 0)",
                params![other_identity.profile_id, generation as i64],
            ).expect("second-profile watermark");
            connection.execute(
                "UPDATE workspace_index_state SET desired_profile_id=?1, active_profile_id=?1
                 WHERE workspace_id='ws-b'",
                [&other_identity.profile_id],
            ).expect("workspace-specific active profile");
            id
        };
        let other_base = service.paths.vectors_dir
            .join(format!("ws-b_{}", other_identity.profile_id));
        let mut other_index = crate::index::vector::VectorIndex::new(
            crate::index::vector::CacheManifest {
                workspace_id: "ws-b".into(),
                profile_id: other_identity.profile_id.clone(),
                dimension: 2,
                metric: "cosine".into(),
                generation,
                outbox_count: 0,
                outbox_max_id: 0,
                count: 0,
                checksum: String::new(),
            },
        ).expect("second-profile index");
        other_index.add(numeric_id as u64, &[1.0, 0.0]).expect("second-profile vector");
        other_index.save(&other_base).expect("second-profile HNSW snapshot");

        let result = service.code_search(semantic_request(&["ws-a", "ws-b"], "vectoronlycrossworkspace"))
            .expect("cross-workspace search");
        let items = &result.data().expect("successful search").items;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].workspace_id, "ws-a");
        assert!(items[0].component_ranks.contains_key("semantic"));
        assert!(result.warnings().iter().any(|warning| warning.code == "INDEX_DEGRADED"));
    }

    #[test]
    fn remote_opt_in_and_missing_credentials_fail_closed_without_leaking_names() {
        let temp = TempDir::new().expect("temporary directory");
        let mut disabled = test_config();
        disabled.embedding.provider = "openai_compatible".into();
        disabled.embedding.model = "text-embedding-3-small".into();
        disabled.embedding.endpoint = Some("https://api.example.test".into());
        let error = WorkspaceService::open(test_paths(&temp), disabled)
            .err().expect("remote opt-in is required");
        assert_eq!(error.code, ErrorCode::InvalidArgument);

        let mut missing_key = test_config();
        missing_key.embedding.provider = "openai_compatible".into();
        missing_key.embedding.model = "text-embedding-3-small".into();
        missing_key.embedding.allow_remote = true;
        missing_key.embedding.endpoint = Some("https://api.example.test".into());
        let key_name = format!("WIM_MISSING_{}", uuid::Uuid::new_v4().simple());
        missing_key.embedding.api_key_env = Some(key_name.clone());
        let error = WorkspaceService::open(test_paths(&temp), missing_key)
            .err().expect("configured missing key fails closed");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(!error.message.contains(&key_name));
    }
    #[test]
    fn branch_transition_withholds_old_search_results_and_stales_existing_cursor() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("branch-workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(
            root.join("first.rs"),
            "fn branch_transition_probe_first() {}\n",
        ).expect("first source");
        std::fs::write(
            root.join("second.rs"),
            "fn branch_transition_probe_second() {}\n",
        ).expect("second source");
        service.register_workspace("branch", &root).expect("workspace registration");
        service.reconcile("branch", true, false).expect("initial index");
        let initial_generation = service.index_generation("branch").expect("initial generation");

        let first_page = service.code_search(CodeSearchRequest {
            query: "branch_transition_probe".into(),
            query_vector: None,
            workspace_ids: vec!["branch".into()],
            profile_id: None,
            languages: Vec::new(),
            path_globs: Vec::new(),
            limit: 1,
            cursor: None,
        }).expect("initial search");
        let cursor = first_page.data().expect("successful first page").next_cursor
            .clone().expect("multiple matches produce a cursor");

        service.begin_rebuild("branch").expect("begin branch rebuild");
        let transitioning: bool = service.connection.lock().query_row(
            "SELECT branch_transition FROM workspace_index_state WHERE workspace_id='branch'",
            [],
            |row| row.get(0),
        ).expect("branch transition marker");
        assert!(transitioning);
        let code = service.code_search(semantic_request(&["branch"], "branch_transition_probe"))
            .expect("reconciling search remains a success");
        assert!(code.data().expect("successful no-hit result").items.is_empty());
        assert!(code.warnings().iter().any(|warning| {
            warning.code == "RECONCILING" && !warning.retryable
        }));
        assert_eq!(
            service.code_search(CodeSearchRequest {
                query: "branch_transition_probe".into(),
                query_vector: None,
                workspace_ids: vec!["branch".into()],
                profile_id: None,
                languages: Vec::new(),
                path_globs: Vec::new(),
                limit: 1,
                cursor: Some(cursor.clone()),
            }).expect("cursor request returns a typed result").error().unwrap().code,
            ErrorCode::StaleCursor,
        );
        let symbols = service.symbol_search(SymbolSearchRequest {
            query: "branch_transition_probe".into(),
            workspace_ids: vec!["branch".into()],
            direction: Some("both".into()),
            limit: 10,
            cursor: None,
        }).expect("reconciling symbol search remains a success");
        assert!(symbols.data().expect("successful no-hit result").items.is_empty());
        assert!(symbols.warnings().iter().any(|warning| {
            warning.code == "RECONCILING" && !warning.retryable
        }));
        let published = service.reconcile("branch", true, false)
            .expect("complete stable branch publication");
        assert!(!published.degraded_semantic);
        let current_generation = service.index_generation("branch")
            .expect("published generation");
        assert!(current_generation > initial_generation);
        let transitioning: bool = service.connection.lock().query_row(
            "SELECT branch_transition FROM workspace_index_state WHERE workspace_id='branch'",
            [],
            |row| row.get(0),
        ).expect("published branch transition marker");
        assert!(!transitioning);
        let current = service.code_search(semantic_request(&["branch"], "branch_transition_probe"))
            .expect("search after stable publication");
        let current_items = &current.data().expect("published hits").items;
        assert_eq!(current_items.len(), 2);
        assert!(current_items.iter().all(|item| item.visible_generation == current_generation));
        let stale_after_publish = service.code_search(CodeSearchRequest {
            query: "branch_transition_probe".into(),
            query_vector: None,
            workspace_ids: vec!["branch".into()],
            profile_id: None,
            languages: Vec::new(),
            path_globs: Vec::new(),
            limit: 1,
            cursor: Some(cursor),
        }).expect("pre-transition cursor result");
        assert_eq!(stale_after_publish.error().unwrap().code, ErrorCode::StaleCursor);
        let stored_symbols = {
            let connection = service.connection.lock();
            let mut statement = connection.prepare(
                "SELECT name FROM symbols WHERE workspace_id='branch' ORDER BY name",
            ).expect("symbol statement");
            statement.query_map([], |row| row.get::<_, String>(0))
                .expect("symbol rows").collect::<Result<Vec<_>, _>>()
                .expect("symbol names")
        };
        assert_eq!(stored_symbols, vec![
            "branch_transition_probe_first".to_owned(),
            "branch_transition_probe_second".to_owned(),
        ]);
        let current_symbols = service.symbol_search(SymbolSearchRequest {
            query: "branch_transition_probe_first".into(),
            workspace_ids: vec!["branch".into()],
            direction: Some("both".into()),
            limit: 10,
            cursor: None,
        }).expect("symbol search after stable publication");
        assert!(!current_symbols.data().expect("published symbols").items.is_empty(),
            "symbol search result: {current_symbols:?}");
    }

    #[test]
    fn startup_reclaims_running_jobs_and_status_reports_running_work() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let root = temp.path().join("recover-jobs");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn recover_jobs() {}\n").expect("source");
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        service.register_workspace("recover", &root).expect("workspace registration");
        service.reconcile("recover", true, false).expect("initial index");
        service.connection.lock().execute(
            "INSERT INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
             VALUES ('recover', 'source.rs', 0, 'queued', ?1)",
            [unix_seconds() as i64],
        ).expect("queued retry");
        service.connection.lock().execute(
            "INSERT INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
             VALUES ('recover', 'source.rs', 0, 'running', ?1)",
            [unix_seconds() as i64],
        ).expect("interrupted running job");
        let active_status = service.index_status("recover").expect("active status");
        assert_eq!(active_status.queued_jobs, 1);
        assert_eq!(active_status.running_jobs, 1);
        drop(service);

        let recovered = WorkspaceService::open_with_provider(
            paths, test_config(), Box::new(DeterministicProvider),
        ).expect("service recovers");
        let status = recovered.index_status("recover").expect("recovered status");
        assert_eq!(status.queued_jobs, 1);
        assert_eq!(status.running_jobs, 0);
        let remaining_jobs: i64 = recovered.connection.lock().query_row(
            "SELECT count(*) FROM index_jobs
             WHERE workspace_id='recover' AND relative_path='source.rs'
               AND state IN ('queued', 'running')",
            [],
            |row| row.get(0),
        ).expect("durable job state");
        assert_eq!(remaining_jobs, 1);
    }

    #[test]
    fn post_commit_vector_snapshot_failure_returns_degraded_success() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("snapshot-failure");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn snapshot_failure() {}\n").expect("source");
        service.register_workspace("snapshot", &root).expect("workspace registration");
        std::fs::remove_dir(&paths.vectors_dir).expect("remove vector directory");
        std::fs::write(&paths.vectors_dir, "not a directory").expect("block vector storage");

        let report = service.reconcile("snapshot", true, false)
            .expect("post-commit snapshot error remains a successful reconciliation");
        assert!(report.degraded_semantic);
        let state: (i64, i64, bool) = service.connection.lock().query_row(
            "SELECT s.visible_generation, g.applied_generation, g.backfill_required
             FROM workspace_index_state s
             LEFT JOIN profile_applied_generations g
               ON g.workspace_id=s.workspace_id AND g.profile_id=s.desired_profile_id
             WHERE s.workspace_id='snapshot'",
            [],
            |row| Ok((row.get(0)?, row.get::<_, Option<i64>>(1)?.unwrap_or(0), row.get::<_, Option<bool>>(2)?.unwrap_or(true))),
        ).expect("committed metadata and profile watermark");
        assert!(state.0 > 0);
        assert!(state.1 < state.0 || state.2);
    }

    #[test]
    fn provider_admission_waits_for_contention_instead_of_failing() {
        let gate = Arc::new(Mutex::new(()));
        let slot = Arc::new(ProviderSlot {
            fingerprint: "test-provider".into(),
            dimension: 2,
            provider: Mutex::new(Box::new(DeterministicProvider)),
            admission: Arc::clone(&gate),
        });
        let held = gate.lock();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            started_tx.send(()).expect("worker started");
            result_tx.send(slot.embed_batch(&["queued".into()])).expect("provider result sent");
        });
        started_rx.recv().expect("worker start signal");
        assert!(matches!(
            result_rx.recv_timeout(std::time::Duration::from_millis(25)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout),
        ));
        drop(held);
        let vectors = result_rx.recv_timeout(std::time::Duration::from_secs(1))
            .expect("provider was admitted")
            .expect("queued embedding");
        worker.join().expect("provider worker");
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].len(), 2);
    }
    #[test]
    fn unregister_archives_prepared_changeset_without_retaining_source_text() {
        let temp = TempDir::new().expect("temporary directory");
        let service = Arc::new(WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens"));
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "private base").expect("source file");
        service.register_workspace("archive", &root).expect("workspace registration");
        let base_hash = test_sha256(b"private base");
        let proposed_hash = test_sha256(b"private private proposed");
        let prepared = service.changeset_prepare("archive", vec![ChangeOperation::Replace {
            path: "source.rs".into(),
            expected_sha256: base_hash.clone(),
            replacements: vec![crate::changeset::Replacement {
                old_text: "base".into(), new_text: "private proposed".into(), replace_all: false,
            }],
        }]).expect("changeset preparation");
        let payload = service.paths.pending_dir.join(format!("{}.json", prepared.operation_id));
        assert!(std::fs::read_to_string(&payload).expect("private payload").contains("private base"));

        service.unregister_workspace("archive").expect("workspace unregister");

        assert!(!payload.exists());
        let archived = service.changeset_status(&prepared.operation_id).expect("archived status");
        assert!(archived.archived);
        assert_eq!(archived.state, "Prepared");
        assert_eq!(archived.source_status, "unregistered");
        assert_eq!(archived.files[0].path, "source.rs");
        assert_eq!(archived.files[0].base_sha256.as_deref(), Some(base_hash.as_str()));
        assert_eq!(archived.files[0].operation, "replace");
        assert_eq!(archived.files[0].state, "pending");
        assert_eq!(archived.files[0].proposed_sha256.as_deref(), Some(proposed_hash.as_str()));
        assert!(archived.audit_events.iter().any(|event| event.event_type == "changeset_prepare"));
        assert!(archived.audit_events.iter().any(|event| event.event_type == "workspace_unregister"));
        assert!(service.changeset_apply(&prepared.operation_id, &prepared.digest).is_err());
        assert!(service.changeset_discard(&prepared.operation_id, &prepared.digest).is_err());
        assert_eq!(std::fs::read_to_string(root.join("source.rs")).expect("source remains"), "private base");
        let archive_json: String = service.connection.lock().query_row(
            "SELECT archive_json FROM unregistered_changesets WHERE operation_id=?1",
            [&prepared.operation_id],
            |row| row.get(0),
        ).expect("archive row");
        assert!(!archive_json.contains("private base"));
        assert!(!archive_json.contains("private proposed"));
    }

    #[test]
    fn startup_retries_unregistration_payload_cleanup_after_unlink_failure() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = Arc::new(WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens"));
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("cleanup", &root).expect("workspace registration");
        let prepared = service.changeset_prepare("cleanup", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "private".into(),
        }]).expect("changeset preparation");
        let payload = paths.pending_dir.join(format!("{}.json", prepared.operation_id));
        std::fs::remove_file(&payload).expect("remove payload before fault");
        std::fs::create_dir(&payload).expect("directory blocks file unlink");

        service.unregister_workspace("cleanup").expect("unregister remains committed");

        let queued: i64 = service.connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup WHERE operation_id=?1",
            [&prepared.operation_id],
            |row| row.get(0),
        ).expect("durable cleanup record");
        assert_eq!(queued, 1);
        drop(service);
        std::fs::remove_dir(&payload).expect("remove unlink fault");
        std::fs::write(&payload, "private stale payload").expect("restore simulated crash residue");
        let reopened = WorkspaceService::open_with_provider(
            paths, test_config(), Box::new(DeterministicProvider),
        ).expect("startup retries cleanup");
        assert!(!payload.exists());
        let queued: i64 = reopened.connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup WHERE operation_id=?1",
            [&prepared.operation_id],
            |row| row.get(0),
        ).expect("cleanup record cleared");
        assert_eq!(queued, 0);
        assert!(reopened.changeset_status(&prepared.operation_id).expect("archived status").archived);
    }

    #[test]
    fn malformed_cleanup_sweeps_only_unreferenced_canonical_payload_files() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let invalid_id = "not-a-uuid";
        service.connection.lock().execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES (?1, 'unregister', 1)",
            [invalid_id],
        ).expect("malformed cleanup record");
        let orphan_id = uuid::Uuid::new_v4().to_string();
        let orphan = paths.pending_dir.join(format!("{orphan_id}.json"));
        std::fs::write(&orphan, b"orphan payload").expect("orphan payload");
        let ordinary = paths.pending_dir.join("notes.txt");
        std::fs::write(&ordinary, b"ordinary file").expect("ordinary file");
        let target = paths.pending_dir.join("symlink-target");
        std::fs::write(&target, b"symlink target").expect("symlink target");
        let link = paths.pending_dir.join(format!("{}.json", uuid::Uuid::new_v4()));
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).expect("payload symlink");

        let report = service.doctor_report().expect("doctor recovers malformed cleanup");

        assert!(!orphan.exists());
        assert_eq!(std::fs::read(&ordinary).expect("ordinary file remains"), b"ordinary file");
        #[cfg(unix)]
        {
            assert!(std::fs::symlink_metadata(&link).expect("symlink remains").file_type().is_symlink());
            assert_eq!(std::fs::read(&target).expect("symlink target remains"), b"symlink target");
        }
        assert!(report.ok);
        assert_eq!(report.payload_cleanup_pending, 0);
        assert_eq!(report.payload_cleanup_unresolved, 0);
    }

    #[test]
    fn malformed_cleanup_preserves_active_and_queued_payloads_but_collects_archive_orphan() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("cleanup-protected", &root).expect("workspace registration");
        let active = service.changeset_prepare("cleanup-protected", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "active".into(),
        }]).expect("active changeset");
        let queued_id = uuid::Uuid::new_v4().to_string();
        let archived_id = uuid::Uuid::new_v4().to_string();
        let queued_path = paths.pending_dir.join(format!("{queued_id}.json"));
        let archived_path = paths.pending_dir.join(format!("{archived_id}.json"));
        std::fs::write(&queued_path, b"queued payload").expect("queued payload");
        std::fs::write(&archived_path, b"archived payload").expect("archived payload");
        let invalid_id = "malformed-cleanup-id";
        let connection = service.connection.lock();
        connection.execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES (?1, 'unregister', 1)",
            [invalid_id],
        ).expect("malformed cleanup record");
        connection.execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES (?1, 'prepare', 1)",
            [&queued_id],
        ).expect("queued payload record");
        connection.execute(
            "INSERT INTO unregistered_changesets
             (operation_id, workspace_id, digest, state, archive_json, archived_at)
             VALUES (?1, 'removed', 'digest', 'Prepared', '{}', 1)",
            [&archived_id],
        ).expect("archive reference");
        drop(connection);
        let active_path = paths.pending_dir.join(format!("{}.json", active.operation_id));

        let report = service.doctor_report().expect("doctor retries cleanup");
        assert!(active_path.is_file());
        assert!(queued_path.is_file());
        assert!(!archived_path.exists());
        assert!(report.ok);
        assert_eq!(report.payload_cleanup_pending, 0);
        assert_eq!(report.payload_cleanup_unresolved, 0);
    }

    #[test]
    fn malformed_cleanup_does_not_treat_terminal_changeset_as_payload_owner() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("cleanup-terminal", &root).expect("workspace registration");
        let terminal = service.changeset_prepare("cleanup-terminal", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "terminal".into(),
        }]).expect("changeset preparation");
        let payload = paths.pending_dir.join(format!("{}.json", terminal.operation_id));
        service.connection.lock().execute(
            "UPDATE pending_changesets SET state='Applied' WHERE operation_id=?1",
            [&terminal.operation_id],
        ).expect("terminal state");
        service.connection.lock().execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES ('malformed-terminal-cleanup', 'terminal', 1)",
            [],
        ).expect("malformed cleanup row");

        let changesets = ChangesetService::for_service(
            &service.connection,
            &paths.pending_dir,
            test_config().limits.changeset_expiry_seconds,
            test_config().limits.max_file_bytes,
        ).expect("changeset cleanup service");
        assert_eq!(changesets.retry_payload_cleanup().expect("malformed cleanup recovery"), 0);
        assert!(!payload.exists(), "malformed cleanup recovered the sole terminal payload");
        let malformed_rows: i64 = service.connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup
             WHERE operation_id='malformed-terminal-cleanup'",
            [],
            |row| row.get(0),
        ).expect("malformed cleanup row count");
        assert_eq!(malformed_rows, 0);

        let report = service.doctor_report().expect("doctor enqueues and cleans terminal payload");
        assert!(!payload.exists());
        assert!(report.ok);
        assert_eq!(report.payload_cleanup_pending, 0);
        assert_eq!(report.payload_cleanup_unresolved, 0);
    }

    #[test]
    fn malformed_cleanup_with_only_referenced_payloads_remains_unresolved() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("cleanup-referenced", &root).expect("workspace registration");
        let active = service.changeset_prepare("cleanup-referenced", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "active".into(),
        }]).expect("active changeset");
        let queued_id = uuid::Uuid::new_v4().to_string();
        let queued_path = paths.pending_dir.join(format!("{queued_id}.json"));
        std::fs::write(&queued_path, b"queued payload").expect("queued payload");
        let connection = service.connection.lock();
        connection.execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES ('malformed-only-protected', 'unregister', 1)",
            [],
        ).expect("malformed cleanup row");
        connection.execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES (?1, 'prepare', 1)",
            [&queued_id],
        ).expect("queued payload reference");
        drop(connection);

        let report = service.doctor_report().expect("doctor reports unresolved row");
        let active_path = paths.pending_dir.join(format!("{}.json", active.operation_id));
        assert!(active_path.is_file());
        assert!(queued_path.is_file());
        assert!(!report.ok);
        assert_eq!(report.payload_cleanup_pending, 1);
        assert_eq!(report.payload_cleanup_unresolved, 1);
    }

    #[test]
    fn malformed_cleanup_enumeration_failure_stays_visible_to_doctor() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        service.connection.lock().execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES ('invalid', 'unregister', 1)",
            [],
        ).expect("malformed cleanup record");
        let moved_pending = paths.pending_dir.with_extension("preserved");
        std::fs::write(paths.pending_dir.join("retained.json"), b"payload remains")
            .expect("pending payload");
        std::fs::rename(&paths.pending_dir, &moved_pending).expect("move pending directory");
        std::fs::write(&paths.pending_dir, b"not a directory").expect("block enumeration");

        let report = service.doctor_report().expect("doctor reports unresolved cleanup");

        assert!(!report.ok);
        assert_eq!(report.payload_cleanup_pending, 1);
        assert_eq!(report.payload_cleanup_unresolved, 1);
        assert_eq!(std::fs::read(&paths.pending_dir).expect("invalid path remains"), b"not a directory");
        assert_eq!(
            std::fs::read(moved_pending.join("retained.json")).expect("retained payload"),
            b"payload remains",
        );
        let failure: Option<String> = service.connection.lock().query_row(
            "SELECT last_error FROM changeset_payload_cleanup WHERE operation_id='invalid'",
            [],
            |row| row.get(0),
        ).expect("retained failure reason");
        assert_eq!(failure.as_deref(), Some("pending_directory_unavailable"));
    }

    #[test]
    fn malformed_cleanup_unlink_failure_retains_queue_and_orphan() {
        let temp = TempDir::new().expect("temporary directory");
        let paths = test_paths(&temp);
        let service = WorkspaceService::open_with_provider(
            paths.clone(), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let orphan_id = uuid::Uuid::new_v4().to_string();
        let orphan = paths.pending_dir.join(format!("{orphan_id}.json"));
        std::fs::write(&orphan, b"orphan payload").expect("orphan payload");
        service.connection.lock().execute(
            "INSERT INTO changeset_payload_cleanup(operation_id, reason, queued_at)
             VALUES ('invalid-unlink-id', 'unregister', 1)",
            [],
        ).expect("malformed cleanup record");
        let changesets = ChangesetService::for_service(
            &service.connection,
            &paths.pending_dir,
            test_config().limits.changeset_expiry_seconds,
            test_config().limits.max_file_bytes,
        ).expect("changeset service");

        changesets.retry_malformed_cleanup_with_unlink_failure("invalid-unlink-id")
            .expect("unlink failure is recorded for retry");

        assert!(orphan.is_file());
        let record: (i64, Option<String>) = service.connection.lock().query_row(
            "SELECT attempts, last_error FROM changeset_payload_cleanup
             WHERE operation_id='invalid-unlink-id'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).expect("malformed cleanup remains queued");
        assert_eq!(record.0, 1);
        assert_eq!(record.1.as_deref(), Some("recovery_unlink_permission_denied"));
    }

    #[test]
    fn unregister_archive_transaction_failure_preserves_workspace_and_payload() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("rollback", &root).expect("workspace registration");
        let prepared = service.changeset_prepare("rollback", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "private".into(),
        }]).expect("changeset preparation");
        let payload = service.paths.pending_dir.join(format!("{}.json", prepared.operation_id));
        service.connection.lock().execute_batch(
            "CREATE TRIGGER fail_changeset_archive BEFORE INSERT ON unregistered_changesets
             BEGIN SELECT RAISE(ABORT, 'archive fault'); END;",
        ).expect("install transaction fault");

        assert!(service.unregister_workspace("rollback").is_err());

        assert!(service.workspace_root("rollback").is_ok());
        assert!(payload.is_file());
        assert!(!service.changeset_status(&prepared.operation_id).expect("active status").archived);
        let cleanup_rows: i64 = service.connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup WHERE operation_id=?1",
            [&prepared.operation_id],
            |row| row.get(0),
        ).expect("cleanup queue remains empty");
        assert_eq!(cleanup_rows, 0);
    }

    #[test]
    fn failed_prepare_transaction_removes_or_queues_private_payload() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("prepare-fault", &root).expect("workspace registration");
        service.connection.lock().execute_batch(
            "CREATE TRIGGER fail_changeset_file BEFORE INSERT ON changeset_files
             BEGIN SELECT RAISE(ABORT, 'file journal fault'); END;",
        ).expect("install transaction fault");

        assert!(service.changeset_prepare("prepare-fault", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "private".into(),
        }]).is_err());

        let pending_files: Vec<_> = std::fs::read_dir(&service.paths.pending_dir)
            .expect("pending directory")
            .collect();
        assert!(pending_files.is_empty());
        let pending_rows: i64 = service.connection.lock().query_row(
            "SELECT count(*) FROM pending_changesets WHERE workspace_id='prepare-fault'",
            [],
            |row| row.get(0),
        ).expect("no prepared row");
        let cleanup_rows: i64 = service.connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup",
            [],
            |row| row.get(0),
        ).expect("no cleanup remains after successful unlink");
        assert_eq!(pending_rows, 0);
        assert_eq!(cleanup_rows, 0);
    }

    #[test]
    fn malformed_archived_json_and_paths_fail_closed() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("malformed", &root).expect("workspace registration");
        let prepared = service.changeset_prepare("malformed", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "private".into(),
        }]).expect("changeset preparation");
        service.unregister_workspace("malformed").expect("workspace unregister");
        service.connection.lock().execute_batch(
            "DROP TRIGGER unregistered_changesets_no_update;",
        ).expect("disable append-only guard for corruption fixture");
        let original_json: String = service.connection.lock().query_row(
            "SELECT archive_json FROM unregistered_changesets WHERE operation_id=?1",
            [&prepared.operation_id],
            |row| row.get(0),
        ).expect("archived json");
        service.connection.lock().execute(
            "UPDATE unregistered_changesets SET archive_json='{' WHERE operation_id=?1",
            [&prepared.operation_id],
        ).expect("corrupt archive json");
        assert!(service.changeset_status(&prepared.operation_id).is_err());

        let mut archive: serde_json::Value =
            serde_json::from_str(&original_json).expect("valid archive fixture");
        archive["outcome"]["files"][0]["path"] = "../escape".into();
        let invalid_path_json = serde_json::to_string(&archive).expect("invalid path fixture");
        service.connection.lock().execute(
            "UPDATE unregistered_changesets SET archive_json=?1 WHERE operation_id=?2",
            rusqlite::params![invalid_path_json, prepared.operation_id],
        ).expect("install invalid archived path");
        assert!(service.changeset_status(&prepared.operation_id).is_err());
    }
    #[test]
    fn discard_removes_prepared_payload_through_durable_cleanup_queue() {
        let temp = TempDir::new().expect("temporary directory");
        let service = WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens");
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("discard", &root).expect("workspace registration");
        let prepared = service.changeset_prepare("discard", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "private".into(),
        }]).expect("changeset preparation");
        let payload = service.paths.pending_dir.join(format!("{}.json", prepared.operation_id));

        let discarded = service.changeset_discard(&prepared.operation_id, &prepared.digest)
            .expect("changeset discard");

        assert_eq!(discarded.state, "Discarded");
        assert!(!payload.exists());
        let queued: i64 = service.connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup WHERE operation_id=?1",
            [&prepared.operation_id],
            |row| row.get(0),
        ).expect("cleanup queue");
        assert_eq!(queued, 0);
    }

    #[test]
    fn expired_changeset_removes_payload_through_durable_cleanup_queue() {
        let temp = TempDir::new().expect("temporary directory");
        let service = Arc::new(WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens"));
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("expired", &root).expect("workspace registration");
        let prepared = service.changeset_prepare("expired", vec![ChangeOperation::Create {
            path: "new.rs".into(), content: "private".into(),
        }]).expect("changeset preparation");
        let payload = service.paths.pending_dir.join(format!("{}.json", prepared.operation_id));
        service.connection.lock().execute(
            "UPDATE pending_changesets SET expires_at=0 WHERE operation_id=?1",
            [&prepared.operation_id],
        ).expect("expire changeset");

        let error = service.changeset_apply(&prepared.operation_id, &prepared.digest)
            .expect_err("expired changeset cannot apply");

        assert_eq!(error.code, ErrorCode::Expired);
        assert!(!payload.exists());
        assert_eq!(service.changeset_status(&prepared.operation_id).expect("expired status").state, "Expired");
        let queued: i64 = service.connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup WHERE operation_id=?1",
            [&prepared.operation_id],
            |row| row.get(0),
        ).expect("cleanup queue");
        assert_eq!(queued, 0);
    }
    #[test]
    fn concurrent_prepare_and_unregister_leaves_no_unarchived_payload() {
        let temp = TempDir::new().expect("temporary directory");
        let service = Arc::new(WorkspaceService::open_with_provider(
            test_paths(&temp), test_config(), Box::new(DeterministicProvider),
        ).expect("service opens"));
        let root = temp.path().join("workspace");
        std::fs::create_dir(&root).expect("workspace directory");
        service.register_workspace("race", &root).expect("workspace registration");
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let prepare_service = Arc::clone(&service);
        let prepare_barrier = Arc::clone(&barrier);
        let prepare = std::thread::spawn(move || {
            prepare_barrier.wait();
            prepare_service.changeset_prepare("race", vec![ChangeOperation::Create {
                path: "new.rs".into(), content: "private".into(),
            }]).ok()
        });
        let unregister_service = Arc::clone(&service);
        let unregister_barrier = Arc::clone(&barrier);
        let unregister = std::thread::spawn(move || {
            unregister_barrier.wait();
            unregister_service.unregister_workspace("race")
        });
        barrier.wait();
        let prepared = prepare.join().expect("prepare worker");
        unregister.join().expect("unregister worker").expect("workspace unregister");

        assert!(std::fs::read_dir(&service.paths.pending_dir)
            .expect("pending directory")
            .next()
            .is_none());
        if let Some(prepared) = prepared {
            let archived = service.changeset_status(&prepared.operation_id).expect("archived status");
            assert!(archived.archived);
            assert_eq!(archived.state, "Prepared");
        }
    }
}
