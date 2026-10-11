use crate::config::{Config, Paths};
use crate::model::{ErrorCode, IndexState, Page, ToolResult, Warning, WorkspaceMode};
use crate::retrieval::search::{CodeSearchRequest, SymbolSearchRequest};
use crate::service::WorkspaceService;
use crate::tool_registry::{
    ChangesetApplyArgs, ChangesetIdArgs, ChangesetPrepareArgs, FileReadArgs, OnboardWorkspaceArgs,
    ReconcileArgs, SearchArgs, SetModeArgs, SymbolSearchArgs, WorkspaceIdArgs, WorkspaceRegisterArgs,
    validate_tool_definitions,
};
use crate::watch::{WATCHER_STOP_TIMEOUT, WatchConfig, WatchStopOutcome, WorkspaceWatcher};
use parking_lot::Mutex;
use rmcp::{ErrorData, Json, ServerHandler, RoleServer, ServiceExt, tool, tool_handler, tool_router, prompt, prompt_handler, prompt_router};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ListResourcesResult, ListResourceTemplatesResult, PaginatedRequestParams, PromptMessage, ProtocolVersion, ReadResourceRequestParams, ReadResourceResponse, ReadResourceResult, Resource, ResourceContents, ResourceTemplate, Role, ServerCapabilities, ServerConfig};
use rmcp::handler::server::router::prompt::PromptRouter;
use schemars::JsonSchema;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::Instant;
use tokio::sync::Notify;

const MAX_RESOURCE_SUBSCRIPTIONS: usize = 128;

#[derive(Clone)]
pub struct WorkspaceIndexServer { tool_router: rmcp::handler::server::tool::ToolRouter<Self>, prompt_router: PromptRouter<Self>, state: Arc<ServerState> }

struct ServerState {
    watchers: Mutex<HashMap<String, WorkspaceWatcher>>,
    service: Arc<WorkspaceService>,
    subscriptions: Mutex<std::collections::HashSet<String>>,
    last_resource_fingerprint: Mutex<HashMap<String, String>>,
    notification_gate: tokio::sync::Mutex<()>,
    blocking_tasks: Arc<BlockingTaskRegistry>,
}
impl Drop for ServerState {
    fn drop(&mut self) {
        if let Err(error) = stop_watchers_until(self, Instant::now()) {
            eprintln!("workspace-index: watcher shutdown persistence failure: {error}");
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BlockingTaskSummary {
    task_id: u64,
    operation: String,
    workspace_id: Option<String>,
    state: &'static str,
}

#[derive(Default)]
struct BlockingTaskRegistry {
    next_id: AtomicU64,
    active: Mutex<HashMap<u64, BlockingTaskSummary>>,
    changed: Notify,
}

impl BlockingTaskRegistry {
    fn start(
        self: &Arc<Self>,
        operation: &str,
        workspace_id: Option<&str>,
    ) -> BlockingTaskGuard {
        let task_id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.active.lock().insert(task_id, BlockingTaskSummary {
            task_id,
            operation: operation.to_owned(),
            workspace_id: workspace_id.map(str::to_owned),
            state: "running",
        });
        BlockingTaskGuard { registry: Arc::clone(self), task_id }
    }

    fn active(&self) -> Vec<BlockingTaskSummary> {
        let mut tasks = self.active.lock().values().cloned().collect::<Vec<_>>();
        tasks.sort_by_key(|task| task.task_id);
        tasks
    }

    #[cfg(test)]
    fn workspace_is_idle(&self, workspace_id: &str) -> bool {
        !self.active.lock().values()
            .any(|task| task.workspace_id.as_deref() == Some(workspace_id))
    }

    #[cfg(test)]
    async fn wait_for_workspace_idle(&self, workspace_id: &str) {
        loop {
            let notified = self.changed.notified();
            if self.workspace_is_idle(workspace_id) {
                return;
            }
            notified.await;
        }
    }
}

struct BlockingTaskGuard {
    registry: Arc<BlockingTaskRegistry>,
    task_id: u64,
}

impl Drop for BlockingTaskGuard {
    fn drop(&mut self) {
        self.registry.active.lock().remove(&self.task_id);
        self.registry.changed.notify_one();
    }
}
fn stop_watchers_until(state: &ServerState, deadline: Instant) -> Result<(), String> {
    let watchers = std::mem::take(&mut *state.watchers.lock());
    let mut failures = Vec::new();
    for (workspace_id, mut watcher) in watchers {
        match watcher.stop_until(deadline) {
            WatchStopOutcome::Stopped => {}
            WatchStopOutcome::TimedOut => {
                let message = format!(
                    "watcher stop timed out for {workspace_id}; worker remains tracked"
                );
                eprintln!("workspace-index: {message}");
                failures.push(message);
            }
            WatchStopOutcome::ReconciliationQueueFailed(error) => {
                let message = format!(
                    "watcher shutdown reconciliation enqueue failed for {workspace_id}: {error}"
                );
                eprintln!("workspace-index: {message}");
                failures.push(message);
            }
        }
    }
    if failures.is_empty() { Ok(()) } else { Err(failures.join("; ")) }
}
async fn stop_watchers_blocking(state: Arc<ServerState>) -> Result<(), String> {
    let deadline = Instant::now() + WATCHER_STOP_TIMEOUT;
    let shutdown_state = Arc::clone(&state);
    WorkspaceIndexServer::run_blocking_for(&state, "watcher_shutdown", None, move || {
        stop_watchers_until(&shutdown_state, deadline)
    }).await.map_err(|error| error.to_string())?
}

#[derive(Debug, Clone)]
struct WorkspaceResourceUri {
    workspace_id: String,
    kind: String,
    limit: usize,
    cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
struct WorkspaceSummary {
    workspace_id: String,
    mode: WorkspaceMode,
    registration_epoch: u64,
    mode_epoch: u64,
    state: IndexState,
    visible_generation: u64,
    building_generation: u64,
    dirty: bool,
    health: String,
    queued_jobs: u64,
    running_jobs: u64,
    manifest_version: String,
    graph_generation: Option<u64>,
    file_count: u64,
    chunk_count: u64,
}


impl WorkspaceIndexServer {
    pub fn open(paths: Paths, config: Config) -> Result<Self, String> {
        let service = Arc::new(WorkspaceService::open(paths, config).map_err(|error| error.to_string())?);
        Self::from_service(service)
    }

    fn from_service(service: Arc<WorkspaceService>) -> Result<Self, String> {
        let server = Self {
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
            state: Arc::new(ServerState {
                watchers: Mutex::new(HashMap::new()),
                service,
                subscriptions: Mutex::new(std::collections::HashSet::new()),
                last_resource_fingerprint: Mutex::new(HashMap::new()),
                notification_gate: tokio::sync::Mutex::new(()),
                blocking_tasks: Arc::new(BlockingTaskRegistry::default()),
            }),
        };
        validate_tool_definitions(&server.tool_router.list_all())?;
        if let Err(error) = server.start_all_watchers() {
            if let Err(stop_error) = stop_watchers_until(
                &server.state,
                Instant::now() + WATCHER_STOP_TIMEOUT,
            ) {
                eprintln!("workspace-index: watcher cleanup after startup failure failed: {stop_error}");
            }
            return Err(format!("workspace watcher startup failed: {error}"));
        }
        Ok(server)
    }
    pub fn registered_tool_names(&self) -> Vec<String> {
        self.tool_router.list_all().into_iter().map(|tool| tool.name.to_string()).collect()
    }
    fn start_all_watchers(&self) -> Result<(), String> {
        for record in self.state.service.list_workspaces().map_err(|error| error.to_string())? {
            self.start_workspace_watcher(&record.id, Path::new(&record.root_path))?;
        }
        Ok(())
    }

    fn start_workspace_watcher(&self, workspace_id: &str, root: &Path) -> Result<(), String> {
        Self::start_workspace_watcher_in_state(&self.state, workspace_id, root)
    }

    fn start_workspace_watcher_in_state(
        state: &ServerState,
        workspace_id: &str,
        root: &Path,
    ) -> Result<(), String> {
        if state.watchers.lock().contains_key(workspace_id) { return Ok(()); }
        let backend = state.service.watch_backend(workspace_id);
        let mut config = WatchConfig::default();
        config.periodic_reconcile_interval = std::time::Duration::from_secs(
            state.service.config().reconciliation_interval_seconds,
        );
        config.max_file_bytes = state.service.config().limits.max_file_bytes;
        config.queue_capacity = state.service.config().limits.watcher_queue_paths;
        let watcher = WorkspaceWatcher::start(workspace_id, root, backend, config).map_err(|error| error.to_string())?;
        state.watchers.lock().insert(workspace_id.to_owned(), watcher);
        Ok(())
    }


    async fn run_blocking_for<T, F>(
        state: &ServerState,
        operation: &str,
        workspace_id: Option<&str>,
        task: F,
    ) -> Result<T, ErrorData>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let tasks = Arc::clone(&state.blocking_tasks);
        let service_guard = Arc::clone(&state.service);
        let operation = operation.to_owned();
        let workspace_id = workspace_id.map(str::to_owned);
        tokio::task::spawn_blocking(move || {
            let _service_guard = service_guard;
            let _task_guard = tasks.start(&operation, workspace_id.as_deref());
            task()
        }).await.map_err(|_| {
            ErrorData::internal_error("Workspace service worker failed", None)
        })
    }
    async fn resolve_workspace_blocking(
        &self,
        requested: Option<String>,
    ) -> Result<String, ToolResult<Value>> {
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "resolve_workspace", None, move || service.resolve_workspace(requested.as_deref())).await {
            Ok(Ok(id)) => Ok(id),
            Ok(Err(error)) => Err(ToolResult::failure(error.code, error.message, error.retryable)),
            Err(_) => Err(ToolResult::failure(ErrorCode::Conflict, "Workspace service worker failed", true)),
        }
    }

    async fn resolve_search_workspaces_blocking(
        &self,
        ids: Vec<String>,
    ) -> Result<Vec<String>, ToolResult<Value>> {
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "resolve_search_workspaces", None, move || service.validate_workspace_ids(ids)).await {
            Ok(Ok(ids)) => Ok(ids),
            Ok(Err(error)) => Err(ToolResult::failure(error.code, error.message, error.retryable)),
            Err(_) => Err(ToolResult::failure(ErrorCode::Conflict, "Workspace service worker failed", true)),
        }
    }
    fn response<T: Serialize>(result: ToolResult<T>) -> Result<Json<Value>, ErrorData> {
        serde_json::to_value(result).map(Json).map_err(|error| ErrorData::internal_error("Failed to serialize tool result", Some(json!({"category": format!("{:?}", error.classify())}))))
    }
    fn failure(code: ErrorCode, message: impl Into<String>, retryable: bool) -> Result<Json<Value>, ErrorData> {
        Self::response::<Value>(ToolResult::failure(code, message, retryable))
    }
}

#[tool_router]
impl WorkspaceIndexServer {
    #[tool(name = "workspace_list", description = "List registered workspaces without exposing absolute roots.")]
    async fn workspace_list(&self) -> Result<Json<Value>, ErrorData> {
        let service = Arc::clone(&self.state.service);
        let items = match Self::run_blocking_for(&self.state, "workspace_list", None, move || {
            let records = service.list_workspaces()?;
            records.into_iter().map(|record| {
                let status = service.index_status(&record.id)?;
                Ok(WorkspaceSummary {
                    workspace_id: record.id,
                    mode: record.mode,
                    registration_epoch: status.registration_epoch,
                    mode_epoch: status.mode_epoch,
                    state: status.state,
                    visible_generation: status.visible_generation,
                    building_generation: status.building_generation,
                    dirty: status.dirty,
                    health: status.health,
                    queued_jobs: status.queued_jobs,
                    running_jobs: status.running_jobs,
                    manifest_version: status.manifest_version,
                    graph_generation: status.graph_generation,
                    file_count: status.file_count,
                    chunk_count: status.chunk_count,
                })
            }).collect::<Result<Vec<_>, crate::service::ServiceError>>()
        }).await? {
            Ok(items) => items,
            Err(error) => return Self::failure(error.code, error.message, error.retryable),
        };
        Self::response(ToolResult::success(Page { items, truncated: false, next_cursor: None }))
    }

    #[tool(name = "workspace_register", description = "Register an explicit workspace_id and absolute directory. New workspaces default to read-only.")]
    async fn workspace_register(&self, Parameters(args): Parameters<WorkspaceRegisterArgs>) -> Result<Json<Value>, ErrorData> {
        let service = Arc::clone(&self.state.service);
        let workspace_id = args.workspace_id;
        let root = Path::new(&args.root).to_path_buf();
        let record = match Self::run_blocking_for(&self.state, "workspace_register", None, move || service.register_workspace(&workspace_id, &root)).await? {
            Ok(record) => record,
            Err(error) => return Self::failure(error.code, error.message, error.retryable),
        };
        let watcher_error = self.start_workspace_watcher(&record.id, Path::new(&record.root_path)).err();
        let result = ToolResult::success(json!({
            "workspace_id": record.id, "mode": record.mode,
            "registration_epoch": record.registration_epoch, "mode_epoch": record.mode_epoch,
        }));
        let result = match watcher_error {
            Some(message) => result.with_warning(Warning {
                code: "INDEX_DEGRADED".into(),
                message,
                retryable: true,
            }),
            None => result,
        };
        Self::response(result)
    }

    #[tool(name = "workspace_set_mode", description = "Set workspace access mode. This build returns UNSUPPORTED for read_write because mutation containment cannot be proven; requires approval.")]
    async fn workspace_set_mode(&self, Parameters(args): Parameters<SetModeArgs>) -> Result<Json<Value>, ErrorData> {
        let service = Arc::clone(&self.state.service);
        let result = Self::run_blocking_for(&self.state, "workspace_set_mode", None, move || service.set_workspace_mode(&args.workspace_id, args.mode)).await?;
        match result {
            Ok(record) => Self::response(ToolResult::success(json!({
                "workspace_id": record.id, "mode": record.mode,
                "registration_epoch": record.registration_epoch, "mode_epoch": record.mode_epoch,
            }))),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }

    #[tool(name = "workspace_unregister", description = "Stop the workspace watcher with a bounded wait, then unregister the workspace. Requires user approval.")]
    async fn workspace_unregister(&self, Parameters(args): Parameters<WorkspaceIdArgs>) -> Result<Json<Value>, ErrorData> {
        let Some(id) = args.workspace_id else { return Self::failure(ErrorCode::WorkspaceRequired, "workspace_id is required", false); };
        let watcher = self.state.watchers.lock().remove(&id);
        let state = Arc::clone(&self.state);
        let service = Arc::clone(&self.state.service);
        let unregister_service = Arc::clone(&service);
        let unregister_id = id.clone();
        let worker_id = id.clone();
        let result = Self::run_blocking_for(&self.state, "workspace_unregister", Some(&id), move || {
            let mut watcher = watcher;
            let had_watcher = watcher.is_some();
            let stop_outcome = watcher.as_mut().map(WorkspaceWatcher::stop);
            match stop_outcome {
                Some(WatchStopOutcome::TimedOut) => {
                    state.watchers.lock().insert(worker_id.clone(), watcher.take().expect("watcher handle retained"));
                    return Err(crate::service::ServiceError {
                        code: ErrorCode::Busy,
                        message: format!(
                            "Watcher is still stopping after the {:?} wait bound; the workspace remains registered and its worker remains tracked.",
                            WATCHER_STOP_TIMEOUT
                        ),
                        retryable: true,
                    });
                }
                Some(WatchStopOutcome::ReconciliationQueueFailed(error)) => {
                    let restart = unregister_service.workspace_root(&worker_id)
                        .and_then(|root| Self::start_workspace_watcher_in_state(&state, &worker_id, &root)
                            .map_err(|restart_error| crate::service::ServiceError {
                                code: ErrorCode::Busy,
                                message: format!("watcher restart failed: {restart_error}"),
                                retryable: true,
                            }));
                    let restart_error = restart.err().map(|restart_error| restart_error.message);
                    let message = match restart_error {
                        Some(restart_error) => format!(
                            "Workspace remains registered because durable shutdown reconciliation could not be queued: {error}; {restart_error}"
                        ),
                        None => format!(
                            "Workspace remains registered because durable shutdown reconciliation could not be queued: {error}"
                        ),
                    };
                    return Err(crate::service::ServiceError {
                        code: ErrorCode::Busy,
                        message,
                        retryable: true,
                    });
                }
                Some(WatchStopOutcome::Stopped) | None => {}
            }
            let result = unregister_service.unregister_workspace(&unregister_id);
            if let Err(error) = result {
            if had_watcher
                && let Err(restart_error) = unregister_service.workspace_root(&worker_id)
                    .and_then(|root| Self::start_workspace_watcher_in_state(&state, &worker_id, &root)
                        .map_err(|message| crate::service::ServiceError {
                            code: ErrorCode::Busy,
                            message,
                            retryable: true,
                        }))
            {
                return Err(crate::service::ServiceError {
                    message: format!("{}; watcher restart failed: {}", error.message, restart_error.message),
                    ..error
                });
            }
                return Err(error);
            }
            Ok(())
        }).await?;
        match result {
            Ok(()) => Self::response(ToolResult::success(json!({"workspace_id": id, "removed": true}))),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }

    #[tool(name = "index_status", description = "Return index health, durable job counts, and active blocking MCP operations that continue if a caller stops waiting.")]
    async fn index_status(&self, Parameters(args): Parameters<WorkspaceIdArgs>) -> Result<Json<Value>, ErrorData> {
        let id = match self.resolve_workspace_blocking(args.workspace_id).await {
            Ok(id) => id,
            Err(result) => return Self::response(result),
        };
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "index_status", None, move || service.index_status(&id)).await? {
            Ok(status) => {
                let mut status = serde_json::to_value(status).map_err(|error| {
                    ErrorData::internal_error("Failed to serialize index status", Some(json!({
                        "category": format!("{:?}", error.classify())
                    })))
                })?;
                let tasks = self.state.blocking_tasks.active();
                let tasks = serde_json::to_value(tasks).map_err(|error| {
                    ErrorData::internal_error("Failed to serialize active task status", Some(json!({
                        "category": format!("{:?}", error.classify())
                    })))
                })?;
                let fields = status.as_object_mut().ok_or_else(|| {
                    ErrorData::internal_error("Index status did not serialize as an object", None)
                })?;
                fields.insert("active_blocking_tasks".into(), tasks);
                Self::response(ToolResult::success(status))
            }
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }

    #[tool(name = "index_reconcile", description = "Reconcile the workspace. Stopping the caller only stops waiting; the tracked worker continues and appears in index_status until completion.")]
    async fn index_reconcile(&self, Parameters(args): Parameters<ReconcileArgs>) -> Result<Json<Value>, ErrorData> {
        let id = match self.resolve_workspace_blocking(args.workspace_id).await {
            Ok(id) => id,
            Err(result) => return Self::response(result),
        };
        let graphify_requested = args.graphify.unwrap_or(false);
        let service = Arc::clone(&self.state.service);
        let workspace_id = id.clone();
        let report = match Self::run_blocking_for(&self.state, "index_reconcile", Some(&id), move || {
            service.reconcile(&workspace_id, args.full.unwrap_or(false), graphify_requested)
        }).await? {
            Ok(report) => report,
            Err(error) => return Self::failure(error.code, error.message, error.retryable),
        };
        let result = ToolResult::success(json!({
            "updated": report.updated, "unchanged": report.unchanged,
            "deleted": report.deleted, "degraded_semantic": report.degraded_semantic,
        }));
        let result = if report.degraded_semantic {
            result.with_warning(Warning {
                code: "INDEX_DEGRADED".into(),
                message: "Index metadata committed, but semantic coverage/profile maintenance is degraded; recovery was queued where possible and active-profile promotion is not implied.".into(),
                retryable: false,
            })
        } else {
            result
        };
        let result = if graphify_requested {
            match self.state.service.graphify_warning(&id) {
                Some(warning) => result.with_warning(warning),
                None => result,
            }
        } else {
            result
        };
        Self::response(result)
    }

    #[tool(name = "code_search", description = "Search selected workspaces with lexical, symbol and optional semantic retrieval; citations include workspace, relative path and line range.")]
    async fn code_search(&self, Parameters(args): Parameters<SearchArgs>) -> Result<Json<Value>, ErrorData> {
        let ids = match self.resolve_search_workspaces_blocking(args.workspace_ids.unwrap_or_default()).await {
            Ok(ids) => ids,
            Err(result) => return Self::response(result),
        };
        let request = CodeSearchRequest {
            query: args.query, query_vector: None, workspace_ids: ids, profile_id: None,
            languages: args.languages.unwrap_or_default(),
            path_globs: args.path_glob.into_iter().collect(),
            limit: args.limit.unwrap_or(10), cursor: args.cursor,
        };
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "code_search", None, move || service.code_search(request)).await? {
            Ok(result) => Self::response(result),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }

    #[tool(name = "symbol_search", description = "Search exact/prefix definitions and indexed reference candidates across explicit workspaces.")]
    async fn symbol_search(&self, Parameters(args): Parameters<SymbolSearchArgs>) -> Result<Json<Value>, ErrorData> {
        let ids = match self.resolve_search_workspaces_blocking(args.workspace_ids.unwrap_or_default()).await {
            Ok(ids) => ids,
            Err(result) => return Self::response(result),
        };
        let request = SymbolSearchRequest {
            query: args.query, workspace_ids: ids, direction: args.direction,
            limit: args.limit.unwrap_or(10), cursor: args.cursor,
        };
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "symbol_search", None, move || service.symbol_search(request)).await? {
            Ok(result) => Self::response(result),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }

    #[tool(name = "file_read", description = "Read UTF-8 source through the workspace filesystem capability. Supports inclusive line ranges and expected SHA-256.")]
    async fn file_read(&self, Parameters(args): Parameters<FileReadArgs>) -> Result<Json<Value>, ErrorData> {
        let id = match self.resolve_workspace_blocking(args.workspace_id).await {
            Ok(id) => id,
            Err(result) => return Self::response(result),
        };
        let service = Arc::clone(&self.state.service);
        let path = Path::new(&args.path).to_path_buf();
        match Self::run_blocking_for(&self.state, "file_read", None, move || service.read_file(
            &id, &path, args.start_line, args.end_line, args.expected_sha256.as_deref(),
        )).await? {
            Ok(result) => Self::response(ToolResult::success(result)),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }
    #[tool(name = "changeset_prepare", description = "Prepare create/replace/delete/move changes in one workspace. Returns a complete bounded approval diff and does not modify source.")]
    async fn changeset_prepare(&self, Parameters(args): Parameters<ChangesetPrepareArgs>) -> Result<Json<Value>, ErrorData> {
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "changeset_prepare", None, move || service.changeset_prepare(&args.workspace_id, args.operations)).await? {
            Ok(prepared) => Self::response(ToolResult::success(prepared)),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }


    #[tool(name = "changeset_apply", description = "Apply a prepared digest-bound changeset. Stopping the caller only stops waiting; inspect changeset_status for completion. Source effects remain disabled until safe parent containment is provable; requires approval.")]
    async fn changeset_apply(&self, Parameters(args): Parameters<ChangesetApplyArgs>) -> Result<Json<Value>, ErrorData> {
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "changeset_apply", None, move || service.changeset_apply(&args.operation_id, &args.digest)).await? {
            Ok(outcome) => Self::response(ToolResult::success(outcome)),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }

    #[tool(name = "changeset_status", description = "Read persisted changeset state and per-file base/proposed/observed hashes.")]
    async fn changeset_status(&self, Parameters(args): Parameters<ChangesetIdArgs>) -> Result<Json<Value>, ErrorData> {
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "changeset_status", None, move || service.changeset_status(&args.operation_id)).await? {
            Ok(outcome) => Self::response(ToolResult::success(outcome)),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }

    #[tool(name = "changeset_discard", description = "Discard only an unapplied prepared changeset. Does not modify source.")]
    async fn changeset_discard(&self, Parameters(args): Parameters<ChangesetApplyArgs>) -> Result<Json<Value>, ErrorData> {
        let service = Arc::clone(&self.state.service);
        match Self::run_blocking_for(&self.state, "changeset_discard", None, move || service.changeset_discard(&args.operation_id, &args.digest)).await? {
            Ok(outcome) => Self::response(ToolResult::success(outcome)),
            Err(error) => Self::failure(error.code, error.message, error.retryable),
        }
    }

}

#[prompt_router]
impl WorkspaceIndexServer {
    #[prompt(name = "onboard-workspace", description = "Inspect a selected workspace status and index to map architecture. workspace_id is optional; require user selection if omitted and multiple workspaces are registered. Cite every claim.")]
    async fn onboard_workspace(&self, Parameters(args): Parameters<OnboardWorkspaceArgs>) -> Vec<PromptMessage> {
        let selection = match args.workspace_id {
            Some(workspace_id) => format!(
                "Use the explicitly supplied workspace_id {:?}; do not select another workspace.",
                workspace_id,
            ),
            None => "Use workspace_list first. If more than one workspace exists, require the user to select an explicit workspace_id before inspecting any workspace.".to_owned(),
        };
        let prompt = format!(
            "{selection} Read the selected workspace manifest and status, then search architecture entry points with code_search. Cite every claim with workspace_id, relative path, line range, and file hash. Do not infer findings from uncited content.",
        );
        vec![PromptMessage::new_text(Role::User, prompt)]
    }
}

#[tool_handler(router = self.tool_router)]
#[prompt_handler(router = self.prompt_router)]
impl ServerHandler for WorkspaceIndexServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().enable_resources().enable_resources_subscribe().enable_prompts().build())
            .with_protocol_version(ProtocolVersion::V_2025_11_25)
            .with_server_info(Implementation::new("workspace-index-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions("Use explicit workspace IDs for multi-workspace queries. Cancelling an MCP request stops waiting; blocking workers continue and are listed in index_status, while durable operation state remains available through index_status or changeset_status. Source mutation is disabled in this build; changeset_apply returns UNSUPPORTED before source effects.")
    }

    fn subscribe(&self, request: rmcp::model::SubscribeRequestParams, _context: rmcp::service::RequestContext<RoleServer>) -> impl Future<Output = Result<(), ErrorData>> + Send + '_ {
        let state = Arc::clone(&self.state);
        async move {
            let _gate = state.notification_gate.lock().await;
            let uri = request.uri;
            let read_state = Arc::clone(&state);
            let read_uri = uri.clone();
            let representation = Self::run_blocking_for(&state, "resource_subscribe", None, move || Self::resource_representation(&read_state, &read_uri)).await??;
            let fingerprint = resource_fingerprint(&representation);
            let mut subscriptions = state.subscriptions.lock();
            if !subscription_capacity_allows(subscriptions.len(), subscriptions.contains(&uri)) {
                Err(ErrorData::invalid_params("Resource subscription limit reached", None))
            } else {
                state.last_resource_fingerprint.lock().insert(uri.clone(), fingerprint);
                subscriptions.insert(uri);
                Ok(())
            }
        }
    }

    async fn unsubscribe(&self, request: rmcp::model::UnsubscribeRequestParams, _context: rmcp::service::RequestContext<RoleServer>) -> Result<(), ErrorData> {
        let _gate = self.state.notification_gate.lock().await;
        self.state.subscriptions.lock().remove(&request.uri);
        self.state.last_resource_fingerprint.lock().remove(&request.uri);
        Ok(())
    }

    fn list_resources(&self, _request: Option<PaginatedRequestParams>, _context: rmcp::service::RequestContext<RoleServer>) -> impl Future<Output = Result<ListResourcesResult, ErrorData>> + Send + '_ {
        let result = ListResourcesResult {
            resources: vec![Resource::new("workspace-index://workspaces", "workspaces").with_description("Registered workspace IDs and safe summary metadata").with_mime_type("application/json")],
            ..Default::default()
        };
        std::future::ready(Ok(result))
    }

    fn list_resource_templates(&self, _request: Option<PaginatedRequestParams>, _context: rmcp::service::RequestContext<RoleServer>) -> impl Future<Output = Result<ListResourceTemplatesResult, ErrorData>> + Send + '_ {
        let result = ListResourceTemplatesResult {
            resource_templates: vec![
                ResourceTemplate::new("workspace-index://workspace/{id}/status", "workspace-status").with_mime_type("application/json"),
                ResourceTemplate::new("workspace-index://workspace/{id}/manifest", "workspace-manifest").with_mime_type("application/json"),
                ResourceTemplate::new("workspace-index://workspace/{id}/graph", "workspace-graph").with_mime_type("application/json"),
            ],
            ..Default::default()
        };
        std::future::ready(Ok(result))
    }

    fn read_resource(&self, request: ReadResourceRequestParams, _context: rmcp::service::RequestContext<RoleServer>) -> impl Future<Output = Result<ReadResourceResponse, ErrorData>> + Send + '_ {
        let state = Arc::clone(&self.state);
        let uri = request.uri;
        async move {
            let read_state = Arc::clone(&state);
            let read_uri = uri.clone();
            let representation = Self::run_blocking_for(&state, "resource_read", None, move || Self::resource_representation(&read_state, &read_uri)).await??;
            Ok(ReadResourceResult::new(vec![ResourceContents::text(representation, uri)]).into())
        }
    }
}

impl WorkspaceIndexServer {
    fn resource_representation(state: &ServerState, uri: &str) -> Result<String, ErrorData> {
        if uri == "workspace-index://workspaces" {
            let records = match state.service.list_workspaces() {
                Ok(records) => records,
                Err(error) => return resource_failure_text(error.code, error.message, error.retryable),
            };
            let mut summaries = Vec::with_capacity(records.len());
            for record in records {
                let status = match state.service.index_status(&record.id) {
                    Ok(status) => status,
                    Err(error) => return resource_failure_text(error.code, error.message, error.retryable),
                };
                summaries.push(WorkspaceSummary {
                    workspace_id: record.id,
                    mode: record.mode,
                    registration_epoch: status.registration_epoch,
                    mode_epoch: status.mode_epoch,
                    state: status.state,
                    visible_generation: status.visible_generation,
                    building_generation: status.building_generation,
                    dirty: status.dirty,
                    health: status.health,
                    queued_jobs: status.queued_jobs,
                    running_jobs: status.running_jobs,
                    manifest_version: status.manifest_version,
                    graph_generation: status.graph_generation,
                    file_count: status.file_count,
                    chunk_count: status.chunk_count,
                });
            }
            return serialize_resource_text(&summaries);
        }
        let Some(resource) = parse_workspace_resource(uri) else {
            return Err(ErrorData::invalid_params("Unsupported workspace resource URI", None));
        };
        if let Err(error) = state.service.resolve_workspace(Some(&resource.workspace_id)) {
            return resource_failure_text(error.code, error.message, error.retryable);
        }
        match resource.kind.as_str() {
            "status" => match state.service.index_status(&resource.workspace_id) {
                Ok(status) => serialize_resource_text(&status),
                Err(error) => resource_failure_text(error.code, error.message, error.retryable),
            },
            "manifest" => match state.service.workspace_manifest(&resource.workspace_id) {
                Ok(manifest) => serialize_resource_text(&manifest),
                Err(error) => resource_failure_text(error.code, error.message, error.retryable),
            },
            "graph" => match state.service.read_graph(
                &resource.workspace_id,
                resource.limit,
                resource.cursor.as_deref(),
            ) {
                Ok((graph, warnings)) => serialize_resource_text(
                    &ToolResult::Success {
                        data: graph,
                        warnings,
                        meta: crate::model::Meta { request_id: String::new() },
                    },
                ),
                Err(error) => resource_failure_text(error.code, error.message, error.retryable),
            },
            _ => Err(ErrorData::invalid_params("Unsupported workspace resource URI", None)),
        }
    }

    pub async fn serve_stdio(paths: Paths, config: Config) -> Result<(), String> {
        let server = tokio::task::spawn_blocking(move || Self::open(paths, config))
            .await.map_err(|error| error.to_string())??;
        let state = Arc::clone(&server.state);
        let running = match server.serve(rmcp::transport::stdio()).await {
            Ok(running) => running,
            Err(error) => {
                stop_watchers_blocking(Arc::clone(&state)).await?;
                return Err(error.to_string());
            }
        };
        let peer = running.peer().clone();
        let polling_state = Arc::clone(&state);
        let polling = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            let state = polling_state;
            loop {
                interval.tick().await;
                let subscriptions = state.subscriptions.lock().clone();
                for uri in subscriptions {
                    let read_state = Arc::clone(&state);
                    let read_uri = uri.clone();
                    let representation = match Self::run_blocking_for(&state, "resource_subscription_update", None, move || {
                        Self::resource_representation(&read_state, &read_uri)
                    }).await.and_then(|result| result) {
                        Ok(representation) => representation,
                        Err(error) => {
                            eprintln!("workspace-index: subscribed resource read failed: {error:?}");
                            continue;
                        }
                    };
                    let fingerprint = resource_fingerprint(&representation);
                    let _gate = state.notification_gate.lock().await;
                    if !state.subscriptions.lock().contains(&uri) {
                        continue;
                    }
                    let changed = state.last_resource_fingerprint.lock().get(&uri)
                        .is_none_or(|last| resource_fingerprint_changed(Some(last), &fingerprint));
                    if !changed { continue; }
                    let notification = rmcp::model::ServerNotification::ResourceUpdatedNotification(
                        rmcp::model::ResourceUpdatedNotification::new(
                            rmcp::model::ResourceUpdatedNotificationParam::new(uri.clone()),
                        ),
                    );
                    match peer.send_notification(notification).await {
                        Ok(()) => {
                            state.last_resource_fingerprint.lock().insert(uri, fingerprint);
                        }
                        Err(error) => eprintln!("workspace-index: resource update notification failed: {error:?}"),
                    }
                }
            }
        });
        let result = running.waiting().await.map_err(|error| error.to_string());
        polling.abort();
        let _ = polling.await;
        stop_watchers_blocking(Arc::clone(&state)).await?;
        result?;
        Ok(())
    }
}

fn parse_workspace_resource(uri: &str) -> Option<WorkspaceResourceUri> {
    let rest = uri.strip_prefix("workspace-index://workspace/")?;
    let (workspace_id, rest) = rest.split_once('/')?;
    if !valid_resource_workspace_id(workspace_id) { return None; }
    let (kind, query) = match rest.split_once('?') {
        Some((kind, query)) => (kind, Some(query)),
        None => (rest, None),
    };
    if !matches!(kind, "status" | "manifest" | "graph") { return None; }
    let mut limit = 100;
    let mut cursor = None;
    if let Some(query) = query {
        if kind != "graph" || query.is_empty() { return None; }
        let mut seen = std::collections::HashSet::new();
        for parameter in query.split('&') {
            let (name, value) = parameter.split_once('=')?;
            if !seen.insert(name) { return None; }
            match name {
                "limit" => {
                    if value.is_empty() { return None; }
                    limit = value.parse::<usize>().ok()?;
                    if !(1..=100).contains(&limit) { return None; }
                }
                "cursor" => cursor = Some(value.to_owned()),
                _ => return None,
            }
        }
    }
    Some(WorkspaceResourceUri {
        workspace_id: workspace_id.to_owned(),
        kind: kind.to_owned(),
        limit,
        cursor,
    })
}

fn valid_resource_workspace_id(value: &str) -> bool {
    let mut bytes = value.bytes();
    matches!(bytes.next(), Some(b'a'..=b'z' | b'0'..=b'9'))
        && value.len() <= 64
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-'))
}
fn serialize_resource_text<T: Serialize>(value: &T) -> Result<String, ErrorData> {
    serde_json::to_string(value).map_err(|error| {
        ErrorData::internal_error(
            "Failed to serialize resource result",
            Some(json!({"category": format!("{:?}", error.classify())})),
        )
    })
}

fn resource_failure_text(
    code: ErrorCode,
    message: impl Into<String>,
    retryable: bool,
) -> Result<String, ErrorData> {
    serialize_resource_text(&ToolResult::<Value>::failure(code, message, retryable))
}

fn subscription_capacity_allows(current: usize, already_subscribed: bool) -> bool {
    already_subscribed || current < MAX_RESOURCE_SUBSCRIPTIONS
}

fn resource_fingerprint(representation: &str) -> String {
    let canonical = serde_json::from_str::<Value>(representation)
        .ok()
        .and_then(|mut value| {
            if let Some(meta) = value.get_mut("meta").and_then(Value::as_object_mut) {
                meta.remove("request_id");
            }
            serde_json::to_vec(&value).ok()
        });
    let bytes = canonical.as_deref().unwrap_or(representation.as_bytes());
    blake3::hash(bytes).to_hex().to_string()
}

fn resource_fingerprint_changed(previous: Option<&str>, current: &str) -> bool {
    previous != Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::WorkspaceIndexStatus;
    #[test]
    fn graph_resource_uris_have_bounded_strict_query_arguments() {
        let parsed = parse_workspace_resource(
            "workspace-index://workspace/team-1/graph?limit=20&cursor=abc.def",
        ).expect("supported graph URI");
        assert_eq!(parsed.workspace_id, "team-1");
        assert_eq!(parsed.kind, "graph");
        assert_eq!(parsed.limit, 20);
        assert_eq!(parsed.cursor.as_deref(), Some("abc.def"));
        assert_eq!(
            parse_workspace_resource("workspace-index://workspace/team-1/graph?cursor=").unwrap().cursor.as_deref(),
            Some(""),
        );
        assert!(parse_workspace_resource("workspace-index://workspace/team/status?limit=2").is_none());
        assert!(parse_workspace_resource("workspace-index://workspace/team/graph?limit=0").is_none());
        assert!(parse_workspace_resource("workspace-index://workspace/team/graph?limit=2&limit=3").is_none());
        assert!(parse_workspace_resource("workspace-index://workspace/team/graph?offset=2").is_none());
        assert!(parse_workspace_resource("workspace-index://workspace/Team/graph").is_none());
        assert!(parse_workspace_resource("https://example.test/graph").is_none());
    }

    #[test]
    fn canonical_status_representation_tracks_all_subscription_fields() {
        let baseline = WorkspaceIndexStatus {
            workspace_id: "repo".into(),
            state: IndexState::Ready,
            visible_generation: 4,
            building_generation: 4,
            dirty: false,
            health: "ready".into(),
            registration_epoch: 1,
            mode_epoch: 1,
            manifest_version: "manifest-a".into(),
            graph_generation: Some(4),
            file_count: 3,
            chunk_count: 5,
            queued_jobs: 0,
            running_jobs: 0,
            active_embedding_profile: Some("profile-a".into()),
        };
        let original = resource_fingerprint(&serde_json::to_string(&baseline).unwrap());
        let variants = [
            WorkspaceIndexStatus { dirty: true, ..baseline.clone() },
            WorkspaceIndexStatus { health: "degraded".into(), ..baseline.clone() },
            WorkspaceIndexStatus { queued_jobs: 1, ..baseline.clone() },
            WorkspaceIndexStatus { running_jobs: 1, ..baseline.clone() },
            WorkspaceIndexStatus { registration_epoch: 2, ..baseline.clone() },
            WorkspaceIndexStatus { mode_epoch: 2, ..baseline.clone() },
            WorkspaceIndexStatus { manifest_version: "manifest-b".into(), ..baseline.clone() },
            WorkspaceIndexStatus { graph_generation: Some(5), ..baseline.clone() },
            WorkspaceIndexStatus { building_generation: 5, ..baseline },
        ];
        for variant in variants {
            let current = resource_fingerprint(&serde_json::to_string(&variant).unwrap());
            assert!(resource_fingerprint_changed(Some(&original), &current));
        }
        assert!(!resource_fingerprint_changed(Some(&original), &original));
        assert!(resource_fingerprint_changed(None, &original));
    }


    #[test]
    fn resource_fingerprint_ignores_only_per_read_request_ids() {
        let first = r#"{"ok":true,"data":{"generation":1},"meta":{"request_id":"first"}}"#;
        let second = r#"{"ok":true,"data":{"generation":1},"meta":{"request_id":"second"}}"#;
        let changed = r#"{"ok":true,"data":{"generation":2},"meta":{"request_id":"third"}}"#;
        let first_fingerprint = resource_fingerprint(first);
        assert_eq!(first_fingerprint, resource_fingerprint(second));
        assert!(resource_fingerprint_changed(
            Some(&first_fingerprint),
            &resource_fingerprint(changed),
        ));
    }
    #[test]
    fn resource_read_failures_remain_error_representations() {
        let text = resource_failure_text(ErrorCode::Conflict, "resource unavailable", false).unwrap();
        let error: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(error["ok"], false);
        let fingerprint = resource_fingerprint(&text);
        assert!(resource_fingerprint_changed(None, &fingerprint));
        assert!(!resource_fingerprint_changed(Some(&fingerprint), &fingerprint));
    }

    #[test]
    fn arbitrary_resource_subscriptions_are_bounded_and_rejected() {
        assert_eq!(MAX_RESOURCE_SUBSCRIPTIONS, 128);
        assert!(subscription_capacity_allows(127, false));
        assert!(!subscription_capacity_allows(128, false));
        assert!(subscription_capacity_allows(128, true));
        assert!(parse_workspace_resource("workspace-index://workspace/repo/unknown").is_none());
        assert!(parse_workspace_resource("workspace-index://workspace/repo/manifest?cursor=raw-offset").is_none());
    }
    #[test]
    fn startup_fails_closed_when_a_registered_workspace_watcher_cannot_start() {
        use crate::index::adapter::EmbeddingProvider;
        use tempfile::TempDir;

        struct TestProvider;
        impl EmbeddingProvider for TestProvider {
            fn fingerprint(&self) -> String { "startup-watcher-test".into() }
            fn dimension(&self) -> usize { 2 }
            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let temp = TempDir::new().expect("temporary directory");
        let paths = Paths::resolve(None, Some(&temp.path().join("data"))).expect("service paths");
        let config = Config { allow_non_git_roots: true, ..Config::default() };
        let service = WorkspaceService::open_with_provider(paths, config, Box::new(TestProvider))
            .expect("service opens");
        let first_root = temp.path().join("a-watched");
        let missing_root = temp.path().join("z-missing");
        std::fs::create_dir_all(&first_root).expect("first workspace");
        std::fs::create_dir_all(&missing_root).expect("second workspace");
        service.register_workspace("first", &first_root).expect("first registration");
        service.register_workspace("second", &missing_root).expect("second registration");
        std::fs::remove_dir_all(&missing_root).expect("simulate unavailable registered root");

        let startup = WorkspaceIndexServer::from_service(Arc::new(service));
        assert!(startup.is_err(), "server must not be returned with a partial watcher fleet");
    }
    fn test_server(service: Arc<WorkspaceService>) -> WorkspaceIndexServer {
        WorkspaceIndexServer {
            tool_router: WorkspaceIndexServer::tool_router(),
            prompt_router: WorkspaceIndexServer::prompt_router(),
            state: Arc::new(ServerState {
                watchers: Mutex::new(HashMap::new()),
                service,
                subscriptions: Mutex::new(std::collections::HashSet::new()),
                last_resource_fingerprint: Mutex::new(HashMap::new()),
                notification_gate: tokio::sync::Mutex::new(()),
                blocking_tasks: Arc::new(BlockingTaskRegistry::default()),
            }),
        }
    }

    struct DeterministicProvider;
    impl crate::index::adapter::EmbeddingProvider for DeterministicProvider {
        fn fingerprint(&self) -> String { "mcp-lifecycle-test".into() }
        fn dimension(&self) -> usize { 2 }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    struct BlockingProvider {
        started: std::sync::mpsc::SyncSender<()>,
        release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
        notified: std::sync::atomic::AtomicBool,
        released: std::sync::atomic::AtomicBool,
    }

    impl crate::index::adapter::EmbeddingProvider for BlockingProvider {
        fn fingerprint(&self) -> String { "mcp-cancellation-test".into() }
        fn dimension(&self) -> usize { 2 }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            use std::sync::atomic::Ordering;
            if !self.released.load(Ordering::Acquire) {
                if !self.notified.swap(true, Ordering::AcqRel) {
                    self.started.send(()).map_err(|error| error.to_string())?;
                }
                self.release.lock().map_err(|error| error.to_string())?
                    .recv().map_err(|error| error.to_string())?;
                self.released.store(true, Ordering::Release);
            }
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_reconcile_waiter_remains_visible_until_worker_finishes() {
        use tempfile::TempDir;

        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        std::fs::write(root.join("source.rs"), "fn cancellation_probe() {}\n").expect("source file");
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let provider = BlockingProvider {
            started: started_tx,
            release: std::sync::Mutex::new(release_rx),
            notified: std::sync::atomic::AtomicBool::new(false),
            released: std::sync::atomic::AtomicBool::new(false),
        };
        let paths = Paths::resolve(None, Some(&temp.path().join("data"))).expect("service paths");
        let service = Arc::new(WorkspaceService::open_with_provider(
            paths,
            Config { allow_non_git_roots: true, ..Config::default() },
            Box::new(provider),
        ).expect("service opens"));
        service.register_workspace("cancel", &root).expect("workspace registration");
        let server = test_server(Arc::clone(&service));
        let worker_state = Arc::clone(&server.state);
        let waiter = tokio::spawn(async move {
            let worker_service = Arc::clone(&worker_state.service);
            WorkspaceIndexServer::run_blocking_for(
                &worker_state,
                "index_reconcile",
                Some("cancel"),
                move || worker_service.reconcile("cancel", true, false),
            ).await
        });
        tokio::task::spawn_blocking(move || started_rx.recv())
            .await.expect("start notification worker")
            .expect("reconciliation entered embedding");
        waiter.abort();
        assert!(waiter.await.expect_err("request waiter cancelled").is_cancelled());

        let Json(status) = server.index_status(Parameters(WorkspaceIdArgs {
            workspace_id: Some("cancel".into()),
        })).await.expect("status query");
        assert!(status["data"]["active_blocking_tasks"].as_array().is_some_and(|tasks| {
            tasks.iter().any(|task| {
                task["operation"] == "index_reconcile"
                    && task["workspace_id"] == "cancel"
                    && task["state"] == "running"
            })
        }));
        release_tx.send(()).expect("release reconciliation");
        server.state.blocking_tasks.wait_for_workspace_idle("cancel").await;
        let Json(status) = server.index_status(Parameters(WorkspaceIdArgs {
            workspace_id: Some("cancel".into()),
        })).await.expect("completed status query");
        assert!(status["data"]["active_blocking_tasks"].as_array().is_some_and(|tasks| {
            !tasks.iter().any(|task| {
                task["operation"] == "index_reconcile" && task["workspace_id"] == "cancel"
            })
        }));
    }

    struct BlockingWatchBackend {
        started: std::sync::mpsc::SyncSender<()>,
        release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
        fail_shutdown_queue: bool,
    }

    impl crate::watch::WatchIndexBackend for BlockingWatchBackend {
        fn apply_incremental(&self, _request: crate::watch::ReconcileRequest) -> Result<(), String> {
            Ok(())
        }
        fn begin_rebuild(&self, _transition: &crate::watch::BranchTransition) -> Result<(), String> {
            Ok(())
        }
        fn reconcile(&self, _request: crate::watch::ReconcileRequest) -> Result<(), String> {
            let _ = self.started.try_send(());
            self.release.lock().map_err(|error| error.to_string())?
                .recv().map_err(|error| error.to_string())
        }
        fn complete_reconciliation(&self, _request: &crate::watch::ReconcileRequest) -> Result<(), String> {
            Ok(())
        }
        fn current_manifest(
            &self,
            _root: &Path,
            _max_file_bytes: u64,
        ) -> Result<crate::watch::Manifest, String> {
            Ok(crate::watch::Manifest::default())
        }
        fn queue_shutdown_reconciliation(&self, _: &str) -> Result<(), String> {
            if self.fail_shutdown_queue {
                Err("injected durable queue failure".into())
            } else {
                Ok(())
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unregister_waits_for_watcher_join_off_the_async_worker() {
        use tempfile::TempDir;

        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let paths = Paths::resolve(None, Some(&temp.path().join("data"))).expect("service paths");
        let service = Arc::new(WorkspaceService::open_with_provider(
            paths,
            Config { allow_non_git_roots: true, ..Config::default() },
            Box::new(DeterministicProvider),
        ).expect("service opens"));
        service.register_workspace("watch-stop", &root).expect("workspace registration");
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let watcher = WorkspaceWatcher::start(
            "watch-stop",
            &root,
            Arc::new(BlockingWatchBackend {
                started: started_tx,
                release: std::sync::Mutex::new(release_rx),
                fail_shutdown_queue: false,
            }),
            crate::watch::WatchConfig::default(),
        ).expect("start blocked watcher");
        let server = test_server(service);
        server.state.watchers.lock().insert("watch-stop".into(), watcher);
        tokio::task::spawn_blocking(move || started_rx.recv())
            .await.expect("start notification worker")
            .expect("watcher entered reconciliation");

        let state = Arc::clone(&server.state);
        let request_server = server.clone();
        let unregister = tokio::spawn(async move {
            request_server.workspace_unregister(Parameters(WorkspaceIdArgs {
                workspace_id: Some("watch-stop".into()),
            })).await
        });
        while state.watchers.lock().contains_key("watch-stop") {
            tokio::task::yield_now().await;
        }
        let (tick_tx, tick_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move { let _ = tick_tx.send(()); });
        tokio::time::timeout(std::time::Duration::from_secs(1), tick_rx)
            .await.expect("async worker stayed responsive")
            .expect("executor heartbeat");
        release_tx.send(()).expect("release blocked watcher");
        let Json(result) = unregister.await.expect("unregister task").expect("unregister tool");
        assert_eq!(result["ok"], true);
        assert_eq!(result["data"]["removed"], true);
        assert!(service_is_unregistered(&state.service));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unregister_keeps_workspace_registered_when_shutdown_enqueue_fails() {
        use tempfile::TempDir;

        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let paths = Paths::resolve(None, Some(&temp.path().join("data"))).expect("service paths");
        let service = Arc::new(WorkspaceService::open_with_provider(
            paths,
            Config { allow_non_git_roots: true, ..Config::default() },
            Box::new(DeterministicProvider),
        ).expect("service opens"));
        service.register_workspace("watch-stop", &root).expect("workspace registration");
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let watcher = WorkspaceWatcher::start(
            "watch-stop",
            &root,
            Arc::new(BlockingWatchBackend {
                started: started_tx,
                release: std::sync::Mutex::new(release_rx),
                fail_shutdown_queue: true,
            }),
            WatchConfig::default(),
        ).expect("start failing watcher");
        let server = test_server(Arc::clone(&service));
        server.state.watchers.lock().insert("watch-stop".into(), watcher);
        tokio::task::spawn_blocking(move || started_rx.recv())
            .await.expect("start notification worker")
            .expect("watcher entered reconciliation");

        let state = Arc::clone(&server.state);
        let request_server = server.clone();
        let unregister = tokio::spawn(async move {
            request_server.workspace_unregister(Parameters(WorkspaceIdArgs {
                workspace_id: Some("watch-stop".into()),
            })).await
        });
        while state.watchers.lock().contains_key("watch-stop") {
            tokio::task::yield_now().await;
        }
        release_tx.send(()).expect("release blocked watcher");
        let Json(result) = unregister.await.expect("unregister task").expect("unregister tool");
        assert_eq!(result["ok"], false);
        assert_eq!(result["error"]["code"], "BUSY");
        assert_eq!(result["error"]["retryable"], true);
        assert!(
            result["error"]["message"].as_str().is_some_and(|message| {
                message.contains("durable shutdown reconciliation could not be queued")
                    && message.contains("injected durable queue failure")
            }),
            "queue failure should be visible in the unregister result: {result}"
        );
        assert!(!service_is_unregistered(&service));
        assert!(state.watchers.lock().contains_key("watch-stop"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_unregister_restarts_watcher_inside_blocking_worker() {
        use tempfile::TempDir;

        let temp = TempDir::new().expect("temporary directory");
        let root = temp.path().join("workspace");
        std::fs::create_dir_all(&root).expect("workspace directory");
        let paths = Paths::resolve(None, Some(&temp.path().join("data"))).expect("service paths");
        let service = Arc::new(WorkspaceService::open_with_provider(
            paths,
            Config { allow_non_git_roots: true, ..Config::default() },
            Box::new(DeterministicProvider),
        ).expect("service opens"));
        service.register_workspace("watch-stop", &root).expect("workspace registration");
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let watcher = WorkspaceWatcher::start(
            "watch-stop",
            &root,
            Arc::new(BlockingWatchBackend {
                started: started_tx,
                release: std::sync::Mutex::new(release_rx),
                fail_shutdown_queue: true,
            }),
            WatchConfig::default(),
        ).expect("start failing watcher");
        let server = test_server(Arc::clone(&service));
        let state = Arc::clone(&server.state);
        server.state.watchers.lock().insert("watch-stop".into(), watcher);
        tokio::task::spawn_blocking(move || started_rx.recv())
            .await.expect("start notification worker")
            .expect("watcher entered reconciliation");

        let request_server = server.clone();
        let unregister = tokio::spawn(async move {
            request_server.workspace_unregister(Parameters(WorkspaceIdArgs {
                workspace_id: Some("watch-stop".into()),
            })).await
        });
        while !state.blocking_tasks.active.lock().values()
            .any(|task| task.operation == "workspace_unregister")
        {
            tokio::task::yield_now().await;
        }
        unregister.abort();
        let join_error = unregister.await.err().expect("MCP future cancelled");
        assert!(join_error.is_cancelled());
        release_tx.send(()).expect("release blocked watcher");

        tokio::time::timeout(
            std::time::Duration::from_secs(3),
            state.blocking_tasks.wait_for_workspace_idle("watch-stop"),
        ).await.expect("unregister blocking worker completed");
        assert!(!service_is_unregistered(&service));
        let watchers = state.watchers.lock();
        let restarted = watchers.get("watch-stop").expect("worker restarted watcher");
        assert!(!restarted.status().stopping, "restarted watcher is active");
    }

    #[test]
    fn server_shutdown_returns_shutdown_queue_failure() {
        use tempfile::TempDir;

        let temp = TempDir::new().expect("temporary directory");
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let watcher = WorkspaceWatcher::start(
            "shutdown-failure",
            temp.path(),
            Arc::new(BlockingWatchBackend {
                started: started_tx,
                release: std::sync::Mutex::new(release_rx),
                fail_shutdown_queue: true,
            }),
            WatchConfig::default(),
        ).expect("start watcher");
        let paths = Paths::resolve(None, Some(&temp.path().join("data"))).expect("service paths");
        let service = WorkspaceService::open_with_provider(
            paths,
            Config { allow_non_git_roots: true, ..Config::default() },
            Box::new(DeterministicProvider),
        ).expect("service opens");
        let server = test_server(Arc::new(service));
        server.state.watchers.lock().insert("shutdown-failure".into(), watcher);
        started_rx.recv_timeout(std::time::Duration::from_secs(2))
            .expect("watcher began reconciliation");
        release_tx.send(()).expect("release watcher");

        let error = stop_watchers_until(
            &server.state,
            Instant::now() + WATCHER_STOP_TIMEOUT,
        ).expect_err("shutdown must propagate persistence failure");
        assert!(error.contains("shutdown-failure"));
        assert!(error.contains("injected durable queue failure"));
    }

    fn service_is_unregistered(service: &WorkspaceService) -> bool {
        service.resolve_workspace(Some("watch-stop")).is_err()
    }
}
