use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use rusqlite::{Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};
use crate::model::{ErrorCode, ToolResult, Page, Warning};
use crate::security::WorkspaceFs;
use super::cursor::{CursorPayload, generate_cursor, verify_cursor};
use sha2::{Digest, Sha256};
use parking_lot::Mutex;
use crate::index::vector::VectorIndex;

const MAX_SEARCH_CANDIDATES: usize = 1000;
// 500 vector IDs plus three metadata binds stays below SQLite's legacy 999-variable limit.
const SEMANTIC_MAPPING_BATCH_SIZE: usize = 500;

const RESULT_SET_TTL: Duration = Duration::from_secs(600);
const MAX_CACHED_RESULT_SETS: usize = 64;
const MAX_CACHED_SEMANTIC_SNAPSHOTS: usize = 64;
const RRF_K: f32 = 60.0;
const EXACT_SYMBOL_BOOST: f32 = 1.5;
const PREFIX_DEFINITION_BOOST: f32 = 1.1;
const PATH_TOKEN_BOOST: f32 = 1.2;
const DEFINITION_BOOST: f32 = 1.1;
const ONE_HOP_DEPENDENCY_BOOST: f32 = 1.1;

type SemanticQueryMap = HashMap<String, (String, Arc<Vec<f32>>)>;

pub struct SearchService<'a> {
    pub conn: &'a Connection,
    pub secret: &'a [u8],
    capabilities: HashMap<String, WorkspaceFs>,
    max_file_bytes: u64,
    max_results: usize,
    max_snippet_bytes: usize,
    max_tool_bytes: usize,
    vectors_dir: PathBuf,
    semantic_queries: SemanticQueryMap,
    result_sets: Arc<SearchResultCache>,
    semantic_snapshots: Arc<SemanticSnapshotCache>,
}

pub struct SearchServiceOptions {
    pub capabilities: HashMap<String, WorkspaceFs>,
    pub max_file_bytes: u64,
    pub max_results: usize,
    pub max_snippet_bytes: usize,
    pub max_tool_bytes: usize,
    pub vectors_dir: PathBuf,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SearchResult {
    pub workspace_id: String,
    pub relative_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub symbol: Option<String>,
    pub kind: Option<String>,
    pub language: Option<String>,
    pub snippet: String,
    pub content_hash: String,
    pub content_hash_algorithm: String,
    pub sha256: String,
    pub visible_generation: u64,
    pub rank: usize,
    pub component_ranks: HashMap<String, usize>,
}

#[derive(Clone)]
pub struct CodeSearchRequest {
    pub query: String,
    pub query_vector: Option<Vec<f32>>,
    pub workspace_ids: Vec<String>,
    pub profile_id: Option<String>,
    pub languages: Vec<String>,
    pub path_globs: Vec<String>,
    pub limit: usize,
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct CandidateKey {
    workspace_id: String,
    relative_path: String,
    chunk_index: usize,
}


#[derive(Default)]
struct SearchContext {
    generations: Vec<(String, u64)>,
    profile_ids_by_workspace: Vec<(String, Option<String>)>,
    transitions: Vec<(String, bool)>,
    registrations: Vec<WorkspaceRegistration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkspaceRegistration {
    workspace_id: String,
    epoch: u64,
    root_path: String,
    device_id: i64,
    inode: i64,
}

#[derive(Clone, Copy)]
struct RankQuery<'a> {
    connection: &'a Connection,
    query: &'a str,
    workspace_ids: &'a [String],
    languages: &'a [String],
    path_matcher: Option<&'a ignore::overrides::Override>,
    pull_limit: usize,
}

#[derive(Clone, Copy)]
struct SemanticRankQuery<'a> {
    connection: &'a Connection,
    queries: &'a SemanticQueryMap,
    workspace_ids: &'a [String],
    languages: &'a [String],
    path_matcher: Option<&'a ignore::overrides::Override>,
    pull_limit: usize,
    vectors_dir: &'a Path,
    snapshots: &'a SemanticSnapshotCache,
}
#[derive(Debug, Clone)]
struct CodeCandidate {
    key: CandidateKey,
    start_line: usize,
    end_line: usize,
    symbol: Option<String>,
    kind: Option<String>,
    language: Option<String>,
    content_hash: String,
    visible_generation: u64,
    component_ranks: HashMap<String, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SymbolMatch {
    Exact,
    Prefix,
}

#[derive(Debug, Clone, Copy)]
struct SymbolRank {
    rank: usize,
    matched: SymbolMatch,
}

#[derive(Debug, Clone)]
struct SymbolCandidate {
    workspace_id: String,
    relative_path: String,
    start_line: usize,
    end_line: usize,
    symbol: String,
    kind: String,
    evidence: String,
    content_hash: String,
    visible_generation: u64,
}

#[derive(Clone)]
enum CachedCandidates {
    Code {
        candidates: Arc<Vec<CodeCandidate>>,
        truncated: bool,
        warnings: Vec<Warning>,
    },
    Symbol {
        candidates: Arc<Vec<SymbolCandidate>>,
        truncated: bool,
        warnings: Vec<Warning>,
    },
}

#[derive(Clone)]
struct CachedResultSet {
    query_digest: String,
    state_digest: String,
    expires_at: Instant,
    candidates: CachedCandidates,
}

#[derive(Default)]
pub(crate) struct SearchResultCache {
    sets: Mutex<HashMap<String, CachedResultSet>>,
}

impl SearchResultCache {
    fn insert(&self, result_set: CachedResultSet) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Instant::now();
        let mut sets = self.sets.lock();
        sets.retain(|_, entry| entry.expires_at > now);
        while sets.len() >= MAX_CACHED_RESULT_SETS {
            let oldest = sets.iter().min_by_key(|(_, entry)| entry.expires_at)
                .map(|(id, _)| id.clone());
            if let Some(oldest) = oldest { sets.remove(&oldest); } else { break; }
        }
        sets.insert(id.clone(), result_set);
        id
    }

    fn get(&self, id: &str) -> Option<CachedResultSet> {
        let mut sets = self.sets.lock();
        if sets.get(id).is_some_and(|entry| entry.expires_at <= Instant::now()) {
            sets.remove(id);
        }
        sets.get(id).cloned()
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct SemanticSnapshotKey {
    workspace_id: String,
    profile_id: String,
    generation: u64,
    checksum: String,
}

#[derive(Default)]
pub(crate) struct SemanticSnapshotCache {
    indexes: Mutex<HashMap<SemanticSnapshotKey, Arc<Mutex<VectorIndex>>>>,
}

impl SemanticSnapshotCache {
    fn get_or_load(
        &self,
        key: SemanticSnapshotKey,
        base_path: &Path,
        manifest: &crate::index::vector::CacheManifest,
    ) -> Option<Arc<Mutex<VectorIndex>>> {
        let mut indexes = self.indexes.lock();
        if let Some(index) = indexes.get(&key) {
            return Some(Arc::clone(index));
        }
        let loaded = VectorIndex::load(base_path, manifest).ok()?;
        if indexes.len() >= MAX_CACHED_SEMANTIC_SNAPSHOTS
            && let Some(oldest) = indexes.keys().next().cloned()
        {
            indexes.remove(&oldest);
        }
        let index = Arc::new(Mutex::new(loaded));
        indexes.insert(key, Arc::clone(&index));
        Some(index)
    }
}

pub(crate) struct SearchExecution<T> {
    pub result: ToolResult<Page<T>>,
    pub stale_paths: Vec<(String, String)>,
    pub generations: Vec<(String, u64)>,
    pub profile_ids_by_workspace: Vec<(String, Option<String>)>,
    transitions: Vec<(String, bool)>,
    registrations: Vec<WorkspaceRegistration>,
}

impl<T> SearchExecution<T> {
    pub(crate) fn still_current(&self, connection: &Connection) -> bool {
        if !self.result.is_success() {
            return true;
        }
        self.generations.iter().all(|(workspace_id, expected_generation)| {
            let expected_transition = self.transitions.iter().find(|(id, _)| id == workspace_id);
            let registration = self.registrations.iter()
                .find(|identity| &identity.workspace_id == workspace_id);
            let current = current_search_state(connection, workspace_id);
            current.is_some_and(|state| {
                state.generation == *expected_generation
                    && expected_transition.is_some_and(|(_, value)| state.branch_transition == *value)
                    && self.profile_ids_by_workspace.iter()
                        .find(|(id, _)| id == workspace_id)
                        .is_some_and(|(_, value)| state.profile_id.as_ref() == value.as_ref())
                    && registration.is_some_and(|expected| expected == &state.registration)
            })
        })
    }
}

struct CurrentSearchState {
    generation: u64,
    branch_transition: bool,
    profile_id: Option<String>,
    registration: WorkspaceRegistration,
}

fn current_search_state(connection: &Connection, workspace_id: &str) -> Option<CurrentSearchState> {
    connection.query_row(
        "SELECT s.visible_generation, s.branch_transition, s.active_profile_id,
                w.registration_epoch, w.root_path, w.device_id, w.inode
         FROM workspace_index_state s
         JOIN workspaces w ON w.id=s.workspace_id
         WHERE s.workspace_id=?1",
        [workspace_id],
        |row| Ok(CurrentSearchState {
            generation: row.get::<_, i64>(0)?.max(0) as u64,
            branch_transition: row.get(1)?,
            profile_id: row.get(2)?,
            registration: WorkspaceRegistration {
                workspace_id: workspace_id.to_owned(),
                epoch: row.get::<_, i64>(3)?.max(0) as u64,
                root_path: row.get(4)?,
                device_id: row.get(5)?,
                inode: row.get(6)?,
            },
        }),
    ).optional().ok().flatten()
}

pub(crate) struct CodeSearchSnapshot {
    failure: Option<ToolResult<Page<SearchResult>>>,
    query: String,
    result_set_id: Option<String>,
    workspace_ids: Vec<String>,
    generations: Vec<(String, u64)>,
    profile_ids_by_workspace: Vec<(String, Option<String>)>,
    transitions: Vec<(String, bool)>,
    registrations: Vec<WorkspaceRegistration>,
    limit: usize,
    offset: usize,
    candidates: Arc<Vec<CodeCandidate>>,
    candidate_set_truncated: bool,
    path_matcher: Option<ignore::overrides::Override>,
    warnings: Vec<Warning>,
    secret: Vec<u8>,
    max_file_bytes: u64,
    max_snippet_bytes: usize,
    max_tool_bytes: usize,
}

pub(crate) struct SymbolSearchSnapshot {
    failure: Option<ToolResult<Page<SymbolSearchResult>>>,
    query: String,
    result_set_id: Option<String>,
    direction: String,
    workspace_ids: Vec<String>,
    generations: Vec<(String, u64)>,
    profile_ids_by_workspace: Vec<(String, Option<String>)>,
    transitions: Vec<(String, bool)>,
    registrations: Vec<WorkspaceRegistration>,
    limit: usize,
    offset: usize,
    candidates: Arc<Vec<SymbolCandidate>>,
    candidate_set_truncated: bool,
    warnings: Vec<Warning>,
    secret: Vec<u8>,
    max_file_bytes: u64,
    max_snippet_bytes: usize,
    max_tool_bytes: usize,
}

fn capability_lines(
    capabilities: &HashMap<String, WorkspaceFs>,
    workspace_id: &str,
    relative_path: &str,
    start_line: usize,
    end_line: usize,
    expected_hash: &str,
    max_file_bytes: u64,
) -> Option<(String, String, String)> {
    if start_line == 0 || end_line < start_line {
        return None;
    }
    let bytes = capabilities.get(workspace_id)?.read(Path::new(relative_path), max_file_bytes).ok()?;
    let content_hash = blake3::hash(&bytes).to_hex().to_string();
    if content_hash != expected_hash {
        return None;
    }
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let source = String::from_utf8(bytes).ok()?;
    let snippet = source_lines(&source, start_line, end_line)?;
    Some((content_hash, sha256, snippet))
}

fn source_lines(source: &str, start_line: usize, end_line: usize) -> Option<String> {
    let mut selected = String::new();
    for (index, line) in source.split_inclusive('\n').enumerate() {
        let line_number = index + 1;
        if (start_line..=end_line).contains(&line_number) {
            selected.push_str(line);
        }
        if line_number >= end_line {
            break;
        }
    }
    (!selected.is_empty()).then_some(selected)
}

fn truncate_snippet(snippet: &mut String, max_bytes: usize) {
    if snippet.len() <= max_bytes { return; }
    let suffix = "\n...[truncated]";
    let end = if max_bytes >= suffix.len() { max_bytes - suffix.len() } else { max_bytes };
    let mut end = end.min(snippet.len());
    while !snippet.is_char_boundary(end) { end -= 1; }
    snippet.truncate(end);
    if max_bytes >= suffix.len() { snippet.push_str(suffix); }
}

fn search_cursor_context(
    query: &str,
    languages: &[String],
    paths: &[String],
    profile_id: Option<&str>,
    query_vector: Option<&[f32]>,
    limit: usize,
) -> String {
    let mut digest = Sha256::new();
    append_cursor_field(&mut digest, "code-search-v2");
    append_cursor_field(&mut digest, query);
    digest.update((languages.len() as u64).to_be_bytes());
    for language in languages { append_cursor_field(&mut digest, language); }
    digest.update((paths.len() as u64).to_be_bytes());
    for path in paths { append_cursor_field(&mut digest, path); }
    match profile_id {
        Some(profile_id) => {
            digest.update([1]);
            append_cursor_field(&mut digest, profile_id.trim());
        }
        None => digest.update([0]),
    }
    match query_vector {
        Some(vector) => {
            digest.update([1]);
            digest.update((vector.len() as u64).to_be_bytes());
            for value in vector { digest.update(value.to_bits().to_be_bytes()); }
        }
        None => digest.update([0]),
    }
    digest.update((limit as u64).to_be_bytes());
    format!("{:x}", digest.finalize())
}

fn symbol_cursor_context(query: &str, direction: &str, limit: usize) -> String {
    let mut digest = Sha256::new();
    append_cursor_field(&mut digest, "symbol-search-v2");
    append_cursor_field(&mut digest, query);
    append_cursor_field(&mut digest, direction);
    digest.update((limit as u64).to_be_bytes());
    format!("{:x}", digest.finalize())
}

fn normalize_symbol_direction(direction: Option<&str>) -> Result<&'static str, ()> {
    match direction.unwrap_or("both").trim() {
        value if value.eq_ignore_ascii_case("incoming") => Ok("incoming"),
        value if value.eq_ignore_ascii_case("outgoing") => Ok("outgoing"),
        value if value.eq_ignore_ascii_case("both") => Ok("both"),
        _ => Err(()),
    }
}

fn append_cursor_field(digest: &mut Sha256, field: &str) {
    digest.update((field.len() as u64).to_be_bytes());
    digest.update(field.as_bytes());
}

fn cursor_state_digest(
    workspace_ids: &[String],
    profile_ids_by_workspace: &[(String, Option<String>)],
    generations: &[(String, u64)],
    transitions: &[(String, bool)],
    registrations: &[WorkspaceRegistration],
) -> String {
    let mut digest = Sha256::new();
    append_cursor_field(&mut digest, "search-state-v5");
    digest.update((workspace_ids.len() as u64).to_be_bytes());
    for workspace_id in workspace_ids { append_cursor_field(&mut digest, workspace_id); }
    digest.update((profile_ids_by_workspace.len() as u64).to_be_bytes());
    for (workspace_id, profile_id) in profile_ids_by_workspace {
        append_cursor_field(&mut digest, workspace_id);
        match profile_id {
            Some(profile_id) => {
                digest.update([1]);
                append_cursor_field(&mut digest, profile_id);
            }
            None => digest.update([0]),
        }
    }
    digest.update((generations.len() as u64).to_be_bytes());
    for (workspace_id, generation) in generations {
        append_cursor_field(&mut digest, workspace_id);
        digest.update(generation.to_be_bytes());
    }
    digest.update((transitions.len() as u64).to_be_bytes());
    for (workspace_id, branch_transition) in transitions {
        append_cursor_field(&mut digest, workspace_id);
        digest.update([u8::from(*branch_transition)]);
    }
    digest.update((registrations.len() as u64).to_be_bytes());
    for registration in registrations {
        append_cursor_field(&mut digest, &registration.workspace_id);
        digest.update(registration.epoch.to_be_bytes());
        append_cursor_field(&mut digest, &registration.root_path);
        digest.update(registration.device_id.to_be_bytes());
        digest.update(registration.inode.to_be_bytes());
    }
    format!("{:x}", digest.finalize())
}
fn workspace_has_branch_transition(transitions: &[(String, bool)]) -> bool {
    transitions.iter().any(|(_, branch_transition)| *branch_transition)
}

fn add_reconciling_warnings(warnings: &mut Vec<Warning>) {
    warnings.push(Warning {
        code: "RECONCILING".into(),
        message: "Workspace branch transition is in progress; search results are withheld until stable publication.".into(),
        retryable: false,
    });
}

fn compile_path_matcher(paths: &[String]) -> Result<Option<ignore::overrides::Override>, String> {
    if paths.is_empty() { return Ok(None); }
    let mut builder = ignore::overrides::OverrideBuilder::new(".");
    for path in paths {
        if path.is_empty() || path.starts_with('!') {
            return Err("path filters must be nonempty positive globs".into());
        }
        builder.add(path).map_err(|_| "path filter is invalid")?;
    }
    let matcher = builder.build().map_err(|_| "path filter is invalid")?;
    if matcher.is_empty() { return Err("path filter is invalid".into()); }
    Ok(Some(matcher))
}

#[cfg(test)]
mod path_glob_tests {
    use super::compile_path_matcher;
    use std::path::Path;

    #[test]
    fn path_filter_globs_or_across_relative_paths() {
        let matcher = compile_path_matcher(&[
            "src/**/*.rs".to_owned(),
            "tests/**/*.rs".to_owned(),
        ]).expect("valid relative-path globs").unwrap();
        assert!(matcher.matched(Path::new("src/one.rs"), false).is_whitelist());
        assert!(matcher.matched(Path::new("src/nested/deep/one.rs"), false).is_whitelist());
        assert!(matcher.matched(Path::new("tests/one.rs"), false).is_whitelist());
        assert!(matcher.matched(Path::new("docs/one.rs"), false).is_ignore());
        assert!(matcher.matched(Path::new("src/one.py"), false).is_ignore());
    }
}

#[cfg(test)]
mod fts_query_plan_tests {
    use super::fts_rank_sql;
    use rusqlite::Connection;

    #[test]
    fn fts_match_drives_rank_query_before_chunk_lookups() {
        let mut connection = Connection::open_in_memory().expect("open in-memory database");
        crate::state::schema::initialize(&mut connection).expect("initialize production schema");

        let sql = format!("EXPLAIN QUERY PLAN {}", fts_rank_sql(10));
        let mut statement = connection.prepare(&sql).expect("prepare production query plan");
        let details = statement
            .query_map(
                rusqlite::params_from_iter(
                    std::iter::once("needle").chain(std::iter::repeat_n("ws", 10)),
                ),
                |row| row.get::<_, String>(3),
            )
            .expect("explain production rank query")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("read production query plan");

        assert!(
            details
                .first()
                .is_some_and(|detail| detail.contains("SCAN f VIRTUAL TABLE INDEX")),
            "expected FTS MATCH to drive the production rank query, got {details:?}"
        );
        assert!(
            details.iter().any(|detail| {
                detail.contains("SEARCH c USING INTEGER PRIMARY KEY (rowid=?)")
            }),
            "expected chunks to be looked up by FTS rowid, got {details:?}"
        );
        assert!(
            details.iter().all(|detail| !detail.contains("SCAN c")),
            "the production rank query must not scan all chunks: {details:?}"
        );
    }
}

impl<'a> SearchService<'a> {
    pub fn with_capabilities(
        conn: &'a Connection,
        secret: &'a [u8],
        options: SearchServiceOptions,
    ) -> Self {
        let SearchServiceOptions {
            capabilities,
            max_file_bytes,
            max_results,
            max_snippet_bytes,
            max_tool_bytes,
            vectors_dir,
        } = options;
        Self {
            conn, secret, capabilities, max_file_bytes, max_results,
            max_snippet_bytes, max_tool_bytes, vectors_dir,
            semantic_queries: HashMap::new(),
            result_sets: Arc::new(SearchResultCache::default()),
            semantic_snapshots: Arc::new(SemanticSnapshotCache::default()),
        }
    }

    pub(crate) fn with_semantic_queries(
        mut self,
        queries: SemanticQueryMap,
    ) -> Self {
        self.semantic_queries = queries;
        self
    }
    pub(crate) fn with_shared_caches(
        mut self,
        result_sets: Arc<SearchResultCache>,
        semantic_snapshots: Arc<SemanticSnapshotCache>,
    ) -> Self {
        self.result_sets = result_sets;
        self.semantic_snapshots = semantic_snapshots;
        self
    }

}
    
impl<'a> SearchService<'a> {
    pub(crate) fn capture_code_search(&self, req: CodeSearchRequest) -> CodeSearchSnapshot {
        let mut workspace_ids = req.workspace_ids.clone();
        workspace_ids.sort();
        workspace_ids.dedup();
        let mut languages = req.languages.clone();
        languages.sort();
        languages.dedup();
        let mut path_globs = req.path_globs.clone();
        path_globs.sort();
        path_globs.dedup();
        let profile_id = req.profile_id.as_deref().map(str::trim);
        let limit = req.limit.clamp(1, self.max_results.clamp(1, 100));
        let mut snapshot = CodeSearchSnapshot {
            failure: None,
            query: search_cursor_context(
                &req.query,
                &languages,
                &path_globs,
                profile_id,
                req.query_vector.as_deref(),
                limit,
            ),
            result_set_id: None,
            workspace_ids: workspace_ids.clone(),
            generations: Vec::new(),
            profile_ids_by_workspace: Vec::new(),
            transitions: Vec::new(),
            registrations: Vec::new(),
            offset: 0,
            limit,
            candidates: Arc::new(Vec::new()),
            candidate_set_truncated: false,
            path_matcher: None,
            warnings: Vec::new(),
            secret: self.secret.to_vec(),
            max_file_bytes: self.max_file_bytes,
            max_snippet_bytes: self.max_snippet_bytes,
            max_tool_bytes: self.max_tool_bytes,
        };
        snapshot.path_matcher = match compile_path_matcher(&path_globs) {
            Ok(matcher) => matcher,
            Err(message) => {
                snapshot.failure = Some(ToolResult::failure(ErrorCode::InvalidArgument, message, false));
                return snapshot;
            }
        };
        if workspace_ids.is_empty() {
            snapshot.failure = Some(ToolResult::failure(
                ErrorCode::WorkspaceRequired,
                "At least one workspace_id is required",
                false,
            ));
            return snapshot;
        }
        match load_search_context(self.conn, &workspace_ids) {
            Ok(context) => {
                snapshot.generations = context.generations;
                snapshot.profile_ids_by_workspace = context.profile_ids_by_workspace;
                snapshot.transitions = context.transitions;
                snapshot.registrations = context.registrations;
            }
            Err(error) => {
                snapshot.failure = Some(ToolResult::failure(error, "Workspace search state is unavailable", false));
                return snapshot;
            }
        }
        if workspace_has_branch_transition(&snapshot.transitions) {
            if req.cursor.is_some() {
                snapshot.failure = Some(ToolResult::failure(
                    ErrorCode::StaleCursor,
                    "Workspace branch transition invalidated the search cursor",
                    false,
                ));
            } else {
                add_reconciling_warnings(&mut snapshot.warnings);
            }
            return snapshot;
        }
        let state_digest = cursor_state_digest(
            &workspace_ids,
            &snapshot.profile_ids_by_workspace,
            &snapshot.generations,
            &snapshot.transitions,
            &snapshot.registrations,
        );
        if let Some(cursor) = &req.cursor {
            let cached = verify_cursor(&snapshot.secret, cursor).ok()
                .and_then(|payload| {
                    if payload.state_digest != state_digest {
                        return None;
                    }
                    let entry = self.result_sets.get(&payload.context_digest)?;
                    (entry.query_digest == snapshot.query && entry.state_digest == state_digest)
                        .then_some((payload, entry))
                });
            let Some((payload, entry)) = cached else {
                snapshot.failure = Some(ToolResult::failure(
                    ErrorCode::StaleCursor,
                    "Index state, query parameters, or cached result set changed",
                    false,
                ));
                return snapshot;
            };
            let CachedCandidates::Code { candidates, truncated, warnings } = entry.candidates else {
                snapshot.failure = Some(ToolResult::failure(
                    ErrorCode::StaleCursor,
                    "Cursor belongs to a different search tool",
                    false,
                ));
                return snapshot;
            };
            if payload.offset >= candidates.len() {
                snapshot.failure = Some(ToolResult::failure(
                    ErrorCode::StaleCursor,
                    "Search cursor offset is outside its frozen result set",
                    false,
                ));
                return snapshot;
            }
            snapshot.offset = payload.offset;
            snapshot.result_set_id = Some(payload.context_digest);
            snapshot.candidates = candidates;
            snapshot.candidate_set_truncated = truncated;
            snapshot.warnings = warnings;
            return snapshot;
        }

        let pull_limit = MAX_SEARCH_CANDIDATES + 1;
        let path_matcher = snapshot.path_matcher.as_ref();
        let rank_query = RankQuery {
            connection: self.conn,
            query: &req.query,
            workspace_ids: &workspace_ids,
            languages: &languages,
            path_matcher,
            pull_limit,
        };
        let mut fts_ranks = HashMap::new();
        let mut symbol_ranks = HashMap::new();
        let mut semantic_ranks = HashMap::new();
        if !req.query.is_empty() {
            capture_fts_ranks(&rank_query, &mut fts_ranks, &mut snapshot.warnings);
            capture_symbol_ranks(&rank_query, &mut symbol_ranks, &mut snapshot.warnings);
        }
        let fallback_queries = if self.semantic_queries.is_empty() {
            match (&req.query_vector, profile_id) {
                (Some(vector), Some(profile_id)) => workspace_ids.iter()
                    .filter(|workspace_id| snapshot.profile_ids_by_workspace.iter()
                        .any(|(id, active)| id == *workspace_id && active.as_deref() == Some(profile_id)))
                    .map(|workspace_id| (workspace_id.clone(), (profile_id.to_owned(), Arc::new(vector.clone()))))
                    .collect(),
                _ => HashMap::new(),
            }
        } else {
            HashMap::new()
        };
        let semantic_queries = if self.semantic_queries.is_empty() {
            &fallback_queries
        } else {
            &self.semantic_queries
        };
        for (workspace_id, profile_id) in &snapshot.profile_ids_by_workspace {
            let visible = snapshot.generations.iter()
                .find(|(id, _)| id == workspace_id)
                .map(|(_, generation)| *generation)
                .unwrap_or_default();
            if (profile_id.is_some() && !semantic_queries.contains_key(workspace_id))
                || (profile_id.is_none() && visible > 0)
            {
                snapshot.warnings.push(Warning {
                    code: "INDEX_DEGRADED".into(),
                    message: format!("Semantic index is unavailable for workspace {workspace_id}"),
                    retryable: false,
                });
            }
        }
        capture_semantic_ranks(
            SemanticRankQuery {
                connection: self.conn,
                queries: semantic_queries,
                workspace_ids: &workspace_ids,
                languages: &languages,
                path_matcher,
                pull_limit,
                vectors_dir: &self.vectors_dir,
                snapshots: &self.semantic_snapshots,
            },
            &mut semantic_ranks,
            &mut snapshot.warnings,
        );
        let channels_truncated = fts_ranks.len() > MAX_SEARCH_CANDIDATES
            || symbol_ranks.len() > MAX_SEARCH_CANDIDATES
            || semantic_ranks.len() > MAX_SEARCH_CANDIDATES;
        let candidates = capture_ranked_code_candidates(
            self.conn,
            &req.query,
            fts_ranks,
            symbol_ranks,
            semantic_ranks,
            &mut snapshot.warnings,
        );
        snapshot.candidate_set_truncated =
            channels_truncated || candidates.len() > MAX_SEARCH_CANDIDATES;
        snapshot.candidates = Arc::new(candidates);
        if !snapshot.candidates.is_empty() {
            snapshot.result_set_id = Some(self.result_sets.insert(CachedResultSet {
                query_digest: snapshot.query.clone(),
                state_digest,
                expires_at: Instant::now() + RESULT_SET_TTL,
                candidates: CachedCandidates::Code {
                    candidates: Arc::clone(&snapshot.candidates),
                    truncated: snapshot.candidate_set_truncated,
                    warnings: snapshot.warnings.clone(),
                },
            }));
        }
        snapshot
    }

    pub fn code_search(&self, req: CodeSearchRequest) -> ToolResult<Page<SearchResult>> {
        self.capture_code_search(req).execute(&self.capabilities).result
    }
}

impl CodeSearchSnapshot {
    pub(crate) fn execute(
        self,
        capabilities: &HashMap<String, WorkspaceFs>,
    ) -> SearchExecution<SearchResult> {
        let mut stale_paths = std::collections::BTreeSet::new();
        if let Some(failure) = self.failure {
            return SearchExecution {
                result: failure,
                stale_paths: Vec::new(),
                generations: self.generations,
                profile_ids_by_workspace: self.profile_ids_by_workspace,
                transitions: self.transitions,
                registrations: self.registrations,
            };
        }
        let mut results = Vec::new();
        let mut result_offsets = Vec::new();
        let mut current_idx = self.offset;
        let mut estimated_payload_bytes = 0;
        while results.len() < self.limit
            && current_idx < self.candidates.len()
            && estimated_payload_bytes < self.max_tool_bytes
        {
            let candidate = &self.candidates[current_idx];
            current_idx += 1;
            let Some((content_hash, sha256, mut snippet)) = capability_lines(
                capabilities,
                &candidate.key.workspace_id,
                &candidate.key.relative_path,
                candidate.start_line,
                candidate.end_line,
                &candidate.content_hash,
                self.max_file_bytes,
            ) else {
                stale_paths.insert((candidate.key.workspace_id.clone(), candidate.key.relative_path.clone()));
                continue;
            };
            truncate_snippet(&mut snippet, self.max_snippet_bytes);
            let result = SearchResult {
                workspace_id: candidate.key.workspace_id.clone(),
                relative_path: candidate.key.relative_path.clone(),
                start_line: candidate.start_line,
                end_line: candidate.end_line,
                symbol: candidate.symbol.clone(),
                kind: candidate.kind.clone(),
                language: candidate.language.clone(),
                snippet,
                content_hash,
                content_hash_algorithm: "blake3".into(),
                sha256,
                visible_generation: candidate.visible_generation,
                rank: results.len() + 1,
                component_ranks: candidate.component_ranks.clone(),
            };
            estimated_payload_bytes += result.snippet.len() + 500;
            results.push(result);
            result_offsets.push(current_idx);
        }
        let state_digest = cursor_state_digest(
            &self.workspace_ids,
            &self.profile_ids_by_workspace,
            &self.generations,
            &self.transitions,
            &self.registrations,
        );
        let mut truncated = current_idx < self.candidates.len() || self.candidate_set_truncated;
        let mut next_cursor = self.result_set_id.as_ref()
            .filter(|_| current_idx < self.candidates.len())
            .map(|result_set_id| generate_cursor(&self.secret, &CursorPayload {
                context_digest: result_set_id.clone(),
                state_digest: state_digest.clone(),
                offset: current_idx,
            }));
        let mut result = ToolResult::success(Page { items: results.clone(), truncated, next_cursor })
            .with_warnings(self.warnings.clone());
        if results.is_empty() {
            result = result.with_warning(Warning {
                code: "EMPTY_RESULT".into(),
                message: "No matching fresh results found.".into(),
                retryable: false,
            });
        }
        if !stale_paths.is_empty() {
            result = result.with_warning(Warning {
                code: "INDEX_DEGRADED".into(),
                message: "Stale search candidates were omitted and queued for refresh.".into(),
                retryable: true,
            });
        }
        while !results.is_empty() {
            let serialized = serde_json::to_string(&result).unwrap_or_default();
            if serialized.len() <= self.max_tool_bytes {
                break;
            }
            results.pop();
            current_idx = result_offsets.pop().unwrap_or(self.offset);
            truncated = current_idx < self.candidates.len() || self.candidate_set_truncated;
            next_cursor = self.result_set_id.as_ref()
                .filter(|_| current_idx < self.candidates.len())
                .map(|result_set_id| generate_cursor(&self.secret, &CursorPayload {
                    context_digest: result_set_id.clone(),
                    state_digest: state_digest.clone(),
                    offset: current_idx,
                }));
            result = ToolResult::success(Page { items: results.clone(), truncated, next_cursor })
                .with_warnings(self.warnings.clone());
            if results.is_empty() {
                result = result.with_warning(Warning {
                    code: "EMPTY_RESULT".into(),
                    message: "No matching fresh results found.".into(),
                    retryable: false,
                });
            }
            if !stale_paths.is_empty() {
                result = result.with_warning(Warning {
                    code: "INDEX_DEGRADED".into(),
                    message: "Stale search candidates were omitted and queued for refresh.".into(),
                    retryable: true,
                });
            }
        }

        SearchExecution {
            result,
            stale_paths: stale_paths.into_iter().collect(),
            generations: self.generations,
            profile_ids_by_workspace: self.profile_ids_by_workspace,
            transitions: self.transitions,
            registrations: self.registrations,
        }
    }
}

fn load_search_context(
    connection: &Connection,
    workspace_ids: &[String],
) -> Result<SearchContext, ErrorCode> {
    let mut context = SearchContext {
        generations: Vec::with_capacity(workspace_ids.len()),
        profile_ids_by_workspace: Vec::with_capacity(workspace_ids.len()),
        transitions: Vec::with_capacity(workspace_ids.len()),
        registrations: Vec::with_capacity(workspace_ids.len()),
    };
    for workspace_id in workspace_ids {
        let state = connection.query_row(
            "SELECT s.visible_generation, s.branch_transition, s.active_profile_id,
                    w.registration_epoch, w.root_path, w.device_id, w.inode
             FROM workspace_index_state s
             JOIN workspaces w ON w.id=s.workspace_id
             WHERE s.workspace_id=?1",
            [workspace_id],
            |row| Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, bool>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
            )),
        ).optional().map_err(|_| ErrorCode::Conflict)?;
        let Some((generation, branch_transition, profile_id, epoch, root_path, device_id, inode)) = state else {
            return Err(ErrorCode::WorkspaceNotFound);
        };
        context.generations.push((workspace_id.clone(), generation.max(0) as u64));
        context.profile_ids_by_workspace.push((workspace_id.clone(), profile_id));
        context.transitions.push((workspace_id.clone(), branch_transition));
        context.registrations.push(WorkspaceRegistration {
            workspace_id: workspace_id.clone(),
            epoch: epoch.max(0) as u64,
            root_path,
            device_id,
            inode,
        });
    }
    Ok(context)
}

fn candidate_matches_filters(
    relative_path: &str,
    language: Option<&str>,
    languages: &[String],
    path_matcher: Option<&ignore::overrides::Override>,
) -> bool {
    let language_matches = languages.is_empty()
        || language.is_some_and(|language| languages.iter().any(|filter| filter == language));
    let path_matches = path_matcher.is_none_or(|matcher| {
        matcher.matched(Path::new(relative_path), false).is_whitelist()
    });
    language_matches && path_matches
}

fn fts_rank_sql(workspace_count: usize) -> String {
    let placeholders = (0..workspace_count).map(|_| "?").collect::<Vec<_>>().join(",");
    format!(
        "SELECT c.workspace_id,c.relative_path,c.chunk_index,c.language
         FROM chunks_fts f
         CROSS JOIN chunks c
         CROSS JOIN files fi
         CROSS JOIN workspace_index_state s
         WHERE chunks_fts MATCH ? AND f.rowid=c.rowid
           AND c.workspace_id IN ({})
           AND fi.workspace_id=c.workspace_id AND fi.relative_path=c.relative_path
           AND s.workspace_id=c.workspace_id
           AND fi.visible_generation=s.visible_generation
         ORDER BY bm25(chunks_fts) ASC,c.workspace_id ASC,c.relative_path ASC,
                  c.start_line ASC,c.chunk_index ASC",
        placeholders,
    )
}

fn capture_fts_ranks(
    context: &RankQuery<'_>,
    ranks: &mut HashMap<CandidateKey, usize>,
    warnings: &mut Vec<Warning>,
) {
    let RankQuery { connection, query, workspace_ids, languages, path_matcher, pull_limit } = *context;
    let sql = fts_rank_sql(workspace_ids.len());
    let mut params: Vec<&dyn rusqlite::ToSql> = vec![&query];
    params.extend(workspace_ids.iter().map(|workspace_id| workspace_id as &dyn rusqlite::ToSql));
    let Ok(mut statement) = connection.prepare(&sql) else {
        warnings.push(sql_warning("FTS query could not be prepared"));
        return;
    };
    let rows = statement.query_map(rusqlite::params_from_iter(params), |row| {
        Ok((
            CandidateKey {
                workspace_id: row.get(0)?,
                relative_path: row.get(1)?,
                chunk_index: row.get::<_, i64>(2)?.max(0) as usize,
            },
            row.get::<_, Option<String>>(3)?,
        ))
    });
    match rows {
        Ok(rows) => for row in rows {
            match row {
                Ok((key, language))
                    if candidate_matches_filters(&key.relative_path, language.as_deref(), languages, path_matcher) =>
                {
                    let rank = ranks.len() + 1;
                    ranks.entry(key).or_insert(rank);
                    if ranks.len() >= pull_limit { break; }
                }
                Ok(_) => {}
                Err(_) => warnings.push(sql_warning("FTS result row could not be read")),
            }
        },
        Err(_) => warnings.push(sql_warning("FTS query failed")),
    }
}

fn capture_symbol_ranks(
    context: &RankQuery<'_>,
    ranks: &mut HashMap<CandidateKey, SymbolRank>,
    warnings: &mut Vec<Warning>,
) {
    let RankQuery { connection, query, workspace_ids, languages, path_matcher, pull_limit } = *context;
    let placeholders = workspace_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT s.workspace_id,s.relative_path,s.name,c.chunk_index,c.language
         FROM symbols s
         JOIN chunks c ON c.workspace_id=s.workspace_id AND c.relative_path=s.relative_path
           AND c.chunk_index=(
             SELECT c2.chunk_index FROM chunks c2
             WHERE c2.workspace_id=s.workspace_id AND c2.relative_path=s.relative_path
               AND c2.start_line<=s.start_line AND c2.end_line>=s.end_line
             ORDER BY c2.start_line ASC,c2.chunk_index ASC LIMIT 1
           )
         JOIN files fi ON s.workspace_id=fi.workspace_id AND s.relative_path=fi.relative_path
         JOIN workspace_index_state st ON st.workspace_id=s.workspace_id
         WHERE (s.name=? OR instr(s.name,?)=1) AND s.workspace_id IN ({})
           AND fi.visible_generation=st.visible_generation
         ORDER BY (s.name=?) DESC,s.workspace_id ASC,s.relative_path ASC,
                  c.start_line ASC,c.chunk_index ASC,s.end_line ASC,s.name ASC,s.symbol_index ASC",
        placeholders,
    );
    let mut params: Vec<&dyn rusqlite::ToSql> = vec![&query, &query];
    params.extend(workspace_ids.iter().map(|workspace_id| workspace_id as &dyn rusqlite::ToSql));
    params.push(&query);
    let Ok(mut statement) = connection.prepare(&sql) else {
        warnings.push(sql_warning("Symbol query could not be prepared"));
        return;
    };
    let rows = statement.query_map(rusqlite::params_from_iter(params), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)?.max(0) as usize,
            row.get::<_, Option<String>>(4)?,
        ))
    });
    let Ok(rows) = rows else {
        warnings.push(sql_warning("Symbol query failed"));
        return;
    };
    for row in rows {
        let Ok((workspace_id, relative_path, name, chunk_index, language)) = row else {
            warnings.push(sql_warning("Symbol result row could not be read"));
            continue;
        };
        if !candidate_matches_filters(&relative_path, language.as_deref(), languages, path_matcher) {
            continue;
        }
        let key = CandidateKey { workspace_id, relative_path, chunk_index };
        let matched = if name == query { SymbolMatch::Exact } else { SymbolMatch::Prefix };
        let rank = ranks.len() + 1;
        ranks.entry(key).or_insert(SymbolRank { rank, matched });
        if ranks.len() >= pull_limit { break; }
    }
}


fn capture_semantic_ranks(
    context: SemanticRankQuery<'_>,
    ranks: &mut HashMap<CandidateKey, usize>,
    warnings: &mut Vec<Warning>,
) -> bool {
    let SemanticRankQuery {
        connection,
        queries,
        workspace_ids,
        languages,
        path_matcher,
        pull_limit,
        vectors_dir,
        snapshots,
    } = context;
    let mut workspace_matches = Vec::new();
    for workspace_id in workspace_ids {
        let Some((profile_id, query_vector)) = queries.get(workspace_id) else { continue };
        if query_vector.iter().any(|value| !value.is_finite()) {
            warnings.push(sql_warning("Query embedding is invalid"));
            continue;
        }
        let context = connection.query_row(
            "SELECT s.active_profile_id, g.applied_generation, s.visible_generation,
                    g.backfill_required, p.dimension, p.metric
             FROM workspace_index_state s
             LEFT JOIN profile_applied_generations g
               ON g.workspace_id=s.workspace_id AND g.profile_id=?1
             LEFT JOIN embedding_profiles p ON p.id=?1
             WHERE s.workspace_id=?2",
            rusqlite::params![profile_id, workspace_id],
            |row| Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<bool>>(3)?,
                row.get::<_, Option<i64>>(4)?,
                row.get::<_, Option<String>>(5)?,
            )),
        );
        let Ok((Some(active_profile), Some(applied), visible, Some(false), Some(dimension), Some(metric))) = context else {
            warnings.push(semantic_degraded(workspace_id));
            continue;
        };
        let visible = visible.max(0) as u64;
        if active_profile != *profile_id
            || applied.max(0) as u64 != visible
            || visible == 0
            || usize::try_from(dimension).ok() != Some(query_vector.len())
        {
            warnings.push(semantic_degraded(workspace_id));
            continue;
        }
        let base_path = vectors_dir.join(format!("{}_{}", workspace_id, profile_id));
        let manifest: crate::index::vector::CacheManifest =
            match std::fs::read_to_string(base_path.with_extension("json"))
                .ok().and_then(|json| serde_json::from_str(&json).ok())
            {
                Some(manifest) => manifest,
                None => {
                    warnings.push(semantic_degraded(workspace_id));
                    continue;
                }
            };
        if manifest.workspace_id != *workspace_id
            || manifest.profile_id != *profile_id
            || manifest.dimension != query_vector.len()
            || manifest.metric != metric
            || manifest.generation != visible
        {
            warnings.push(semantic_degraded(workspace_id));
            continue;
        }
        let key = SemanticSnapshotKey {
            workspace_id: workspace_id.clone(),
            profile_id: profile_id.clone(),
            generation: visible,
            checksum: manifest.checksum.clone(),
        };
        let Some(index) = snapshots.get_or_load(key, &base_path, &manifest) else {
            warnings.push(semantic_degraded(workspace_id));
            continue;
        };
        let index_size = index.lock().size();
        let mut requested = pull_limit.min(index_size);
        let mut candidates = Vec::new();
        loop {
            let matches = match index.lock().search(query_vector, requested) {
                Ok(matches) => matches,
                Err(_) => {
                    warnings.push(semantic_degraded(workspace_id));
                    candidates.clear();
                    break;
                }
            };
            let Some(filtered) = map_semantic_matches(
                connection, workspace_id, profile_id, visible, &matches,
                languages, path_matcher,
            ) else {
                warnings.push(semantic_degraded(workspace_id));
                candidates.clear();
                break;
            };
            candidates = filtered;
            if candidates.len() >= pull_limit || requested >= index_size {
                break;
            }
            requested = requested.saturating_mul(2).max(requested + 1).min(index_size);
        }
        candidates.sort_by(|left, right| {
            right.1.total_cmp(&left.1)
                .then_with(|| left.0.relative_path.cmp(&right.0.relative_path))
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.0.chunk_index.cmp(&right.0.chunk_index))
        });
        workspace_matches.push(candidates);
    }
    let max_rank = workspace_matches.iter().map(Vec::len).max().unwrap_or_default();
    let mut all_results = Vec::new();
    for rank in 0..max_rank {
        let mut tied = workspace_matches.iter()
            .filter_map(|matches| matches.get(rank).cloned())
            .collect::<Vec<_>>();
        tied.sort_by(|left, right| {
            left.0.workspace_id.cmp(&right.0.workspace_id)
                .then_with(|| left.0.relative_path.cmp(&right.0.relative_path))
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.0.chunk_index.cmp(&right.0.chunk_index))
        });
        all_results.extend(tied);
    }
    for (index, (key, _, _)) in all_results.into_iter().take(pull_limit).enumerate() {
        ranks.insert(key, index + 1);
    }
    true
}

fn semantic_degraded(workspace_id: &str) -> Warning {
    Warning {
        code: "INDEX_DEGRADED".into(),
        message: format!("Semantic index is not current for workspace {workspace_id}"),
        retryable: false,
    }
}

fn map_semantic_matches(
    connection: &Connection,
    workspace_id: &str,
    profile_id: &str,
    visible_generation: u64,
    matches: &[(u64, f32)],
    languages: &[String],
    path_matcher: Option<&ignore::overrides::Override>,
) -> Option<Vec<(CandidateKey, f32, usize)>> {
    if matches.is_empty() {
        return Some(Vec::new());
    }
    // Invalid HNSW IDs make the mapping ambiguous; do not present surviving IDs as healthy.
    if matches
        .iter()
        .any(|(numeric_id, _)| !matches!(i64::try_from(*numeric_id), Ok(id) if id > 0))
    {
        return None;
    }

    let visible_generation = i64::try_from(visible_generation).ok()?;
    let id_placeholders = (1..=SEMANTIC_MAPPING_BATCH_SIZE)
        .map(|slot| format!("?{slot}"))
        .collect::<Vec<_>>()
        .join(",");
    let workspace_parameter = SEMANTIC_MAPPING_BATCH_SIZE + 1;
    let profile_parameter = SEMANTIC_MAPPING_BATCH_SIZE + 2;
    let generation_parameter = SEMANTIC_MAPPING_BATCH_SIZE + 3;
    let sql = format!(
        "SELECT e.numeric_vector_id,e.relative_path,e.chunk_index,c.start_line,c.language
         FROM chunk_embeddings e
         JOIN chunks c ON c.workspace_id=e.workspace_id
           AND c.relative_path=e.relative_path AND c.chunk_index=e.chunk_index
         JOIN files fi ON e.workspace_id=fi.workspace_id AND e.relative_path=fi.relative_path
         WHERE e.numeric_vector_id IN ({id_placeholders})
           AND e.workspace_id=?{workspace_parameter}
           AND e.profile_id=?{profile_parameter}
           AND e.generation=fi.file_generation
           AND fi.visible_generation=?{generation_parameter}"
    );
    let mut statement = connection.prepare(&sql).ok()?;
    let mut candidates = Vec::with_capacity(matches.len());
    let mut emitted_ids = HashSet::with_capacity(matches.len());

    for batch in matches.chunks(SEMANTIC_MAPPING_BATCH_SIZE) {
        let mut seen_ids = HashSet::with_capacity(batch.len());
        let ids = batch
            .iter()
            .filter_map(|(numeric_id, _)| i64::try_from(*numeric_id).ok())
            .filter(|numeric_id| seen_ids.insert(*numeric_id))
            .collect::<Vec<_>>();
        let mut parameters = ids
            .into_iter()
            .map(rusqlite::types::Value::Integer)
            .collect::<Vec<_>>();
        parameters.resize(SEMANTIC_MAPPING_BATCH_SIZE, rusqlite::types::Value::Null);
        parameters.extend([
            rusqlite::types::Value::Text(workspace_id.to_owned()),
            rusqlite::types::Value::Text(profile_id.to_owned()),
            rusqlite::types::Value::Integer(visible_generation),
        ]);

        let rows = statement
            .query_map(rusqlite::params_from_iter(parameters), |row| {
                let numeric_id = row.get::<_, i64>(0)?;
                let relative_path = row.get::<_, String>(1)?;
                let chunk_index = usize::try_from(row.get::<_, i64>(2)?)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;
                let start_line = usize::try_from(row.get::<_, i64>(3)?)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;
                let language = row.get::<_, Option<String>>(4)?;
                if numeric_id <= 0
                    || !is_normalized_semantic_path(&relative_path)
                    || start_line == 0
                {
                    return Err(rusqlite::Error::InvalidQuery);
                }
                Ok((numeric_id, relative_path, chunk_index, start_line, language))
            })
            .ok()?;
        let mut mappings = HashMap::with_capacity(batch.len());
        for row in rows {
            let (numeric_id, relative_path, chunk_index, start_line, language) = row.ok()?;
            if mappings
                .insert(
                    numeric_id,
                    (relative_path, chunk_index, start_line, language),
                )
                .is_some()
            {
                return None;
            }
        }

        for (numeric_id, distance) in batch {
            let numeric_id = i64::try_from(*numeric_id).ok()?;
            if !emitted_ids.insert(numeric_id) {
                continue;
            }
            // Missing joined rows are ordinary stale/non-visible vectors and are intentionally omitted.
            let Some((relative_path, chunk_index, start_line, language)) =
                mappings.get(&numeric_id)
            else {
                continue;
            };
            if !candidate_matches_filters(
                relative_path,
                language.as_deref(),
                languages,
                path_matcher,
            ) {
                continue;
            }
            candidates.push((
                CandidateKey {
                    workspace_id: workspace_id.to_owned(),
                    relative_path: relative_path.clone(),
                    chunk_index: *chunk_index,
                },
                1.0 / (1.0 + distance),
                *start_line,
            ));
        }
    }
    Some(candidates)
}

fn is_normalized_semantic_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.ends_with('/')
        && !path.contains('\\')
        && !path.contains('\0')
        && path
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

fn reciprocal_rank_contribution(rank: usize) -> f32 {
    1.0 / (RRF_K + rank as f32)
}

fn query_has_matching_path_token(query: &str, path: &str) -> bool {
    query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .any(|query_token| {
            path.split(|character: char| !character.is_alphanumeric())
                .any(|path_token| path_token.eq_ignore_ascii_case(query_token))
        })
}

fn capture_ranked_code_candidates(
    connection: &Connection,
    query: &str,
    fts_ranks: HashMap<CandidateKey, usize>,
    symbol_ranks: HashMap<CandidateKey, SymbolRank>,
    semantic_ranks: HashMap<CandidateKey, usize>,
    warnings: &mut Vec<Warning>,
) -> Vec<CodeCandidate> {
    let mut keys = HashSet::new();
    keys.extend(fts_ranks.keys().cloned());
    keys.extend(symbol_ranks.keys().cloned());
    keys.extend(semantic_ranks.keys().cloned());

    let mut ranked = Vec::with_capacity(keys.len());
    for key in keys {
        let mut score = 0.0;
        let mut component_ranks = HashMap::new();
        if let Some(rank) = fts_ranks.get(&key) {
            score += reciprocal_rank_contribution(*rank);
            component_ranks.insert("fts".to_owned(), *rank);
        }
        let symbol_match = symbol_ranks.get(&key).copied();
        if let Some(symbol_rank) = symbol_match {
            score += reciprocal_rank_contribution(symbol_rank.rank);
            component_ranks.insert("symbol".to_owned(), symbol_rank.rank);
        }
        if let Some(rank) = semantic_ranks.get(&key) {
            score += reciprocal_rank_contribution(*rank);
            component_ranks.insert("semantic".to_owned(), *rank);
        }

        let candidate = match load_code_candidate(connection, &key, query) {
            Ok(Some(candidate)) => candidate,
            Ok(None) => continue,
            Err(_) => {
                warnings.push(sql_warning("Ranked search candidate could not be read"));
                continue;
            }
        };
        if !query.is_empty() {
            match symbol_match.map(|matched| matched.matched) {
                Some(SymbolMatch::Exact) => score *= EXACT_SYMBOL_BOOST,
                Some(SymbolMatch::Prefix) => score *= PREFIX_DEFINITION_BOOST,
                None => {}
            }
            if symbol_match.is_some() { score *= DEFINITION_BOOST; }
            if query_has_matching_path_token(query, &key.relative_path) {
                score *= PATH_TOKEN_BOOST;
            }
            if candidate.one_hop_dependency {
                score *= ONE_HOP_DEPENDENCY_BOOST;
            }
        }
        ranked.push((CodeCandidate {
            key,
            start_line: candidate.start_line,
            end_line: candidate.end_line,
            symbol: candidate.symbol,
            kind: candidate.kind,
            language: candidate.language,
            content_hash: candidate.content_hash,
            visible_generation: candidate.visible_generation,
            component_ranks,
        }, score));
    }
    ranked.sort_by(|(left, left_score), (right, right_score)| {
        right_score.total_cmp(left_score)
            .then_with(|| left.key.workspace_id.cmp(&right.key.workspace_id))
            .then_with(|| left.key.relative_path.cmp(&right.key.relative_path))
            .then_with(|| left.start_line.cmp(&right.start_line))
            .then_with(|| left.key.chunk_index.cmp(&right.key.chunk_index))
    });
    ranked.into_iter().map(|(candidate, _)| candidate).collect()
}

fn load_code_candidate(
    connection: &Connection,
    key: &CandidateKey,
    query: &str,
) -> rusqlite::Result<Option<HydratedCodeCandidate>> {
    connection.query_row(
        "SELECT c.start_line,c.end_line,c.language,f.content_hash,f.visible_generation,
                (SELECT s.name FROM symbols s WHERE s.workspace_id=c.workspace_id AND s.relative_path=c.relative_path
                 AND s.start_line>=c.start_line AND s.end_line<=c.end_line
                 ORDER BY s.start_line ASC,s.symbol_index ASC LIMIT 1),
                (SELECT s.kind FROM symbols s WHERE s.workspace_id=c.workspace_id AND s.relative_path=c.relative_path
                 AND s.start_line>=c.start_line AND s.end_line<=c.end_line
                 ORDER BY s.start_line ASC,s.symbol_index ASC LIMIT 1),
                EXISTS (
                  SELECT 1 FROM dependency_edges d
                  WHERE d.workspace_id=c.workspace_id AND d.relative_path=c.relative_path
                    AND d.start_line<=c.end_line AND d.end_line>=c.start_line
                    AND (
                      d.name=?4 OR d.source_symbol=?4
                      OR (d.source_symbol IS NULL AND EXISTS (
                        SELECT 1 FROM symbols src
                        WHERE src.workspace_id=d.workspace_id AND src.relative_path=d.relative_path
                          AND src.name=?4 AND src.start_line<=d.start_line AND src.end_line>=d.start_line
                      ))
                    )
                )
         FROM chunks c
         JOIN files f ON c.workspace_id=f.workspace_id AND c.relative_path=f.relative_path
         JOIN workspace_index_state st ON st.workspace_id=c.workspace_id
         WHERE c.workspace_id=?1 AND c.relative_path=?2 AND c.chunk_index=?3
           AND f.visible_generation=st.visible_generation",
        rusqlite::params![key.workspace_id, key.relative_path, key.chunk_index as i64, query],
        |row| {
            Ok(HydratedCodeCandidate {
                start_line: row.get::<_, i64>(0)?.max(0) as usize,
                end_line: row.get::<_, i64>(1)?.max(0) as usize,
                language: row.get(2)?,
                content_hash: row.get(3)?,
                visible_generation: row.get::<_, i64>(4)?.max(0) as u64,
                symbol: row.get(5)?,
                kind: row.get(6)?,
                one_hop_dependency: row.get(7)?,
            })
        },
    )
    .optional()
}

struct HydratedCodeCandidate {
    start_line: usize,
    end_line: usize,
    language: Option<String>,
    content_hash: String,
    visible_generation: u64,
    symbol: Option<String>,
    kind: Option<String>,
    one_hop_dependency: bool,
}

fn sql_warning(message: &str) -> Warning {
    Warning { code: "SQL_ERROR".into(), message: message.into(), retryable: false }
}

#[derive(Clone)]
pub struct SymbolSearchRequest {
    pub query: String,
    pub workspace_ids: Vec<String>,
    pub direction: Option<String>,
    pub limit: usize,
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SymbolSearchResult {
    pub workspace_id: String,
    pub relative_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub symbol: String,
    pub kind: String,
    pub evidence: String,
    pub snippet: String,
    pub content_hash: String,
    pub content_hash_algorithm: String,
    pub sha256: String,
    pub visible_generation: u64,
}

impl<'a> SearchService<'a> {
    pub(crate) fn capture_symbol_search(&self, req: SymbolSearchRequest) -> SymbolSearchSnapshot {
        let mut workspace_ids = req.workspace_ids.clone();
        workspace_ids.sort();
        workspace_ids.dedup();
        let direction_result = normalize_symbol_direction(req.direction.as_deref());
        let direction = direction_result.unwrap_or("both").to_string();
        let limit = req.limit.clamp(1, self.max_results.clamp(1, 100));
        let mut snapshot = SymbolSearchSnapshot {
            failure: None,
            query: symbol_cursor_context(&req.query, &direction, limit),
            result_set_id: None,
            direction,
            workspace_ids: workspace_ids.clone(),
            generations: Vec::new(),
            profile_ids_by_workspace: Vec::new(),
            transitions: Vec::new(),
            registrations: Vec::new(),
            offset: 0,
            limit,
            candidates: Arc::new(Vec::new()),
            candidate_set_truncated: false,
            warnings: Vec::new(),
            secret: self.secret.to_vec(),
            max_file_bytes: self.max_file_bytes,
            max_snippet_bytes: self.max_snippet_bytes,
            max_tool_bytes: self.max_tool_bytes,
        };
        if direction_result.is_err() {
            snapshot.failure = Some(ToolResult::failure(
                ErrorCode::InvalidArgument,
                "direction must be incoming, outgoing, or both",
                false,
            ));
            return snapshot;
        }
        if workspace_ids.is_empty() {
            snapshot.failure = Some(ToolResult::failure(
                ErrorCode::WorkspaceRequired,
                "At least one workspace_id is required",
                false,
            ));
            return snapshot;
        }
        match load_search_context(self.conn, &workspace_ids) {
            Ok(context) => {
                snapshot.generations = context.generations;
                snapshot.profile_ids_by_workspace = context.profile_ids_by_workspace;
                snapshot.transitions = context.transitions;
                snapshot.registrations = context.registrations;
            }
            Err(error) => {
                snapshot.failure = Some(ToolResult::failure(error, "Workspace search state is unavailable", false));
                return snapshot;
            }
        }
        if workspace_has_branch_transition(&snapshot.transitions) {
            if req.cursor.is_some() {
                snapshot.failure = Some(ToolResult::failure(
                    ErrorCode::StaleCursor,
                    "Workspace branch transition invalidated the search cursor",
                    false,
                ));
            } else {
                add_reconciling_warnings(&mut snapshot.warnings);
            }
            return snapshot;
        }
        let state_digest = cursor_state_digest(
            &workspace_ids,
            &snapshot.profile_ids_by_workspace,
            &snapshot.generations,
            &snapshot.transitions,
            &snapshot.registrations,
        );
        if let Some(cursor) = &req.cursor {
            let cached = verify_cursor(&snapshot.secret, cursor).ok()
                .and_then(|payload| {
                    if payload.state_digest != state_digest {
                        return None;
                    }
                    let entry = self.result_sets.get(&payload.context_digest)?;
                    (entry.query_digest == snapshot.query && entry.state_digest == state_digest)
                        .then_some((payload, entry))
                });
            let Some((payload, entry)) = cached else {
                snapshot.failure = Some(ToolResult::failure(
                    ErrorCode::StaleCursor,
                    "Index state, query parameters, or cached result set changed",
                    false,
                ));
                return snapshot;
            };
            let CachedCandidates::Symbol { candidates, truncated, warnings } = entry.candidates else {
                snapshot.failure = Some(ToolResult::failure(
                    ErrorCode::StaleCursor,
                    "Cursor belongs to a different search tool",
                    false,
                ));
                return snapshot;
            };
            if payload.offset >= candidates.len() {
                snapshot.failure = Some(ToolResult::failure(
                    ErrorCode::StaleCursor,
                    "Search cursor offset is outside its frozen result set",
                    false,
                ));
                return snapshot;
            }
            snapshot.offset = payload.offset;
            snapshot.result_set_id = Some(payload.context_digest);
            snapshot.candidates = candidates;
            snapshot.candidate_set_truncated = truncated;
            snapshot.warnings = warnings;
            return snapshot;
        }
        let (candidates, truncated) = capture_symbol_candidates(
            self.conn,
            &req.query,
            &snapshot.direction,
            &workspace_ids,
            MAX_SEARCH_CANDIDATES + 1,
            &mut snapshot.warnings,
        );
        snapshot.candidate_set_truncated = truncated;
        snapshot.candidates = Arc::new(candidates);
        if !snapshot.candidates.is_empty() {
            snapshot.result_set_id = Some(self.result_sets.insert(CachedResultSet {
                query_digest: snapshot.query.clone(),
                state_digest,
                expires_at: Instant::now() + RESULT_SET_TTL,
                candidates: CachedCandidates::Symbol {
                    candidates: Arc::clone(&snapshot.candidates),
                    truncated,
                    warnings: snapshot.warnings.clone(),
                },
            }));
        }
        snapshot
    }

    pub fn symbol_search(&self, req: SymbolSearchRequest) -> ToolResult<Page<SymbolSearchResult>> {
        self.capture_symbol_search(req).execute(&self.capabilities).result
    }
}

impl SymbolSearchSnapshot {
    pub(crate) fn execute(
        self,
        capabilities: &HashMap<String, WorkspaceFs>,
    ) -> SearchExecution<SymbolSearchResult> {
        let mut stale_paths = std::collections::BTreeSet::new();
        if let Some(failure) = self.failure {
            return SearchExecution {
                result: failure,
                stale_paths: Vec::new(),
                generations: self.generations,
                profile_ids_by_workspace: self.profile_ids_by_workspace,
                transitions: self.transitions,
                registrations: self.registrations,
            };
        }
        let mut results = Vec::new();
        let mut result_offsets = Vec::new();
        let mut current_idx = self.offset;
        let mut estimated_payload_bytes = 0;
        while results.len() < self.limit
            && current_idx < self.candidates.len()
            && estimated_payload_bytes < self.max_tool_bytes
        {
            let candidate = &self.candidates[current_idx];
            current_idx += 1;
            let Some((content_hash, sha256, mut snippet)) = capability_lines(
                capabilities,
                &candidate.workspace_id,
                &candidate.relative_path,
                candidate.start_line,
                candidate.end_line,
                &candidate.content_hash,
                self.max_file_bytes,
            ) else {
                stale_paths.insert((candidate.workspace_id.clone(), candidate.relative_path.clone()));
                continue;
            };
            truncate_snippet(&mut snippet, self.max_snippet_bytes);
            let result = SymbolSearchResult {
                workspace_id: candidate.workspace_id.clone(),
                relative_path: candidate.relative_path.clone(),
                start_line: candidate.start_line,
                end_line: candidate.end_line,
                symbol: candidate.symbol.clone(),
                kind: candidate.kind.clone(),
                evidence: candidate.evidence.clone(),
                snippet,
                content_hash,
                content_hash_algorithm: "blake3".into(),
                sha256,
                visible_generation: candidate.visible_generation,
            };
            estimated_payload_bytes += result.snippet.len() + 500;
            results.push(result);
            result_offsets.push(current_idx);
        }
        let state_digest = cursor_state_digest(
            &self.workspace_ids,
            &self.profile_ids_by_workspace,
            &self.generations,
            &self.transitions,
            &self.registrations,
        );
        let mut truncated = current_idx < self.candidates.len() || self.candidate_set_truncated;
        let mut next_cursor = self.result_set_id.as_ref()
            .filter(|_| current_idx < self.candidates.len())
            .map(|result_set_id| generate_cursor(&self.secret, &CursorPayload {
                context_digest: result_set_id.clone(),
                state_digest: state_digest.clone(),
                offset: current_idx,
            }));
        let mut result = ToolResult::success(Page { items: results.clone(), truncated, next_cursor })
            .with_warnings(self.warnings.clone());
        if results.is_empty() {
            result = result.with_warning(Warning {
                code: "EMPTY_RESULT".into(),
                message: "No matching fresh results found.".into(),
                retryable: false,
            });
        }
        if !stale_paths.is_empty() {
            result = result.with_warning(Warning {
                code: "INDEX_DEGRADED".into(),
                message: "Stale search candidates were omitted and queued for refresh.".into(),
                retryable: true,
            });
        }
        while !results.is_empty() {
            let serialized = serde_json::to_string(&result).unwrap_or_default();
            if serialized.len() <= self.max_tool_bytes {
                break;
            }
            results.pop();
            current_idx = result_offsets.pop().unwrap_or(self.offset);
            truncated = current_idx < self.candidates.len() || self.candidate_set_truncated;
            next_cursor = self.result_set_id.as_ref()
                .filter(|_| current_idx < self.candidates.len())
                .map(|result_set_id| generate_cursor(&self.secret, &CursorPayload {
                    context_digest: result_set_id.clone(),
                    state_digest: state_digest.clone(),
                    offset: current_idx,
                }));
            result = ToolResult::success(Page { items: results.clone(), truncated, next_cursor })
                .with_warnings(self.warnings.clone());
            if results.is_empty() {
                result = result.with_warning(Warning {
                    code: "EMPTY_RESULT".into(),
                    message: "No matching fresh results found.".into(),
                    retryable: false,
                });
            }
            if !stale_paths.is_empty() {
                result = result.with_warning(Warning {
                    code: "INDEX_DEGRADED".into(),
                    message: "Stale search candidates were omitted and queued for refresh.".into(),
                    retryable: true,
                });
            }
        }
        SearchExecution {
            result,
            stale_paths: stale_paths.into_iter().collect(),
            generations: self.generations,
            profile_ids_by_workspace: self.profile_ids_by_workspace,
            transitions: self.transitions,
            registrations: self.registrations,
        }
    }
}

fn capture_symbol_candidates(
    connection: &Connection,
    query: &str,
    direction: &str,
    workspace_ids: &[String],
    pull_limit: usize,
    warnings: &mut Vec<Warning>,
) -> (Vec<SymbolCandidate>, bool) {
    let placeholders = workspace_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let include_definitions = direction != "incoming";
    let dependency_match = if direction == "outgoing" {
        "(d.source_symbol=? OR (d.source_symbol IS NULL AND EXISTS (
            SELECT 1 FROM symbols src
            WHERE src.workspace_id=d.workspace_id AND src.relative_path=d.relative_path
              AND src.name=? AND src.start_line<=d.start_line AND src.end_line>=d.start_line
        )))"
    } else {
        "(d.name=? OR instr(d.name,?)=1)"
    };
    let definition_query = format!(
        "SELECT s.workspace_id,s.relative_path,s.start_line,s.end_line,s.name,s.kind,'definition' AS evidence
         FROM symbols s
         JOIN files f1 ON s.workspace_id=f1.workspace_id AND s.relative_path=f1.relative_path
         JOIN workspace_index_state w1 ON w1.workspace_id=s.workspace_id
         WHERE (s.name=? OR instr(s.name,?)=1) AND s.workspace_id IN ({})
           AND f1.visible_generation=w1.visible_generation",
        placeholders
    );
    let dependency_query = format!(
        "SELECT d.workspace_id,d.relative_path,d.start_line,d.end_line,d.name,'reference' AS kind,d.evidence
         FROM dependency_edges d
         JOIN files f2 ON d.workspace_id=f2.workspace_id AND d.relative_path=f2.relative_path
         JOIN workspace_index_state w2 ON w2.workspace_id=d.workspace_id
         WHERE {dependency_match} AND d.workspace_id IN ({})
           AND f2.visible_generation=w2.visible_generation",
        placeholders
    );
    let sql = match include_definitions {
        true => format!("{definition_query} UNION ALL {dependency_query}"),
        false => dependency_query,
    };
    let sql = format!(
        "SELECT r.workspace_id,r.relative_path,r.start_line,r.end_line,r.name,r.kind,r.evidence,
                COALESCE((SELECT c.chunk_index FROM chunks c
                          WHERE c.workspace_id=r.workspace_id AND c.relative_path=r.relative_path
                            AND c.start_line<=r.start_line AND c.end_line>=r.end_line
                          ORDER BY c.start_line ASC,c.chunk_index ASC LIMIT 1),-1) AS chunk_index
         FROM ({sql}) AS r
         ORDER BY (r.name=?) DESC,r.workspace_id ASC,r.relative_path ASC,r.start_line ASC,
                  chunk_index ASC,r.end_line ASC,r.name ASC,r.kind ASC,r.evidence ASC"
    );
    let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
    if include_definitions {
        params.extend([&query as &dyn rusqlite::ToSql, &query]);
        params.extend(workspace_ids.iter().map(|id| id as &dyn rusqlite::ToSql));
    }
    if direction == "outgoing" {
        params.push(&query);
        params.push(&query);
    } else {
        params.extend([&query as &dyn rusqlite::ToSql, &query]);
    }
    params.extend(workspace_ids.iter().map(|id| id as &dyn rusqlite::ToSql));
    params.push(&query);
    let Ok(mut statement) = connection.prepare(&sql) else {
        warnings.push(sql_warning("Symbol query could not be prepared"));
        return (Vec::new(), false);
    };
    let rows = statement.query_map(rusqlite::params_from_iter(params), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?.max(0) as usize,
            row.get::<_, i64>(3)?.max(0) as usize,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, i64>(7)?,
        ))
    });
    let Ok(rows) = rows else {
        warnings.push(sql_warning("Symbol query failed"));
        return (Vec::new(), false);
    };
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    for row in rows {
        let Ok((workspace_id, relative_path, start_line, end_line, symbol, kind, evidence, _chunk_index)) = row else {
            warnings.push(sql_warning("Symbol result row could not be read"));
            continue;
        };
        if !seen.insert((
            workspace_id.clone(), relative_path.clone(), start_line, end_line,
            symbol.clone(), kind.clone(), evidence.clone(),
        )) {
            continue;
        }
        let metadata = connection.query_row(
            "SELECT f.content_hash,f.visible_generation
             FROM files f JOIN workspace_index_state st ON st.workspace_id=f.workspace_id
             WHERE f.workspace_id=?1 AND f.relative_path=?2
               AND f.visible_generation=st.visible_generation
               AND EXISTS (SELECT 1 FROM chunks c WHERE c.workspace_id=f.workspace_id
                 AND c.relative_path=f.relative_path AND c.start_line<=?3 AND c.end_line>=?4)
             LIMIT 1",
            rusqlite::params![workspace_id, relative_path, start_line as i64, end_line as i64],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?.max(0) as u64)),
        );
        match metadata {
            Ok((content_hash, visible_generation)) => candidates.push(SymbolCandidate {
                workspace_id,
                relative_path,
                start_line,
                end_line,
                symbol,
                kind,
                evidence,
                content_hash,
                visible_generation,
            }),
            Err(rusqlite::Error::QueryReturnedNoRows) => {}
            Err(_) => warnings.push(sql_warning("Symbol candidate metadata could not be read")),
        }
        if candidates.len() >= pull_limit { break; }
    }
    let truncated = candidates.len() > MAX_SEARCH_CANDIDATES;
    candidates.truncate(MAX_SEARCH_CANDIDATES);
    (candidates, truncated)
}
