use rusqlite::{params, Connection};
use std::process::Command;
use std::path::Path;
use tempfile::TempDir;
use workspace_index_mcp::model::WorkspaceMode;
use workspace_index_mcp::state::{
    schema, OwnerLock, RegistryOptions, StateError, WorkspaceRegistry,
};

fn init_git_repo(path: &Path) {
    Command::new("git")
        .arg("init")
        .arg(path)
        .output()
        .expect("Failed to init git repo");
}
const V3_SCHEMA_FIXTURE: &str = r#"
CREATE TABLE workspaces (
    id TEXT PRIMARY KEY, root_path TEXT NOT NULL, device_id INTEGER NOT NULL,
    inode INTEGER NOT NULL, mode TEXT NOT NULL, registration_epoch INTEGER NOT NULL,
    mode_epoch INTEGER NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE TABLE embedding_profiles (
    id TEXT PRIMARY KEY, provider_kind TEXT NOT NULL, endpoint TEXT, model TEXT NOT NULL,
    revision TEXT, dimension INTEGER NOT NULL, metric TEXT NOT NULL, state TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE files (
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL, content_hash TEXT NOT NULL, size_bytes INTEGER NOT NULL,
    visible_generation INTEGER NOT NULL, file_generation INTEGER NOT NULL,
    parser_version INTEGER NOT NULL, chunker_version INTEGER NOT NULL,
    PRIMARY KEY (workspace_id, relative_path)
);
CREATE TABLE workspace_index_state (
    workspace_id TEXT PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE chunks (
    workspace_id TEXT NOT NULL, relative_path TEXT NOT NULL, chunk_index INTEGER NOT NULL,
    start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, content TEXT NOT NULL,
    symbol_context TEXT, language TEXT, chunk_hash TEXT,
    FOREIGN KEY (workspace_id, relative_path)
        REFERENCES files(workspace_id, relative_path) ON DELETE CASCADE,
    PRIMARY KEY (workspace_id, relative_path, chunk_index)
);
CREATE TABLE chunk_embeddings (
    workspace_id TEXT NOT NULL, relative_path TEXT NOT NULL, chunk_index INTEGER NOT NULL,
    profile_id TEXT NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL, vector_id TEXT NOT NULL, embedding BLOB NOT NULL,
    PRIMARY KEY (workspace_id, relative_path, chunk_index, profile_id),
    FOREIGN KEY (workspace_id, relative_path, chunk_index)
        REFERENCES chunks(workspace_id, relative_path, chunk_index) ON DELETE CASCADE
);
CREATE TABLE vector_ops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL, op_type TEXT NOT NULL, vector_id TEXT NOT NULL, embedding BLOB
);
CREATE TABLE index_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 1,
    state TEXT NOT NULL DEFAULT 'queued', created_at INTEGER NOT NULL,
    UNIQUE (workspace_id, relative_path, state)
);
INSERT INTO workspaces VALUES ('legacy', '/repo', 1, 2, 'read_only', 1, 1, 1, 1);
INSERT INTO embedding_profiles VALUES
    ('legacy-profile', 'fastembed', NULL, 'bge', 'rev-1', 3, 'cosine', 'active', 1);
INSERT INTO workspace_index_state VALUES ('legacy', 8, 1);
INSERT INTO files VALUES ('legacy', 'src/lib.rs', 'hash', 10, 7, 7, 1, 1);
INSERT INTO chunks VALUES ('legacy', 'src/lib.rs', 0, 1, 1, 'fn main() {}', NULL, 'rust', 'chunk');
INSERT INTO chunk_embeddings VALUES
    ('legacy', 'src/lib.rs', 0, 'legacy-profile', 7, 'legacy-vector', X'00000000');
INSERT INTO vector_ops (workspace_id, profile_id, generation, op_type, vector_id, embedding)
    VALUES ('legacy', 'legacy-profile', 7, 'add', 'legacy-vector', X'00000000');
INSERT INTO index_jobs (workspace_id, relative_path, priority, state, created_at)
    VALUES ('legacy', 'src/lib.rs', 0, 'queued', 1);
PRAGMA user_version = 3;
"#;

fn create_v3_schema(conn: &Connection) {
    conn.execute_batch(V3_SCHEMA_FIXTURE).unwrap();
}
fn assert_schema_settings(conn: &Connection) {
    let journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0)).unwrap();
    assert_eq!(journal_mode.to_uppercase(), "WAL");
    let foreign_keys: i64 = conn.query_row("PRAGMA foreign_keys", [], |row| row.get(0)).unwrap();
    let synchronous: i64 = conn.query_row("PRAGMA synchronous", [], |row| row.get(0)).unwrap();
    assert_eq!(foreign_keys, 1);
    assert_eq!(synchronous, 2);
}

fn assert_v3_index_rows_preserved(conn: &Connection) {
    let legacy_rows: (String, String) = conn.query_row(
        "SELECT f.relative_path, c.content FROM files f
         JOIN chunks c ON c.workspace_id = f.workspace_id AND c.relative_path = f.relative_path
         WHERE f.workspace_id = 'legacy'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(legacy_rows, ("src/lib.rs".into(), "fn main() {}".into()));
}


fn assert_v3_profile_backfill(conn: &Connection) {
    let watermark: (i64, i64) = conn.query_row(
        "SELECT applied_generation, backfill_required
         FROM profile_applied_generations
         WHERE workspace_id='legacy' AND profile_id='legacy-profile'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(watermark, (0, 1));
}

fn assert_v3_numeric_vector_ids(conn: &Connection) {
    let ids: (Option<i64>, Option<i64>) = conn.query_row(
        "SELECT (SELECT numeric_vector_id FROM chunk_embeddings),
                (SELECT numeric_vector_id FROM vector_ops)",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert!(ids.0.is_some_and(|id| id > 0));
    assert_eq!(ids.0, ids.1);
}

fn assert_v3_jobs_remain_durable(conn: &Connection) {
    let queued_path: String = conn.query_row(
        "SELECT relative_path FROM index_jobs WHERE state='queued'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(queued_path, "src/lib.rs");
    assert!(conn.execute(
        "UPDATE workspace_index_state SET visible_generation=6 WHERE workspace_id='legacy'",
        [],
    ).is_err());
}


#[test]
fn test_schema_initialization_and_reopen() {
    let temp = TempDir::new().unwrap();
    let mut conn = Connection::open(temp.path().join("db.sqlite")).unwrap();
    schema::initialize(&mut conn).unwrap();
    let count: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='workspaces'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 1);
    schema::initialize(&mut conn).unwrap();
    assert_schema_settings(&conn);
    let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0)).unwrap();
    assert_eq!(version, schema::CURRENT_SCHEMA_VERSION);
}

#[test]
fn v6_upgrade_defaults_branch_transition_without_clearing_startup_dirty_state() {
    let mut conn = Connection::open_in_memory().expect("in-memory v6 database");
    conn.execute_batch(
        "CREATE TABLE chunk_embeddings (
            workspace_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            numeric_vector_id INTEGER NOT NULL,
            generation INTEGER NOT NULL
        );
        CREATE TABLE workspace_index_state (
            workspace_id TEXT PRIMARY KEY,
            generation INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL DEFAULT 0,
            building_generation INTEGER NOT NULL DEFAULT 0,
            visible_generation INTEGER NOT NULL DEFAULT 0,
            dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0, 1)),
            health TEXT NOT NULL DEFAULT 'unknown',
            desired_profile_id TEXT,
            active_profile_id TEXT
        );
        INSERT INTO workspace_index_state
            (workspace_id, generation, building_generation, visible_generation, dirty, health)
        VALUES ('startup', 8, 8, 7, 1, 'reconcile_required');
        PRAGMA user_version = 6;",
    )
    .expect("v6 fixture");

    schema::initialize(&mut conn).expect("v7 migration");

    let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("schema version");
    let (branch_transition, dirty, visible_generation): (bool, bool, i64) = conn.query_row(
        "SELECT branch_transition, dirty, visible_generation
         FROM workspace_index_state WHERE workspace_id='startup'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).expect("migrated workspace state");
    assert_eq!(version, schema::CURRENT_SCHEMA_VERSION);
    assert!(!branch_transition);
    assert!(dirty);
    assert_eq!(visible_generation, 7);
}

#[test]
fn test_v3_migration_preserves_snapshot_jobs_and_assigns_stable_vector_ids() {
    let temp = TempDir::new().unwrap();
    let mut conn = Connection::open(temp.path().join("db.sqlite")).unwrap();
    create_v3_schema(&conn);
    schema::initialize(&mut conn).unwrap();

    let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0)).unwrap();
    assert_eq!(version, schema::CURRENT_SCHEMA_VERSION);
    assert_v3_index_rows_preserved(&conn);
    assert_v3_profile_backfill(&conn);
    assert_v3_numeric_vector_ids(&conn);
    assert_v3_jobs_remain_durable(&conn);
    let epoch: (i64, i64) = conn.query_row(
        "SELECT registration_epoch, mode_epoch FROM workspace_registration_epochs WHERE id='legacy'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(epoch, (1, 1));

    schema::initialize(&mut conn).unwrap();
    let stable_ids: i64 = conn.query_row("SELECT count(*) FROM stable_vector_ids", [], |row| row.get(0)).unwrap();
    assert_eq!(stable_ids, 1);
}

#[test]
fn test_owner_lock_contention() {
    let temp = TempDir::new().unwrap();
    let data_dir = temp.path().join("data");
    
    let lock1 = OwnerLock::acquire(&data_dir).unwrap();
    
    // Second acquisition should fail with Busy
    let err = OwnerLock::acquire(&data_dir).unwrap_err();
    assert!(matches!(err, StateError::Busy));
    
    // Drop lock1, should succeed now
    drop(lock1);
    let _lock2 = OwnerLock::acquire(&data_dir).unwrap();
}

#[test]
fn test_registry_invalid_id() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db.sqlite");
    let mut registry = WorkspaceRegistry::open(&db_path, RegistryOptions::default()).unwrap();
    
    let root = temp.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    init_git_repo(&root);
    
    // Invalid IDs
    assert!(matches!(
        registry.register("Invalid", &root).unwrap_err(),
        StateError::InvalidWorkspaceId(_)
    ));
    assert!(matches!(
        registry.register("a space", &root).unwrap_err(),
        StateError::InvalidWorkspaceId(_)
    ));
    assert!(matches!(
        registry.register("-start-dash", &root).unwrap_err(),
        StateError::InvalidWorkspaceId(_)
    ));
}

#[test]
fn test_registry_git_worktree_requirement() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db.sqlite");
    
    let root = temp.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    
    // Default options (requires git)
    let mut registry = WorkspaceRegistry::open(&db_path, RegistryOptions::default()).unwrap();
    assert!(matches!(
        registry.register("test-repo", &root).unwrap_err(),
        StateError::NotGitWorktree(_)
    ));
    
    // With allow_non_git_roots
    let mut registry_allow = WorkspaceRegistry::open(&db_path, RegistryOptions { allow_non_git_roots: true }).unwrap();
    let record = registry_allow.register("test-repo", &root).unwrap();
    assert_eq!(record.id, "test-repo");
}

#[test]
fn existing_non_git_registration_conflicts_when_git_identity_appears() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db.sqlite");
    let root = temp.path().join("repo");
    std::fs::create_dir(&root).unwrap();
    let mut registry = WorkspaceRegistry::open(
        &db_path,
        RegistryOptions { allow_non_git_roots: true },
    ).unwrap();
    let original = registry.register("repo", &root).unwrap();
    assert_eq!(original.git_dir, None);

    init_git_repo(&root);

    assert!(matches!(
        registry.register("repo", &root).unwrap_err(),
        StateError::WorkspaceOverlap(_)
    ));
    assert!(matches!(
        registry.set_mode("repo", WorkspaceMode::ReadOnly).unwrap_err(),
        StateError::WorkspaceOverlap(_)
    ));
    assert_eq!(registry.list().unwrap(), vec![original]);
}

#[test]
fn test_registry_idempotence_and_write_mode_fails_closed() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db.sqlite");
    let mut registry = WorkspaceRegistry::open(&db_path, RegistryOptions::default()).unwrap();

    let root = temp.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    init_git_repo(&root);

    let first = registry.register("test-repo", &root).unwrap();
    assert_eq!(first.mode, WorkspaceMode::ReadOnly);
    assert_eq!(first.registration_epoch, 1);
    assert_eq!(first.mode_epoch, 1);
    assert_eq!(registry.register("test-repo", &root).unwrap(), first);

    assert!(matches!(
        registry.register_with_mode("test-repo", &root, WorkspaceMode::ReadWrite).unwrap_err(),
        StateError::WorkspaceOverlap(_)
    ));
    assert!(matches!(
        registry.set_mode("test-repo", WorkspaceMode::ReadWrite).unwrap_err(),
        StateError::Unsupported
    ));

    let read = registry.set_mode("test-repo", WorkspaceMode::ReadOnly).unwrap();
    assert_eq!(read, first);
}

#[test]
fn test_registry_overlap_rejection() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db.sqlite");
    let mut registry = WorkspaceRegistry::open(&db_path, RegistryOptions::default()).unwrap();
    
    let parent = temp.path().join("parent");
    let child = parent.join("child");
    
    std::fs::create_dir_all(&parent).unwrap();
    std::fs::create_dir_all(&child).unwrap();
    init_git_repo(&parent);
    init_git_repo(&child);
    
    // Register parent
    registry.register("parent", &parent).unwrap();
    
    // Register child (descendant overlap)
    assert!(matches!(
        registry.register("child", &child).unwrap_err(),
        StateError::WorkspaceOverlap(_)
    ));
    
    // Register parent again with different ID (exact overlap)
    assert!(matches!(
        registry.register("parent-dup", &parent).unwrap_err(),
        StateError::WorkspaceOverlap(_)
    ));
    
    // Unregister parent, then register child
    registry.unregister("parent").unwrap();
    registry.register("child", &child).unwrap();
    
    // Register parent (ancestor overlap)
    assert!(matches!(
        registry.register("parent", &parent).unwrap_err(),
        StateError::WorkspaceOverlap(_)
    ));
}

#[test]
fn test_registry_identity_replacement_requires_unregister() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db.sqlite");
    let mut registry = WorkspaceRegistry::open(&db_path, RegistryOptions::default()).unwrap();

    let root1 = temp.path().join("repo1");
    let root2 = temp.path().join("repo2");
    std::fs::create_dir_all(&root1).unwrap();
    std::fs::create_dir_all(&root2).unwrap();
    init_git_repo(&root1);
    init_git_repo(&root2);

    let first = registry.register("test-repo", &root1).unwrap();
    assert_eq!(first.registration_epoch, 1);
    assert!(matches!(
        registry.register("test-repo", &root2).unwrap_err(),
        StateError::WorkspaceOverlap(_)
    ));

    let still_registered = registry.list().unwrap();
    assert_eq!(still_registered.len(), 1);
    assert_eq!(still_registered[0], first);

    registry.unregister("test-repo").unwrap();
    let re_registered = registry.register("test-repo", &root2).unwrap();
    assert_eq!(re_registered.registration_epoch, 2);
    assert_eq!(re_registered.root_path, root2.canonicalize().unwrap().to_string_lossy());
}

#[test]
fn test_registration_epoch_tombstone_survives_database_reopen() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db.sqlite");
    let root = temp.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    init_git_repo(&root);

    let mut registry = WorkspaceRegistry::open(&db_path, RegistryOptions::default()).unwrap();
    let initial = registry.register("test-repo", &root).unwrap();
    assert_eq!(initial.registration_epoch, 1);
    registry.unregister("test-repo").unwrap();
    drop(registry);

    let mut reopened = WorkspaceRegistry::open(&db_path, RegistryOptions::default()).unwrap();
    let record = reopened.register("test-repo", &root).unwrap();
    assert_eq!(record.registration_epoch, 2);
    assert_eq!(record.mode_epoch, 1);
}

#[test]
fn test_registry_identifies_linked_git_worktrees() {
    let temp = TempDir::new().unwrap();
    let main = temp.path().join("main");
    let linked = temp.path().join("linked");
    std::fs::create_dir_all(&main).unwrap();
    assert!(Command::new("git").args(["init", "--initial-branch=main"]).arg(&main).status().unwrap().success());
    std::fs::write(main.join("tracked.txt"), "commit").unwrap();
    assert!(Command::new("git").args(["-C"]).arg(&main).args(["add", "tracked.txt"]).status().unwrap().success());
    assert!(Command::new("git").args(["-C"]).arg(&main)
        .args(["-c", "user.name=Workspace Test", "-c", "user.email=test@example.invalid", "commit", "-m", "initial"])
        .status().unwrap().success());
    assert!(Command::new("git").args(["-C"]).arg(&main)
        .args(["worktree", "add", "-b", "linked-branch"]).arg(&linked).arg("HEAD")
        .status().unwrap().success());
    assert!(std::fs::symlink_metadata(linked.join(".git")).unwrap().is_file());

    let db_path = temp.path().join("db.sqlite");
    let mut registry = WorkspaceRegistry::open(&db_path, RegistryOptions::default()).unwrap();
    let main_record = registry.register("main", &main).unwrap();
    let linked_record = registry.register("linked", &linked).unwrap();

    assert_ne!(main_record.git_dir, linked_record.git_dir);
    assert_eq!(main_record.git_common_dir, linked_record.git_common_dir);
    assert_eq!(main_record.git_head_commit, linked_record.git_head_commit);
    assert_eq!(main_record.git_head_ref.as_deref(), Some("refs/heads/main"));
    assert_eq!(linked_record.git_head_ref.as_deref(), Some("refs/heads/linked-branch"));
}
#[test]
fn v7_upgrade_adds_archive_and_payload_cleanup_without_workspace_foreign_keys() {
    let mut conn = Connection::open_in_memory().expect("v7 database");
    conn.execute_batch(
        "PRAGMA foreign_keys=ON;
         CREATE TABLE chunk_embeddings (
            workspace_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            numeric_vector_id INTEGER NOT NULL,
            generation INTEGER NOT NULL
         );
         CREATE TABLE workspaces (id TEXT PRIMARY KEY);
         CREATE TABLE pending_changesets (
            operation_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            digest TEXT NOT NULL, state TEXT NOT NULL, expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL
         );
         CREATE TABLE changeset_files (
            operation_id TEXT NOT NULL REFERENCES pending_changesets(operation_id) ON DELETE CASCADE,
            file_index INTEGER NOT NULL, op_kind TEXT NOT NULL, relative_path TEXT NOT NULL,
            destination_path TEXT, base_sha256 TEXT, proposed_sha256 TEXT, observed_sha256 TEXT,
            effect_state TEXT NOT NULL, intent_at INTEGER, observed_at INTEGER,
            PRIMARY KEY (operation_id, file_index)
         );
         CREATE TABLE audit_events (
            id INTEGER PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            operation_id TEXT, event_type TEXT NOT NULL, outcome TEXT NOT NULL, created_at INTEGER NOT NULL
         );
         INSERT INTO workspaces VALUES ('legacy');
         INSERT INTO pending_changesets VALUES ('op-1', 'legacy', 'digest', 'Prepared', 99, 10);
         INSERT INTO changeset_files VALUES ('op-1', 0, 'replace', 'src/lib.rs', NULL, 'base', 'new', NULL, 'pending', NULL, NULL);
         INSERT INTO audit_events VALUES (1, 'legacy', 'op-1', 'changeset_prepare', 'prepared', 10);
         PRAGMA user_version=7;",
    ).expect("v7 fixture");

    schema::initialize(&mut conn).expect("migration to current schema");

    let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("schema version");
    let pending: (String, String) = conn.query_row(
        "SELECT operation_id, state FROM pending_changesets WHERE operation_id='op-1'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).expect("pre-migration changeset remains");
    let archive_foreign_keys: i64 = conn.query_row(
        "SELECT count(*) FROM pragma_foreign_key_list('unregistered_changesets')",
        [],
        |row| row.get(0),
    ).expect("archive foreign keys");
    let cleanup_foreign_keys: i64 = conn.query_row(
        "SELECT count(*) FROM pragma_foreign_key_list('changeset_payload_cleanup')",
        [],
        |row| row.get(0),
    ).expect("cleanup foreign keys");
    let archive_triggers: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master
         WHERE type='trigger' AND name IN ('unregistered_changesets_no_update','unregistered_changesets_no_delete')",
        [],
        |row| row.get(0),
    ).expect("append-only archive guards");
    let recovery_target_column: i64 = conn.query_row(
        "SELECT count(*) FROM pragma_table_info('changeset_payload_cleanup')
         WHERE name='recovery_target_id' AND \"notnull\"=0",
        [],
        |row| row.get(0),
    ).expect("nullable recovery target");
    assert_eq!(version, schema::CURRENT_SCHEMA_VERSION);
    assert_eq!(pending, ("op-1".into(), "Prepared".into()));
    assert_eq!(archive_foreign_keys, 0);
    assert_eq!(cleanup_foreign_keys, 0);
    assert_eq!(archive_triggers, 2);
    assert_eq!(recovery_target_column, 1);
}

#[test]
fn v8_upgrade_adds_nullable_recovery_target_id() {
    let mut conn = Connection::open_in_memory().expect("v8 database");
    conn.execute_batch(
        "CREATE TABLE chunk_embeddings (
            workspace_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            numeric_vector_id INTEGER NOT NULL,
            generation INTEGER NOT NULL
         );
         CREATE TABLE changeset_payload_cleanup (
            operation_id TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            queued_at INTEGER NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT
         );
         INSERT INTO changeset_payload_cleanup VALUES ('malformed', 'terminal', 10, 2, 'prior_error');
         PRAGMA user_version=8;",
    ).expect("v8 fixture");

    schema::initialize(&mut conn).expect("v9 migration");

    let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0)).unwrap();
    let (target, attempts, error): (Option<String>, i64, String) = conn.query_row(
        "SELECT recovery_target_id, attempts, last_error
         FROM changeset_payload_cleanup WHERE operation_id='malformed'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).expect("cleanup row preserved");
    assert_eq!(version, schema::CURRENT_SCHEMA_VERSION);
    assert_eq!(target, None);
    assert_eq!(attempts, 2);
    assert_eq!(error, "prior_error");
}

#[test]
fn v9_upgrade_indexes_semantic_numeric_vector_lookups() {
    let mut conn = Connection::open_in_memory().expect("v9 database");
    schema::initialize(&mut conn).expect("create current schema for v9 fixture");
    conn.execute(
        "INSERT INTO workspaces (
            id, root_path, device_id, inode, mode, registration_epoch, mode_epoch,
            created_at, updated_at
         ) VALUES ('workspace', '/repo', 1, 2, 'read_only', 1, 1, 1, 1)",
        [],
    ).expect("workspace fixture");
    conn.execute(
        "INSERT INTO embedding_profiles (
            id, provider_kind, endpoint, model, revision, dimension, metric, state, created_at
         ) VALUES ('profile', 'fastembed', NULL, 'test-model', 'test-revision', 3, 'cosine', 'active', 1)",
        [],
    ).expect("profile fixture");
    conn.execute(
        "INSERT INTO files (
            workspace_id, relative_path, content_hash, size_bytes, visible_generation,
            file_generation, parser_version, chunker_version
         ) VALUES ('workspace', 'src/lib.rs', 'file-hash', 12, 7, 7, 1, 1)",
        [],
    ).expect("file fixture");
    conn.execute(
        "INSERT INTO chunks (
            workspace_id, relative_path, chunk_index, start_line, end_line, content,
            symbol_context, language, chunk_hash
         ) VALUES ('workspace', 'src/lib.rs', 0, 3, 4, 'fn answer() {}', NULL, 'rust', 'chunk-hash')",
        [],
    ).expect("chunk fixture");
    conn.execute(
        "INSERT INTO chunk_embeddings (
            workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding
         ) VALUES ('workspace', 'src/lib.rs', 0, 'profile', 7, 'vector-stable', X'00000000')",
        [],
    ).expect("embedding fixture");

    conn.execute_batch(
        "DROP INDEX idx_chunk_embeddings_semantic_lookup;
         PRAGMA user_version=9;",
    ).expect("model v9 by removing only the v10 index");
    schema::initialize(&mut conn).expect("v10 migration");

    let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0)).unwrap();
    assert_eq!(version, 10);
    let preserved: (String, String, String, String, Option<i64>) = conn.query_row(
        "SELECT e.relative_path, c.content, fi.content_hash, e.vector_id, e.numeric_vector_id
         FROM chunk_embeddings e
         JOIN chunks c ON c.workspace_id=e.workspace_id
           AND c.relative_path=e.relative_path AND c.chunk_index=e.chunk_index
         JOIN files fi ON e.workspace_id=fi.workspace_id AND e.relative_path=fi.relative_path
         WHERE e.workspace_id='workspace' AND e.profile_id='profile'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
    ).expect("seeded embedding and source rows preserved");
    assert_eq!(
        preserved,
        (
            "src/lib.rs".into(),
            "fn answer() {}".into(),
            "file-hash".into(),
            "vector-stable".into(),
            Some(1),
        )
    );
    let index_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master
         WHERE type='index' AND name='idx_chunk_embeddings_semantic_lookup'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(index_count, 1);

    let plan = conn.prepare(
        "EXPLAIN QUERY PLAN
         SELECT e.relative_path,e.chunk_index,c.start_line,c.language
         FROM chunk_embeddings e
         JOIN chunks c ON c.workspace_id=e.workspace_id
           AND c.relative_path=e.relative_path AND c.chunk_index=e.chunk_index
         JOIN files fi ON e.workspace_id=fi.workspace_id AND e.relative_path=fi.relative_path
         WHERE e.workspace_id=?1 AND e.profile_id=?2 AND e.numeric_vector_id=?3
           AND e.generation=fi.file_generation AND fi.visible_generation=?4",
    ).unwrap().query_map(params!["workspace", "profile", 1_i64, 7_i64], |row| row.get::<_, String>(3))
        .unwrap().collect::<rusqlite::Result<Vec<_>>>()
        .unwrap().join("\n");
    assert!(
        plan.contains("idx_chunk_embeddings_semantic_lookup"),
        "production semantic lookup should use its index:\n{plan}"
    );
}
