use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;
use rusqlite::Connection;

use workspace_index_mcp::index::{
    build_scanner, is_binary, line_chunker, IndexPipeline, IndexWorkspaceRequest,
    UnavailableEmbeddingProvider, EmbeddingProvider,
};
use workspace_index_mcp::index::scanner::{read_scanned_file, ExclusionReason, ScanEntry};
use workspace_index_mcp::security::WorkspaceFs;
use workspace_index_mcp::state::schema;

fn insert_registered_workspace(connection: &Connection, id: &str, root: &Path) {
    let metadata = fs::metadata(root).unwrap();
    #[cfg(unix)]
    let (device, inode) = {
        use std::os::unix::fs::MetadataExt;
        (metadata.dev(), metadata.ino())
    };
    #[cfg(not(unix))]
    let (device, inode) = (0, 0);
    connection.execute(
        "INSERT INTO workspaces (id, root_path, device_id, inode, mode, registration_epoch, mode_epoch, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, 'ReadOnly', 1, 1, 0, 0)",
        rusqlite::params![id, root.to_string_lossy(), device as i64, inode as i64],
    ).unwrap();
}

fn registered_workspace(connection: &Connection, id: &str) -> WorkspaceFs {
    let identity = connection.query_row(
        "SELECT root_path, device_id, inode FROM workspaces WHERE id=?1",
        [id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
    ).unwrap();
    WorkspaceFs::open_registered(Path::new(&identity.0), identity.1 as u64, identity.2 as u64).unwrap()
}

#[test]
fn test_scanner_exclusions() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("workspace");
    let data = root.join(".workspace-index-data");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(&data).unwrap();
    File::create(root.join(".git/config")).unwrap();
    File::create(root.join("normal.rs")).unwrap();
    File::create(root.join(".env")).unwrap();
    File::create(root.join("secret.pem")).unwrap();
    File::create(root.join("id_rsa")).unwrap();
    File::create(root.join("credentials.rs")).unwrap();
    File::create(data.join("indexed.rs")).unwrap();
    File::create(root.join("unsupported.txt")).unwrap();

    let mut large = File::create(root.join("large.rs")).unwrap();
    large.write_all(&vec![b'a'; 1024 * 1024 + 10]).unwrap();
    let mut ignore = File::create(root.join(".workspace-indexignore")).unwrap();
    writeln!(ignore, "ignored.rs").unwrap();
    File::create(root.join("ignored.rs")).unwrap();

    let scanner = build_scanner(&root, &data).unwrap();
    let capability = WorkspaceFs::open(&root).unwrap();
    let mut found = Vec::new();
    let mut excluded = Vec::new();
    for result in scanner {
        let entry = result.unwrap();
        match read_scanned_file(&capability, &root, &entry, 1024 * 1024).unwrap() {
            ScanEntry::Included(file) => {
                found.push(file.relative_path.to_str().unwrap().to_string());
            }
            ScanEntry::Excluded { relative_path, reason } => {
                excluded.push((relative_path.to_str().unwrap().to_string(), reason));
            }
        }
    }
    found.sort();

    assert!(found.contains(&"normal.rs".to_string()));
    assert!(!found.iter().any(|path| path.starts_with(".git/")));
    assert!(!found.contains(&"ignored.rs".to_string()));
    assert!(!found.iter().any(|path| path.starts_with(".workspace-index-data/")));
    assert!(excluded.contains(&(".env".to_string(), ExclusionReason::Sensitive)));
    assert!(excluded.contains(&("secret.pem".to_string(), ExclusionReason::Sensitive)));
    assert!(excluded.contains(&("id_rsa".to_string(), ExclusionReason::Sensitive)));
    assert!(excluded.contains(&("credentials.rs".to_string(), ExclusionReason::Sensitive)));
    assert!(excluded.contains(&("large.rs".to_string(), ExclusionReason::Oversized)));
    assert!(excluded.contains(&("unsupported.txt".to_string(), ExclusionReason::Unsupported)));
}

#[cfg(unix)]
#[test]
fn scanner_and_capability_read_reject_innocuous_hard_link_to_env() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("workspace");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join(".env"), "private hard-link content").unwrap();
    fs::hard_link(root.join(".env"), root.join("src/config.rs")).unwrap();
    let scanner = build_scanner(&root, &root.join("index-data")).unwrap();
    let capability = WorkspaceFs::open(&root).unwrap();

    assert!(matches!(
        capability.read(Path::new("src/config.rs"), 1024),
        Err(workspace_index_mcp::security::workspace_fs::WorkspaceFsError::SensitivePath)
    ));
    let alias = scanner
        .filter_map(Result::ok)
        .find(|entry| entry.path().ends_with("src/config.rs"))
        .unwrap();
    assert!(matches!(
        read_scanned_file(&capability, &root, &alias, 1024).unwrap(),
        ScanEntry::Excluded { reason: ExclusionReason::Sensitive, .. }
    ));
}

#[test]
fn test_binary_detection_uses_scanned_bytes() {
    assert!(is_binary(b"hello\0world"));
    assert!(!is_binary(b"hello world"));
}

#[test]
fn test_line_chunker() {
    let mut source = String::new();
    for i in 1..=250 {
        source.push_str(&format!("Line {}\n", i));
    }
    
    let chunks = line_chunker(&source, 120, 20);
    
    assert_eq!(chunks.len(), 3);
    
    assert_eq!(chunks[0].index, 0);
    assert_eq!(chunks[0].start_line, 1);
    assert_eq!(chunks[0].end_line, 120);
    
    assert_eq!(chunks[1].index, 1);
    assert_eq!(chunks[1].start_line, 101); // 120 - 20 + 1
    assert_eq!(chunks[1].end_line, 220);
    
    assert_eq!(chunks[2].index, 2);
    assert_eq!(chunks[2].start_line, 201); // 220 - 20 + 1
    assert_eq!(chunks[2].end_line, 250);
}

#[test]
fn test_pipeline_persists_complete_file_state_atomically() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("workspace");
    fs::create_dir_all(&root).unwrap();
    let mut conn = Connection::open(temp.path().join("db.sqlite")).unwrap();
    schema::initialize(&mut conn).unwrap();
    insert_registered_workspace(&conn, "ws1", &root);
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, endpoint, model, revision, dimension, metric, state, created_at)
         VALUES ('prof1', 'deterministic', NULL, 'test', NULL, 2, 'cosine', 'Active', 0)",
        [],
    ).unwrap();
    fs::write(root.join("test.rs"), "fn persisted() {}\n").unwrap();
    let workspace = registered_workspace(&conn, "ws1");

    let tx = conn.transaction().unwrap();
    let pipeline = IndexPipeline::new(&tx);
    let report = pipeline.index_workspace(IndexWorkspaceRequest {
        workspace_id: "ws1",
        workspace: &workspace,
        excluded_data_root: &temp.path().join("data"),
        provider: &DeterministicProvider,
        profile_id: "prof1",
        generation: 1,
        max_file_bytes: 1024 * 1024,
    }).unwrap();
    assert_eq!(report.updated, 1);
    tx.commit().unwrap();

    let file_generation: i64 = conn.query_row(
        "SELECT file_generation FROM files WHERE workspace_id='ws1' AND relative_path='test.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(file_generation, 1);
    let chunk_count: i64 = conn.query_row(
        "SELECT count(*) FROM chunks WHERE workspace_id='ws1' AND relative_path='test.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    let (op_type, embedding): (String, Vec<u8>) = conn.query_row(
        "SELECT op_type, embedding FROM vector_ops WHERE workspace_id='ws1'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(chunk_count, 1);
    assert_eq!(op_type, "add");
    assert_eq!(embedding.len(), 2 * std::mem::size_of::<f32>());
}

#[test]
fn public_vector_operations_reject_invalid_type_payload_and_dimension() {
    let temp = TempDir::new().unwrap();
    let mut conn = Connection::open(temp.path().join("db.sqlite")).unwrap();
    schema::initialize(&mut conn).unwrap();
    let root = temp.path().join("workspace");
    fs::create_dir_all(&root).unwrap();
    insert_registered_workspace(&conn, "ws1", &root);
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, model, dimension, metric, state, created_at)
         VALUES ('prof1', 'deterministic', 'test', 2, 'cosine', 'Active', 0)",
        [],
    ).unwrap();
    let tx = conn.transaction().unwrap();
    let pipeline = IndexPipeline::new(&tx);

    assert!(pipeline.enqueue_vector_op("ws1", "prof1", 1, "add", "v1", None).is_err());
    assert!(pipeline.enqueue_vector_op("ws1", "prof1", 1, "delete", "v1", Some(&[1.0, 2.0])).is_err());
    assert!(pipeline.enqueue_vector_op("ws1", "prof1", 1, "add", "v1", Some(&[1.0])).is_err());
    assert!(pipeline.enqueue_vector_op("ws1", "prof1", 1, "add", "v1", Some(&[1.0, f32::NAN])).is_err());
    assert!(pipeline.enqueue_vector_op("ws1", "prof1", 1, "add", "v1", Some(&[1.0, f32::INFINITY])).is_err());
    assert!(pipeline.enqueue_vector_op("ws1", "prof1", 1, "replace", "v1", None).is_err());
    assert!(pipeline.enqueue_vector_op("ws1", "missing", 1, "delete", "v1", None).is_err());

    pipeline.enqueue_vector_op("ws1", "prof1", 1, "add", "v1", Some(&[1.0, 2.0])).unwrap();
    pipeline.enqueue_vector_op("ws1", "prof1", 2, "delete", "v1", None).unwrap();
    tx.commit().unwrap();
    let operation_count: i64 = conn.query_row("SELECT count(*) FROM vector_ops", [], |row| row.get(0)).unwrap();
    assert_eq!(operation_count, 2);
}

#[test]
fn test_unavailable_embedding_provider() {
    let provider = UnavailableEmbeddingProvider;
    assert_eq!(provider.fingerprint(), "unavailable");
    assert_eq!(provider.dimension(), 0);
    assert!(provider.embed_batch(&["test".to_string()]).is_err());
}

struct DeterministicProvider;
impl EmbeddingProvider for DeterministicProvider {
    fn fingerprint(&self) -> String { "deterministic".to_string() }
    fn dimension(&self) -> usize { 2 }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts.iter().map(|t| vec![t.len() as f32, 1.0]).collect())
    }
}

#[test]
fn test_orchestration_edit_reindex() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    
    let db_path = root.join("db.sqlite");
    let mut conn = Connection::open(&db_path).unwrap();
    schema::initialize(&mut conn).unwrap();
    
    // Insert dummy workspace and profile
    insert_registered_workspace(&conn, "ws1", root);
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, endpoint, model, revision, dimension, metric, state, created_at)
         VALUES ('prof1', 'deterministic', NULL, 'none', NULL, 2, 'cosine', 'Active', 0)",
        [],
    ).unwrap();
    
    // Create initial file
    let file_path = root.join("test.rs");
    fs::write(&file_path, "fn hello() { println!(\"world\"); }").unwrap();
    
    let provider = DeterministicProvider;
    let workspace = registered_workspace(&conn, "ws1");
    let excluded_data_root = root.join(".workspace-index-data");
    
    // First index
    {
        let tx = conn.transaction().unwrap();
        let pipeline = IndexPipeline::new(&tx);
        let report = pipeline.index_workspace(IndexWorkspaceRequest {
            workspace_id: "ws1", workspace: &workspace, excluded_data_root: &excluded_data_root,
            provider: &provider, profile_id: "prof1", generation: 1, max_file_bytes: 1024 * 1024,
        }).unwrap();
        assert_eq!(report.updated, 1);
        assert_eq!(report.deleted, 0);
        tx.commit().unwrap();
    }
    
    // Verify lexical rows
    let count: i64 = conn.query_row("SELECT count(*) FROM chunks_fts", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 1); // one chunk
    let count: i64 = conn.query_row("SELECT count(*) FROM symbols", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 1); // fn hello
    
    // Verify vector ops
    let ops: Vec<(String, String)> = conn.prepare("SELECT op_type, vector_id FROM vector_ops ORDER BY id").unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(ops, vec![("add".to_string(), "test.rs#0".to_string())]);
    
    // Edit file
    fs::write(&file_path, "fn hello2() { println!(\"world2\"); }").unwrap();
    
    // Reindex
    {
        let tx = conn.transaction().unwrap();
        let pipeline = IndexPipeline::new(&tx);
        let report = pipeline.index_workspace(IndexWorkspaceRequest {
            workspace_id: "ws1", workspace: &workspace, excluded_data_root: &excluded_data_root,
            provider: &provider, profile_id: "prof1", generation: 2, max_file_bytes: 1024 * 1024,
        }).unwrap();
        assert_eq!(report.updated, 1);
        assert_eq!(report.deleted, 0);
        tx.commit().unwrap();
    }
    
    // Verify lexical rows updated
    let count: i64 = conn.query_row("SELECT count(*) FROM chunks_fts", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 1); // still one chunk
    let sym_name: String = conn.query_row("SELECT name FROM symbols", [], |row| row.get(0)).unwrap();
    assert_eq!(sym_name, "hello2");
    
    // Verify vector ops
    let ops: Vec<(String, String)> = conn.prepare("SELECT op_type, vector_id FROM vector_ops ORDER BY id").unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(ops, vec![
        ("add".to_string(), "test.rs#0".to_string()),
        ("delete".to_string(), "test.rs#0".to_string()),
        ("add".to_string(), "test.rs#0".to_string())
    ]);
    
    // Delete file
    fs::remove_file(&file_path).unwrap();
    
    // Reindex
    {
        let tx = conn.transaction().unwrap();
        let pipeline = IndexPipeline::new(&tx);
        let report = pipeline.index_workspace(IndexWorkspaceRequest {
            workspace_id: "ws1", workspace: &workspace, excluded_data_root: &excluded_data_root,
            provider: &provider, profile_id: "prof1", generation: 3, max_file_bytes: 1024 * 1024,
        }).unwrap();
        assert_eq!(report.updated, 0);
        assert_eq!(report.deleted, 1);
        tx.commit().unwrap();
    }
    
    // Verify lexical rows deleted
    let count: i64 = conn.query_row("SELECT count(*) FROM chunks_fts", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 0);
    let count: i64 = conn.query_row("SELECT count(*) FROM symbols", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 0);
    
    // Verify vector ops
    let ops: Vec<(String, String)> = conn.prepare("SELECT op_type, vector_id FROM vector_ops ORDER BY id").unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(ops, vec![
        ("add".to_string(), "test.rs#0".to_string()),
        ("delete".to_string(), "test.rs#0".to_string()),
        ("add".to_string(), "test.rs#0".to_string()),
        ("delete".to_string(), "test.rs#0".to_string())
    ]);
}

#[test]
fn test_orchestration_degraded_semantic() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    
    let db_path = root.join("db.sqlite");
    let mut conn = Connection::open(&db_path).unwrap();
    schema::initialize(&mut conn).unwrap();
    
    insert_registered_workspace(&conn, "ws1", root);
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, endpoint, model, revision, dimension, metric, state, created_at)
         VALUES ('prof1', 'unavailable', NULL, 'none', NULL, 0, 'cosine', 'Active', 0)",
        [],
    ).unwrap();
    
    let file_path = root.join("test.rs");
    fs::write(&file_path, "fn hello() { println!(\"world\"); }").unwrap();
    
    let provider = UnavailableEmbeddingProvider;
    let workspace = registered_workspace(&conn, "ws1");
    let excluded_data_root = root.join(".workspace-index-data");
    
    let tx = conn.transaction().unwrap();
    let pipeline = IndexPipeline::new(&tx);
    let report = pipeline.index_workspace(IndexWorkspaceRequest {
        workspace_id: "ws1", workspace: &workspace, excluded_data_root: &excluded_data_root,
        provider: &provider, profile_id: "prof1", generation: 1, max_file_bytes: 1024 * 1024,
    }).unwrap();
    
    assert_eq!(report.updated, 1);
    assert!(report.degraded_semantic);
    tx.commit().unwrap();
    
    // Lexical index should still be healthy
    let count: i64 = conn.query_row("SELECT count(*) FROM chunks_fts", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 1);
    let count: i64 = conn.query_row("SELECT count(*) FROM symbols", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 1);
    
    // No vector ops should be enqueued
    let count: i64 = conn.query_row("SELECT count(*) FROM vector_ops", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn reconciliation_batches_large_files_without_losing_embeddings() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("workspace");
    fs::create_dir_all(&root).unwrap();
    let mut conn = Connection::open(temp.path().join("db.sqlite")).unwrap();
    schema::initialize(&mut conn).unwrap();
    insert_registered_workspace(&conn, "ws1", &root);
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, model, dimension, metric, state, created_at)
         VALUES ('prof1', 'deterministic', 'test', 2, 'cosine', 'Active', 0)",
        [],
    ).unwrap();
    let source = (0..40)
        .map(|index| format!("fn function_{index}() {{ let value = {index}; }}\n"))
        .collect::<String>();
    fs::write(root.join("many.rs"), source).unwrap();
    let workspace = registered_workspace(&conn, "ws1");
    let tx = conn.transaction().unwrap();
    let report = IndexPipeline::new(&tx)
        .index_workspace(IndexWorkspaceRequest {
            workspace_id: "ws1", workspace: &workspace, excluded_data_root: &temp.path().join("data"),
            provider: &LimitedBatchProvider, profile_id: "prof1", generation: 1,
            max_file_bytes: 1024 * 1024,
        }).unwrap();
    assert!(!report.degraded_semantic);
    tx.commit().unwrap();

    let chunks: i64 = conn.query_row(
        "SELECT count(*) FROM chunks WHERE workspace_id='ws1' AND relative_path='many.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    let embeddings: i64 = conn.query_row(
        "SELECT count(*) FROM chunk_embeddings WHERE workspace_id='ws1' AND relative_path='many.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    let additions: i64 = conn.query_row(
        "SELECT count(*) FROM vector_ops WHERE workspace_id='ws1' AND op_type='add'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert!(chunks > 32);
    assert_eq!(embeddings, chunks);
    assert_eq!(additions, chunks);
}

#[test]
fn invalid_embedding_subbatch_discards_all_vectors_for_profile() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("workspace");
    fs::create_dir_all(&root).unwrap();
    let mut conn = Connection::open(temp.path().join("db.sqlite")).unwrap();
    schema::initialize(&mut conn).unwrap();
    insert_registered_workspace(&conn, "ws1", &root);
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, model, dimension, metric, state, created_at)
         VALUES ('prof1', 'deterministic', 'test', 2, 'cosine', 'Active', 0)",
        [],
    ).unwrap();
    let source = (0..40)
        .map(|index| format!("fn invalid_{index}() {{ let value = {index}; }}\n"))
        .collect::<String>();
    fs::write(root.join("invalid.rs"), source).unwrap();
    let workspace = registered_workspace(&conn, "ws1");
    let provider = InvalidSecondBatchProvider { calls: std::sync::atomic::AtomicUsize::new(0) };
    let tx = conn.transaction().unwrap();
    let report = IndexPipeline::new(&tx)
        .index_workspace(IndexWorkspaceRequest {
            workspace_id: "ws1", workspace: &workspace, excluded_data_root: &temp.path().join("data"),
            provider: &provider, profile_id: "prof1", generation: 1, max_file_bytes: 1024 * 1024,
        }).unwrap();
    assert!(report.degraded_semantic);
    tx.commit().unwrap();

    let lexical_chunks: i64 = conn.query_row(
        "SELECT count(*) FROM chunks WHERE workspace_id='ws1' AND relative_path='invalid.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    let embeddings: i64 = conn.query_row(
        "SELECT count(*) FROM chunk_embeddings WHERE workspace_id='ws1' AND relative_path='invalid.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    let vector_ops: i64 = conn.query_row(
        "SELECT count(*) FROM vector_ops WHERE workspace_id='ws1'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert!(lexical_chunks > 32);
    assert_eq!(embeddings, 0);
    assert_eq!(vector_ops, 0);
}

struct InvalidSecondBatchProvider {
    calls: std::sync::atomic::AtomicUsize,
}
impl EmbeddingProvider for InvalidSecondBatchProvider {
    fn fingerprint(&self) -> String { "invalid-second-batch".into() }
    fn dimension(&self) -> usize { 2 }
    fn batch_size(&self) -> usize { 32 }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 1 {
            return Ok(vec![vec![f32::NAN, 1.0]; texts.len()]);
        }
        Ok(texts.iter().map(|text| vec![text.len() as f32, 1.0]).collect())
    }
}
#[test]
fn failed_embedding_subbatch_keeps_lexical_chunks_without_partial_vectors() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("workspace");
    fs::create_dir_all(&root).unwrap();
    let mut conn = Connection::open(temp.path().join("db.sqlite")).unwrap();
    schema::initialize(&mut conn).unwrap();
    insert_registered_workspace(&conn, "ws1", &root);
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, model, dimension, metric, state, created_at)
         VALUES ('prof1', 'deterministic', 'test', 2, 'cosine', 'Active', 0)",
        [],
    ).unwrap();
    let source = (0..40)
        .map(|index| format!("fn function_{index}() {{ let value = {index}; }}\n"))
        .collect::<String>();
    fs::write(root.join("many.rs"), source).unwrap();
    let workspace = registered_workspace(&conn, "ws1");
    let provider = FailsOnSecondBatch { calls: std::sync::atomic::AtomicUsize::new(0) };
    let tx = conn.transaction().unwrap();
    let report = IndexPipeline::new(&tx)
        .index_workspace(IndexWorkspaceRequest {
            workspace_id: "ws1", workspace: &workspace, excluded_data_root: &temp.path().join("data"),
            provider: &provider, profile_id: "prof1", generation: 1, max_file_bytes: 1024 * 1024,
        }).unwrap();
    assert!(report.degraded_semantic);
    tx.commit().unwrap();

    let chunks: i64 = conn.query_row(
        "SELECT count(*) FROM chunks WHERE workspace_id='ws1' AND relative_path='many.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    let symbols: i64 = conn.query_row(
        "SELECT count(*) FROM symbols WHERE workspace_id='ws1' AND relative_path='many.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    let embeddings: i64 = conn.query_row(
        "SELECT count(*) FROM chunk_embeddings WHERE workspace_id='ws1' AND relative_path='many.rs'",
        [],
        |row| row.get(0),
    ).unwrap();
    let operations: i64 = conn.query_row(
        "SELECT count(*) FROM vector_ops WHERE workspace_id='ws1'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert!(chunks > 32);
    assert!(symbols > 0);
    assert_eq!(embeddings, 0);
    assert_eq!(operations, 0);
}

struct FailsOnSecondBatch {
    calls: std::sync::atomic::AtomicUsize,
}
impl EmbeddingProvider for FailsOnSecondBatch {
    fn fingerprint(&self) -> String { "failure-in-middle".into() }
    fn dimension(&self) -> usize { 2 }
    fn batch_size(&self) -> usize { 32 }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 1 {
            return Err("injected second-batch failure".into());
        }
        Ok(texts.iter().map(|text| vec![text.len() as f32, 1.0]).collect())
    }
}

struct LimitedBatchProvider;
impl EmbeddingProvider for LimitedBatchProvider {
    fn fingerprint(&self) -> String { "limited-batch".into() }
    fn dimension(&self) -> usize { 2 }
    fn batch_size(&self) -> usize { 32 }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.len() > self.batch_size() {
            return Err("batch exceeds provider limit".into());
        }
        Ok(texts.iter().map(|text| vec![text.len() as f32, 1.0]).collect())
    }
}

