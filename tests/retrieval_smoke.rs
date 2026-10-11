use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;
use rusqlite::Connection;

use workspace_index_mcp::security::WorkspaceFs;
use workspace_index_mcp::state::schema;
use workspace_index_mcp::retrieval::search::{
    CodeSearchRequest, SearchService, SearchServiceOptions, SymbolSearchRequest,
};
use workspace_index_mcp::model::ErrorCode;

fn setup_db() -> (Connection, TempDir, TempDir) {
    let temp1 = TempDir::new().unwrap();
    let temp2 = TempDir::new().unwrap();
    let mut conn = Connection::open_in_memory().unwrap();
    schema::initialize(&mut conn).unwrap();
    for (id, root) in [("ws1", temp1.path()), ("ws2", temp2.path())] {
        let (device, inode) = root_identity(root);
        conn.execute(
            "INSERT INTO workspaces (id, root_path, device_id, inode, mode, registration_epoch, mode_epoch, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 'read_only', 1, 1, 0, 0)",
            rusqlite::params![id, root.to_str().unwrap(), device as i64, inode as i64],
        ).unwrap();
    }
    conn.execute(
        "INSERT INTO workspace_index_state (workspace_id, generation, building_generation, visible_generation, updated_at) VALUES ('ws1', 1, 1, 1, 0), ('ws2', 1, 1, 1, 0)",
        [],
    ).unwrap();

    (conn, temp1, temp2)
}

#[cfg(unix)]
fn root_identity(root: &Path) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::metadata(root).unwrap();
    (metadata.dev(), metadata.ino())
}

#[cfg(not(unix))]
fn root_identity(_root: &Path) -> (u64, u64) { (0, 0) }

fn search_service<'a>(conn: &'a Connection, secret: &'a [u8], workspace_ids: &[&str], vectors_dir: std::path::PathBuf) -> SearchService<'a> {
    let capabilities = workspace_ids.iter().filter_map(|id| {
        let identity = conn.query_row(
            "SELECT root_path, device_id, inode FROM workspaces WHERE id=?1",
            [id], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
        ).ok()?;
        let capability = WorkspaceFs::open_registered(Path::new(&identity.0), identity.1 as u64, identity.2 as u64)
            .expect("registered root capability");
        Some(((*id).to_string(), capability))
    }).collect::<HashMap<_, _>>();
    SearchService::with_capabilities(conn, secret, SearchServiceOptions {
        capabilities,
        max_file_bytes: 1_048_576,
        max_results: 100,
        max_snippet_bytes: 16 * 1024,
        max_tool_bytes: 450 * 1024,
        vectors_dir,
    })
}
    
#[test]
fn test_retrieval_ordering_and_boosts() {
    let (conn, temp, _) = setup_db();
    let file1_path = temp.path().join("main.rs");
    let mut f1 = File::create(&file1_path).unwrap();
    f1.write_all(b"fn target_function() {}").unwrap();
    
    let file2_path = temp.path().join("other.rs");
    let mut f2 = File::create(&file2_path).unwrap();
    f2.write_all(b"fn something_else() { target_function(); }").unwrap();
    
    let hash1 = "hash1";
    let hash2 = "hash2";
    
    // Insert files
    conn.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation) 
         VALUES ('ws1', 'main.rs', ?1, 100, 1)",
        rusqlite::params![hash1]
    ).unwrap();
    
    conn.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation) 
         VALUES ('ws1', 'other.rs', ?1, 100, 1)",
        rusqlite::params![hash2]
    ).unwrap();
    
    // Insert chunks
    conn.execute(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content, language, chunk_hash)
         VALUES ('ws1', 'main.rs', 0, 1, 1, 'fn target_function() {}', 'rust', 'ch1')",
         []
    ).unwrap();
    
    conn.execute(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content, language, chunk_hash)
         VALUES ('ws1', 'other.rs', 0, 1, 1, 'fn something_else() { target_function(); }', 'rust', 'ch2')",
         []
    ).unwrap();
    
    // Insert symbols
    conn.execute(
        "INSERT INTO symbols (workspace_id, relative_path, symbol_index, name, kind, byte_start, byte_end, start_line, end_line)
         VALUES ('ws1', 'main.rs', 0, 'target_function', 'function', 0, 20, 1, 1)",
         []
    ).unwrap();
    
    // FTS trigger already populated chunks_fts
    
    let secret = b"test_secret";
    let service = search_service(&conn, secret, &["ws1"], temp.path().to_path_buf());
    
    let req = CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "target_function".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: None, 
        limit: 10,
        cursor: None,
    };
    
    // Wait, the content hash needs to match the actual file hash for freshness check
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fn target_function() {}");
    let actual_hash1 = hasher.finalize().to_hex().to_string();
    
    let mut hasher2 = blake3::Hasher::new();
    hasher2.update(b"fn something_else() { target_function(); }");
    let actual_hash2 = hasher2.finalize().to_hex().to_string();
    
    conn.execute("UPDATE files SET content_hash = ?1 WHERE relative_path = 'main.rs'", rusqlite::params![actual_hash1]).unwrap();
    conn.execute("UPDATE files SET content_hash = ?1 WHERE relative_path = 'other.rs'", rusqlite::params![actual_hash2]).unwrap();
    
    let result = service.code_search(req).into_data().unwrap();
    let items = result.items;
    
    assert_eq!(items.len(), 2);
    // exact symbol boost should make main.rs rank higher
    assert_eq!(items[0].relative_path, "main.rs");
    assert_eq!(items[1].relative_path, "other.rs");
}

#[test]
fn test_explicit_workspaces() {
    let (conn, temp, temp2) = setup_db();
    
    let file1_path = temp.path().join("main.rs");
    let mut f1 = File::create(&file1_path).unwrap();
    f1.write_all(b"fn shared_name() {}").unwrap();
    
    let file2_path = temp2.path().join("main.rs");
    let mut f2 = File::create(&file2_path).unwrap();
    f2.write_all(b"fn shared_name() {}").unwrap();
    
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fn shared_name() {}");
    let actual_hash = hasher.finalize().to_hex().to_string();
    
    conn.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation) 
         VALUES ('ws1', 'main.rs', ?1, 100, 1), ('ws2', 'main.rs', ?1, 100, 1)",
        rusqlite::params![actual_hash]
    ).unwrap();
    
    conn.execute(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content)
         VALUES ('ws1', 'main.rs', 0, 1, 1, 'fn shared_name() {}'),
                ('ws2', 'main.rs', 0, 1, 1, 'fn shared_name() {}')",
         []
    ).unwrap();
    
    let secret = b"test_secret";
    let service = search_service(&conn, secret, &["ws2"], temp.path().to_path_buf());
    
    let req = CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "shared_name".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws2".to_string()],
        profile_id: None, 
        limit: 10,
        cursor: None,
    };
    
    let result = service.code_search(req).into_data().unwrap();
    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].workspace_id, "ws2");
}

#[test]
fn test_cursor_tampering_and_staleness() {
    let (conn, temp, _) = setup_db();
    
    let file1_path = temp.path().join("main.rs");
    let mut f1 = File::create(&file1_path).unwrap();
    f1.write_all(b"fn test() {}").unwrap();
    
    let file2_path = temp.path().join("main2.rs");
    let mut f2 = File::create(&file2_path).unwrap();
    f2.write_all(b"fn test something() {}").unwrap();
    
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fn test() {}");
    let actual_hash = hasher.finalize().to_hex().to_string();
    
    let mut hasher2 = blake3::Hasher::new();
    hasher2.update(b"fn test something() {}");
    let actual_hash2 = hasher2.finalize().to_hex().to_string();
    
    conn.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation) 
         VALUES ('ws1', 'main.rs', ?1, 100, 1),
                ('ws1', 'main2.rs', ?2, 100, 1)",
        rusqlite::params![actual_hash, actual_hash2]
    ).unwrap();
    
    conn.execute(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content)
         VALUES ('ws1', 'main.rs', 0, 1, 1, 'fn test() {}'),
                ('ws1', 'main2.rs', 0, 1, 1, 'fn test something() {}')",
         []
    ).unwrap();
    
    let secret = b"test_secret";
    let service = search_service(&conn, secret, &["ws1"], temp.path().to_path_buf());
    
    let req = CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "test".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: None, 
        limit: 1, // trigger pagination
        cursor: None,
    };
    
    let result = service.code_search(req).into_data().unwrap();
    let cursor = result.next_cursor.unwrap();
    
    // Tamper cursor
    let mut tampered = cursor.clone();
    tampered.replace_range(0..1, "0");
    
    let req2 = CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "test".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: None, 
        limit: 1,
        cursor: Some(tampered),
    };
    
    let err = service.code_search(req2).into_error().unwrap();
    assert_eq!(err.code, ErrorCode::StaleCursor);
    
    // Stale generation
    conn.execute("UPDATE workspace_index_state SET building_generation = 2, visible_generation = 2 WHERE workspace_id = 'ws1'", []).unwrap();
    
    let req3 = CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "test".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: None, 
        limit: 1,
        cursor: Some(cursor),
    };
    
    let err = service.code_search(req3).into_error().unwrap();
    assert_eq!(err.code, ErrorCode::StaleCursor);
}

#[test]
fn code_cursor_rejects_workspace_id_re_registered_to_another_root() {
    let (conn, root_a, _root_b) = setup_db();
    for name in ["needle_a.rs", "needle_b.rs"] {
        add_file_chunk(
            &conn,
            root_a.path(),
            "ws1",
            name,
            &format!("fn {}() {{}}", name.trim_end_matches(".rs")),
            1,
            1,
        );
    }
    let service = search_service(
        &conn,
        b"test_secret",
        &["ws1"],
        root_a.path().to_path_buf(),
    );
    let request = |cursor| CodeSearchRequest {
        query: "needle".into(),
        query_vector: None,
        workspace_ids: vec!["ws1".into()],
        profile_id: None,
        languages: vec![],
        path_globs: vec![],
        limit: 1,
        cursor,
    };

    let first = service.code_search(request(None)).into_data().unwrap();
    let cursor = first.next_cursor.expect("second result needs a cursor");
    let replacement_root = TempDir::new().unwrap();
    let (device, inode) = root_identity(replacement_root.path());
    conn.execute(
        "UPDATE workspaces
         SET root_path=?1, device_id=?2, inode=?3, registration_epoch=registration_epoch+1
         WHERE id='ws1'",
        rusqlite::params![replacement_root.path().to_str().unwrap(), device as i64, inode as i64],
    )
    .unwrap();

    let stale = service.code_search(request(Some(cursor))).into_error().unwrap();
    assert_eq!(stale.code, ErrorCode::StaleCursor);
}

#[test]
fn test_stale_file_suppression() {
    let (conn, temp, _) = setup_db();
    let file1_path = temp.path().join("main.rs");
    let mut f1 = File::create(&file1_path).unwrap();
    f1.write_all(b"fn old_content() {}").unwrap();
    
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fn old_content() {}");
    let actual_hash = hasher.finalize().to_hex().to_string();
    
    conn.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation) 
         VALUES ('ws1', 'main.rs', ?1, 100, 1)",
        rusqlite::params![actual_hash]
    ).unwrap();
    
    conn.execute(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content)
         VALUES ('ws1', 'main.rs', 0, 1, 1, 'fn old_content() {}')",
         []
    ).unwrap();
    
    let secret = b"test_secret";
    let service = search_service(&conn, secret, &["ws1"], temp.path().to_path_buf());
    
    // Now change the file on disk so the hash mismatches
    let mut f1 = File::create(&file1_path).unwrap();
    f1.write_all(b"fn new_content() {}").unwrap();
    
    let req = CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "old_content".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: None, 
        limit: 10,
        cursor: None,
    };
    
    let result = service.code_search(req);
    let data = result.data().unwrap();
    // Should be suppressed
    assert_eq!(data.items.len(), 0);
    assert!(result.warnings().iter().any(|w| w.code == "INDEX_DEGRADED"));
    
}

fn add_file_chunk(
    conn: &Connection,
    root: &std::path::Path,
    workspace_id: &str,
    relative_path: &str,
    content: &str,
    visible_generation: i64,
    file_generation: i64,
) {
    fs::write(root.join(relative_path), content).unwrap();
    let content_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    conn.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation, file_generation)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            workspace_id,
            relative_path,
            content_hash,
            content.len() as i64,
            visible_generation,
            file_generation
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content)
         VALUES (?1, ?2, 0, 1, 1, ?3)",
        rusqlite::params![workspace_id, relative_path, content],
    )
    .unwrap();
}

fn add_embedding_profile(conn: &Connection) {
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, model, dimension, metric, state, created_at)
         VALUES ('profile-1', 'test', 'test-model', 1, 'cosine', 'active', 0)",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE workspace_index_state
         SET desired_profile_id='profile-1', active_profile_id='profile-1'
         WHERE workspace_id='ws1'",
        [],
    ).unwrap();
}

#[test]
fn test_rejects_empty_and_unknown_workspace_lists() {
    let (conn, temp, _) = setup_db();
    let service = search_service(&conn, b"test_secret", &[], temp.path().to_path_buf());

    let empty = service.code_search(CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "anything".to_string(),
        query_vector: None,
        workspace_ids: Vec::new(),
        profile_id: None, 
        limit: 10,
        cursor: None,
    });
    assert_eq!(empty.into_error().unwrap().code, ErrorCode::WorkspaceRequired);

    let unknown = service.symbol_search(SymbolSearchRequest { direction: None,
        query: "anything".to_string(),
        workspace_ids: vec!["missing".to_string()],
        limit: 10,
        cursor: None,
    });
    assert_eq!(unknown.into_error().unwrap().code, ErrorCode::WorkspaceNotFound);
}

#[test]
fn test_excludes_files_outside_visible_generation_and_stale_embeddings() {
    let (conn, temp, _) = setup_db();
    add_embedding_profile(&conn);
    add_file_chunk(
        &conn,
        temp.path(),
        "ws1",
        "future.rs",
        "future generation content",
        2,
        2,
    );
    let embedding = 1.0f32.to_le_bytes();
    conn.execute(
        "INSERT INTO chunk_embeddings (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding)
         VALUES ('ws1', 'future.rs', 0, 'profile-1', 2, 'not-visible', ?1)",
        rusqlite::params![embedding.as_slice()],
    )
    .unwrap();

    let result = search_service(&conn, b"test_secret", &["ws1"], temp.path().to_path_buf()).code_search(CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "no_lexical_match".to_string(),
        query_vector: Some(vec![1.0]),
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: Some("profile-1".to_string()),
        limit: 10,
        cursor: None,
    });
    
    assert!(result.into_data().unwrap().items.is_empty());
}


#[test]
fn test_symlinked_freshness_path_is_suppressed() {
    let (conn, temp, _) = setup_db();
    let target_content = "symlink_target content";
    fs::write(temp.path().join("target.rs"), target_content).unwrap();
    std::os::unix::fs::symlink("target.rs", temp.path().join("linked.rs")).unwrap();
    
    let hash = blake3::hash(target_content.as_bytes()).to_hex().to_string();
    conn.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation, file_generation)
         VALUES ('ws1', 'linked.rs', ?1, 100, 1, 1)",
        rusqlite::params![hash],
    ).unwrap();
    conn.execute(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content)
         VALUES ('ws1', 'linked.rs', 0, 1, 1, ?1)",
        rusqlite::params![target_content],
    ).unwrap();

    let result = search_service(&conn, b"test_secret", &["ws1"], temp.path().to_path_buf()).code_search(CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "symlink_target".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: None, 
        limit: 10,
        cursor: None,
    });
    
    assert!(result.into_data().unwrap().items.is_empty());
}

#[test]
fn search_never_returns_content_reached_through_an_outside_root_symlink() {
    let (conn, root, _) = setup_db();
    let outside_dir = tempfile::TempDir::new().unwrap();
    let outside_target = outside_dir.path().join("secret.rs");
    let outside_content = "secret_outside_workspace content";
    fs::write(&outside_target, outside_content).unwrap();
    
    add_file_chunk(
        &conn, root.path(), "ws1", "linked.rs", outside_content, 1, 1,
    );
    fs::remove_file(root.path().join("linked.rs")).unwrap();
    std::os::unix::fs::symlink(&outside_target, root.path().join("linked.rs")).unwrap();

    let result = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf()).code_search(CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "secret_outside_workspace".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: None, 
        limit: 10,
        cursor: None,
    });
    
    assert!(result.into_data().unwrap().items.is_empty());
}

#[test]
fn test_code_and_symbol_pages_stay_within_payload_byte_limit() {
    let (conn, temp, _) = setup_db();
    let filler = "x".repeat(16 * 1024 - "needle ".len());
    let content = format!("needle {filler}");
    for index in 0..40 {
        let relative_path = format!("chunk-{index:02}.rs");
        add_file_chunk(
            &conn,
            temp.path(),
            "ws1",
            &relative_path,
            &content,
            1,
            1,
        );
        conn.execute(
            "INSERT INTO symbols (workspace_id, relative_path, symbol_index, name, kind, byte_start, byte_end, start_line, end_line)
             VALUES ('ws1', ?1, 0, ?2, 'function', 0, 1, 1, 1)",
            rusqlite::params![relative_path, format!("needle_symbol_{index:02}")],
        )
        .unwrap();
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], temp.path().to_path_buf());

    let code_result = service.code_search(CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "needle".to_string(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: None, 
        limit: 100,
        cursor: None,
    });
    let code_json = serde_json::to_vec(&code_result).unwrap();
    assert!(code_json.len() <= 512 * 1024);
    assert!(code_result.data().unwrap().truncated);

    let symbol_result = service.symbol_search(SymbolSearchRequest { direction: None,
        query: "needle".to_string(),
        workspace_ids: vec!["ws1".to_string()], 
        limit: 100,
        cursor: None,
    });
    let symbol_json = serde_json::to_vec(&symbol_result).unwrap();
    assert!(symbol_json.len() <= 512 * 1024);
    assert!(symbol_result.data().unwrap().truncated);
}


#[test]
fn symbol_search_filters_directional_candidates_and_binds_cursor_direction() {
    let (conn, temp, _) = setup_db();
    let source = "caller invokes target and other\n";
    add_file_chunk(&conn, temp.path(), "ws1", "symbols.rs", source, 1, 1);
    for (index, name) in ["caller", "target"].iter().enumerate() {
        conn.execute(
            "INSERT INTO symbols (workspace_id, relative_path, symbol_index, name, kind, byte_start, byte_end, start_line, end_line)
             VALUES ('ws1', 'symbols.rs', ?1, ?2, 'function', 0, 1, 1, 1)",
            rusqlite::params![index as i64, name],
        ).unwrap();
    }
    for (index, name, evidence, source_symbol) in [
        (0_i64, "target", "call_candidate", Some("caller")),
        (1_i64, "other", "import", Some("caller")),
    ] {
        conn.execute(
            "INSERT INTO dependency_edges (workspace_id, relative_path, edge_index, evidence, name, source_symbol, byte_start, byte_end, start_line, end_line)
             VALUES ('ws1', 'symbols.rs', ?1, ?2, ?3, ?4, 0, 1, 1, 1)",
            rusqlite::params![index, evidence, name, source_symbol],
        ).unwrap();
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], temp.path().to_path_buf());

    let incoming = service.symbol_search(SymbolSearchRequest {
        direction: Some(" incoming ".into()),
        query: "target".into(),
        workspace_ids: vec!["ws1".into()],
        limit: 10,
        cursor: None,
    }).into_data().unwrap();
    assert_eq!(incoming.items.len(), 1);
    assert_eq!(incoming.items[0].evidence, "call_candidate");

    let outgoing = service.symbol_search(SymbolSearchRequest {
        direction: Some("outgoing".into()),
        query: "caller".into(),
        workspace_ids: vec!["ws1".into()],
        limit: 10,
        cursor: None,
    }).into_data().unwrap();
    assert_eq!(
        outgoing.items.iter().map(|item| item.evidence.as_str()).collect::<Vec<_>>(),
        vec!["definition", "import", "call_candidate"]
    );
    assert!(outgoing.items.iter().any(|item| item.symbol == "target" && item.evidence == "call_candidate"));

    let both = service.symbol_search(SymbolSearchRequest {
        direction: None,
        query: "target".into(),
        workspace_ids: vec!["ws1".into()],
        limit: 10,
        cursor: None,
    }).into_data().unwrap();
    let both_evidence = both.items.iter().map(|item| item.evidence.as_str()).collect::<std::collections::HashSet<_>>();
    assert!(both_evidence.contains("definition"));
    assert!(both_evidence.contains("call_candidate"));
}

#[test]
fn symbol_cursor_rejects_a_changed_direction() {
    let (conn, temp, _) = setup_db();
    for name in ["needle_one", "needle_two"] {
        let path = format!("{name}.rs");
        add_file_chunk(&conn, temp.path(), "ws1", &path, "needle", 1, 1);
        conn.execute(
            "INSERT INTO symbols (workspace_id, relative_path, symbol_index, name, kind, byte_start, byte_end, start_line, end_line)
             VALUES ('ws1', ?1, 0, ?2, 'function', 0, 1, 1, 1)",
            rusqlite::params![path, name],
        ).unwrap();
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], temp.path().to_path_buf());
    let cursor = service.symbol_search(SymbolSearchRequest {
        direction: None,
        query: "needle".into(),
        workspace_ids: vec!["ws1".into()],
        limit: 1,
        cursor: None,
    }).into_data().unwrap().next_cursor.unwrap();
    let changed_direction = service.symbol_search(SymbolSearchRequest {
        direction: Some("outgoing".into()),
        query: "needle".into(),
        workspace_ids: vec!["ws1".into()],
        limit: 1,
        cursor: Some(cursor),
    });
    assert_eq!(changed_direction.into_error().unwrap().code, ErrorCode::StaleCursor);
}
#[test]
fn test_symbol_cursor_binds_active_profile_and_visible_generation() {
    let (conn, temp, _) = setup_db();
    add_file_chunk(
        &conn,
        temp.path(),
        "ws1",
        "one.rs",
        "needle one",
        1,
        1,
    );
    add_file_chunk(
        &conn,
        temp.path(),
        "ws1",
        "two.rs",
        "needle two",
        1,
        1,
    );
    for (index, path) in ["one.rs", "two.rs"].iter().enumerate() {
        conn.execute(
            "INSERT INTO symbols (workspace_id, relative_path, symbol_index, name, kind, byte_start, byte_end, start_line, end_line)
             VALUES ('ws1', ?1, 0, ?2, 'function', 0, 1, 1, 1)",
            rusqlite::params![path, format!("needle_{index}")],
        )
        .unwrap();
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], temp.path().to_path_buf());
    let first_page = service.symbol_search(SymbolSearchRequest { direction: None,
        query: "needle".to_string(),
        workspace_ids: vec!["ws1".to_string()], 
        limit: 1,
        cursor: None,
    });
    let cursor = first_page.into_data().unwrap().next_cursor.unwrap();

    add_embedding_profile(&conn);
    let stale_profile_cursor = service.symbol_search(SymbolSearchRequest { direction: None,
        query: "needle".to_string(),
        workspace_ids: vec!["ws1".to_string()], 
        limit: 1,
        cursor: Some(cursor),
    });
    assert_eq!(stale_profile_cursor.into_error().unwrap().code, ErrorCode::StaleCursor);

    let profile_bound_page = service.symbol_search(SymbolSearchRequest { direction: None,
        query: "needle".to_string(),
        workspace_ids: vec!["ws1".to_string()], 
        limit: 1,
        cursor: None,
    });
    let profile_bound_cursor = profile_bound_page.into_data().unwrap().next_cursor.unwrap();
    conn.execute(
        "UPDATE workspace_index_state SET building_generation = 2, visible_generation = 2 WHERE workspace_id = 'ws1'",
        [],
    )
    .unwrap();
    let stale_generation_cursor = service.symbol_search(SymbolSearchRequest { direction: None,
        query: "needle".to_string(),
        workspace_ids: vec!["ws1".to_string()], 
        limit: 1,
        cursor: Some(profile_bound_cursor),
    });
    assert_eq!(stale_generation_cursor.into_error().unwrap().code, ErrorCode::StaleCursor);
}


#[test]
fn test_fused_score_ties_have_deterministic_order() {
    let (conn, temp, _) = setup_db();
    add_embedding_profile(&conn);
    add_file_chunk(
        &conn,
        temp.path(),
        "ws1",
        "z.rs",
        "semantic-only content",
        1,
        1,
    );
    add_file_chunk(
        &conn,
        temp.path(),
        "ws1",
        "a.rs",
        "needle lexical content",
        1,
        1,
    );
    let embedding = 1.0f32.to_le_bytes();
    conn.execute(
        "INSERT INTO chunk_embeddings (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding)
         VALUES ('ws1', 'z.rs', 0, 'profile-1', 1, 'semantic-tie', ?1)",
        rusqlite::params![embedding.as_slice()],
    )
    .unwrap();
    
    let numeric_id: i64 = conn.query_row(
        "SELECT numeric_vector_id FROM chunk_embeddings WHERE workspace_id = 'ws1' AND relative_path = 'z.rs'",
        [],
        |row| row.get(0),
    ).unwrap();

    conn.execute(
        "INSERT INTO profile_applied_generations (workspace_id, profile_id, applied_generation) VALUES ('ws1', 'profile-1', 1)",
        [],
    ).unwrap();
    
    let manifest = workspace_index_mcp::index::vector::CacheManifest {
        workspace_id: "ws1".to_string(),
        profile_id: "profile-1".to_string(),
        dimension: 1,
        metric: "cosine".to_string(),
        generation: 1,
        outbox_count: 0,
        outbox_max_id: 0,
        count: 1,
        checksum: "dummy".to_string(),
    };
    let mut index = workspace_index_mcp::index::vector::VectorIndex::new(manifest).unwrap();
    index.add(numeric_id as u64, &[1.0]).unwrap();
    let base_path = temp.path().join("ws1_profile-1");
    index.save(&base_path).unwrap();

    let service = search_service(&conn, b"test_secret", &["ws1"], temp.path().to_path_buf());
    let request = || CodeSearchRequest { languages: vec![], path_globs: vec![],
        query: "needle".to_string(),
        query_vector: Some(vec![1.0]),
        workspace_ids: vec!["ws1".to_string()], 
        profile_id: Some("profile-1".to_string()),
        limit: 10,
        cursor: None,
    };

    for _ in 0..2 {
        let res = service.code_search(request());
        let result = res.into_data().unwrap();
        let paths: Vec<_> = result.items.iter().map(|item| item.relative_path.as_str()).collect();
        assert_eq!(paths, vec!["a.rs", "z.rs"]);
    }
}
#[test]
fn test_exhaustive_pagination_fetches_additional_batches() {
    let (conn, temp, _) = setup_db();
    for index in 0..60 {
        let relative_path = format!("chunk-{index:02}.rs");
        add_file_chunk(
            &conn,
            temp.path(),
            "ws1",
            &relative_path,
            "needle content",
            1,
            1,
        );
    }
    let secret = b"test_secret";
    let service = search_service(&conn, secret, &["ws1"], temp.path().to_path_buf());

    let mut total_results = 0;
    let mut cursor = None;
    
    for _ in 0..10 {
        let req = CodeSearchRequest { languages: vec![], path_globs: vec![],
            query: "needle".to_string(),
            query_vector: None,
            workspace_ids: vec!["ws1".to_string()], 
            profile_id: None, 
            limit: 10,
            cursor,
        };
        
        let result = service.code_search(req);
        let page = result.data().unwrap();
        total_results += page.items.len();
        
        if !page.truncated {
            break;
        }
        cursor = page.next_cursor.clone();
    }
    
    assert_eq!(total_results, 60);
}


#[test]
fn large_query_cursor_is_bounded_decodable_and_query_bound() {
    let (conn, root, _) = setup_db();
    for index in 0..2 {
        let relative_path = format!("large-query-{index}.rs");
        add_file_chunk(
            &conn,
            root.path(),
            "ws1",
            &relative_path,
            "needle content",
            1,
            1,
        );
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let query = format!("needle{}", " ".repeat(70 * 1024));
    let request = |query: String, cursor| CodeSearchRequest {
        query,
        query_vector: None,
        workspace_ids: vec!["ws1".to_owned()],
        profile_id: None,
        languages: vec![],
        path_globs: vec![],
        limit: 1,
        cursor,
    };

    let first = service.code_search(request(query.clone(), None)).into_data().unwrap();
    let cursor = first.next_cursor.expect("second matching candidate needs a cursor");
    assert!(cursor.len() <= 64 * 1024);

    let second = service
        .code_search(request(query.clone(), Some(cursor.clone())))
        .into_data()
        .unwrap();
    assert_eq!(second.items.len(), 1);

    let stale = service
        .code_search(request(format!("different{}", " ".repeat(70 * 1024)), Some(cursor)))
        .into_error()
        .unwrap();
    assert_eq!(stale.code, ErrorCode::StaleCursor);
}

#[test]
fn repeated_path_globs_match_any_path_and_bind_cursors() {
    let (conn, root, _) = setup_db();
    for relative_path in ["src/first.rs", "tests/second.rs", "docs/third.rs"] {
        let parent = root.path().join(relative_path).parent().unwrap().to_path_buf();
        fs::create_dir_all(parent).unwrap();
        add_file_chunk(&conn, root.path(), "ws1", relative_path, "fn target_marker() {}", 1, 1);
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let request = |path_globs, cursor| CodeSearchRequest {
        query: "target_marker".to_owned(),
        query_vector: None,
        workspace_ids: vec!["ws1".to_owned()],
        profile_id: None,
        languages: vec![],
        path_globs,
        limit: 1,
        cursor,
    };

    let first = service.code_search(request(
        vec!["src/**/*.rs".to_owned(), "tests/**/*.rs".to_owned()],
        None,
    )).into_data().unwrap();
    assert_eq!(first.items.len(), 1);
    assert!(first.truncated);
    let cursor = first.next_cursor.unwrap();

    let second = service.code_search(request(
        vec!["src/**/*.rs".to_owned(), "tests/**/*.rs".to_owned()],
        Some(cursor.clone()),
    )).into_data().unwrap();
    assert_eq!(second.items.len(), 1);
    assert_ne!(first.items[0].relative_path, second.items[0].relative_path);
    assert!(["src/first.rs", "tests/second.rs"].contains(&first.items[0].relative_path.as_str()));
    assert!(["src/first.rs", "tests/second.rs"].contains(&second.items[0].relative_path.as_str()));

    let stale = service.code_search(request(
        vec!["src/**/*.rs".to_owned()],
        Some(cursor),
    )).into_error().unwrap();
    assert_eq!(stale.code, ErrorCode::StaleCursor);
}
#[test]
fn code_and_symbol_search_reject_malformed_utf8_and_oversized_cursors() {
    let (conn, temp, _) = setup_db();
    let service = search_service(&conn, b"test_secret", &["ws1"], temp.path().to_path_buf());
    let cursors = [
        "aé000.00".to_owned(),
        "000é0.00".to_owned(),
        "00.00aé0".to_owned(),
        "0".repeat(1024 * 1024),
    ];

    for cursor in cursors {
        let code = service.code_search(CodeSearchRequest {
            languages: vec![],
            path_globs: vec![],
            query: "needle".into(),
            query_vector: None,
            workspace_ids: vec!["ws1".into()],
            profile_id: None,
            limit: 1,
            cursor: Some(cursor.clone()),
        });
        assert_eq!(code.into_error().unwrap().code, ErrorCode::StaleCursor);

        let symbol = service.symbol_search(SymbolSearchRequest {
            direction: None,
            query: "needle".into(),
            workspace_ids: vec!["ws1".into()],
            limit: 1,
            cursor: Some(cursor),
        });
        assert_eq!(symbol.into_error().unwrap().code, ErrorCode::StaleCursor);
    }
}
#[test]
fn symbol_pages_terminate_at_candidate_cap_without_repeating_a_cursor() {
    let (conn, root, _) = setup_db();
    for index in 0..1001 {
        let name = format!("needle_{index:04}");
        let relative_path = format!("symbols/{name}.rs");
        fs::create_dir_all(root.path().join("symbols")).unwrap();
        let content = format!("fn {name}() {{}}\n");
        add_file_chunk(&conn, root.path(), "ws1", &relative_path, &content, 1, 1);
        conn.execute(
            "INSERT INTO symbols
             (workspace_id,relative_path,symbol_index,name,kind,byte_start,byte_end,start_line,end_line)
             VALUES ('ws1',?1,0,?2,'function',0,?3,1,1)",
            rusqlite::params![relative_path, name, content.len() as i64],
        ).unwrap();
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let mut cursor = None;
    let mut previous_cursor: Option<String> = None;
    let mut names = std::collections::HashSet::new();
    let mut page_count = 0;
    let final_truncated = loop {
        let page = service.symbol_search(SymbolSearchRequest {
            query: "needle".into(),
            workspace_ids: vec!["ws1".into()],
            direction: None,
            limit: 100,
            cursor: cursor.clone(),
        }).into_data().unwrap();
        page_count += 1;
        for item in page.items {
            assert!(names.insert(item.symbol), "symbol result repeated across pages");
        }
        match page.next_cursor {
            Some(next) => {
                assert_ne!(Some(next.as_str()), previous_cursor.as_deref());
                previous_cursor = Some(next.clone());
                cursor = Some(next);
            }
            None => break page.truncated,
        }
        assert!(page_count <= 10, "pagination must terminate at the candidate cap");
    };
    assert_eq!(names.len(), 1000);
    assert!(final_truncated);
    assert_eq!(page_count, 10);
}

#[test]
fn symbol_cursor_uses_a_frozen_ranked_result_set() {
    let (conn, root, _) = setup_db();
    for name in ["needle_a", "needle_b"] {
        let relative_path = format!("{name}.rs");
        let content = format!("fn {name}() {{}}\n");
        add_file_chunk(&conn, root.path(), "ws1", &relative_path, &content, 1, 1);
        conn.execute(
            "INSERT INTO symbols
             (workspace_id,relative_path,symbol_index,name,kind,byte_start,byte_end,start_line,end_line)
             VALUES ('ws1',?1,0,?2,'function',0,?3,1,1)",
            rusqlite::params![relative_path, name, content.len() as i64],
        ).unwrap();
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let first = service.symbol_search(SymbolSearchRequest {
        query: "needle".into(), workspace_ids: vec!["ws1".into()],
        direction: None, limit: 1, cursor: None,
    }).into_data().unwrap();
    assert_eq!(first.items[0].symbol, "needle_a");
    let cursor = first.next_cursor.expect("symbol page cursor");

    let content = "fn needle_0() {}\n";
    add_file_chunk(&conn, root.path(), "ws1", "needle_0.rs", content, 1, 1);
    conn.execute(
        "INSERT INTO symbols
         (workspace_id,relative_path,symbol_index,name,kind,byte_start,byte_end,start_line,end_line)
         VALUES ('ws1','needle_0.rs',0,'needle_0','function',0,?1,1,1)",
        [content.len() as i64],
    ).unwrap();
    let second = service.symbol_search(SymbolSearchRequest {
        query: "needle".into(), workspace_ids: vec!["ws1".into()],
        direction: None, limit: 1, cursor: Some(cursor),
    }).into_data().unwrap();
    assert_eq!(second.items[0].symbol, "needle_b");
    assert!(second.next_cursor.is_none());
    assert!(!second.truncated);
}

#[test]
fn language_and_path_filters_apply_before_channel_candidate_caps() {
    let (conn, root, _) = setup_db();
    fs::create_dir_all(root.path().join("outside")).unwrap();
    for index in 0..1001 {
        let relative_path = format!("outside/{index:04}.rs");
        add_file_chunk(
            &conn,
            root.path(),
            "ws1",
            &relative_path,
            "fn target_marker() {}",
            1,
            1,
        );
    }
    fs::create_dir_all(root.path().join("wanted")).unwrap();
    add_file_chunk(
        &conn,
        root.path(),
        "ws1",
        "wanted/target.rs",
        "fn target_marker() {}",
        1,
        1,
    );
    conn.execute(
        "UPDATE chunks SET language='rust' WHERE workspace_id='ws1'",
        [],
    ).unwrap();
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let result = service.code_search(CodeSearchRequest {
        query: "target_marker".into(),
        query_vector: None,
        workspace_ids: vec!["ws1".into()],
        profile_id: None,
        languages: vec!["rust".into()],
        path_globs: vec!["wanted/**/*.rs".into()],
        limit: 10,
        cursor: None,
    }).into_data().unwrap();
    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].relative_path, "wanted/target.rs");
}
#[test]
fn code_cursor_binds_filters_semantic_inputs_and_limit() {
    let (conn, root, _) = setup_db();
    for path in ["needle_a.rs", "needle_b.rs"] {
        add_file_chunk(&conn, root.path(), "ws1", path, "fn needle() {}", 1, 1);
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let request = |cursor, languages: Vec<String>, path_globs: Vec<String>, query_vector, profile_id, limit| {
        CodeSearchRequest {
            query: "needle".into(),
            query_vector,
            workspace_ids: vec!["ws1".into()],
            profile_id,
            languages,
            path_globs,
            limit,
            cursor,
        }
    };
    let first = service.code_search(request(None, vec![], vec![], Some(vec![0.25]), None, 1))
        .into_data().unwrap();
    let cursor = first.next_cursor.expect("second match needs a cursor");

    for changed in [
        request(Some(cursor.clone()), vec!["rust".into()], vec![], Some(vec![0.25]), None, 1),
        request(Some(cursor.clone()), vec![], vec!["src/**/*.rs".into()], Some(vec![0.25]), None, 1),
        request(Some(cursor.clone()), vec![], vec![], Some(vec![0.5]), None, 1),
        request(Some(cursor.clone()), vec![], vec![], Some(vec![0.25, 0.5]), None, 1),
        request(Some(cursor.clone()), vec![], vec![], Some(vec![0.25]), Some("profile-1".into()), 1),
        request(Some(cursor.clone()), vec![], vec![], Some(vec![0.25]), None, 2),
    ] {
        assert_eq!(
            service.code_search(changed).into_error().unwrap().code,
            ErrorCode::StaleCursor
        );
    }

    let continuation = service.code_search(request(Some(cursor), vec![], vec![], Some(vec![0.25]), None, 1))
        .into_data().unwrap();
    assert_eq!(continuation.items.len(), 1);
}

#[test]
fn symbol_cursor_binds_effective_limit() {
    let (conn, root, _) = setup_db();
    for name in ["needle_a", "needle_b"] {
        let path = format!("{name}.rs");
        add_file_chunk(&conn, root.path(), "ws1", &path, "needle", 1, 1);
        add_symbol_definition(&conn, "ws1", &path, name, 1);
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let first = service.symbol_search(SymbolSearchRequest {
        direction: None,
        query: "needle".into(),
        workspace_ids: vec!["ws1".into()],
        limit: 1,
        cursor: None,
    }).into_data().unwrap();
    let cursor = first.next_cursor.expect("second symbol needs a cursor");
    let stale = service.symbol_search(SymbolSearchRequest {
        direction: None,
        query: "needle".into(),
        workspace_ids: vec!["ws1".into()],
        limit: 2,
        cursor: Some(cursor.clone()),
    });
    assert_eq!(stale.into_error().unwrap().code, ErrorCode::StaleCursor);

    let continuation = service.symbol_search(SymbolSearchRequest {
        direction: None,
        query: "needle".into(),
        workspace_ids: vec!["ws1".into()],
        limit: 1,
        cursor: Some(cursor),
    }).into_data().unwrap();
    assert_eq!(continuation.items.len(), 1);
}

#[test]
fn code_cursor_uses_a_frozen_ranked_result_set() {
    let (conn, root, _) = setup_db();
    for name in ["needle_a", "needle_b"] {
        let relative_path = format!("{name}.rs");
        let content = format!("fn {name}() {{}}\n");
        add_file_chunk(&conn, root.path(), "ws1", &relative_path, &content, 1, 1);
        conn.execute(
            "INSERT INTO symbols
             (workspace_id,relative_path,symbol_index,name,kind,byte_start,byte_end,start_line,end_line)
             VALUES ('ws1',?1,0,?2,'function',0,?3,1,1)",
            rusqlite::params![relative_path, name, content.len() as i64],
        ).unwrap();
    }
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let request = |cursor| CodeSearchRequest {
        query: "needle".into(),
        query_vector: None,
        workspace_ids: vec!["ws1".into()],
        profile_id: None,
        languages: Vec::new(),
        path_globs: Vec::new(),
        limit: 1,
        cursor,
    };
    let first = service.code_search(request(None)).into_data().unwrap();
    let first_path = first.items[0].relative_path.clone();
    let cursor = first.next_cursor.expect("code search cursor");

    let content = "fn needle_0() {}\n";
    add_file_chunk(&conn, root.path(), "ws1", "needle_0.rs", content, 1, 1);
    conn.execute(
        "INSERT INTO symbols
         (workspace_id,relative_path,symbol_index,name,kind,byte_start,byte_end,start_line,end_line)
         VALUES ('ws1','needle_0.rs',0,'needle_0','function',0,?1,1,1)",
        [content.len() as i64],
    ).unwrap();
    let second = service.code_search(request(Some(cursor))).into_data().unwrap();
    assert_ne!(second.items[0].relative_path, first_path);
    assert_ne!(second.items[0].relative_path, "needle_0.rs");
}
#[test]
fn exact_symbol_match_outweighs_a_prefix_definition() {
    let (conn, root, _) = setup_db();
    let prefix_content = "// needle fn needleExtra() {}";
    let exact_content = "fn needle() {} // filler alpha beta gamma delta epsilon zeta";
    add_file_chunk(&conn, root.path(), "ws1", "a-prefix.rs", prefix_content, 1, 1);
    add_file_chunk(&conn, root.path(), "ws1", "z-exact.rs", exact_content, 1, 1);
    add_symbol_definition(&conn, "ws1", "a-prefix.rs", "needleExtra", 1);
    add_symbol_definition(&conn, "ws1", "z-exact.rs", "needle", 1);

    let result = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf())
        .code_search(CodeSearchRequest {
            query: "needle".into(),
            query_vector: None,
            workspace_ids: vec!["ws1".into()],
            profile_id: None,
            languages: vec![],
            path_globs: vec![],
            limit: 10,
            cursor: None,
        })
        .into_data()
        .unwrap();

    assert_eq!(
        result.items.iter().map(|item| item.relative_path.as_str()).collect::<Vec<_>>(),
        vec!["z-exact.rs", "a-prefix.rs"]
    );
}

#[test]
fn path_boost_requires_a_matching_path_token() {
    let (conn, root, _) = setup_db();
    for path in ["aaa.rs", "catalogue.rs", "log.rs"] {
        add_file_chunk(&conn, root.path(), "ws1", path, "fn log() {}\n", 1, 1);
    }
    let result = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf())
        .code_search(CodeSearchRequest {
            query: "log".into(),
            query_vector: None,
            workspace_ids: vec!["ws1".into()],
            profile_id: None,
            languages: vec![],
            path_globs: vec![],
            limit: 10,
            cursor: None,
        })
        .into_data()
        .unwrap();

    assert_eq!(
        result.items.iter().map(|item| item.relative_path.as_str()).collect::<Vec<_>>(),
        vec!["log.rs", "aaa.rs", "catalogue.rs"]
    );
}

#[test]
fn one_hop_dependency_boost_changes_fused_code_ranking() {
    let (conn, root, _) = setup_db();
    for path in ["aaa.rs", "zzz.rs"] {
        add_file_chunk(&conn, root.path(), "ws1", path, "fn needle() {}\n", 1, 1);
    }
    add_dependency_edge(&conn, "ws1", "zzz.rs", "needle");
    let result = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf())
        .code_search(CodeSearchRequest {
            query: "needle".into(),
            query_vector: None,
            workspace_ids: vec!["ws1".into()],
            profile_id: None,
            languages: vec![],
            path_globs: vec![],
            limit: 10,
            cursor: None,
        })
        .into_data()
        .unwrap();

    assert_eq!(result.items[0].relative_path, "zzz.rs");
    assert_eq!(result.items[1].relative_path, "aaa.rs");
}

#[test]
fn definitions_rank_above_call_candidates() {
    let (conn, root, _) = setup_db();
    add_file_chunk(&conn, root.path(), "ws1", "a-call.rs", "fn caller() { needle(); }\n", 1, 1);
    add_file_chunk(
        &conn,
        root.path(),
        "ws1",
        "z-definition.rs",
        "fn needle() {} // filler alpha beta gamma delta epsilon\n",
        1,
        1,
    );
    add_symbol_definition(&conn, "ws1", "z-definition.rs", "needle", 1);
    add_dependency_edge(&conn, "ws1", "a-call.rs", "needle");
    let result = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf())
        .code_search(CodeSearchRequest {
            query: "needle".into(),
            query_vector: None,
            workspace_ids: vec!["ws1".into()],
            profile_id: None,
            languages: vec![],
            path_globs: vec![],
            limit: 10,
            cursor: None,
        })
        .into_data()
        .unwrap();

    assert_eq!(result.items[0].relative_path, "z-definition.rs");
    assert_eq!(result.items[1].relative_path, "a-call.rs");
}

#[test]
fn code_candidate_ties_follow_workspace_path_start_line_and_chunk_id() {
    let (conn, root, root2) = setup_db();
    let multi_line_source = "needle first\nneedle second\n";
    fs::write(root.path().join("a.rs"), multi_line_source).unwrap();
    let content_hash = blake3::hash(multi_line_source.as_bytes()).to_hex().to_string();
    conn.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation, file_generation)
         VALUES ('ws1', 'a.rs', ?1, ?2, 1, 1)",
        rusqlite::params![content_hash, multi_line_source.len() as i64],
    )
    .unwrap();
    for (chunk_index, start_line, end_line) in [(4, 1, 1), (2, 1, 2), (0, 2, 2)] {
        conn.execute(
            "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content)
             VALUES ('ws1', 'a.rs', ?1, ?2, ?3, 'needle')",
            rusqlite::params![chunk_index, start_line, end_line],
        )
        .unwrap();
    }
    add_file_chunk(&conn, root.path(), "ws1", "z.rs", "needle\n", 1, 1);
    add_file_chunk(&conn, root2.path(), "ws2", "a.rs", "needle\n", 1, 1);
    let service = search_service(
        &conn,
        b"test_secret",
        &["ws1", "ws2"],
        root.path().to_path_buf(),
    );
    let request = CodeSearchRequest {
        query: "needle".into(),
        query_vector: None,
        workspace_ids: vec!["ws1".into(), "ws2".into()],
        profile_id: None,
        languages: vec![],
        path_globs: vec![],
        limit: 10,
        cursor: None,
    };

    let result = service.code_search(request.clone()).into_data().unwrap();
    let ordered: Vec<_> = result.items.iter()
        .map(|item| (item.workspace_id.as_str(), item.relative_path.as_str(), item.start_line))
        .collect();
    assert_eq!(
        ordered,
        vec![
            ("ws1", "a.rs", 1),
            ("ws1", "a.rs", 1),
            ("ws1", "a.rs", 2),
            ("ws1", "z.rs", 1),
            ("ws2", "a.rs", 1),
        ]
    );
    assert_eq!(result.items[0].snippet, "needle first\nneedle second\n");
    assert_eq!(result.items[1].snippet, "needle first\n");

    let repeated = service.code_search(request).into_data().unwrap();
    assert_eq!(
        repeated.items.iter().map(|item| (
            item.workspace_id.as_str(),
            item.relative_path.as_str(),
            item.start_line,
            item.snippet.as_str(),
        )).collect::<Vec<_>>(),
        result.items.iter().map(|item| (
            item.workspace_id.as_str(),
            item.relative_path.as_str(),
            item.start_line,
            item.snippet.as_str(),
        )).collect::<Vec<_>>()
    );
}

#[test]
fn symbol_search_rejects_unrecognized_direction() {
    let (conn, root, _) = setup_db();
    let service = search_service(&conn, b"test_secret", &["ws1"], root.path().to_path_buf());
    let result = service.symbol_search(SymbolSearchRequest {
        query: "needle".into(),
        workspace_ids: vec!["ws1".into()],
        direction: Some("sideways".into()),
        limit: 10,
        cursor: None,
    });

    assert_eq!(result.into_error().unwrap().code, ErrorCode::InvalidArgument);
}

fn add_symbol_definition(
    connection: &Connection,
    workspace_id: &str,
    relative_path: &str,
    name: &str,
    line: i64,
) {
    connection.execute(
        "INSERT INTO symbols (workspace_id, relative_path, symbol_index, name, kind, byte_start, byte_end, start_line, end_line)
         VALUES (?1, ?2, 0, ?3, 'function', 0, 1, ?4, ?4)",
        rusqlite::params![workspace_id, relative_path, name, line],
    ).unwrap();
}

fn add_dependency_edge(
    connection: &Connection,
    workspace_id: &str,
    relative_path: &str,
    name: &str,
) {
    connection.execute(
        "INSERT INTO dependency_edges (workspace_id, relative_path, edge_index, evidence, name, source_symbol, byte_start, byte_end, start_line, end_line)
         VALUES (?1, ?2, 0, 'call_candidate', ?3, NULL, 0, 1, 1, 1)",
        rusqlite::params![workspace_id, relative_path, name],
    ).unwrap();
}

#[test]
fn semantic_candidates_keep_hnsw_order_across_batched_mapping_and_pages() {
    const CANDIDATE_COUNT: usize = 501;

    let (conn, root, _) = setup_db();
    conn.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, model, dimension, metric, state, created_at)
         VALUES ('profile-1', 'test', 'test-model', 2, 'cosine', 'active', 0)",
        [],
    ).unwrap();
    conn.execute(
        "UPDATE workspace_index_state
         SET desired_profile_id='profile-1', active_profile_id='profile-1'
         WHERE workspace_id='ws1'",
        [],
    ).unwrap();

    let manifest = workspace_index_mcp::index::vector::CacheManifest {
        workspace_id: "ws1".to_owned(),
        profile_id: "profile-1".to_owned(),
        dimension: 2,
        metric: "cosine".to_owned(),
        generation: 1,
        outbox_count: 0,
        outbox_max_id: 0,
        count: CANDIDATE_COUNT as u64,
        checksum: "semantic-batch-regression".to_owned(),
    };
    let mut index = workspace_index_mcp::index::vector::VectorIndex::new(manifest).unwrap();
    let mut expected_paths = Vec::with_capacity(CANDIDATE_COUNT);

    for vector_rank in 0..CANDIDATE_COUNT {
        let relative_path = format!("candidate-{:03}.rs", CANDIDATE_COUNT - vector_rank - 1);
        add_file_chunk(
            &conn,
            root.path(),
            "ws1",
            &relative_path,
            &format!("semantic payload {vector_rank}"),
            1,
            1,
        );
        conn.execute(
            "UPDATE chunks SET language=?3 WHERE workspace_id=?1 AND relative_path=?2",
            rusqlite::params![
                "ws1",
                relative_path,
                if vector_rank % 2 == 0 { "rust" } else { "python" },
            ],
        )
        .unwrap();

        let vector = [1.0_f32, vector_rank as f32 * 0.01];
        let embedding = vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        conn.execute(
            "INSERT INTO chunk_embeddings
                (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding)
             VALUES ('ws1', ?1, 0, 'profile-1', 1, ?2, ?3)",
            rusqlite::params![
                relative_path,
                format!("semantic-vector-{vector_rank}"),
                embedding,
            ],
        )
        .unwrap();

        let numeric_id: i64 = conn
            .query_row(
                "SELECT numeric_vector_id FROM chunk_embeddings
                 WHERE workspace_id='ws1' AND relative_path=?1 AND profile_id='profile-1'",
                [&relative_path],
                |row| row.get(0),
            )
            .unwrap();
        index.add(numeric_id as u64, &vector).unwrap();
        if vector_rank % 2 == 0 && vector_rank >= 401 {
            expected_paths.push(relative_path);
        }
    }

    conn.execute(
        "INSERT INTO profile_applied_generations
            (workspace_id, profile_id, applied_generation)
         VALUES ('ws1', 'profile-1', 1)",
        [],
    )
    .unwrap();
    let vectors_dir = root.path().to_path_buf();
    index.save(&vectors_dir.join("ws1_profile-1")).unwrap();

    let service = search_service(&conn, b"test_secret", &["ws1"], vectors_dir);
    let mut cursor = None;
    let mut actual_paths = Vec::with_capacity(CANDIDATE_COUNT);
    let mut expected_semantic_rank = 1;

    loop {
        let page = service
            .code_search(CodeSearchRequest {
                query: "unmatchedneedle".to_owned(),
                query_vector: Some(vec![1.0, 0.0]),
                workspace_ids: vec!["ws1".to_owned()],
                profile_id: Some("profile-1".to_owned()),
                languages: vec!["rust".to_owned()],
                path_globs: vec!["candidate-0??.rs".to_owned()],
                limit: 10,
                cursor: cursor.clone(),
            })
            .into_data()
            .unwrap();

        for item in page.items {
            assert_eq!(
                item.component_ranks.get("semantic"),
                Some(&expected_semantic_rank)
            );
            actual_paths.push(item.relative_path);
            expected_semantic_rank += 1;
        }

        if !page.truncated {
            break;
        }
        cursor = page.next_cursor;
        assert!(cursor.is_some(), "a truncated page must provide a cursor");
    }

    assert_eq!(actual_paths, expected_paths);
}
fn assert_invalid_semantic_mapping_is_degraded(
    corrupt_mapping: fn(&Connection, &Path, i64),
    include_zero_hnsw_id: bool,
) {
    let (conn, root, _) = setup_db();
    add_embedding_profile(&conn);
    add_file_chunk(
        &conn,
        root.path(),
        "ws1",
        "semantic.rs",
        "semantic-only payload",
        1,
        1,
    );
    conn.execute(
        "INSERT INTO chunk_embeddings
            (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding)
         VALUES ('ws1', 'semantic.rs', 0, 'profile-1', 1, 'semantic-vector', X'0000803F00000000')",
        [],
    )
    .unwrap();
    let numeric_id: i64 = conn
        .query_row(
            "SELECT numeric_vector_id FROM chunk_embeddings WHERE vector_id='semantic-vector'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    corrupt_mapping(&conn, root.path(), numeric_id);
    conn.execute(
        "INSERT INTO profile_applied_generations
            (workspace_id, profile_id, applied_generation)
         VALUES ('ws1', 'profile-1', 1)",
        [],
    )
    .unwrap();

    let manifest = workspace_index_mcp::index::vector::CacheManifest {
        workspace_id: "ws1".to_owned(),
        profile_id: "profile-1".to_owned(),
        dimension: 2,
        metric: "cosine".to_owned(),
        generation: 1,
        outbox_count: 0,
        outbox_max_id: 0,
        count: 1,
        checksum: "invalid-semantic-mapping".to_owned(),
    };
    let mut index = workspace_index_mcp::index::vector::VectorIndex::new(manifest).unwrap();
    index.add(numeric_id as u64, &[1.0, 0.0]).unwrap();
    if include_zero_hnsw_id {
        index.add(0, &[0.9, 0.1]).unwrap();
    }
    let vectors_dir = root.path().to_path_buf();
    index.save(&vectors_dir.join("ws1_profile-1")).unwrap();
    let service = search_service(&conn, b"test_secret", &["ws1"], vectors_dir);

    let result = service.code_search(CodeSearchRequest {
        query: "no-lexical-match".to_owned(),
        query_vector: Some(vec![1.0, 0.0]),
        workspace_ids: vec!["ws1".to_owned()],
        profile_id: Some("profile-1".to_owned()),
        languages: vec![],
        path_globs: vec![],
        limit: 10,
        cursor: None,
    });
    assert!(result.data().unwrap().items.is_empty());
    assert!(result
        .warnings()
        .iter()
        .any(|warning| warning.code == "INDEX_DEGRADED"));
}

#[test]
fn duplicate_semantic_database_mapping_is_degraded() {
    assert_invalid_semantic_mapping_is_degraded(|connection, root, numeric_id| {
        add_file_chunk(
            connection,
            root,
            "ws1",
            "duplicate.rs",
            "duplicate payload",
            1,
            1,
        );
        connection.execute(
            "INSERT INTO chunk_embeddings
                (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding)
             VALUES ('ws1', 'duplicate.rs', 0, 'profile-1', 1, 'duplicate-vector', X'0000803F00000000')",
            [],
        ).unwrap();
        connection.execute(
            "UPDATE chunk_embeddings SET numeric_vector_id=?1 WHERE vector_id='duplicate-vector'",
            [numeric_id],
        ).unwrap();
    }, false);
}

#[test]
fn invalid_semantic_metadata_conversion_is_degraded() {
    assert_invalid_semantic_mapping_is_degraded(|connection, _root, _numeric_id| {
        connection.execute(
            "UPDATE chunks SET start_line='invalid'
             WHERE workspace_id='ws1' AND relative_path='semantic.rs'",
            [],
        ).unwrap();
    }, false);
}

#[test]
fn nonpositive_hnsw_id_degrades_semantic_mapping() {
    assert_invalid_semantic_mapping_is_degraded(|_, _, _| {}, true);
}
