use rusqlite::{params, Connection};
use tempfile::tempdir;
use workspace_index_mcp::config::Config;
use workspace_index_mcp::index::sequencer::{SequencerError, VectorWatermark};
use workspace_index_mcp::index::{
    CacheManifest, ProfileManager, Sequencer, VectorError, VectorIndex,
};

const WORKSPACE_ID: &str = "ws1";
const PROFILE_ID: &str = "prof1";
const PROFILE_DIMENSION: i64 = 2;

#[test]
fn test_remote_embedder_respects_allow_remote() {
    let mut config = Config::default();
    config.embedding.provider = "ollama".to_string();
    config.embedding.allow_remote = false;
    assert!(!config.embedding.allow_remote);
}

fn setup_connection(visible_generation: i64) -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    workspace_index_mcp::state::schema::initialize(&mut conn).unwrap();
    conn.execute(
        "INSERT INTO workspaces
            (id, root_path, device_id, inode, mode, registration_epoch, mode_epoch, created_at, updated_at)
         VALUES (?1, '/tmp', 0, 0, 'ReadOnly', 1, 1, 0, 0)",
        [WORKSPACE_ID],
    ).unwrap();
    insert_profile(&conn, PROFILE_ID, "active", PROFILE_DIMENSION);
    conn.execute(
        "INSERT INTO workspace_index_state
            (workspace_id, generation, updated_at, building_generation, visible_generation,
             dirty, health, desired_profile_id, active_profile_id)
         VALUES (?1, ?2, 0, ?2, ?2, 0, 'ready', ?3, ?3)",
        params![WORKSPACE_ID, visible_generation, PROFILE_ID],
    ).unwrap();
    conn.execute(
        "INSERT INTO files
            (workspace_id, relative_path, content_hash, size_bytes, visible_generation,
             file_generation, parser_version, chunker_version)
         VALUES (?1, 'src/lib.rs', 'hash', 8, ?2, ?2, 1, 1)",
        params![WORKSPACE_ID, visible_generation],
    ).unwrap();
    conn.execute(
        "INSERT INTO chunks
            (workspace_id, relative_path, chunk_index, start_line, end_line, content, language, chunk_hash)
         VALUES (?1, 'src/lib.rs', 0, 1, 1, 'fn f() {}', 'rust', 'chunk-hash')",
        [WORKSPACE_ID],
    ).unwrap();
    conn
}

fn insert_profile(conn: &Connection, id: &str, state: &str, dimension: i64) {
    conn.execute(
        "INSERT INTO embedding_profiles
            (id, provider_kind, model, dimension, metric, state, created_at)
         VALUES (?1, 'fastembed', 'test-model', ?2, 'cosine', ?3, 0)",
        params![id, dimension, state],
    ).unwrap();
}

fn set_applied_generation(conn: &Connection, profile_id: &str, generation: i64, backfill: bool) {
    conn.execute(
        "INSERT INTO profile_applied_generations
            (workspace_id, profile_id, applied_generation, backfill_required, updated_at)
         VALUES (?1, ?2, ?3, ?4, 0)",
        params![WORKSPACE_ID, profile_id, generation, backfill],
    ).unwrap();
}

fn insert_embedding(
    conn: &Connection,
    profile_id: &str,
    generation: i64,
    embedding: &[u8],
    operation_generation: Option<i64>,
    operation_embedding: Option<&[u8]>,
) {
    let vector_id = format!("src/lib.rs#{profile_id}");
    conn.execute(
        "INSERT INTO chunk_embeddings
            (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding)
         VALUES (?1, 'src/lib.rs', 0, ?2, ?3, ?4, ?5)",
        params![WORKSPACE_ID, profile_id, generation, vector_id, embedding],
    ).unwrap();
    if let Some(operation_generation) = operation_generation {
        conn.execute(
            "INSERT INTO vector_ops
                (workspace_id, profile_id, generation, op_type, vector_id, embedding)
             VALUES (?1, ?2, ?3, 'add', ?4, ?5)",
            params![WORKSPACE_ID, profile_id, operation_generation, vector_id, operation_embedding],
        ).unwrap();
    }
}

fn vector_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn make_sequencer(vectors_dir: &std::path::Path) -> Sequencer {
    Sequencer::new(
        WORKSPACE_ID.to_owned(),
        PROFILE_ID.to_owned(),
        vectors_dir.to_path_buf(),
        PROFILE_DIMENSION as usize,
        "cosine".to_owned(),
    )
}

fn verified_watermark(
    conn: &mut Connection,
    profile_id: &str,
) -> (tempfile::TempDir, VectorWatermark) {
    let cache_dir = tempdir().unwrap();
    let sequencer = Sequencer::new(
        WORKSPACE_ID.to_owned(),
        profile_id.to_owned(),
        cache_dir.path().to_path_buf(),
        PROFILE_DIMENSION as usize,
        "cosine".to_owned(),
    );
    let watermark = sequencer.sync(conn).unwrap();
    (cache_dir, watermark)
}

fn applied_generation(conn: &Connection, profile_id: &str) -> i64 {
    conn.query_row(
        "SELECT applied_generation FROM profile_applied_generations
         WHERE workspace_id=?1 AND profile_id=?2",
        params![WORKSPACE_ID, profile_id],
        |row| row.get(0),
    ).unwrap()
}

#[test]
fn sequencer_replays_little_endian_vectors_and_carries_forward_valid_no_op_generations() {
    let dir = tempdir().unwrap();
    let mut conn = setup_connection(1);
    let bytes = vector_bytes(&[1.0, -2.5]);
    insert_embedding(&conn, PROFILE_ID, 1, &bytes, Some(1), Some(&bytes));
    set_applied_generation(&conn, PROFILE_ID, 0, false);
    let sequencer = make_sequencer(dir.path());

    assert_eq!(sequencer.sync(&mut conn).unwrap().generation(), 1);
    assert_eq!(applied_generation(&conn, PROFILE_ID), 1);
    let numeric_id: i64 = conn.query_row(
        "SELECT numeric_vector_id FROM chunk_embeddings WHERE workspace_id=?1 AND profile_id=?2",
        params![WORKSPACE_ID, PROFILE_ID],
        |row| row.get(0),
    ).unwrap();
    let generation_one = CacheManifest {
        workspace_id: WORKSPACE_ID.to_owned(),
        profile_id: PROFILE_ID.to_owned(),
        dimension: PROFILE_DIMENSION as usize,
        metric: "cosine".to_owned(),
        generation: 1,
        outbox_count: 0,
        outbox_max_id: 0,
        count: 0,
        checksum: String::new(),
    };
    let cache_base = dir.path().join(format!("{WORKSPACE_ID}_{PROFILE_ID}"));
    let first_snapshot = VectorIndex::load(&cache_base, &generation_one).unwrap();
    assert_eq!(first_snapshot.search(&[1.0, -2.5], 1).unwrap()[0].0, numeric_id as u64);

    conn.execute(
        "UPDATE workspace_index_state SET generation=2, building_generation=2, visible_generation=2
         WHERE workspace_id=?1",
        [WORKSPACE_ID],
    ).unwrap();
    conn.execute(
        "UPDATE files SET visible_generation=2, file_generation=2 WHERE workspace_id=?1",
        [WORKSPACE_ID],
    ).unwrap();
    conn.execute(
        "UPDATE chunk_embeddings SET generation=2 WHERE workspace_id=?1 AND profile_id=?2",
        params![WORKSPACE_ID, PROFILE_ID],
    ).unwrap();

    assert_eq!(sequencer.sync(&mut conn).unwrap().generation(), 2);
    assert_eq!(applied_generation(&conn, PROFILE_ID), 2);
    let generation_two = CacheManifest { generation: 2, ..generation_one };
    let carried_snapshot = VectorIndex::load(&cache_base, &generation_two).unwrap();
    assert_eq!(carried_snapshot.search(&[1.0, -2.5], 1).unwrap()[0].0, numeric_id as u64);
}

#[test]
fn sequencer_rebuilds_when_outbox_changes_without_a_generation_change() {
    let dir = tempdir().unwrap();
    let mut conn = setup_connection(1);
    let bytes = vector_bytes(&[1.0, 0.0]);
    insert_embedding(&conn, PROFILE_ID, 1, &bytes, Some(1), Some(&bytes));
    set_applied_generation(&conn, PROFILE_ID, 0, false);
    let sequencer = make_sequencer(dir.path());
    assert_eq!(sequencer.sync(&mut conn).unwrap().generation(), 1);

    conn.execute(
        "INSERT INTO vector_ops
            (workspace_id, profile_id, generation, op_type, vector_id, embedding)
         VALUES (?1, ?2, 1, 'add', 'src/lib.rs#prof1', ?3)",
        params![WORKSPACE_ID, PROFILE_ID, bytes],
    ).unwrap();
    assert_eq!(sequencer.sync(&mut conn).unwrap().generation(), 1);

    let cache_base = dir.path().join(format!("{WORKSPACE_ID}_{PROFILE_ID}"));
    let manifest_path = cache_base.with_extension("json");
    let manifest: CacheManifest =
        serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
    assert_eq!(manifest.outbox_count, 2);
    assert_eq!(manifest.outbox_max_id, 2);
    assert_eq!(applied_generation(&conn, PROFILE_ID), 1);
}

#[test]
fn sequencer_rejects_trailing_malformed_nan_and_infinite_outbox_vectors() {
    let valid = vector_bytes(&[1.0, 0.0]);
    let cases = [
        vec![0, 0, 0, 0, 0, 0, 0],
        [valid.as_slice(), &[7][..]].concat(),
        vector_bytes(&[f32::NAN, 0.0]),
        vector_bytes(&[f32::INFINITY, 0.0]),
    ];
    for bad_bytes in cases {
        let dir = tempdir().unwrap();
        let mut conn = setup_connection(1);
        insert_embedding(&conn, PROFILE_ID, 1, &valid, Some(1), Some(&bad_bytes));
        set_applied_generation(&conn, PROFILE_ID, 0, false);

        let result = make_sequencer(dir.path()).sync(&mut conn);
        assert!(matches!(result, Err(SequencerError::InvalidOperation { .. })));
        assert_eq!(applied_generation(&conn, PROFILE_ID), 0);
        assert!(!dir.path().join(format!("{WORKSPACE_ID}_{PROFILE_ID}.json")).exists());
    }
}

#[test]
fn sequencer_checks_stored_profile_dimension_before_vector_add() {
    let dir = tempdir().unwrap();
    let mut conn = setup_connection(1);
    let bytes = vector_bytes(&[1.0, 0.0]);
    insert_embedding(&conn, PROFILE_ID, 1, &bytes, Some(1), Some(&bytes));
    set_applied_generation(&conn, PROFILE_ID, 0, false);
    let wrong_dimension = Sequencer::new(
        WORKSPACE_ID.to_owned(),
        PROFILE_ID.to_owned(),
        dir.path().to_path_buf(),
        3,
        "cosine".to_owned(),
    );

    assert!(matches!(
        wrong_dimension.sync(&mut conn),
        Err(SequencerError::ProfileConfiguration { .. })
    ));
    assert_eq!(applied_generation(&conn, PROFILE_ID), 0);
}

#[test]
fn sequencer_does_not_advance_watermark_when_snapshot_save_fails() {
    let dir = tempdir().unwrap();
    let blocked_dir = dir.path().join("not-a-directory");
    std::fs::write(&blocked_dir, b"block snapshot directory").unwrap();
    let mut conn = setup_connection(1);
    let bytes = vector_bytes(&[1.0, 0.0]);
    insert_embedding(&conn, PROFILE_ID, 1, &bytes, Some(1), Some(&bytes));
    set_applied_generation(&conn, PROFILE_ID, 0, false);

    let result = make_sequencer(&blocked_dir).sync(&mut conn);
    assert!(matches!(result, Err(SequencerError::Vector(VectorError::Save(_)))));
    assert_eq!(applied_generation(&conn, PROFILE_ID), 0);
}

#[test]
fn sequencer_refuses_no_op_carry_forward_without_current_generation_coverage() {
    let dir = tempdir().unwrap();
    let mut conn = setup_connection(1);
    let bytes = vector_bytes(&[1.0, 0.0]);
    insert_embedding(&conn, PROFILE_ID, 1, &bytes, Some(1), Some(&bytes));
    set_applied_generation(&conn, PROFILE_ID, 1, false);
    conn.execute(
        "UPDATE workspace_index_state SET generation=2, building_generation=2, visible_generation=2
         WHERE workspace_id=?1",
        [WORKSPACE_ID],
    ).unwrap();
    conn.execute(
        "UPDATE files SET visible_generation=2, file_generation=2 WHERE workspace_id=?1",
        [WORKSPACE_ID],
    ).unwrap();

    assert!(matches!(
        make_sequencer(dir.path()).sync(&mut conn),
        Err(SequencerError::IncompleteCoverage { expected: 1, actual: 0 })
    ));
    assert_eq!(applied_generation(&conn, PROFILE_ID), 1);
}

fn prepare_building_profile(conn: &Connection, backfill: bool) {
    insert_profile(conn, "prof2", "building", PROFILE_DIMENSION);
    insert_profile(conn, "prof3", "building", PROFILE_DIMENSION);
    conn.execute(
        "UPDATE workspace_index_state SET desired_profile_id='prof2' WHERE workspace_id=?1",
        [WORKSPACE_ID],
    ).unwrap();
    set_applied_generation(conn, "prof2", 1, backfill);
    let bytes = vector_bytes(&[0.5, 1.0]);
    insert_embedding(conn, "prof2", 1, &bytes, None, None);
}

fn prepare_third_profile(conn: &Connection) {
    set_applied_generation(conn, "prof3", 1, false);
    let bytes = vector_bytes(&[0.25, 0.75]);
    insert_embedding(conn, "prof3", 1, &bytes, None, None);
    conn.execute(
        "UPDATE workspace_index_state SET desired_profile_id='prof3' WHERE workspace_id=?1",
        [WORKSPACE_ID],
    ).unwrap();
}

fn active_profile_id(conn: &Connection) -> String {
    conn.query_row(
        "SELECT active_profile_id FROM workspace_index_state WHERE workspace_id=?1",
        [WORKSPACE_ID],
        |row| row.get(0),
    ).unwrap()
}

#[test]
fn profile_promotion_requires_desired_profile_and_matching_graph_profile() {
    let mut wrong_desired = setup_connection(1);
    prepare_building_profile(&wrong_desired, false);
    let (_cache_dir, watermark) = verified_watermark(&mut wrong_desired, "prof2");
    let manager = ProfileManager::new(WORKSPACE_ID.to_owned());
    assert!(!manager.try_promote(&mut wrong_desired, "prof3", &watermark).unwrap());
    assert_eq!(active_profile_id(&wrong_desired), PROFILE_ID);

    let mut wrong_graph_profile = setup_connection(1);
    prepare_building_profile(&wrong_graph_profile, false);
    prepare_third_profile(&wrong_graph_profile);
    let (_cache_dir, watermark) = verified_watermark(&mut wrong_graph_profile, "prof2");
    assert!(!manager.try_promote(&mut wrong_graph_profile, "prof3", &watermark).unwrap());
    assert_eq!(active_profile_id(&wrong_graph_profile), PROFILE_ID);

    let mut complete = setup_connection(1);
    prepare_building_profile(&complete, false);
    let (_cache_dir, watermark) = verified_watermark(&mut complete, "prof2");
    assert!(manager.try_promote(&mut complete, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&complete), "prof2");
}

#[test]
fn profile_promotion_rejects_a_stale_graph_generation() {
    let mut conn = setup_connection(1);
    prepare_building_profile(&conn, false);
    let (_cache_dir, old_watermark) = verified_watermark(&mut conn, "prof2");
    conn.execute(
        "UPDATE workspace_index_state SET generation=2, building_generation=2, visible_generation=2
         WHERE workspace_id=?1",
        [WORKSPACE_ID],
    ).unwrap();
    conn.execute(
        "UPDATE files SET visible_generation=2, file_generation=2 WHERE workspace_id=?1",
        [WORKSPACE_ID],
    ).unwrap();
    conn.execute(
        "UPDATE chunk_embeddings SET generation=2 WHERE workspace_id=?1 AND profile_id='prof2'",
        [WORKSPACE_ID],
    ).unwrap();
    conn.execute(
        "UPDATE profile_applied_generations SET applied_generation=2
         WHERE workspace_id=?1 AND profile_id='prof2'",
        [WORKSPACE_ID],
    ).unwrap();

    let manager = ProfileManager::new(WORKSPACE_ID.to_owned());
    assert!(!manager.try_promote(&mut conn, "prof2", &old_watermark).unwrap());
    assert_eq!(active_profile_id(&conn), PROFILE_ID);
}

#[test]
fn profile_promotion_rejects_backfill_and_incomplete_generation_coverage() {
    let mut conn = setup_connection(1);
    prepare_building_profile(&conn, true);
    let (_cache_dir, watermark) = verified_watermark(&mut conn, "prof2");
    let manager = ProfileManager::new(WORKSPACE_ID.to_owned());
    assert!(!manager.try_promote(&mut conn, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&conn), PROFILE_ID);

    conn.execute(
        "UPDATE profile_applied_generations SET backfill_required=0
         WHERE workspace_id=?1 AND profile_id='prof2'",
        [WORKSPACE_ID],
    ).unwrap();
    conn.execute(
        "UPDATE chunk_embeddings SET generation=0
         WHERE workspace_id=?1 AND profile_id='prof2'",
        [WORKSPACE_ID],
    ).unwrap();
    assert!(!manager.try_promote(&mut conn, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&conn), PROFILE_ID);

    let non_finite = vector_bytes(&[f32::NAN, 0.0]);
    conn.execute(
        "UPDATE chunk_embeddings SET generation=1, embedding=?1
         WHERE workspace_id=?2 AND profile_id='prof2'",
        params![non_finite, WORKSPACE_ID],
    ).unwrap();
    assert!(!manager.try_promote(&mut conn, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&conn), PROFILE_ID);
}

#[test]
fn profile_promotion_rejects_missing_numeric_ids_and_infinite_vectors() {
    let manager = ProfileManager::new(WORKSPACE_ID.to_owned());
    let mut missing_id = setup_connection(1);
    prepare_building_profile(&missing_id, false);
    let (_cache_dir, watermark) = verified_watermark(&mut missing_id, "prof2");
    missing_id.execute(
        "UPDATE chunk_embeddings SET numeric_vector_id=NULL
         WHERE workspace_id=?1 AND profile_id='prof2'",
        [WORKSPACE_ID],
    ).unwrap();
    missing_id.execute(
        "DELETE FROM stable_vector_ids WHERE workspace_id=?1 AND profile_id='prof2'",
        [WORKSPACE_ID],
    ).unwrap();
    assert!(!manager.try_promote(&mut missing_id, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&missing_id), PROFILE_ID);

    let mut infinite = setup_connection(1);
    prepare_building_profile(&infinite, false);
    let (_cache_dir, watermark) = verified_watermark(&mut infinite, "prof2");
    let non_finite = vector_bytes(&[f32::INFINITY, 0.0]);
    infinite.execute(
        "UPDATE chunk_embeddings SET embedding=?1
         WHERE workspace_id=?2 AND profile_id='prof2'",
        params![non_finite, WORKSPACE_ID],
    ).unwrap();
    assert!(!manager.try_promote(&mut infinite, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&infinite), PROFILE_ID);
}

#[test]
fn profile_promotion_cas_race_rolls_back_candidate_activation() {
    let mut conn = setup_connection(1);
    prepare_building_profile(&conn, false);
    let (_cache_dir, watermark) = verified_watermark(&mut conn, "prof2");
    conn.execute_batch(
        "CREATE TRIGGER change_desired_during_promotion
         AFTER UPDATE OF active_profile_id ON workspace_index_state
         BEGIN
             UPDATE workspace_index_state SET desired_profile_id='prof3'
             WHERE workspace_id='ws1';
         END;",
    ).unwrap();
    let manager = ProfileManager::new(WORKSPACE_ID.to_owned());

    assert!(!manager.try_promote(&mut conn, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&conn), PROFILE_ID);
    let candidate_state: String = conn.query_row(
        "SELECT state FROM embedding_profiles WHERE id='prof2'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(candidate_state, "building");
}

#[test]
fn profile_promotion_rejects_changed_outbox_after_snapshot_creation() {
    let mut conn = setup_connection(1);
    prepare_building_profile(&conn, false);
    let (_cache_dir, watermark) = verified_watermark(&mut conn, "prof2");
    let bytes = vector_bytes(&[0.5, 1.0]);
    conn.execute(
        "INSERT INTO vector_ops
            (workspace_id, profile_id, generation, op_type, vector_id, embedding)
         VALUES (?1, 'prof2', 1, 'add', 'src/lib.rs#prof2', ?2)",
        params![WORKSPACE_ID, bytes],
    ).unwrap();

    let manager = ProfileManager::new(WORKSPACE_ID.to_owned());
    assert!(!manager.try_promote(&mut conn, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&conn), PROFILE_ID);
}

#[test]
fn profile_promotion_rejects_a_profile_that_is_no_longer_building() {
    let mut conn = setup_connection(1);
    prepare_building_profile(&conn, false);
    let (_cache_dir, watermark) = verified_watermark(&mut conn, "prof2");
    conn.execute(
        "UPDATE embedding_profiles SET state='failed' WHERE id='prof2'",
        [],
    ).unwrap();

    let manager = ProfileManager::new(WORKSPACE_ID.to_owned());
    assert!(!manager.try_promote(&mut conn, "prof2", &watermark).unwrap());
    assert_eq!(active_profile_id(&conn), PROFILE_ID);
}
