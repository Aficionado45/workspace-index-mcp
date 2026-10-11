use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use rusqlite::{Connection, params};
use tempfile::TempDir;
use workspace_index_mcp::index::vector::{CacheManifest, VectorIndex};
use workspace_index_mcp::retrieval::search::{CodeSearchRequest, SearchService, SearchServiceOptions};
use workspace_index_mcp::security::WorkspaceFs;
use workspace_index_mcp::state::schema;

const WORKSPACES: usize = 10;
const CHUNKS: usize = 250_000;
const DIMENSION: usize = 32;
const PROFILE_ID: &str = "synthetic-deterministic-v1";
const QUERY: &str = "synthetic retrieval benchmark needle";
const LIMIT: usize = 10;
const P95_TARGET_MS: f64 = 750.0;

fn main() -> Result<(), Box<dyn Error>> {
    let samples = sample_count()?;
    eprintln!(
        "[retrieval-bench] stage=configuration status=complete samples={samples} workspaces={WORKSPACES} chunks={CHUNKS} chunks_per_workspace={}",
        CHUNKS / WORKSPACES
    );

    let corpus_started = Instant::now();
    let corpus = Corpus::build()?;
    eprintln!(
        "[retrieval-bench] stage=corpus_setup status=complete elapsed_ms={:.3}",
        corpus_started.elapsed().as_secs_f64() * 1_000.0
    );

    let search = corpus.search_service();
    let warmups = 10;
    eprintln!("[retrieval-bench] stage=warmup status=start repeated_samples={warmups}");
    let warmup_started = Instant::now();
    let warm_hits = validate_search(&search, search_request())?;
    for _ in 1..warmups {
        std::hint::black_box(validate_search(&search, search_request())?);
    }
    eprintln!(
        "[retrieval-bench] stage=warmup status=complete elapsed_ms={:.3} hits={}",
        warmup_started.elapsed().as_secs_f64() * 1_000.0,
        warm_hits
    );

    eprintln!("[retrieval-bench] stage=query_samples status=start samples={samples}");
    let query_started = Instant::now();
    let progress_interval = samples.div_ceil(10);
    let mut timings = Vec::with_capacity(samples);
    for sample in 0..samples {
        timings.push(measure(&search, search_request())?);
        let completed = sample + 1;
        if completed % progress_interval == 0 || completed == samples {
            eprintln!(
                "[retrieval-bench] stage=query_samples status=progress completed={completed}/{samples} elapsed_ms={:.3}",
                query_started.elapsed().as_secs_f64() * 1_000.0
            );
        }
    }
    eprintln!(
        "[retrieval-bench] stage=query_samples status=complete elapsed_ms={:.3} samples={samples}",
        query_started.elapsed().as_secs_f64() * 1_000.0
    );
    report(samples, &timings);
    Ok(())
}

fn sample_count() -> Result<usize, Box<dyn Error>> {
    let count = env::args().nth(1).map_or(Ok(100), |value| value.parse())?;
    if count == 0 {
        return Err("sample count must be greater than zero".into());
    }
    Ok(count)
}


struct Corpus {
    connection: Connection,
    roots: Vec<(String, TempDir)>,
    vectors_dir: TempDir,
}

impl Corpus {
    fn build() -> Result<Self, Box<dyn Error>> {
        eprintln!("[retrieval-bench] stage=corpus_root_setup status=start workspaces={WORKSPACES}");
        let roots_started = Instant::now();
        let roots = create_roots()?;
        eprintln!(
            "[retrieval-bench] stage=corpus_root_setup status=complete elapsed_ms={:.3} workspaces={}",
            roots_started.elapsed().as_secs_f64() * 1_000.0,
            roots.len()
        );

        let metadata_started = Instant::now();
        let vectors_dir = TempDir::new()?;
        let mut connection = Connection::open_in_memory()?;
        schema::initialize(&mut connection)?;
        seed_schema(&connection, &roots)?;
        eprintln!(
            "[retrieval-bench] stage=sqlite_metadata_seed status=complete elapsed_ms={:.3} workspaces={}",
            metadata_started.elapsed().as_secs_f64() * 1_000.0,
            roots.len()
        );
        seed_records(&mut connection, &roots, vectors_dir.path())?;
        Ok(Self { connection, roots, vectors_dir })
    }

    fn search_service(&self) -> SearchService<'_> {
        let capabilities = self.roots.iter().filter_map(|(id, root)| {
            let (device, inode) = root_identity(root.path()).ok()?;
            let workspace_fs = WorkspaceFs::open_registered(root.path(), device, inode).ok()?;
            Some((id.clone(), workspace_fs))
        }).collect::<HashMap<_, _>>();
        SearchService::with_capabilities(
            &self.connection,
            b"benchmark-only-cursor-secret-32-bytes-minimum",
            SearchServiceOptions {
                capabilities,
                max_file_bytes: 1_048_576,
                max_results: 100,
                max_snippet_bytes: 16 * 1024,
                max_tool_bytes: 450 * 1024,
                vectors_dir: self.vectors_dir.path().to_path_buf(),
            },
        )
    }
}

const CHUNKS_PER_SOURCE_FILE: usize = 256;

fn create_roots() -> Result<Vec<(String, TempDir)>, Box<dyn Error>> {
    (0..WORKSPACES).map(|index| {
        let root = tempfile::Builder::new().prefix("workspace-index-bench-").tempdir()?;
        Ok((format!("ws{index:02}"), root))
    }).collect()
}

fn source_line(workspace_number: usize, chunk_index: usize) -> String {
    let terms = if chunk_index == 0 {
        QUERY
    } else {
        "corpus baseline record"
    };
    format!(
        "// {terms} workspace{workspace_number} component{} item{chunk_index:05}",
        chunk_index % 64
    )
}

fn seed_schema(connection: &Connection, roots: &[(String, TempDir)]) -> Result<(), Box<dyn Error>> {
    connection.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, model, revision, dimension, metric, state, created_at) VALUES (?1, 'synthetic', 'deterministic-hash-v1', 'seed-1', ?2, 'cosine', 'active', 1)",
        params![PROFILE_ID, DIMENSION as i64],
    )?;
    for (id, root) in roots {
        let (device, inode) = root_identity(root.path())?;
        connection.execute(
            "INSERT INTO workspaces (id, root_path, device_id, inode, mode, registration_epoch, mode_epoch, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, 'read_only', 1, 1, 1, 1)",
            params![id, root.path().to_string_lossy(), device as i64, inode as i64],
        )?;
        connection.execute(
            "INSERT INTO workspace_index_state (workspace_id, generation, building_generation, visible_generation, updated_at, active_profile_id) VALUES (?1, 1, 1, 1, 1, ?2)",
            params![id, PROFILE_ID],
        )?;
        connection.execute(
            "INSERT INTO profile_applied_generations (workspace_id, profile_id, applied_generation, updated_at) VALUES (?1, ?2, 1, 1)",
            params![id, PROFILE_ID],
        )?;
    }
    Ok(())
}

fn seed_records(
    connection: &mut Connection,
    roots: &[(String, TempDir)],
    vectors_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let chunks_per_workspace = CHUNKS / roots.len();
    let mut next_numeric_id = 1_u64;
    let mut first_numeric_ids = Vec::with_capacity(roots.len());

    eprintln!("[retrieval-bench] stage=sqlite_chunk_seed status=start chunks={CHUNKS}");
    let sqlite_started = Instant::now();
    let transaction = connection.transaction()?;
    for (workspace_number, (workspace_id, root)) in roots.iter().enumerate() {
        let workspace_started = Instant::now();
        first_numeric_ids.push(next_numeric_id);
        seed_workspace_records(
            &transaction,
            root.path(),
            workspace_id,
            workspace_number,
            chunks_per_workspace,
            &mut next_numeric_id,
        )?;
        eprintln!(
            "[retrieval-bench] stage=sqlite_workspace_seed status=complete workspace={workspace_id} chunks={chunks_per_workspace} elapsed_ms={:.3}",
            workspace_started.elapsed().as_secs_f64() * 1_000.0
        );
    }
    transaction.commit()?;
    eprintln!(
        "[retrieval-bench] stage=sqlite_chunk_seed status=complete elapsed_ms={:.3} chunks={CHUNKS} workspaces={}",
        sqlite_started.elapsed().as_secs_f64() * 1_000.0,
        roots.len()
    );

    eprintln!("[retrieval-bench] stage=hnsw_build status=start workspaces={}", roots.len());
    let hnsw_started = Instant::now();
    for (workspace_number, (workspace_id, _)) in roots.iter().enumerate() {
        build_workspace_index(
            workspace_id,
            workspace_number,
            chunks_per_workspace,
            first_numeric_ids[workspace_number],
            vectors_dir,
        )?;
    }
    eprintln!(
        "[retrieval-bench] stage=hnsw_build status=complete elapsed_ms={:.3} workspaces={}",
        hnsw_started.elapsed().as_secs_f64() * 1_000.0,
        roots.len()
    );
    Ok(())
}

fn seed_workspace_records(
    connection: &Connection,
    root: &Path,
    workspace_id: &str,
    workspace_number: usize,
    count: usize,
    next_numeric_id: &mut u64,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(root.join("synthetic"))?;
    for file_start in (0..count).step_by(CHUNKS_PER_SOURCE_FILE) {
        let file_end = (file_start + CHUNKS_PER_SOURCE_FILE).min(count);
        let file_index = file_start / CHUNKS_PER_SOURCE_FILE;
        let path = format!("synthetic/part-{file_index:04}.rs");
        let contents = (file_start..file_end)
            .map(|chunk_index| source_line(workspace_number, chunk_index))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";

        fs::write(root.join(&path), contents.as_bytes())?;
        let file_hash = blake3::hash(contents.as_bytes()).to_hex().to_string();
        connection.execute(
            "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation, file_generation) VALUES (?1, ?2, ?3, ?4, 1, 1)",
            params![workspace_id, path, file_hash, contents.len() as i64],
        )?;

        for chunk_index in file_start..file_end {
            let line = chunk_index - file_start + 1;
            insert_chunk(connection, workspace_id, &path, chunk_index, line, workspace_number)?;
            insert_embedding(connection, workspace_id, &path, chunk_index, workspace_number, *next_numeric_id)?;
            *next_numeric_id += 1;
        }
    }
    Ok(())
}

fn build_workspace_index(
    workspace_id: &str,
    workspace_number: usize,
    count: usize,
    first_numeric_id: u64,
    vectors_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let index = VectorIndex::new(CacheManifest {
        workspace_id: workspace_id.to_string(),
        profile_id: PROFILE_ID.into(),
        dimension: DIMENSION,
        metric: "cosine".into(),
        generation: 1,
        outbox_count: 0,
        outbox_max_id: 0,
        count: 0,
        checksum: String::new(),
    })?;

    eprintln!(
        "[retrieval-bench] stage=hnsw_workspace_build status=start workspace={workspace_id} vectors={count}"
    );
    let build_started = Instant::now();
    for chunk_index in 0..count {
        index.add(
            first_numeric_id + chunk_index as u64,
            &synthetic_embedding(workspace_number, chunk_index),
        )?;
    }
    eprintln!(
        "[retrieval-bench] stage=hnsw_workspace_build status=complete workspace={workspace_id} vectors={count} elapsed_ms={:.3}",
        build_started.elapsed().as_secs_f64() * 1_000.0
    );

    let mut index = index;
    let save_started = Instant::now();
    index.save(&vectors_dir.join(format!("{workspace_id}_{PROFILE_ID}")))?;
    eprintln!(
        "[retrieval-bench] stage=hnsw_workspace_save status=complete workspace={workspace_id} vectors={count} elapsed_ms={:.3}",
        save_started.elapsed().as_secs_f64() * 1_000.0
    );
    Ok(())
}

fn insert_chunk(
    connection: &Connection,
    workspace_id: &str,
    path: &str,
    chunk_index: usize,
    line: usize,
    workspace_number: usize,
) -> rusqlite::Result<()> {
    let content = source_line(workspace_number, chunk_index);
    connection.execute(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content, language, chunk_hash) VALUES (?1, ?2, ?3, ?4, ?4, ?5, 'rust', ?6)",
        params![workspace_id, path, chunk_index as i64, line as i64, content, blake3::hash(content.as_bytes()).to_hex().to_string()],
    )?;
    Ok(())
}

fn insert_embedding(
    connection: &Connection,
    workspace_id: &str,
    path: &str,
    chunk_index: usize,
    workspace_number: usize,
    numeric_id: u64,
) -> rusqlite::Result<()> {
    let vector_id = format!("synthetic-{workspace_id}-{chunk_index:05}");
    let vector = synthetic_embedding(workspace_number, chunk_index);
    let bytes = vector.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    connection.execute(
        "INSERT INTO chunk_embeddings (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding) VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6)",
        params![workspace_id, path, chunk_index as i64, PROFILE_ID, vector_id, bytes],
    )?;
    let assigned: i64 = connection.query_row(
        "SELECT id FROM stable_vector_ids WHERE workspace_id=?1 AND profile_id=?2 AND vector_id=?3",
        params![workspace_id, PROFILE_ID, vector_id],
        |row| row.get(0),
    )?;
    if assigned as u64 != numeric_id {
        return Err(rusqlite::Error::InvalidParameterName(format!(
            "unexpected synthetic vector ID: {assigned} != {numeric_id}"
        )));
    }
    Ok(())
}

fn synthetic_embedding(workspace: usize, item: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; DIMENSION];
    vector[(workspace * 3 + item % 13) % DIMENSION] = 1.0;
    vector[(item.wrapping_mul(17) + workspace) % DIMENSION] = 0.25;
    vector
}

fn search_request() -> CodeSearchRequest {
    CodeSearchRequest {
        query: QUERY.into(),
        query_vector: Some(synthetic_embedding(0, 0)),
        workspace_ids: (0..WORKSPACES).map(|index| format!("ws{index:02}")).collect(),
        profile_id: Some(PROFILE_ID.into()),
        languages: Vec::new(),
        path_globs: Vec::new(),
        limit: LIMIT,
        cursor: None,
    }
}

fn measure(
    search: &SearchService<'_>,
    request: CodeSearchRequest,
) -> Result<Duration, std::io::Error> {
    let started = Instant::now();
    std::hint::black_box(validate_search(search, request)?);
    Ok(started.elapsed())
}

fn validate_search(
    search: &SearchService<'_>,
    request: CodeSearchRequest,
) -> Result<usize, std::io::Error> {
    let result = search
        .code_search(request)
        .into_data()
        .ok_or_else(|| std::io::Error::other("synthetic benchmark search failed"))?;
    let hits = result.items.len();
    if hits != LIMIT {
        return Err(std::io::Error::other(format!(
            "expected {LIMIT} benchmark hits, received {hits}"
        )));
    }
    Ok(hits)
}

fn report(samples: usize, timings: &[Duration]) {
    let mut nanos = timings.iter().map(Duration::as_nanos).collect::<Vec<_>>();
    nanos.sort_unstable();
    let p50 = percentile_ms(&nanos, 0.50);
    let p95 = percentile_ms(&nanos, 0.95);
    println!("{}", serde_json::json!({
        "benchmark": "retrieval_synthetic",
        "provider": "synthetic",
        "model": "deterministic-hash-v1",
        "semantic_quality_claimed": false,
        "workspaces": WORKSPACES,
        "chunks": CHUNKS,
        "embedding_dimension": DIMENSION,
        "query": QUERY,
        "limit": LIMIT,
        "samples": samples,
        "warmup_samples": 10,
        "hardware": cpu_description(),
        "os": os_description(),
        "p50_ms": p50,
        "p95_ms": p95,
        "p95_target_ms": P95_TARGET_MS,
        "threshold_met": p95 <= P95_TARGET_MS
    }));
}

fn percentile_ms(sorted_nanos: &[u128], percentile: f64) -> f64 {
    let index = ((sorted_nanos.len() - 1) as f64 * percentile).ceil() as usize;
    sorted_nanos[index] as f64 / 1_000_000.0
}

fn cpu_description() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| info.lines().find(|line| line.starts_with("model name"))?.split_once(':').map(|(_, value)| value.trim().to_string()))
        .unwrap_or_else(|| env::consts::ARCH.to_string())
}

fn os_description() -> String {
    fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|release| release.lines()
            .find(|line| line.starts_with("PRETTY_NAME="))
            .map(|line| line.trim_start_matches("PRETTY_NAME=").trim_matches('"').to_string()))
        .unwrap_or_else(|| env::consts::OS.to_string())
}

#[cfg(unix)]
fn root_identity(path: &Path) -> std::io::Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::metadata(path)?;
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn root_identity(_path: &Path) -> std::io::Result<(u64, u64)> {
    Ok((0, 0))
}
