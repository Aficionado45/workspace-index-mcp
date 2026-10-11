use rusqlite::Connection;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use tempfile::TempDir;
use workspace_index_mcp::index::{EmbeddingProvider, IndexPipeline, IndexWorkspaceRequest};
use workspace_index_mcp::security::WorkspaceFs;
use workspace_index_mcp::state::schema;

struct DeterministicProvider;
impl EmbeddingProvider for DeterministicProvider {
    fn fingerprint(&self) -> String { "deterministic".into() }
    fn dimension(&self) -> usize { 2 }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts.iter().map(|text| vec![text.len() as f32, 1.0]).collect())
    }
}

#[cfg(unix)]
#[test]
fn codeindex_rust_plugin_labels_dependency_ranges_as_source_regions() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("repo");
    std::fs::create_dir(&root).unwrap();
    let source = "pub struct Demo;\nimpl std::fmt::Display for Demo { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, \"demo\") } }\n";
    std::fs::write(root.join("lib.rs"), source).unwrap();
    let mut connection = Connection::open(temp.path().join("index.db")).unwrap();
    schema::initialize(&mut connection).unwrap();
    let metadata = std::fs::metadata(&root).unwrap();
    connection.execute(
        "INSERT INTO workspaces (id, root_path, device_id, inode, mode, registration_epoch, mode_epoch, created_at, updated_at)
         VALUES ('repo', ?1, ?2, ?3, 'read_only', 1, 1, 0, 0)",
        rusqlite::params![root.to_string_lossy(), metadata.dev() as i64, metadata.ino() as i64],
    ).unwrap();
    connection.execute(
        "INSERT INTO embedding_profiles (id, provider_kind, model, dimension, metric, state, created_at)
         VALUES ('test', 'deterministic', 'test', 2, 'cosine', 'Active', 0)",
        [],
    ).unwrap();
    let workspace = WorkspaceFs::open_registered(&root, metadata.dev(), metadata.ino()).unwrap();
    let tx = connection.transaction().unwrap();
    let pipeline = IndexPipeline::new(&tx);
    pipeline.index_workspace(IndexWorkspaceRequest {
        workspace_id: "repo",
        workspace: &workspace,
        excluded_data_root: &temp.path().join("data"),
        provider: &DeterministicProvider,
        profile_id: "test",
        generation: 1,
        max_file_bytes: 1024 * 1024,
    }).unwrap();
    tx.commit().unwrap();

    let mut statement = connection.prepare(
        "SELECT name, byte_start, byte_end, start_line, end_line FROM dependency_edges
         WHERE workspace_id='repo' AND evidence='source_region'",
    ).unwrap();
    let regions: Vec<(String, usize, usize, usize, usize)> = statement.query_map([], |row| {
        let region = |column| {
            usize::try_from(row.get::<_, i64>(column)?)
                .map_err(|error| rusqlite::Error::FromSqlConversionFailure(
                    column,
                    rusqlite::types::Type::Integer,
                    Box::new(error),
                ))
        };
        Ok((row.get(0)?, region(1)?, region(2)?, region(3)?, region(4)?))
    }).unwrap().collect::<Result<_, _>>().unwrap();
    assert!(regions.iter().any(|(name, start, end, first_line, last_line)| {
        name.contains("Display") && start < end && *first_line == 2 && *last_line == 2
    }));
}
