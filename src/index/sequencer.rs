use std::collections::HashSet;
use std::path::PathBuf;

use rusqlite::{params, Connection, OptionalExtension, Transaction};

use crate::index::vector::{
    decode_vector_bytes, validate_vector_bytes, CacheManifest, SnapshotDurability, VectorError,
    VectorIndex,
};

#[derive(Debug, thiserror::Error)]
pub enum SequencerError {
    #[error(transparent)]
    Database(#[from] rusqlite::Error),
    #[error(transparent)]
    Vector(#[from] VectorError),
    #[error("workspace index state is missing for {0}")]
    MissingWorkspaceState(String),
    #[error("embedding profile is missing: {0}")]
    MissingProfile(String),
    #[error("profile configuration mismatch: expected dimension {expected_dimension} and metric {expected_metric}, got dimension {actual_dimension} and metric {actual_metric}")]
    ProfileConfiguration {
        expected_dimension: usize,
        actual_dimension: i64,
        expected_metric: String,
        actual_metric: String,
    },
    #[error("invalid vector operation row {row_id}: {reason}")]
    InvalidOperation { row_id: i64, reason: String },
    #[error("invalid embedding for {path} chunk {chunk_index}: {reason}")]
    InvalidLiveEmbedding {
        path: String,
        chunk_index: i64,
        reason: String,
    },
    #[error("embedding coverage is incomplete: expected {expected} active chunks, found {actual} current vectors")]
    IncompleteCoverage { expected: usize, actual: usize },
    #[error("vector operations are newer than the visible workspace generation")]
    FutureVectorOperations,
    #[error("applied vector generation is newer than the visible workspace generation")]
    WatermarkAhead,
    #[error("index source changed while its snapshot was being persisted")]
    SourceChanged,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorWatermark {
    profile_id: String,
    generation: u64,
    operation_count: i64,
    max_operation_id: i64,
    max_operation_generation: i64,
}

impl VectorWatermark {
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn outbox_identity(&self) -> (i64, i64, i64) {
        (
            self.operation_count,
            self.max_operation_id,
            self.max_operation_generation,
        )
    }
}

pub struct Sequencer {
    workspace_id: String,
    profile_id: String,
    vectors_dir: PathBuf,
    dimension: usize,
    metric: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceIdentity {
    data_version: i64,
    visible_generation: i64,
    desired_profile_id: Option<String>,
    active_profile_id: Option<String>,
    profile_dimension: i64,
    profile_metric: String,
    profile_state: String,
    applied_generation: Option<i64>,
    applied_operation_count: i64,
    applied_max_operation_id: i64,
    backfill_required: bool,
    operation_count: i64,
    max_operation_id: i64,
    max_operation_generation: i64,
}

struct SourceSnapshot {
    identity: SourceIdentity,
    operations: Vec<RawVectorOperation>,
    active_chunk_count: usize,
    active_embeddings: Vec<RawActiveEmbedding>,
}

struct RawVectorOperation {
    id: i64,
    operation: String,
    numeric_id: Option<i64>,
    embedding: Option<Vec<u8>>,
    generation: i64,
}

struct RawActiveEmbedding {
    path: String,
    chunk_index: i64,
    numeric_id: Option<i64>,
    generation: Option<i64>,
    embedding: Option<Vec<u8>>,
}

struct ParsedOperation<'a> {
    numeric_id: u64,
    embedding: Option<&'a [u8]>,
}

impl Sequencer {
    pub fn new(
        workspace_id: String,
        profile_id: String,
        vectors_dir: PathBuf,
        dimension: usize,
        metric: String,
    ) -> Self {
        Self {
            workspace_id,
            profile_id,
            vectors_dir,
            dimension,
            metric,
        }
    }

    fn base_path(&self) -> PathBuf {
        self.vectors_dir
            .join(format!("{}_{}", self.workspace_id, self.profile_id))
    }

    pub fn sync(&self, conn: &mut Connection) -> Result<VectorWatermark, SequencerError> {
        self.sync_with_durability(conn, &FilesystemDurability)
    }

    fn sync_with_durability(
        &self,
        conn: &mut Connection,
        durability: &impl SnapshotDurability,
    ) -> Result<VectorWatermark, SequencerError> {
        let snapshot = self.capture_snapshot(conn)?;
        let visible = generation(snapshot.identity.visible_generation)?;
        let applied = generation(snapshot.identity.applied_generation.unwrap_or(0))?;
        if applied > visible {
            return Err(SequencerError::WatermarkAhead);
        }
        self.validate_profile(&snapshot.identity)?;

        validate_active_embeddings(&snapshot, visible, self.dimension)?;
        validate_operations(&snapshot.operations, self.dimension, visible)?;
        let loaded = self.load_cached(applied, &snapshot.identity, durability);
        let no_op_generation = visible > applied && snapshot.operations.is_empty();
        let stale_count = visible == applied
            && loaded.as_ref().is_some_and(|index| index.size() != snapshot.active_chunk_count);
        let mut rebuild = loaded.is_none() || no_op_generation || stale_count;
        let mut index = if rebuild {
            self.rebuild_from_embeddings(visible, &snapshot, self.dimension, &snapshot.identity)?
        } else {
            let loaded_index = loaded.ok_or(SequencerError::SourceChanged)?;
            apply_operations(&loaded_index, &snapshot.operations, self.dimension, visible)?;
            loaded_index
        };

        if index.size() != snapshot.active_chunk_count {
            index = self.rebuild_from_embeddings(visible, &snapshot, self.dimension, &snapshot.identity)?;
            rebuild = true;
        }
        index.manifest.generation = visible;
        index.manifest.outbox_count = generation(snapshot.identity.operation_count)?;
        index.manifest.outbox_max_id = generation(snapshot.identity.max_operation_id)?;

        let requires_save = rebuild || !snapshot.operations.is_empty() || visible > applied;
        if !requires_save {
            confirm_source_unchanged(conn, &self.workspace_id, &self.profile_id, &snapshot.identity)?;
            return Ok(self.watermark(&snapshot.identity, visible));
        }

        index.save_with_durability(&self.base_path(), durability)?;
        self.advance_watermark(conn, &snapshot.identity, visible)?;
        Ok(self.watermark(&snapshot.identity, visible))
    }

    fn capture_snapshot(&self, conn: &mut Connection) -> Result<SourceSnapshot, SequencerError> {
        let tx = conn.transaction()?;
        let identity = read_source_identity(&tx, &self.workspace_id, &self.profile_id)?;
        let visible = generation(identity.visible_generation)?;
        let applied = generation(identity.applied_generation.unwrap_or(0))?;
        if applied > visible {
            return Err(SequencerError::WatermarkAhead);
        }
        if identity.max_operation_generation < 0
            || identity.max_operation_generation > identity.visible_generation
        {
            return Err(SequencerError::FutureVectorOperations);
        }
        let operations = read_operations(
            &tx,
            &self.workspace_id,
            &self.profile_id,
            applied,
            visible,
        )?;
        let (active_chunk_count, active_embeddings) =
            read_active_embeddings(&tx, &self.workspace_id, &self.profile_id, visible)?;
        tx.commit()?;
        Ok(SourceSnapshot {
            identity,
            operations,
            active_chunk_count,
            active_embeddings,
        })
    }

    fn validate_profile(&self, identity: &SourceIdentity) -> Result<(), SequencerError> {
        if self.dimension == 0
            || usize::try_from(identity.profile_dimension).ok() != Some(self.dimension)
            || identity.profile_metric != self.metric
        {
            return Err(SequencerError::ProfileConfiguration {
                expected_dimension: self.dimension,
                actual_dimension: identity.profile_dimension,
                expected_metric: self.metric.clone(),
                actual_metric: identity.profile_metric.clone(),
            });
        }
        Ok(())
    }

    fn load_cached(
        &self,
        generation: u64,
        identity: &SourceIdentity,
        durability: &impl SnapshotDurability,
    ) -> Option<VectorIndex> {
        let base_path = self.base_path();
        let index = VectorIndex::load(&base_path, &self.expected_manifest(generation)).ok()?;
        VectorIndex::sync_snapshot(&base_path, durability).ok()?;
        if index.manifest.outbox_count != u64::try_from(identity.applied_operation_count).ok()?
            || index.manifest.outbox_max_id
                != u64::try_from(identity.applied_max_operation_id).ok()?
        {
            return None;
        }
        Some(index)
    }

    fn expected_manifest(&self, generation: u64) -> CacheManifest {
        CacheManifest {
            workspace_id: self.workspace_id.clone(),
            profile_id: self.profile_id.clone(),
            dimension: self.dimension,
            metric: self.metric.clone(),
            generation,
            outbox_count: 0,
            outbox_max_id: 0,
            count: 0,
            checksum: String::new(),
        }
    }

    fn watermark(&self, identity: &SourceIdentity, generation: u64) -> VectorWatermark {
        VectorWatermark {
            profile_id: self.profile_id.clone(),
            generation,
            operation_count: identity.operation_count,
            max_operation_id: identity.max_operation_id,
            max_operation_generation: identity.max_operation_generation,
        }
    }

    fn rebuild_from_embeddings(
        &self,
        target_generation: u64,
        snapshot: &SourceSnapshot,
        dimension: usize,
        identity: &SourceIdentity,
    ) -> Result<VectorIndex, SequencerError> {
        let mut index = VectorIndex::new(self.expected_manifest(target_generation))?;
        for row in &snapshot.active_embeddings {
            let Some((numeric_id, bytes)) = current_active_embedding(row, target_generation) else {
                return Err(SequencerError::IncompleteCoverage {
                    expected: snapshot.active_chunk_count,
                    actual: index.size(),
                });
            };
            let numeric_id = positive_numeric_id(numeric_id).map_err(|reason| {
                invalid_live_embedding(row, reason)
            })?;
            let vector = decode_vector_bytes(bytes, dimension)
                .map_err(|error| invalid_live_embedding(row, error.to_string()))?;
            index.add(numeric_id, &vector)?;
        }
        index.manifest.outbox_count = generation(identity.operation_count)?;
        index.manifest.outbox_max_id = generation(identity.max_operation_id)?;
        Ok(index)
    }

    fn advance_watermark(
        &self,
        conn: &mut Connection,
        expected: &SourceIdentity,
        generation: u64,
    ) -> Result<(), SequencerError> {
        let tx = conn.transaction()?;
        if read_source_identity(&tx, &self.workspace_id, &self.profile_id)? != *expected {
            return Err(SequencerError::SourceChanged);
        }
        update_applied_generation(&tx, &self.workspace_id, &self.profile_id, expected, generation)?;
        tx.commit()?;
        Ok(())
    }
}

struct FilesystemDurability;

impl SnapshotDurability for FilesystemDurability {
    fn sync_file(&self, file: &std::fs::File) -> std::io::Result<()> {
        file.sync_all()
    }

    fn sync_directory(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::File::open(path)?.sync_all()
    }
}

fn read_source_identity(
    conn: &Connection,
    workspace_id: &str,
    profile_id: &str,
) -> Result<SourceIdentity, SequencerError> {
    let data_version = conn.query_row("PRAGMA data_version", [], |row| row.get(0))?;
    let (visible_generation, desired_profile_id, active_profile_id) = conn
        .query_row(
            "SELECT visible_generation, desired_profile_id, active_profile_id
             FROM workspace_index_state WHERE workspace_id=?1",
            [workspace_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?
        .ok_or_else(|| SequencerError::MissingWorkspaceState(workspace_id.to_owned()))?;
    let (profile_dimension, profile_metric, profile_state) = conn
        .query_row(
            "SELECT dimension, metric, state FROM embedding_profiles WHERE id=?1",
            [profile_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?
        .ok_or_else(|| SequencerError::MissingProfile(profile_id.to_owned()))?;
    let applied = conn
        .query_row(
            "SELECT applied_generation, backfill_required
             FROM profile_applied_generations WHERE workspace_id=?1 AND profile_id=?2",
            params![workspace_id, profile_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, bool>(1)?)),
        )
        .optional()?;
    let (applied_generation, backfill_required) =
        applied.map_or((None, false), |(generation, backfill)| (Some(generation), backfill));
    let (applied_operation_count, applied_max_operation_id) = conn.query_row(
        "SELECT COUNT(*), COALESCE(MAX(id), 0)
         FROM vector_ops WHERE workspace_id=?1 AND profile_id=?2 AND generation<=?3",
        params![workspace_id, profile_id, applied_generation.unwrap_or(0)],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let (operation_count, max_operation_id, max_operation_generation) = conn.query_row(
        "SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(MAX(generation), 0)
         FROM vector_ops WHERE workspace_id=?1 AND profile_id=?2",
        params![workspace_id, profile_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;

    Ok(SourceIdentity {
        data_version,
        visible_generation,
        desired_profile_id,
        active_profile_id,
        profile_dimension,
        profile_metric,
        profile_state,
        applied_generation,
        applied_operation_count,
        applied_max_operation_id,
        backfill_required,
        operation_count,
        max_operation_id,
        max_operation_generation,
    })
}

fn read_operations(
    conn: &Connection,
    workspace_id: &str,
    profile_id: &str,
    applied_generation: u64,
    visible_generation: u64,
) -> Result<Vec<RawVectorOperation>, SequencerError> {
    let mut statement = conn.prepare(
        "SELECT op.id, op.op_type, COALESCE(op.numeric_vector_id, ids.id), op.embedding, op.generation
         FROM vector_ops op
         LEFT JOIN stable_vector_ids ids
           ON ids.workspace_id=op.workspace_id AND ids.profile_id=op.profile_id
          AND ids.vector_id=op.vector_id
         WHERE op.workspace_id=?1 AND op.profile_id=?2
           AND op.generation>?3 AND op.generation<=?4
         ORDER BY op.generation ASC, op.id ASC",
    )?;
    statement
        .query_map(
            params![
                workspace_id,
                profile_id,
                applied_generation as i64,
                visible_generation as i64
            ],
            |row| {
                Ok(RawVectorOperation {
                    id: row.get(0)?,
                    operation: row.get(1)?,
                    numeric_id: row.get(2)?,
                    embedding: row.get(3)?,
                    generation: row.get(4)?,
                })
            },
        )?
        .collect::<Result<Vec<_>, _>>()
        .map_err(SequencerError::from)
}

fn read_active_embeddings(
    conn: &Connection,
    workspace_id: &str,
    profile_id: &str,
    visible_generation: u64,
) -> Result<(usize, Vec<RawActiveEmbedding>), SequencerError> {
    let active_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM chunks c JOIN files f
           ON f.workspace_id=c.workspace_id AND f.relative_path=c.relative_path
         WHERE c.workspace_id=?1 AND f.visible_generation=?2",
        params![workspace_id, visible_generation as i64],
        |row| row.get(0),
    )?;
    let mut statement = conn.prepare(
        "SELECT c.relative_path, c.chunk_index,
                COALESCE(e.numeric_vector_id, ids.id), e.generation, e.embedding
         FROM chunks c JOIN files f
           ON f.workspace_id=c.workspace_id AND f.relative_path=c.relative_path
         LEFT JOIN chunk_embeddings e
           ON e.workspace_id=c.workspace_id AND e.relative_path=c.relative_path
          AND e.chunk_index=c.chunk_index AND e.profile_id=?2
         LEFT JOIN stable_vector_ids ids
           ON ids.workspace_id=e.workspace_id AND ids.profile_id=e.profile_id
          AND ids.vector_id=e.vector_id
         WHERE c.workspace_id=?1 AND f.visible_generation=?3
         ORDER BY c.relative_path, c.chunk_index",
    )?;
    let embeddings = statement
        .query_map(
            params![workspace_id, profile_id, visible_generation as i64],
            |row| {
                Ok(RawActiveEmbedding {
                    path: row.get(0)?,
                    chunk_index: row.get(1)?,
                    numeric_id: row.get(2)?,
                    generation: row.get(3)?,
                    embedding: row.get(4)?,
                })
            },
        )?
        .collect::<Result<Vec<_>, _>>()?;
    let active_count = usize::try_from(active_count).map_err(|_| SequencerError::IncompleteCoverage {
        expected: 0,
        actual: embeddings.len(),
    })?;
    Ok((active_count, embeddings))
}

fn current_active_embedding(
    row: &RawActiveEmbedding,
    visible_generation: u64,
) -> Option<(i64, &[u8])> {
    match (row.numeric_id, row.generation, row.embedding.as_deref()) {
        (Some(id), Some(generation), Some(bytes)) if generation == visible_generation as i64 => {
            Some((id, bytes))
        }
        _ => None,
    }
}

fn validate_active_embeddings(
    snapshot: &SourceSnapshot,
    visible_generation: u64,
    dimension: usize,
) -> Result<(), SequencerError> {
    let mut vector_ids = HashSet::with_capacity(snapshot.active_chunk_count);
    let mut covered = 0;
    for row in &snapshot.active_embeddings {
        let Some((numeric_id, bytes)) = current_active_embedding(row, visible_generation) else {
            continue;
        };
        let numeric_id = positive_numeric_id(numeric_id)
            .map_err(|reason| invalid_live_embedding(row, reason))?;
        if !vector_ids.insert(numeric_id) {
            return Err(invalid_live_embedding(row, "numeric vector ID is duplicated"));
        }
        validate_vector_bytes(bytes, dimension)
            .map_err(|error| invalid_live_embedding(row, error.to_string()))?;
        covered += 1;
    }
    if covered != snapshot.active_chunk_count {
        return Err(SequencerError::IncompleteCoverage {
            expected: snapshot.active_chunk_count,
            actual: covered,
        });
    }
    Ok(())
}

fn invalid_live_embedding(
    row: &RawActiveEmbedding,
    reason: impl Into<String>,
) -> SequencerError {
    SequencerError::InvalidLiveEmbedding {
        path: row.path.clone(),
        chunk_index: row.chunk_index,
        reason: reason.into(),
    }
}

fn validate_operations(
    rows: &[RawVectorOperation],
    dimension: usize,
    visible_generation: u64,
) -> Result<(), SequencerError> {
    for row in rows {
        let _ = parse_operation(row, dimension, visible_generation)?;
    }
    Ok(())
}

fn parse_operation<'a>(
    row: &'a RawVectorOperation,
    dimension: usize,
    visible_generation: u64,
) -> Result<ParsedOperation<'a>, SequencerError> {
    if row.generation <= 0 || row.generation > visible_generation as i64 {
        return invalid_operation(row, "operation generation is invalid");
    }
    let numeric_id = row
        .numeric_id
        .ok_or_else(|| SequencerError::InvalidOperation {
            row_id: row.id,
            reason: "numeric vector ID is missing".to_owned(),
        })
        .and_then(|id| {
            positive_numeric_id(id).map_err(|reason| SequencerError::InvalidOperation {
                row_id: row.id,
                reason,
            })
        })?;
    let embedding = match (row.operation.as_str(), row.embedding.as_deref()) {
        ("add", Some(bytes)) => {
            validate_vector_bytes(bytes, dimension).map_err(|error| {
                SequencerError::InvalidOperation {
                    row_id: row.id,
                    reason: error.to_string(),
                }
            })?;
            Some(bytes)
        }
        ("delete", None) => None,
        ("add", None) => return invalid_operation(row, "add operation has no embedding"),
        ("delete", Some(_)) => return invalid_operation(row, "delete operation has an embedding"),
        (other, _) => return invalid_operation(row, &format!("unknown operation type {other}")),
    };
    Ok(ParsedOperation { numeric_id, embedding })
}

fn invalid_operation<T>(
    row: &RawVectorOperation,
    reason: &str,
) -> Result<T, SequencerError> {
    Err(SequencerError::InvalidOperation {
        row_id: row.id,
        reason: reason.to_owned(),
    })
}

fn positive_numeric_id(id: i64) -> Result<u64, String> {
    u64::try_from(id)
        .ok()
        .filter(|id| *id > 0)
        .ok_or_else(|| "numeric vector ID must be positive".to_owned())
}

fn apply_operations(
    index: &VectorIndex,
    rows: &[RawVectorOperation],
    dimension: usize,
    visible_generation: u64,
) -> Result<(), SequencerError> {
    for row in rows {
        let operation = parse_operation(row, dimension, visible_generation)?;
        if let Some(bytes) = operation.embedding {
            let vector = decode_vector_bytes(bytes, dimension).map_err(|error| {
                SequencerError::InvalidOperation {
                    row_id: row.id,
                    reason: error.to_string(),
                }
            })?;
            let _ = index.remove(operation.numeric_id);
            index.add(operation.numeric_id, &vector)?;
        } else {
            let _ = index.remove(operation.numeric_id);
        }
    }
    Ok(())
}

fn confirm_source_unchanged(
    conn: &mut Connection,
    workspace_id: &str,
    profile_id: &str,
    expected: &SourceIdentity,
) -> Result<(), SequencerError> {
    let tx = conn.transaction()?;
    if read_source_identity(&tx, workspace_id, profile_id)? != *expected {
        return Err(SequencerError::SourceChanged);
    }
    tx.commit()?;
    Ok(())
}

fn update_applied_generation(
    tx: &Transaction<'_>,
    workspace_id: &str,
    profile_id: &str,
    expected: &SourceIdentity,
    generation: u64,
) -> Result<(), SequencerError> {
    let generation = i64::try_from(generation).map_err(|_| SequencerError::WatermarkAhead)?;
    if expected.applied_generation == Some(generation) {
        return Ok(());
    }
    let changed = match expected.applied_generation {
        Some(previous) => tx.execute(
            "UPDATE profile_applied_generations
             SET applied_generation=?1, updated_at=strftime('%s','now')
             WHERE workspace_id=?2 AND profile_id=?3
               AND applied_generation=?4 AND backfill_required=?5",
            params![
                generation,
                workspace_id,
                profile_id,
                previous,
                expected.backfill_required
            ],
        )?,
        None => tx.execute(
            "INSERT INTO profile_applied_generations
                (workspace_id, profile_id, applied_generation, backfill_required, updated_at)
             VALUES (?1, ?2, ?3, 0, strftime('%s','now'))
             ON CONFLICT(workspace_id, profile_id) DO NOTHING",
            params![workspace_id, profile_id, generation],
        )?,
    };
    if changed != 1 {
        return Err(SequencerError::SourceChanged);
    }
    Ok(())
}

fn generation(value: i64) -> Result<u64, SequencerError> {
    u64::try_from(value).map_err(|_| SequencerError::WatermarkAhead)
}
