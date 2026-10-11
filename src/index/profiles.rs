use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use crate::index::sequencer::VectorWatermark;
 



#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileState {
    Building,
    Active,
    Failed,
    Retired,
}

impl ProfileState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProfileState::Building => "building",
            ProfileState::Active => "active",
            ProfileState::Failed => "failed",
            ProfileState::Retired => "retired",
        }
    }
}

pub struct ProfileManager {
    workspace_id: String,
}

struct PromotionPreflight {
    visible_generation: i64,
    desired_profile_id: Option<String>,
    active_profile_id: Option<String>,
    profile_state: String,
    applied_generation: i64,
    backfill_required: bool,
    outbox_identity: (i64, i64, i64),
    active_chunk_count: i64,
    _coverage_digest: blake3::Hash,
    data_version: i64,
}

impl ProfileManager {
    pub fn new(workspace_id: String) -> Self {
        Self { workspace_id }
    }

    pub fn try_promote(
        &self,
        conn: &mut Connection,
        expected_profile_id: &str,
        graph: &VectorWatermark,
    ) -> Result<bool, String> {
        if graph.profile_id() != expected_profile_id {
            return Ok(false);
        }
        let Some(preflight) = preflight_coverage(conn, &self.workspace_id, expected_profile_id)?
        else {
            return Ok(false);
        };
        if preflight.desired_profile_id.as_deref() != Some(expected_profile_id)
            || (preflight.profile_state != "building" && preflight.profile_state != "active")
            || preflight.active_profile_id.as_deref() == Some(expected_profile_id)
            || u64::try_from(preflight.visible_generation).ok() != Some(graph.generation())
            || preflight.applied_generation != preflight.visible_generation
            || preflight.backfill_required
            || preflight.outbox_identity != graph.outbox_identity()
        {
            return Ok(false);
        }

        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| error.to_string())?;
        let data_version: i64 = tx
            .query_row("PRAGMA data_version", [], |row| row.get(0))
            .map_err(|error| error.to_string())?;
        // The preflight digest is tied to data_version; a commit from another connection
        // changes this token without requiring another vector-BLOB scan under the writer lock.
        if data_version != preflight.data_version {
            tx.rollback().map_err(|error| error.to_string())?;
            return Ok(false);
        }
        let state = tx
            .query_row(
                "SELECT visible_generation, desired_profile_id, active_profile_id
                 FROM workspace_index_state WHERE workspace_id=?1",
                [&self.workspace_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let Some((visible_generation, desired_profile_id, active_profile_id)) = state else {
            return Ok(false);
        };
        if visible_generation != preflight.visible_generation
            || desired_profile_id != preflight.desired_profile_id
            || active_profile_id != preflight.active_profile_id
        {
            tx.rollback().map_err(|error| error.to_string())?;
            return Ok(false);
        }
        let outbox_identity = read_outbox_identity(&tx, &self.workspace_id, expected_profile_id)?;
        if outbox_identity != preflight.outbox_identity {
            tx.rollback().map_err(|error| error.to_string())?;
            return Ok(false);
        }
        let profile_state: Option<String> = tx
            .query_row(
                "SELECT state FROM embedding_profiles WHERE id=?1",
                [expected_profile_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| error.to_string())?;
        if profile_state.as_deref() != Some(preflight.profile_state.as_str()) {
            tx.rollback().map_err(|error| error.to_string())?;
            return Ok(false);
        }
        let applied = tx
            .query_row(
                "SELECT applied_generation, backfill_required
                 FROM profile_applied_generations WHERE workspace_id=?1 AND profile_id=?2",
                params![self.workspace_id, expected_profile_id],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, bool>(1)?)),
            )
            .optional()
            .map_err(|error| error.to_string())?;
        if applied != Some((preflight.applied_generation, preflight.backfill_required))
            || preflight.backfill_required
            || preflight.applied_generation != visible_generation
            || !coverage_shape_matches(
                &tx,
                &self.workspace_id,
                expected_profile_id,
                visible_generation,
                preflight.active_chunk_count,
            )?
        {
            tx.rollback().map_err(|error| error.to_string())?;
            return Ok(false);
        }

        let already_active = preflight.profile_state == "active";
        let changed = tx
            .execute(
                "UPDATE workspace_index_state SET active_profile_id=?1
                 WHERE workspace_id=?2 AND desired_profile_id=?1
                   AND visible_generation=?3 AND active_profile_id IS ?4",
                params![expected_profile_id, self.workspace_id, visible_generation, active_profile_id],
            )
            .map_err(|error| error.to_string())?;
        if changed != 1 {
            tx.rollback().map_err(|error| error.to_string())?;
            return Ok(false);
        }
        if !already_active {
            let profile_changed = tx
                .execute(
                    "UPDATE embedding_profiles SET state='active'
                     WHERE id=?1 AND state='building'",
                    [expected_profile_id],
                )
                .map_err(|error| error.to_string())?;
            if profile_changed != 1 {
                tx.rollback().map_err(|error| error.to_string())?;
                return Ok(false);
            }
        }
        let still_current: (i64, Option<String>, Option<String>) = tx
            .query_row(
                "SELECT visible_generation, desired_profile_id, active_profile_id
                 FROM workspace_index_state WHERE workspace_id=?1",
                [&self.workspace_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(|error| error.to_string())?;
        if still_current.0 != visible_generation
            || still_current.1.as_deref() != Some(expected_profile_id)
            || still_current.2.as_deref() != Some(expected_profile_id)
        {
            tx.rollback().map_err(|error| error.to_string())?;
            return Ok(false);
        }
        tx.commit().map_err(|error| error.to_string())?;
        Ok(true)
    }
}

fn preflight_coverage(
    conn: &mut Connection,
    workspace_id: &str,
    profile_id: &str,
) -> Result<Option<PromotionPreflight>, String> {
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(|error| error.to_string())?;
    let data_version: i64 = tx
        .query_row("PRAGMA data_version", [], |row| row.get(0))
        .map_err(|error| error.to_string())?;
    let state = tx.query_row(
        "SELECT visible_generation, desired_profile_id, active_profile_id
         FROM workspace_index_state WHERE workspace_id=?1",
        [workspace_id],
        |row| Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
        )),
    ).optional().map_err(|error| error.to_string())?;
    let Some((visible_generation, desired_profile_id, active_profile_id)) = state else {
        tx.commit().map_err(|error| error.to_string())?;
        return Ok(None);
    };
    let profile = tx.query_row(
        "SELECT dimension, state FROM embedding_profiles WHERE id=?1",
        [profile_id],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
    ).optional().map_err(|error| error.to_string())?;
    let Some((dimension, profile_state)) = profile else {
        tx.commit().map_err(|error| error.to_string())?;
        return Ok(None);
    };
    if dimension <= 0 || (profile_state != "building" && profile_state != "active") {
        return Ok(None);
    }
    let applied = tx.query_row(
        "SELECT applied_generation, backfill_required
         FROM profile_applied_generations WHERE workspace_id=?1 AND profile_id=?2",
        params![workspace_id, profile_id],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, bool>(1)?)),
    ).optional().map_err(|error| error.to_string())?;
    let Some((applied_generation, backfill_required)) = applied else {
        tx.commit().map_err(|error| error.to_string())?;
        return Ok(None);
    };
    let outbox_identity = read_outbox_identity(&tx, workspace_id, profile_id)?;
    let active_chunk_count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM chunks c JOIN files f
           ON f.workspace_id=c.workspace_id AND f.relative_path=c.relative_path
         WHERE c.workspace_id=?1 AND f.visible_generation=?2",
        params![workspace_id, visible_generation],
        |row| row.get(0),
    ).map_err(|error| error.to_string())?;
    let (covered, coverage_digest) = {
        let mut statement = tx.prepare(
            "SELECT c.relative_path, c.chunk_index, e.generation,
                    COALESCE(e.numeric_vector_id, ids.id), ids.id, e.embedding
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
        ).map_err(|error| error.to_string())?;
        let rows = statement.query_map(
            params![workspace_id, profile_id, visible_generation],
            |row| Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, Option<i64>>(4)?,
                row.get::<_, Option<Vec<u8>>>(5)?,
            )),
        ).map_err(|error| error.to_string())?;
        let mut hasher = blake3::Hasher::new();
        let mut covered = 0_i64;
        for row in rows {
            let (path, chunk_index, generation, numeric_id, stable_id, embedding) =
                row.map_err(|error| error.to_string())?;
            let (Some(generation), Some(numeric_id), Some(stable_id), Some(embedding)) =
                (generation, numeric_id, stable_id, embedding)
            else {
                return Ok(None);
            };
            if generation != visible_generation
                || numeric_id <= 0
                || stable_id != numeric_id
                || embedding.len() != usize::try_from(dimension).unwrap_or(0).saturating_mul(4)
                || !embedding.as_chunks::<4>().0.iter().all(|bytes| {
                    f32::from_le_bytes(*bytes).is_finite()
                })
            {
                return Ok(None);
            }
            hash_part(&mut hasher, workspace_id.as_bytes());
            hash_part(&mut hasher, path.as_bytes());
            hasher.update(&chunk_index.to_le_bytes());
            hash_part(&mut hasher, profile_id.as_bytes());
            hasher.update(&generation.to_le_bytes());
            hasher.update(&numeric_id.to_le_bytes());
            hash_part(&mut hasher, &embedding);
            covered += 1;
        }
        (covered, hasher.finalize())
    };
    if covered != active_chunk_count || backfill_required || applied_generation != visible_generation {
        return Ok(None);
    }
    tx.commit().map_err(|error| error.to_string())?;
    Ok(Some(PromotionPreflight {
        visible_generation,
        desired_profile_id,
        active_profile_id,
        profile_state,
        applied_generation,
        backfill_required,
        outbox_identity,
        active_chunk_count,
        _coverage_digest: coverage_digest,
        data_version,
    }))
}

fn read_outbox_identity(
    conn: &Connection,
    workspace_id: &str,
    profile_id: &str,
) -> Result<(i64, i64, i64), String> {
    conn.query_row(
        "SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(MAX(generation), 0)
         FROM vector_ops WHERE workspace_id=?1 AND profile_id=?2",
        params![workspace_id, profile_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).map_err(|error| error.to_string())
}

fn coverage_shape_matches(
    tx: &Transaction<'_>,
    workspace_id: &str,
    profile_id: &str,
    visible_generation: i64,
    expected_count: i64,
) -> Result<bool, String> {
    let (chunk_count, covered_count, distinct_ids): (i64, i64, i64) = tx.query_row(
        "SELECT COUNT(*),
                SUM(CASE WHEN e.generation=?3 AND ids.id>0
                          AND (e.numeric_vector_id IS NULL OR e.numeric_vector_id=ids.id)
                          THEN 1 ELSE 0 END),
                COUNT(DISTINCT ids.id)
         FROM chunks c JOIN files f
           ON f.workspace_id=c.workspace_id AND f.relative_path=c.relative_path
         LEFT JOIN chunk_embeddings e
           ON e.workspace_id=c.workspace_id AND e.relative_path=c.relative_path
          AND e.chunk_index=c.chunk_index AND e.profile_id=?2
         LEFT JOIN stable_vector_ids ids
           ON ids.workspace_id=e.workspace_id AND ids.profile_id=e.profile_id
          AND ids.vector_id=e.vector_id
         WHERE c.workspace_id=?1 AND f.visible_generation=?3",
        params![workspace_id, profile_id, visible_generation],
        |row| Ok((row.get(0)?, row.get::<_, Option<i64>>(1)?.unwrap_or(0), row.get(2)?)),
    ).map_err(|error| error.to_string())?;
    Ok(chunk_count == expected_count && covered_count == expected_count && distinct_ids == expected_count)
}

fn hash_part(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

