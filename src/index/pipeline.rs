use rusqlite::{OptionalExtension, Transaction};
use crate::index::adapter::EmbeddingProvider;
use crate::index::reconcile::{
    existing_snapshot, persist_workspace, prepare_workspace_profiles, ProfileProvider,
    WorkspacePreparation,
};
use crate::security::WorkspaceFs;
use crate::state::{Result, StateError};
use std::path::Path;

#[derive(Debug, Default, Clone)]
pub struct IndexReport {
    pub updated: usize,
    pub unchanged: usize,
    pub excluded: usize,
    pub deleted: usize,
    pub degraded_semantic: bool,
}

pub struct IndexWorkspaceRequest<'a> {
    pub workspace_id: &'a str,
    pub workspace: &'a WorkspaceFs,
    pub excluded_data_root: &'a Path,
    pub provider: &'a dyn EmbeddingProvider,
    pub profile_id: &'a str,
    pub generation: u64,
    pub max_file_bytes: u64,
}


pub struct IndexPipeline<'a> {
    tx: &'a Transaction<'a>,
}

impl<'a> IndexPipeline<'a> {
    pub fn new(tx: &'a Transaction<'a>) -> Self {
        Self { tx }
    }
    
    
    pub fn enqueue_vector_op(
        &self,
        workspace_id: &str,
        profile_id: &str,
        generation: u64,
        op_type: &str,
        vector_id: &str,
        embedding: Option<&[f32]>,
    ) -> Result<()> {
        if generation == 0 || workspace_id.is_empty() || profile_id.is_empty() || vector_id.is_empty() {
            return Err(StateError::InvalidVectorOperation);
        }
        let dimension = self.tx.query_row(
            "SELECT dimension FROM embedding_profiles WHERE id=?1",
            [profile_id],
            |row| row.get::<_, i64>(0),
        ).optional()?.filter(|dimension| *dimension > 0)
            .ok_or(StateError::InvalidVectorOperation)?;
        let dimension = usize::try_from(dimension).map_err(|_| StateError::InvalidVectorOperation)?;
        let embedding_bytes = match (op_type, embedding) {
            ("add", Some(vector))
                if vector.len() == dimension && vector.iter().all(|value| value.is_finite()) =>
            {
                let mut bytes = Vec::with_capacity(std::mem::size_of_val(vector));
                for value in vector {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Some(bytes)
            }
            ("delete", None) => None,
            _ => return Err(StateError::InvalidVectorOperation),
        };
        self.tx.execute(
            "INSERT INTO vector_ops (workspace_id, profile_id, generation, op_type, vector_id, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                workspace_id,
                profile_id,
                generation as i64,
                op_type,
                vector_id,
                embedding_bytes,
            ],
        )?;
        Ok(())
    }
    
    pub fn index_workspace(&self, request: IndexWorkspaceRequest<'_>) -> Result<IndexReport> {
        let IndexWorkspaceRequest {
            workspace_id,
            workspace,
            excluded_data_root,
            provider,
            profile_id,
            generation,
            max_file_bytes,
        } = request;
        let snapshot = existing_snapshot(self.tx, workspace_id, profile_id)?;
        let profiles = [ProfileProvider { profile_id, provider }];
        let plan = prepare_workspace_profiles(WorkspacePreparation {
            workspace_id,
            workspace,
            excluded_data_root,
            profiles: &profiles,
            generation,
            max_file_bytes,
            force_all: false,
            force_paths: &[],
            snapshot,
        })?;
        persist_workspace(self.tx, &plan)?;
        Ok(plan.report)
    }
}
