use crate::model::{ErrorCode, Warning, WorkspaceMode};
use crate::security::workspace_fs::{WorkspaceFs, WorkspaceFsError};
use crate::state::registry::{resolve_git_identity, GitIdentity};
use crate::state::schema;
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions, ReadDir};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_OPERATIONS: usize = 32;
const MAX_PROPOSED_BYTES: usize = 2 * 1024 * 1024;
const MAX_DIFF_BYTES: usize = 512 * 1024;
const DEFAULT_EXPIRY_SECONDS: u64 = 600;
const MAX_FILE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct Replacement { pub old_text: String, pub new_text: String, pub replace_all: bool }

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChangeOperation {
    Create { path: String, content: String },
    Replace { path: String, expected_sha256: String, replacements: Vec<Replacement> },
    Delete { path: String, expected_sha256: String },
    Move { from_path: String, to_path: String, expected_sha256: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreparedOperation {
    operation: ChangeOperation,
    base_content: Option<String>,
    proposed_content: Option<String>,
    base_sha256: Option<String>,
    proposed_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Payload { operations: Vec<PreparedOperation> }

#[derive(Debug, Clone, Serialize)]
struct CanonicalManifest {
    version: u32,
    operation_id: String,
    workspace_id: String,
    registration_epoch: i64,
    mode_epoch: i64,
    root_device: u64,
    root_inode: u64,
    head_ref: Option<String>,
    head_commit: Option<String>,
    expires_at: u64,
    created_at: u64,
    files: Vec<ManifestFile>,
}
#[derive(Debug, Clone, Serialize)]
struct ManifestFile { kind: String, path: String, destination_path: Option<String>, base_sha256: Option<String>, proposed_sha256: Option<String>, operation: ChangeOperation }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedChangeSet {
    pub operation_id: String,
    pub digest: String,
    pub expires_at: u64,
    created_at: u64,
    pub workspace_id: String,
    pub registration_epoch: i64,
    pub mode_epoch: i64,
    pub root_device: u64,
    pub root_inode: u64,
    pub head_ref: Option<String>,
    pub head_commit: Option<String>,
    pub summary: Vec<String>,
    pub unified_diff: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FileOutcome {
    pub file_index: i64,
    pub operation: String,
    pub path: String,
    pub destination_path: Option<String>,
    pub state: String,
    pub base_sha256: Option<String>,
    pub proposed_sha256: Option<String>,
    pub observed_sha256: Option<String>,
    pub intent_at: Option<i64>,
    pub observed_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChangesetAuditEvent {
    pub event_id: Option<i64>,
    pub event_type: String,
    pub outcome: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChangesetOutcome {
    pub operation_id: String,
    pub digest: String,
    pub state: String,
    pub source_status: String,
    pub index_status: String,
    pub files: Vec<FileOutcome>,
    pub warnings: Vec<Warning>,
    pub archived: bool,
    pub audit_events: Vec<ChangesetAuditEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchivedChangeset {
    version: u32,
    workspace_id: String,
    operation_id: String,
    digest: String,
    original_state: String,
    archived_at: i64,
    expires_at: i64,
    created_at: i64,
    registration_epoch: i64,
    mode_epoch: i64,
    root_device: i64,
    root_inode: i64,
    head_ref: Option<String>,
    head_commit: Option<String>,
    outcome: ChangesetOutcome,
}

#[derive(Debug)]
struct PendingRow {
    workspace_id: String,
    digest: String,
    state: String,
    expires_at: u64,
    created_at: u64,
    registration_epoch: i64,
    mode_epoch: i64,
    root_device: u64,
    root_inode: u64,
    head_ref: Option<String>,
    head_commit: Option<String>,
    payload_path: PathBuf,
    root_path: String,
    mode: WorkspaceMode,
}

#[derive(Debug)]
struct PendingArchiveRow {
    operation_id: String,
    workspace_id: String,
    digest: String,
    state: String,
    expires_at: i64,
    created_at: i64,
    registration_epoch: i64,
    mode_epoch: i64,
    root_device: i64,
    root_inode: i64,
    head_ref: Option<String>,
    head_commit: Option<String>,
}
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ChangesetError { pub code: ErrorCode, pub message: String }
impl ChangesetError { fn new(code: ErrorCode, message: impl Into<String>) -> Self { Self { code, message: message.into() } } }

pub struct ChangesetService<'a> {
    conn: &'a parking_lot::Mutex<Connection>,
    pending_dir: PathBuf,
    expiry_seconds: u64,
    max_file_bytes: u64,
}
impl<'a> ChangesetService<'a> {
    pub fn new(conn: &'a parking_lot::Mutex<Connection>, pending_dir: impl Into<PathBuf>) -> Result<Self, ChangesetError> {
        Self::with_limits(conn, pending_dir, DEFAULT_EXPIRY_SECONDS, MAX_FILE_BYTES)
    }

    pub fn with_limits(
        conn: &'a parking_lot::Mutex<Connection>,
        pending_dir: impl Into<PathBuf>,
        expiry_seconds: u64,
        max_file_bytes: u64,
    ) -> Result<Self, ChangesetError> {
        Self::build(conn, pending_dir.into(), expiry_seconds, max_file_bytes, true, true)
    }

    pub(crate) fn for_service(
        conn: &'a parking_lot::Mutex<Connection>,
        pending_dir: impl Into<PathBuf>,
        expiry_seconds: u64,
        max_file_bytes: u64,
    ) -> Result<Self, ChangesetError> {
        Self::build(conn, pending_dir.into(), expiry_seconds, max_file_bytes, false, false)
    }

    fn build(
        conn: &'a parking_lot::Mutex<Connection>,
        pending_dir: PathBuf,
        expiry_seconds: u64,
        max_file_bytes: u64,
        initialize_schema: bool,
        recover: bool,
    ) -> Result<Self, ChangesetError> {
        if initialize_schema {
            schema::initialize(&mut conn.lock()).map_err(|error| ChangesetError::new(ErrorCode::Conflict, error.to_string()))?;
        }
        if initialize_schema {
            create_private_dir(&pending_dir).map_err(io_error)?;
        } else {
            open_pending_directory(&pending_dir).map_err(io_error)?;
        }
        let service = Self { conn, pending_dir, expiry_seconds, max_file_bytes };
        if recover {
            service.recover_interrupted()?;
            service.expire_prepared_payloads()?;
            service.mark_abandoned_prepare_reservations()?;
            service.enqueue_terminal_payloads()?;
            let pending = service.retry_payload_cleanup()?;
            if pending > 0 {
                eprintln!("workspace-index: startup retained private payload cleanup work");
            }
        }
        Ok(service)
    }

    pub fn prepare(&self, workspace_id: &str, operations: Vec<ChangeOperation>) -> Result<PreparedChangeSet, ChangesetError> {
        if operations.is_empty() || operations.len() > MAX_OPERATIONS {
            return Err(ChangesetError::new(ErrorCode::InvalidArgument, "operation count must be between 1 and 32"));
        }
        let record = self.workspace_record(workspace_id)?;
        let root = PathBuf::from(&record.root_path);
        let fs = WorkspaceFs::open_registered(&root, record.device_id, record.inode).map_err(fs_error)?;
        let git = current_git_identity(&root, &record)?;
        let mut touched = BTreeSet::new();
        let mut prepared = Vec::with_capacity(operations.len());
        let mut proposed_bytes = 0usize;
        let mut diff = String::new();
        let mut summaries = Vec::with_capacity(operations.len());
        let mut manifest_files = Vec::with_capacity(operations.len());

        for operation in operations {
            let item = self.prepare_operation(&fs, operation, &mut touched, &mut proposed_bytes)?;
            diff.push_str(&operation_diff(&item));
            if diff.len() > MAX_DIFF_BYTES {
                return Err(ChangesetError::new(ErrorCode::InvalidArgument, "complete unified diff exceeds 512 KiB"));
            }
            let (kind, path, destination) = operation_identity(&item.operation);
            summaries.push(match destination {
                Some(dest) => format!("{kind}: {path} -> {dest}"),
                None => format!("{kind}: {path}"),
            });
            manifest_files.push(manifest_file(&item));
            prepared.push(item);
        }

        let operation_id = uuid::Uuid::new_v4().to_string();
        let now = unix_seconds();
        let expires_at = now + self.expiry_seconds;
        let head_ref = git.as_ref().and_then(|identity| identity.head_ref.clone());
        let head_commit = git.as_ref().and_then(|identity| identity.head_commit.clone());
        let manifest = CanonicalManifest {
            version: 1, operation_id: operation_id.clone(), workspace_id: workspace_id.into(),
            registration_epoch: record.registration_epoch, mode_epoch: record.mode_epoch,
            root_device: record.device_id, root_inode: record.inode,
            head_ref, head_commit, expires_at, created_at: now, files: manifest_files,
        };
        let canonical = serde_json::to_vec(&manifest).map_err(json_error)?;
        let digest = hex(&Sha256::digest(&canonical));
        let payload = serde_json::to_vec(&Payload { operations: prepared.clone() }).map_err(json_error)?;
        let payload_path = self.pending_dir.join(format!("{operation_id}.json"));
        self.enqueue_payload_cleanup(&operation_id, "prepare", now as i64)?;
        if let Err(error) = write_private_payload(&payload_path, &payload) {
            self.retry_failed_prepare_best_effort(&operation_id);
            return Err(io_error(error));
        }
        if let Err(error) = self.persist_prepared(&record, &manifest, &digest, &prepared, &payload_path) {
            self.retry_failed_prepare_best_effort(&operation_id);
            return Err(error);
        }

        Ok(PreparedChangeSet {
            operation_id, digest, expires_at, created_at: now, workspace_id: workspace_id.into(),
            registration_epoch: record.registration_epoch, mode_epoch: record.mode_epoch,
            root_device: record.device_id, root_inode: record.inode,
            head_ref: manifest.head_ref, head_commit: manifest.head_commit,
            summary: summaries, unified_diff: diff,
        })
    }

    fn persist_prepared(
        &self,
        record: &WorkspaceRow,
        manifest: &CanonicalManifest,
        digest: &str,
        prepared: &[PreparedOperation],
        payload_path: &Path,
    ) -> Result<(), ChangesetError> {
        let mut guard = self.conn.lock();
        let transaction = guard.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        transaction.execute(
            "INSERT INTO pending_changesets (operation_id, workspace_id, digest, state, expires_at, created_at, registration_epoch, mode_epoch, root_device, root_inode, head_ref, head_commit, payload_path) VALUES (?1, ?2, ?3, 'Prepared', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![manifest.operation_id, manifest.workspace_id, digest, manifest.expires_at as i64, manifest.created_at as i64,
                record.registration_epoch, record.mode_epoch, record.device_id as i64,
                record.inode as i64, manifest.head_ref, manifest.head_commit, payload_path.to_string_lossy()],
        ).map_err(db_error)?;
        for (index, item) in prepared.iter().enumerate() {
            let (kind, path, destination) = operation_identity(&item.operation);
            transaction.execute(
                "INSERT INTO changeset_files (operation_id, file_index, op_kind, relative_path, destination_path, base_sha256, proposed_sha256, effect_state) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending')",
                params![manifest.operation_id, index as i64, kind, path, destination, item.base_sha256, item.proposed_sha256],
            ).map_err(db_error)?;
        }
        transaction.execute(
            "INSERT INTO audit_events (workspace_id, operation_id, event_type, outcome, created_at) VALUES (?1, ?2, 'changeset_prepare', 'prepared', ?3)",
            params![manifest.workspace_id, manifest.operation_id, manifest.created_at as i64],
        ).map_err(db_error)?;
        transaction.execute(
            "DELETE FROM changeset_payload_cleanup WHERE operation_id=?1",
            [&manifest.operation_id],
        ).map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }

    fn enqueue_payload_cleanup(
        &self,
        operation_id: &str,
        reason: &str,
        queued_at: i64,
    ) -> Result<(), ChangesetError> {
        let mut guard = self.conn.lock();
        let transaction = guard.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        enqueue_payload_cleanup_in_transaction(&transaction, operation_id, reason, queued_at).map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }

    pub(crate) fn retry_payload_cleanup(&self) -> Result<usize, ChangesetError> {
        let rows = {
            let connection = self.conn.lock();
            let mut statement = connection.prepare(
                "SELECT operation_id, reason FROM changeset_payload_cleanup
                 ORDER BY queued_at, operation_id",
            ).map_err(db_error)?;
            statement.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
                .map_err(db_error)?
                .collect::<Result<Vec<_>, _>>().map_err(db_error)?
        };
        let operation_ids: Vec<_> = rows.iter()
            .filter(|(id, reason)| cleanup_row_is_retryable(id, reason))
            .map(|(id, _)| id.clone())
            .collect();
        for operation_id in operation_ids {
            self.retry_one_payload_cleanup(&operation_id)?;
        }
        let remaining = {
            let connection = self.conn.lock();
            let mut statement = connection.prepare(
                "SELECT operation_id, reason FROM changeset_payload_cleanup",
            ).map_err(db_error)?;
            statement.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
                .map_err(db_error)?
                .collect::<Result<Vec<_>, _>>().map_err(db_error)?
        }.iter()
            .filter(|(id, reason)| cleanup_row_is_retryable(id, reason))
            .count();
        Ok(remaining)
    }

    fn retry_one_payload_cleanup(&self, operation_id: &str) -> Result<(), ChangesetError> {
        if !canonical_operation_id(operation_id) {
            return self.recover_malformed_payload_cleanup(operation_id);
        }
        let directory = match open_pending_directory(&self.pending_dir) {
            Ok(directory) => directory,
            Err(error) => return self.record_payload_cleanup_failure(
                operation_id, &format!("cleanup_directory_{}", io_error_kind(&error)),
            ),
        };
        let file_name = format!("{operation_id}.json");
        match pending_entry_type(&directory, OsStr::new(&file_name)) {
            Ok(Some(rustix::fs::FileType::RegularFile)) => {}
            Ok(None) => return self.sync_handle_and_delete_payload_cleanup_record(&directory, operation_id),
            Ok(Some(_)) => return self.record_payload_cleanup_failure(operation_id, "payload_not_regular"),
            Err(error) => return self.record_payload_cleanup_failure(
                operation_id, &format!("cleanup_metadata_{}", io_error_kind(&error)),
            ),
        }
        if let Err(error) = unlink_pending_entry(&directory, OsStr::new(&file_name)) {
            return self.record_payload_cleanup_failure(
                operation_id, &format!("cleanup_unlink_{}", io_error_kind(&error)),
            );
        }
        self.sync_handle_and_delete_payload_cleanup_record(&directory, operation_id)
    }

    fn recover_malformed_payload_cleanup(&self, operation_id: &str) -> Result<(), ChangesetError> {
        self.recover_malformed_payload_cleanup_with(operation_id, unlink_pending_entry)
    }

    fn recover_malformed_payload_cleanup_with(
        &self,
        operation_id: &str,
        mut unlink: impl FnMut(&File, &OsStr) -> io::Result<()>,
    ) -> Result<(), ChangesetError> {
        const MAX_PENDING_ENTRIES: usize = 10_000;
        let directory = match open_pending_directory(&self.pending_dir) {
            Ok(directory) => directory,
            Err(error) => return self.record_payload_cleanup_failure(
                operation_id, &format!("recovery_directory_{}", io_error_kind(&error)),
            ),
        };
        if let Some(target_id) = self.payload_cleanup_target(operation_id)? {
            return self.recover_persisted_target(operation_id, &target_id, &directory, &mut unlink);
        }
        let entries = match read_pending_entries(&directory) {
            Ok(entries) => entries,
            Err(error) => return self.record_payload_cleanup_failure(
                operation_id, &format!("recovery_scan_{}", io_error_kind(&error)),
            ),
        };
        let mut seen = 0usize;
        let mut candidates = Vec::new();
        for entry in entries {
            seen = seen.saturating_add(1);
            if seen > MAX_PENDING_ENTRIES {
                return self.record_payload_cleanup_failure(operation_id, "recovery_scan_limit");
            }
            let file_name = match entry {
                Ok(entry) => entry.file_name(),
                Err(error) => return self.record_payload_cleanup_failure(
                    operation_id, &format!("recovery_scan_{}", io_error_kind(&error)),
                ),
            };
            let Some(id) = file_name.to_str().and_then(canonical_payload_id) else { continue };
            let file_type = match pending_entry_type(&directory, &file_name) {
                Ok(Some(file_type)) => file_type,
                Ok(None) => return self.record_payload_cleanup_failure(
                    operation_id, "recovery_metadata_not_found",
                ),
                Err(error) => return self.record_payload_cleanup_failure(
                    operation_id, &format!("recovery_metadata_{}", io_error_kind(&error)),
                ),
            };
            if file_type != rustix::fs::FileType::RegularFile { continue; }
            let candidate_path = self.pending_dir.join(&file_name);
            match self.payload_is_referenced(&id, &candidate_path) {
                Ok(true) => continue,
                Ok(false) => candidates.push(id),
                Err(_) => return self.record_payload_cleanup_failure(
                    operation_id, "recovery_reference_check_failed",
                ),
            }
        }
        let target_id = match candidates.as_slice() {
            [target] => target,
            [] => return self.record_payload_cleanup_failure(operation_id, "recovery_no_orphan_found"),
            _ => return self.record_payload_cleanup_failure(operation_id, "recovery_ambiguous_orphans"),
        };
        let target_id = match self.persist_payload_cleanup_target(operation_id, target_id)? {
            Some(target) => target,
            None => return self.record_payload_cleanup_failure(operation_id, "recovery_target_conflict"),
        };
        self.recover_persisted_target(operation_id, &target_id, &directory, &mut unlink)
    }

    fn payload_cleanup_target(&self, operation_id: &str) -> Result<Option<String>, ChangesetError> {
        self.conn.lock().query_row(
            "SELECT recovery_target_id FROM changeset_payload_cleanup WHERE operation_id=?1",
            [operation_id],
            |row| row.get(0),
        ).map_err(db_error)
    }

    fn persist_payload_cleanup_target(
        &self,
        operation_id: &str,
        target_id: &str,
    ) -> Result<Option<String>, ChangesetError> {
        let mut connection = self.conn.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        transaction.execute(
            "UPDATE changeset_payload_cleanup SET recovery_target_id=?1
             WHERE operation_id=?2 AND recovery_target_id IS NULL",
            params![target_id, operation_id],
        ).map_err(db_error)?;
        let saved: Option<String> = transaction.query_row(
            "SELECT recovery_target_id FROM changeset_payload_cleanup WHERE operation_id=?1",
            [operation_id],
            |row| row.get(0),
        ).map_err(db_error)?;
        transaction.commit().map_err(db_error)?;
        Ok(saved.filter(|saved| saved == target_id))
    }

    fn recover_persisted_target(
        &self,
        operation_id: &str,
        target_id: &str,
        directory: &File,
        unlink: &mut impl FnMut(&File, &OsStr) -> io::Result<()>,
    ) -> Result<(), ChangesetError> {
        if !canonical_operation_id(target_id) {
            return self.record_payload_cleanup_failure(operation_id, "recovery_target_invalid");
        }
        let file_name = format!("{target_id}.json");
        match pending_entry_type(directory, OsStr::new(&file_name)) {
            Ok(None) => return self.sync_handle_and_delete_payload_cleanup_record(directory, operation_id),
            Ok(Some(rustix::fs::FileType::RegularFile)) => {}
            Ok(Some(_)) => return self.record_payload_cleanup_failure(operation_id, "recovery_target_not_regular"),
            Err(error) => return self.record_payload_cleanup_failure(
                operation_id, &format!("recovery_target_metadata_{}", io_error_kind(&error)),
            ),
        }
        let candidate_path = self.pending_dir.join(&file_name);
        match self.payload_is_referenced(target_id, &candidate_path) {
            Ok(false) => {}
            Ok(true) => return self.record_payload_cleanup_failure(operation_id, "recovery_target_referenced"),
            Err(_) => return self.record_payload_cleanup_failure(
                operation_id, "recovery_target_reference_check_failed",
            ),
        }
        if let Err(error) = unlink(directory, OsStr::new(&file_name)) {
            return self.record_payload_cleanup_failure(
                operation_id, &format!("recovery_unlink_{}", io_error_kind(&error)),
            );
        }
        self.sync_handle_and_delete_payload_cleanup_record(directory, operation_id)
    }

    fn payload_is_referenced(&self, id: &str, path: &Path) -> Result<bool, ChangesetError> {
        let path = path.to_str().ok_or_else(|| {
            ChangesetError::new(ErrorCode::Conflict, "pending payload path is not valid UTF-8")
        })?;
        let connection = self.conn.lock();
        // Only states that still need their serialized operations own payload bytes.
        // Archives retain status and hashes, not payload bytes or payload_path ownership.
        connection.query_row(
            "SELECT
                EXISTS(SELECT 1 FROM pending_changesets
                       WHERE payload_path=?1 AND state IN ('Prepared','Applying'))
                OR EXISTS(SELECT 1 FROM changeset_payload_cleanup WHERE operation_id=?2)",
            params![path, id],
            |row| row.get(0),
        ).map_err(db_error)
    }

    #[cfg(test)]
    pub(crate) fn retry_malformed_cleanup_with_unlink_failure(
        &self,
        operation_id: &str,
    ) -> Result<(), ChangesetError> {
        self.recover_malformed_payload_cleanup_with(operation_id, |_, _| {
            Err(io::Error::from(io::ErrorKind::PermissionDenied))
        })
    }

    fn sync_handle_and_delete_payload_cleanup_record(
        &self,
        directory: &File,
        operation_id: &str,
    ) -> Result<(), ChangesetError> {
        match directory.sync_all() {
            Ok(()) => self.delete_payload_cleanup_record(operation_id),
            Err(error) => self.record_payload_cleanup_failure(
                operation_id, &format!("directory_sync_{}", io_error_kind(&error)),
            ),
        }
    }

    fn delete_payload_cleanup_record(&self, operation_id: &str) -> Result<(), ChangesetError> {
        self.conn.lock().execute(
            "DELETE FROM changeset_payload_cleanup WHERE operation_id=?1",
            [operation_id],
        ).map(|_| ()).map_err(db_error)
    }

    fn record_payload_cleanup_failure(
        &self,
        operation_id: &str,
        error_kind: &str,
    ) -> Result<(), ChangesetError> {
        self.conn.lock().execute(
            "UPDATE changeset_payload_cleanup
             SET attempts=attempts+1, last_error=?1 WHERE operation_id=?2",
            params![error_kind, operation_id],
        ).map(|_| ()).map_err(db_error)
    }

    fn retry_failed_prepare_best_effort(&self, operation_id: &str) {
        let queued = self.conn.lock().execute(
            "UPDATE changeset_payload_cleanup SET reason='prepare_failed'
             WHERE operation_id=?1 AND reason='prepare'",
            [operation_id],
        );
        let cleanup = self.retry_one_payload_cleanup(operation_id);
        if queued.is_err() || cleanup.is_err() {
            eprintln!("workspace-index: failed-prepare payload cleanup remains durably queued");
        }
    }
    fn retry_payload_cleanup_best_effort(&self) {
        match self.retry_payload_cleanup() {
            Ok(0) => {}
            Ok(_) | Err(_) => eprintln!("workspace-index: private changeset payload cleanup remains durably queued"),
        }
    }
    fn prepare_operation(&self, fs: &WorkspaceFs, operation: ChangeOperation, touched: &mut BTreeSet<String>, total_bytes: &mut usize) -> Result<PreparedOperation, ChangesetError> {
        let prepared = match operation {
            ChangeOperation::Create { path, content } => {
                validate_relative_path(&path)?; reserve_path(touched, &path)?;
                fs.resolve_parent(Path::new(&path)).map_err(fs_error)?;
                if content.as_bytes().contains(&0) { return Err(ChangesetError::new(ErrorCode::InvalidArgument, "binary content is not allowed")); }
                if read_optional(fs, &path, self.max_file_bytes)?.is_some() { return Err(ChangesetError::new(ErrorCode::Conflict, "create target already exists")); }
                *total_bytes = total_bytes.saturating_add(content.len());
                PreparedOperation { operation: ChangeOperation::Create { path, content: content.clone() }, base_content: None, proposed_sha256: Some(sha256(content.as_bytes())), proposed_content: Some(content), base_sha256: None }
            }
            ChangeOperation::Replace { path, expected_sha256, replacements } => {
                validate_relative_path(&path)?; reserve_path(touched, &path)?; validate_sha256(&expected_sha256)?;
                let base = read_required(fs, &path, self.max_file_bytes)?;
                let base_hash = sha256(base.as_bytes());
                if base_hash != expected_sha256 { return Err(ChangesetError::new(ErrorCode::Conflict, "replace preimage SHA-256 mismatch")); }
                let mut proposed = base.clone();
                for replacement in &replacements {
                    if replacement.replace_all || replacement.old_text.is_empty() { return Err(ChangesetError::new(ErrorCode::InvalidArgument, "replacement must specify nonempty old_text and replace_all=false")); }
                    let mut occurrences = proposed.match_indices(&replacement.old_text);
                    if occurrences.next().is_none() || occurrences.next().is_some() { return Err(ChangesetError::new(ErrorCode::Conflict, "old_text must occur exactly once")); }
                    proposed = proposed.replacen(&replacement.old_text, &replacement.new_text, 1);
                }
                *total_bytes = total_bytes.saturating_add(proposed.len());
                PreparedOperation { operation: ChangeOperation::Replace { path, expected_sha256, replacements }, base_content: Some(base), base_sha256: Some(base_hash), proposed_sha256: Some(sha256(proposed.as_bytes())), proposed_content: Some(proposed) }
            }
            ChangeOperation::Delete { path, expected_sha256 } => {
                validate_relative_path(&path)?; reserve_path(touched, &path)?; validate_sha256(&expected_sha256)?;
                let base = read_required(fs, &path, self.max_file_bytes)?;
                let hash = sha256(base.as_bytes());
                if hash != expected_sha256 { return Err(ChangesetError::new(ErrorCode::Conflict, "delete preimage SHA-256 mismatch")); }
                PreparedOperation { operation: ChangeOperation::Delete { path, expected_sha256 }, base_content: Some(base), base_sha256: Some(hash), proposed_sha256: None, proposed_content: None }
            }
            ChangeOperation::Move { from_path, to_path, expected_sha256 } => {
                validate_relative_path(&from_path)?; validate_relative_path(&to_path)?; validate_sha256(&expected_sha256)?;
                reserve_path(touched, &from_path)?; reserve_path(touched, &to_path)?;
                if from_path == to_path { return Err(ChangesetError::new(ErrorCode::InvalidArgument, "move source and destination must differ")); }
                let base = read_required(fs, &from_path, self.max_file_bytes)?;
                let hash = sha256(base.as_bytes());
                if hash != expected_sha256 { return Err(ChangesetError::new(ErrorCode::Conflict, "move preimage SHA-256 mismatch")); }
                fs.resolve_parent(Path::new(&to_path)).map_err(fs_error)?;
                if read_optional(fs, &to_path, self.max_file_bytes)?.is_some() { return Err(ChangesetError::new(ErrorCode::Conflict, "move destination already exists")); }
                PreparedOperation { operation: ChangeOperation::Move { from_path, to_path, expected_sha256 }, base_content: Some(base.clone()), proposed_content: Some(base), base_sha256: Some(hash.clone()), proposed_sha256: Some(hash) }
            }
        };
        if *total_bytes > MAX_PROPOSED_BYTES { return Err(ChangesetError::new(ErrorCode::InvalidArgument, "proposed bytes exceed 2 MiB")); }
        Ok(prepared)
    }

    pub fn apply(&self, operation_id: &str, digest: &str) -> Result<ChangesetOutcome, ChangesetError> {
        self.reject_if_archived(operation_id)?;
        let row = self.pending_row(operation_id)?;
        if row.digest != digest { return Err(ChangesetError::new(ErrorCode::Conflict, "changeset digest mismatch")); }
        if row.state == "Applying" { return Err(ChangesetError::new(ErrorCode::Busy, "changeset is already being applied")); }
        if row.state != "Prepared" { return self.outcome(operation_id); }
        if unix_seconds() >= row.expires_at {
            self.finish_terminal(operation_id, &row, "Expired", "changeset_expiry", "expired")?;
            self.retry_payload_cleanup_best_effort();
            return Err(ChangesetError::new(ErrorCode::Expired, "changeset has expired"));
        }
        if row.mode != WorkspaceMode::ReadWrite { return Err(ChangesetError::new(ErrorCode::ReadOnly, "workspace is read-only")); }
        let now_record = self.workspace_record(&row.workspace_id)?;
        if now_record.root_path != row.root_path || now_record.device_id != row.root_device
            || now_record.inode != row.root_inode || now_record.registration_epoch != row.registration_epoch
            || now_record.mode_epoch != row.mode_epoch || now_record.mode != WorkspaceMode::ReadWrite
        {
            return Err(ChangesetError::new(ErrorCode::Conflict, "workspace identity or mode changed after prepare"));
        }
        let git = current_git_identity(Path::new(&row.root_path), &now_record)?;
        if git.as_ref().and_then(|identity| identity.head_ref.as_deref()) != row.head_ref.as_deref()
            || git.as_ref().and_then(|identity| identity.head_commit.as_deref()) != row.head_commit.as_deref()
        {
            return Err(ChangesetError::new(ErrorCode::Conflict, "Git branch or HEAD changed after prepare"));
        }
        let root_fs = WorkspaceFs::open_registered(
            Path::new(&row.root_path), row.root_device, row.root_inode,
        ).map_err(fs_error)?;
        let expected_payload_path = exact_payload_path(&self.pending_dir, operation_id)
            .map_err(|_| ChangesetError::new(ErrorCode::Conflict, "changeset payload path is invalid"))?;
        if row.payload_path != expected_payload_path {
            return Err(ChangesetError::new(ErrorCode::Conflict, "changeset payload path is invalid"));
        }
        let payload_bytes = fs::read(&expected_payload_path).map_err(io_error)?;
        let payload: Payload = serde_json::from_slice(&payload_bytes).map_err(json_error)?;
        if payload.operations.len() > MAX_OPERATIONS || payload.operations.iter().any(|item| !prepared_operation_valid(item)) {
            return Err(ChangesetError::new(ErrorCode::Conflict, "changeset payload failed integrity checks"));
        }
        let canonical = CanonicalManifest {
            version: 1, operation_id: operation_id.into(), workspace_id: row.workspace_id.clone(),
            registration_epoch: row.registration_epoch, mode_epoch: row.mode_epoch, root_device: row.root_device,
            root_inode: row.root_inode, head_ref: row.head_ref.clone(), head_commit: row.head_commit.clone(),
            expires_at: row.expires_at, created_at: row.created_at, files: payload.operations.iter().map(manifest_file).collect(),
        };
        let actual_digest = hex(&Sha256::digest(serde_json::to_vec(&canonical).map_err(json_error)?));
        if actual_digest != row.digest { return Err(ChangesetError::new(ErrorCode::Conflict, "changeset payload digest mismatch")); }
        // Check for uncertain paths
        let mut has_uncertain = false;
        for item in &payload.operations {
            let (_, path, destination) = operation_identity(&item.operation);
            let mut check_path = |p: &str| -> Result<(), ChangesetError> {
                let count: i64 = self.conn.lock().query_row(
                    "SELECT count(*) FROM changeset_files c JOIN pending_changesets p ON c.operation_id = p.operation_id WHERE p.workspace_id = ?1 AND (c.relative_path = ?2 OR c.destination_path = ?2) AND c.effect_state IN ('unknown', 'intent') AND c.intent_at > ?3 AND c.operation_id != ?4",
                    params![row.workspace_id, p, row.created_at as i64, operation_id],
                    |r| r.get(0)
                ).unwrap_or(0);
                if count > 0 { has_uncertain = true; }
                Ok(())
            };
            check_path(path)?;
            if let Some(dest) = destination { check_path(dest)?; }
        }
        if has_uncertain {
            return Err(ChangesetError::new(ErrorCode::Conflict, "path has uncertain effects from a later operation; prepare a new changeset to capture current conditions"));
        }

        if !WorkspaceFs::mutation_supported() {
            return Err(ChangesetError::new(
                ErrorCode::Unsupported,
                "source mutation backend cannot prevent concurrent parent-directory relocation",
            ));
        }

        let claim = self.conn.lock().execute(
            "UPDATE pending_changesets SET state = 'Applying'
             WHERE operation_id = ?1 AND digest = ?2 AND state = 'Prepared'",
            params![operation_id, digest],
        ).map_err(db_error)?;
        if claim != 1 { return Err(ChangesetError::new(ErrorCode::Busy, "changeset is already being applied")); }
        let mut completed = 0usize;
        for (index, item) in payload.operations.iter().enumerate() {
            self.conn.lock().execute("UPDATE changeset_files SET effect_state = 'intent', intent_at = ?1 WHERE operation_id = ?2 AND file_index = ?3", params![unix_seconds() as i64, operation_id, index as i64]).map_err(db_error)?;
            let applied = apply_operation(&root_fs, item);
            match applied {
                Ok(()) => {
                    completed += 1;
                    let (_, path, destination) = operation_identity(&item.operation);
                    let observed = destination.or(match &item.operation { ChangeOperation::Create { path, .. } | ChangeOperation::Replace { path, .. } => Some(path.as_str()), _ => None });
                    
                    let observed_hash_result = match observed {
                        Some(p) => root_fs.sha256(Path::new(p), self.max_file_bytes).map(Some).map_err(|e| e.to_string()),
                        None => Ok(None),
                    };
                    
                    let journaling_result = observed_hash_result.and_then(|observed_hash| {
                        self.conn.lock().execute("UPDATE changeset_files SET effect_state = 'applied', observed_sha256 = ?1, observed_at = ?2 WHERE operation_id = ?3 AND file_index = ?4", params![observed_hash, unix_seconds() as i64, operation_id, index as i64]).map_err(|e| e.to_string())?;
                        self.conn.lock().execute("INSERT OR IGNORE INTO index_jobs (workspace_id, relative_path, priority, state, created_at) VALUES (?1, ?2, 0, 'queued', ?3)", params![row.workspace_id, path, unix_seconds() as i64]).map_err(|e| e.to_string())?;
                        if let Some(destination) = destination { self.conn.lock().execute("INSERT OR IGNORE INTO index_jobs (workspace_id, relative_path, priority, state, created_at) VALUES (?1, ?2, 0, 'queued', ?3)", params![row.workspace_id, destination, unix_seconds() as i64]).map_err(|e| e.to_string())?; }
                        Ok(())
                    });

                    if let Err(error) = journaling_result {
                        let state = if completed > 1 { "Partial" } else { "EffectUnknown" };
                        let _ = self.conn.lock().execute(
                            "UPDATE changeset_files SET effect_state='unknown', observed_at=?1 WHERE operation_id=?2 AND file_index=?3",
                            params![unix_seconds() as i64, operation_id, index as i64],
                        );
                        if self.finish_terminal(operation_id, &row, state, "changeset_apply", state).is_ok() {
                            self.retry_payload_cleanup_best_effort();
                        }
                        let mut outcome = self.outcome(operation_id)?;
                        outcome.source_status = if completed > 1 { "partial" } else { "effect_unknown" }.into();
                        outcome.index_status = "degraded".into();
                        outcome.warnings = vec![Warning { code: state.into(), message: format!("Observation or journaling failed: {}", error), retryable: false }];
                        return Ok(outcome);
                    }
                }
                Err(error) => {
                    let state = if completed > 0 { "Partial" } else { "EffectUnknown" };
                    self.conn.lock().execute(
                        "UPDATE changeset_files SET effect_state='unknown', observed_at=?1 WHERE operation_id=?2 AND file_index=?3",
                        params![unix_seconds() as i64, operation_id, index as i64],
                    ).map_err(db_error)?;
                    self.finish_terminal(operation_id, &row, state, "changeset_apply", state)?;
                    self.retry_payload_cleanup_best_effort();
                    let mut outcome = self.outcome(operation_id)?;
                    outcome.source_status = if completed > 0 { "partial" } else { "effect_unknown" }.into();
                    outcome.index_status = "degraded".into();
                    outcome.warnings = vec![Warning { code: state.into(), message: error.to_string(), retryable: false }];
                    return Ok(outcome);
                }
            }
        }
        self.finish_terminal(operation_id, &row, "Applied", "changeset_apply", "applied")?;
        self.retry_payload_cleanup_best_effort();
        let mut outcome = self.outcome(operation_id)?;
        outcome.source_status = "applied".into();
        outcome.index_status = "degraded".into();
        
        Ok(outcome)
    }

    pub fn status(&self, operation_id: &str) -> Result<ChangesetOutcome, ChangesetError> {
        if self.expire_prepared_for_status(operation_id)? {
            self.retry_payload_cleanup_best_effort();
        }
        self.outcome(operation_id)
    }

    fn expire_prepared_for_status(&self, operation_id: &str) -> Result<bool, ChangesetError> {
        let now = unix_seconds() as i64;
        let mut guard = self.conn.lock();
        let transaction = guard.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        let workspace_id = transaction.query_row(
            "SELECT workspace_id FROM pending_changesets
             WHERE operation_id=?1 AND state='Prepared' AND expires_at<=?2",
            params![operation_id, now],
            |row| row.get::<_, String>(0),
        ).optional().map_err(db_error)?;
        let Some(workspace_id) = workspace_id else {
            transaction.commit().map_err(db_error)?;
            return Ok(false);
        };
        let changed = transaction.execute(
            "UPDATE pending_changesets SET state='Expired'
             WHERE operation_id=?1 AND state='Prepared' AND expires_at<=?2",
            params![operation_id, now],
        ).map_err(db_error)?;
        if changed == 1 {
            enqueue_payload_cleanup_in_transaction(&transaction, operation_id, "expired", now).map_err(db_error)?;
            transaction.execute(
                "INSERT INTO audit_events (workspace_id, operation_id, event_type, outcome, created_at)
                 VALUES (?1, ?2, 'changeset_expiry', 'expired', ?3)",
                params![workspace_id, operation_id, now],
            ).map_err(db_error)?;
        }
        transaction.commit().map_err(db_error)?;
        Ok(changed == 1)
    }

    pub fn discard(&self, operation_id: &str, digest: &str) -> Result<ChangesetOutcome, ChangesetError> {
        self.reject_if_archived(operation_id)?;
        let row = self.pending_row(operation_id)?;
        if row.digest != digest { return Err(ChangesetError::new(ErrorCode::Conflict, "changeset digest mismatch")); }
        let mut guard = self.conn.lock();
        let transaction = guard.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        let changed = transaction.execute(
            "UPDATE pending_changesets SET state='Discarded'
             WHERE operation_id=?1 AND digest=?2 AND state='Prepared'",
            params![operation_id, digest],
        ).map_err(db_error)?;
        if changed != 1 { return Err(ChangesetError::new(ErrorCode::Busy, "only prepared changesets can be discarded")); }
        let now = unix_seconds() as i64;
        enqueue_payload_cleanup_in_transaction(&transaction, operation_id, "discarded", now).map_err(db_error)?;
        transaction.execute(
            "INSERT INTO audit_events (workspace_id, operation_id, event_type, outcome, created_at)
             VALUES (?1, ?2, 'changeset_discard', 'discarded', ?3)",
            params![row.workspace_id, operation_id, now],
        ).map_err(db_error)?;
        transaction.commit().map_err(db_error)?;
        drop(guard);
        self.retry_payload_cleanup_best_effort();
        self.outcome(operation_id)
    }

    fn reject_if_archived(&self, operation_id: &str) -> Result<(), ChangesetError> {
        let archived = self.conn.lock().query_row(
            "SELECT EXISTS(SELECT 1 FROM unregistered_changesets WHERE operation_id=?1)",
            [operation_id],
            |row| row.get::<_, bool>(0),
        ).map_err(db_error)?;
        if archived {
            return Err(ChangesetError::new(
                ErrorCode::Conflict,
                "an unregistered changeset is archived and cannot be applied or discarded",
            ));
        }
        Ok(())
    }

    fn finish_terminal(
        &self,
        operation_id: &str,
        row: &PendingRow,
        state: &str,
        event_type: &str,
        event_outcome: &str,
    ) -> Result<(), ChangesetError> {
        let now = unix_seconds() as i64;
        let mut guard = self.conn.lock();
        let transaction = guard.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        let updated = transaction.execute(
            "UPDATE pending_changesets SET state=?1 WHERE operation_id=?2",
            params![state, operation_id],
        ).map_err(db_error)?;
        if updated != 1 { return Err(ChangesetError::new(ErrorCode::WorkspaceNotFound, "changeset not found")); }
        enqueue_payload_cleanup_in_transaction(&transaction, operation_id, "terminal", now).map_err(db_error)?;
        transaction.execute(
            "INSERT INTO audit_events (workspace_id, operation_id, event_type, outcome, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![row.workspace_id, operation_id, event_type, event_outcome, now],
        ).map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }
    fn recover_interrupted(&self) -> Result<(), ChangesetError> {
        let ids = {
            let guard = self.conn.lock();
            let mut stmt = guard.prepare("SELECT operation_id FROM pending_changesets WHERE state = 'Applying'").map_err(db_error)?;
            stmt.query_map([], |row| row.get::<_, String>(0)).map_err(db_error)?.collect::<Result<Vec<_>, _>>().map_err(db_error)?
        };
        for operation_id in ids { self.recover_one(&operation_id)?; }
        Ok(())
    }
    pub(crate) fn expire_prepared_payloads(&self) -> Result<(), ChangesetError> {
        let now = unix_seconds() as i64;
        let mut guard = self.conn.lock();
        let transaction = guard.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        let expired = {
            let mut statement = transaction.prepare(
                "SELECT operation_id, workspace_id FROM pending_changesets
                 WHERE state='Prepared' AND expires_at<=?1",
            ).map_err(db_error)?;
            statement.query_map([now], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
                .map_err(db_error)?.collect::<Result<Vec<_>, _>>().map_err(db_error)?
        };
        for (operation_id, workspace_id) in expired {
            let changed = transaction.execute(
                "UPDATE pending_changesets SET state='Expired'
                 WHERE operation_id=?1 AND state='Prepared' AND expires_at<=?2",
                params![operation_id, now],
            ).map_err(db_error)?;
            if changed == 1 {
                enqueue_payload_cleanup_in_transaction(&transaction, &operation_id, "expired", now).map_err(db_error)?;
                transaction.execute(
                    "INSERT INTO audit_events (workspace_id, operation_id, event_type, outcome, created_at)
                     VALUES (?1, ?2, 'changeset_expiry', 'expired', ?3)",
                    params![workspace_id, operation_id, now],
                ).map_err(db_error)?;
            }
        }
        transaction.commit().map_err(db_error)
    }
    pub(crate) fn enqueue_terminal_payloads(&self) -> Result<(), ChangesetError> {
        let now = unix_seconds() as i64;
        let mut guard = self.conn.lock();
        let transaction = guard.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        let operation_ids = {
            let mut statement = transaction.prepare(
                "SELECT operation_id FROM pending_changesets
                 WHERE state IN ('Applied','Partial','EffectUnknown','Expired','Discarded')",
            ).map_err(db_error)?;
            statement.query_map([], |row| row.get::<_, String>(0)).map_err(db_error)?
                .collect::<Result<Vec<_>, _>>().map_err(db_error)?
        };
        for operation_id in operation_ids {
            enqueue_payload_cleanup_in_transaction(&transaction, &operation_id, "terminal", now).map_err(db_error)?;
        }
        transaction.commit().map_err(db_error)
    }
    fn mark_abandoned_prepare_reservations(&self) -> Result<(), ChangesetError> {
        let mut guard = self.conn.lock();
        let transaction = guard.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        transaction.execute(
            "UPDATE changeset_payload_cleanup SET reason='prepare_abandoned'
             WHERE reason='prepare'",
            [],
        ).map_err(db_error)?;
        transaction.commit().map_err(db_error)
    }

    fn recover_one(&self, operation_id: &str) -> Result<(), ChangesetError> {
        let row = self.pending_row(operation_id)?;
        let workspace = WorkspaceFs::open_registered(
            Path::new(&row.root_path), row.root_device, row.root_inode,
        ).map_err(fs_error)?;
        let files = {
            let guard = self.conn.lock();
            let mut stmt = guard.prepare("SELECT file_index, op_kind, relative_path, destination_path, base_sha256, proposed_sha256, effect_state FROM changeset_files WHERE operation_id = ?1 ORDER BY file_index").map_err(db_error)?;
            stmt.query_map([operation_id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, Option<String>>(3)?, r.get::<_, Option<String>>(4)?, r.get::<_, Option<String>>(5)?, r.get::<_, String>(6)?))).map_err(db_error)?.collect::<Result<Vec<_>, _>>().map_err(db_error)?
        };
        let mut applied_count = 0usize;
        for (index, kind, path, destination, base, proposed, current_state) in &files {
            if current_state == "applied" { applied_count += 1; continue; }
            let source_now = current_sha256(&workspace, path, self.max_file_bytes)?;
            let destination_now = match destination { Some(path) => current_sha256(&workspace, path, self.max_file_bytes)?, None => None };
            let (applied, unchanged) = match kind.as_str() {
                "create" => (source_now.as_deref() == proposed.as_deref() && proposed.is_some(), source_now.is_none()),
                "replace" => (source_now.as_deref() == proposed.as_deref() && proposed.is_some(), source_now.as_deref() == base.as_deref()),
                "delete" => (source_now.is_none(), source_now.as_deref() == base.as_deref()),
                "move" => (source_now.is_none() && destination_now.as_deref() == proposed.as_deref() && proposed.is_some(), source_now.as_deref() == base.as_deref() && destination_now.is_none()),
                _ => (false, false),
            };
            let file_state = if applied { applied_count += 1; "applied" } else if unchanged { "not_applied" } else { "unknown" };
            let observed = if destination.is_some() { destination_now.as_deref() } else { source_now.as_deref() };
            self.conn.lock().execute("UPDATE changeset_files SET effect_state = ?1, observed_sha256 = ?2, observed_at = ?3 WHERE operation_id = ?4 AND file_index = ?5", params![file_state, observed, unix_seconds() as i64, operation_id, index]).map_err(db_error)?;
            if applied {
                self.conn.lock().execute("INSERT OR IGNORE INTO index_jobs (workspace_id, relative_path, priority, state, created_at) VALUES (?1, ?2, 0, 'queued', ?3)", params![row.workspace_id, path, unix_seconds() as i64]).map_err(db_error)?;
                if let Some(destination) = destination { self.conn.lock().execute("INSERT OR IGNORE INTO index_jobs (workspace_id, relative_path, priority, state, created_at) VALUES (?1, ?2, 0, 'queued', ?3)", params![row.workspace_id, destination, unix_seconds() as i64]).map_err(db_error)?; }
            }
        }
        let state = if applied_count == files.len() { "Applied" } else if applied_count > 0 { "Partial" } else { "EffectUnknown" };
        self.finish_terminal(operation_id, &row, state, "changeset_recovery", state)?;
        self.retry_payload_cleanup_best_effort();
        Ok(())
    }

    fn workspace_record(&self, workspace_id: &str) -> Result<WorkspaceRow, ChangesetError> {
        self.conn.lock().query_row(
            "SELECT root_path, device_id, inode, mode, registration_epoch, mode_epoch, git_dir, git_common_dir
             FROM workspaces WHERE id=?1",
            [workspace_id],
            |row| {
                let mode: String = row.get(3)?;
                Ok(WorkspaceRow {
                    root_path: row.get(0)?,
                    device_id: row.get::<_, i64>(1)? as u64,
                    inode: row.get::<_, i64>(2)? as u64,
                    mode: if mode == "read_write" { WorkspaceMode::ReadWrite } else { WorkspaceMode::ReadOnly },
                    registration_epoch: row.get(4)?,
                    mode_epoch: row.get(5)?,
                    git_dir: row.get(6)?,
                    git_common_dir: row.get(7)?,
                })
            },
        ).map_err(|error| if error == rusqlite::Error::QueryReturnedNoRows {
            ChangesetError::new(ErrorCode::WorkspaceNotFound, "workspace not found")
        } else { db_error(error) })
    }

    fn pending_row(&self, id: &str) -> Result<PendingRow, ChangesetError> {
        self.conn.lock().query_row("SELECT p.workspace_id, p.digest, p.state, p.expires_at, p.created_at, p.registration_epoch, p.mode_epoch, p.root_device, p.root_inode, p.head_ref, p.head_commit, p.payload_path, w.root_path, w.mode FROM pending_changesets p JOIN workspaces w ON w.id = p.workspace_id WHERE p.operation_id = ?1", [id], |r| Ok(PendingRow { workspace_id: r.get(0)?, digest: r.get(1)?, state: r.get(2)?, expires_at: r.get::<_, i64>(3)? as u64, created_at: r.get::<_, i64>(4)? as u64, registration_epoch: r.get(5)?, mode_epoch: r.get(6)?, root_device: r.get::<_, i64>(7)? as u64, root_inode: r.get::<_, i64>(8)? as u64, head_ref: r.get(9)?, head_commit: r.get(10)?, payload_path: PathBuf::from(r.get::<_, String>(11)?), root_path: r.get(12)?, mode: if r.get::<_, String>(13)? == "read_write" { WorkspaceMode::ReadWrite } else { WorkspaceMode::ReadOnly } })).map_err(|error| if error == rusqlite::Error::QueryReturnedNoRows { ChangesetError::new(ErrorCode::Conflict, "changeset not found") } else { db_error(error) })
    }

    fn outcome(&self, id: &str) -> Result<ChangesetOutcome, ChangesetError> {
        let archived = {
            self.conn.lock().query_row(
                "SELECT workspace_id, digest, state, archive_json, archived_at
                 FROM unregistered_changesets WHERE operation_id=?1",
                [id],
                |row| Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                )),
            ).optional().map_err(db_error)?
        };
        if let Some((workspace_id, digest, state, archive_json, archived_at)) = archived {
            return deserialize_archived_outcome(id, &workspace_id, &digest, &state, &archive_json, archived_at);
        }

        let row = self.pending_row(id)?;
        let connection = self.conn.lock();
        let files = load_file_outcomes(&connection, id)?;
        let audit_events = load_audit_events(&connection, id)?;
        Ok(outcome_from_parts(id, row.digest, row.state, files, audit_events))
    }
}
pub(crate) fn archive_workspace_changesets(
    transaction: &Transaction<'_>,
    workspace_id: &str,
    archived_at: i64,
) -> rusqlite::Result<()> {
    let rows = {
        let mut statement = transaction.prepare(
            "SELECT operation_id, workspace_id, digest, state, expires_at, created_at,
                    registration_epoch, mode_epoch, root_device, root_inode, head_ref, head_commit
             FROM pending_changesets WHERE workspace_id=?1 ORDER BY operation_id",
        )?;
        statement.query_map([workspace_id], |row| Ok(PendingArchiveRow {
            operation_id: row.get(0)?,
            workspace_id: row.get(1)?,
            digest: row.get(2)?,
            state: row.get(3)?,
            expires_at: row.get(4)?,
            created_at: row.get(5)?,
            registration_epoch: row.get(6)?,
            mode_epoch: row.get(7)?,
            root_device: row.get(8)?,
            root_inode: row.get(9)?,
            head_ref: row.get(10)?,
            head_commit: row.get(11)?,
        }))?.collect::<rusqlite::Result<Vec<_>>>()?
    };
    for row in rows {
        archive_pending_changeset(transaction, &row, archived_at)?;
    }
    Ok(())
}

fn archive_pending_changeset(
    transaction: &Transaction<'_>,
    row: &PendingArchiveRow,
    archived_at: i64,
) -> rusqlite::Result<()> {
    let files = load_file_outcomes(transaction, &row.operation_id).map_err(sql_changeset_error)?;
    let mut audit_events = load_audit_events(transaction, &row.operation_id).map_err(sql_changeset_error)?;
    let mut outcome = outcome_from_parts(&row.operation_id, row.digest.clone(), row.state.clone(), files, audit_events.clone());
    outcome.archived = true;
    outcome.source_status = "unregistered".into();
    outcome.warnings.push(Warning {
        code: "WORKSPACE_UNREGISTERED".into(),
        message: "Workspace was unregistered; this archived changeset cannot be applied or discarded.".into(),
        retryable: false,
    });
    audit_events.push(ChangesetAuditEvent {
        event_id: None,
        event_type: "workspace_unregister".into(),
        outcome: "archived".into(),
        created_at: archived_at,
    });
    outcome.audit_events = audit_events;
    let archive = ArchivedChangeset {
        version: 1,
        workspace_id: row.workspace_id.clone(),
        operation_id: row.operation_id.clone(),
        digest: row.digest.clone(),
        original_state: row.state.clone(),
        archived_at,
        expires_at: row.expires_at,
        created_at: row.created_at,
        registration_epoch: row.registration_epoch,
        mode_epoch: row.mode_epoch,
        root_device: row.root_device,
        root_inode: row.root_inode,
        head_ref: row.head_ref.clone(),
        head_commit: row.head_commit.clone(),
        outcome,
    };
    let archive_json = serde_json::to_string(&archive)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    transaction.execute(
        "INSERT INTO unregistered_changesets
            (operation_id, workspace_id, digest, state, archive_json, archived_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![row.operation_id, row.workspace_id, row.digest, row.state, archive_json, archived_at],
    )?;
    enqueue_payload_cleanup_in_transaction(transaction, &row.operation_id, "unregister", archived_at)
}

fn deserialize_archived_outcome(
    operation_id: &str,
    workspace_id: &str,
    digest: &str,
    state: &str,
    archive_json: &str,
    archived_at: i64,
) -> Result<ChangesetOutcome, ChangesetError> {
    let malformed = || ChangesetError::new(ErrorCode::Conflict, "archived changeset record is malformed");
    let archive: ArchivedChangeset = serde_json::from_str(archive_json).map_err(|_| malformed())?;
    let outcome = archive.outcome;
    if archive.version != 1
        || archive.operation_id != operation_id
        || archive.workspace_id != workspace_id
        || archive.digest != digest
        || archive.original_state != state
        || archive.archived_at != archived_at
        || outcome.operation_id != operation_id
        || outcome.digest != digest
        || outcome.state != state
        || !outcome.archived
        || outcome.source_status != "unregistered"
        || !outcome.audit_events.iter().any(|event| event.event_type == "workspace_unregister")
    {
        return Err(malformed());
    }
    if outcome.files.iter().any(|file| {
        validate_relative_path(&file.path).is_err()
            || file.destination_path.as_deref().is_some_and(|path| validate_relative_path(path).is_err())
            || [&file.base_sha256, &file.proposed_sha256, &file.observed_sha256]
                .into_iter().any(|hash| hash.as_deref().is_some_and(|value| validate_sha256(value).is_err()))
    }) {
        return Err(ChangesetError::new(ErrorCode::Conflict, "archived changeset contains an invalid path or hash"));
    }
    Ok(outcome)
}

fn load_file_outcomes(
    connection: &Connection,
    operation_id: &str,
) -> Result<Vec<FileOutcome>, ChangesetError> {
    let mut statement = connection.prepare(
        "SELECT file_index, op_kind, relative_path, destination_path, effect_state,
                base_sha256, proposed_sha256, observed_sha256, intent_at, observed_at
         FROM changeset_files WHERE operation_id=?1 ORDER BY file_index",
    ).map_err(db_error)?;
    statement.query_map([operation_id], |row| Ok(FileOutcome {
        file_index: row.get(0)?,
        operation: row.get(1)?,
        path: row.get(2)?,
        destination_path: row.get(3)?,
        state: row.get(4)?,
        base_sha256: row.get(5)?,
        proposed_sha256: row.get(6)?,
        observed_sha256: row.get(7)?,
        intent_at: row.get(8)?,
        observed_at: row.get(9)?,
    })).map_err(db_error)?.collect::<Result<Vec<_>, _>>().map_err(db_error)
}
fn load_audit_events(
    connection: &Connection,
    operation_id: &str,
) -> Result<Vec<ChangesetAuditEvent>, ChangesetError> {
    let mut statement = connection.prepare(
        "SELECT id, event_type, outcome, created_at
         FROM audit_events WHERE operation_id=?1 ORDER BY id",
    ).map_err(db_error)?;
    statement.query_map([operation_id], |row| Ok(ChangesetAuditEvent {
        event_id: Some(row.get(0)?),
        event_type: row.get(1)?,
        outcome: row.get(2)?,
        created_at: row.get(3)?,
    })).map_err(db_error)?.collect::<Result<Vec<_>, _>>().map_err(db_error)
}

fn outcome_from_parts(
    operation_id: &str,
    digest: String,
    state: String,
    files: Vec<FileOutcome>,
    audit_events: Vec<ChangesetAuditEvent>,
) -> ChangesetOutcome {
    let source_status = match state.as_str() {
        "Applied" => "applied",
        "Partial" => "partial",
        "EffectUnknown" => "effect_unknown",
        "Prepared" => "prepared",
        "Applying" => "applying",
        "Expired" => "expired",
        "Discarded" => "discarded",
        _ => "unknown",
    }.to_owned();
    let is_degraded = matches!(state.as_str(), "Applied" | "Partial" | "EffectUnknown");
    let index_status = if is_degraded { "degraded" } else { "not_applicable" };
    let mut warnings = Vec::new();
    if state == "Applied" {
        warnings.push(Warning { code: "INDEX_DEGRADED".into(), message: "Source changes applied; index reconciliation is queued.".into(), retryable: false });
    } else if state == "Partial" || state == "EffectUnknown" {
        warnings.push(Warning { code: state.to_ascii_uppercase(), message: "A source effect is partial or uncertain; do not retry this changeset.".into(), retryable: false });
    }
    ChangesetOutcome {
        operation_id: operation_id.into(),
        digest,
        state,
        source_status,
        index_status: index_status.into(),
        files,
        warnings,
        archived: false,
        audit_events,
    }
}

fn enqueue_payload_cleanup_in_transaction(
    transaction: &Transaction<'_>,
    operation_id: &str,
    reason: &str,
    queued_at: i64,
) -> rusqlite::Result<()> {
    transaction.execute(
        "INSERT INTO changeset_payload_cleanup (operation_id, reason, queued_at)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(operation_id) DO UPDATE SET reason=excluded.reason",
        params![operation_id, reason, queued_at],
    )?;
    Ok(())
}

// The opened directory handle anchors enumeration, metadata, unlink, and sync.
// A same-UID process can still replace an entry between statat and unlinkat.
fn open_pending_directory(pending_dir: &Path) -> io::Result<File> {
    let parent_path = pending_dir.parent().ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    let basename = pending_dir.file_name().ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    let parent = File::open(parent_path)?;
    #[cfg(unix)]
    {
        use rustix::fs::{openat, Mode, OFlags};
        openat(
            &parent,
            basename,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        ).map(File::from).map_err(io::Error::from)
    }
    #[cfg(not(unix))]
    {
        let _ = (parent, basename);
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

fn read_pending_entries(directory: &File) -> io::Result<ReadDir> {
    #[cfg(target_os = "linux")]
    let descriptor_path = {
        use std::os::fd::AsRawFd;
        PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd()))
    };
    #[cfg(target_os = "macos")]
    let descriptor_path = {
        use std::os::fd::AsRawFd;
        PathBuf::from(format!("/dev/fd/{}", directory.as_raw_fd()))
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = directory;
        return Err(io::Error::from(io::ErrorKind::Unsupported));
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        fs::read_dir(descriptor_path)
    }
}

fn pending_entry_type(directory: &File, name: &OsStr) -> io::Result<Option<rustix::fs::FileType>> {
    use rustix::fs::{statat, AtFlags, FileType};
    match statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(metadata) => Ok(Some(FileType::from_raw_mode(metadata.st_mode))),
        Err(error) => {
            let error = io::Error::from(error);
            if error.kind() == io::ErrorKind::NotFound { Ok(None) } else { Err(error) }
        }
    }
}

fn unlink_pending_entry(directory: &File, name: &OsStr) -> io::Result<()> {
    rustix::fs::unlinkat(directory, name, rustix::fs::AtFlags::empty())
        .map_err(io::Error::from)
}

fn exact_payload_path(pending_dir: &Path, operation_id: &str) -> Result<PathBuf, ()> {
    if !canonical_operation_id(operation_id) { return Err(()); }
    let path = pending_dir.join(format!("{operation_id}.json"));

    (path.parent() == Some(pending_dir)).then_some(path).ok_or(())
}
fn canonical_operation_id(operation_id: &str) -> bool {
    uuid::Uuid::parse_str(operation_id)
        .is_ok_and(|parsed| parsed.to_string() == operation_id)
}

fn cleanup_row_is_retryable(operation_id: &str, reason: &str) -> bool {
    reason != "prepare" || !canonical_operation_id(operation_id)
}
fn canonical_payload_id(file_name: &str) -> Option<String> {
    let id = file_name.strip_suffix(".json")?;
    let parsed = uuid::Uuid::parse_str(id).ok()?;
    (parsed.to_string() == id).then(|| id.to_owned())
}


fn io_error_kind(error: &std::io::Error) -> &'static str {
    match error.kind() {
        std::io::ErrorKind::PermissionDenied => "permission_denied",
        std::io::ErrorKind::NotADirectory => "not_a_directory",

        std::io::ErrorKind::IsADirectory => "is_directory",
        std::io::ErrorKind::InvalidInput => "invalid_input",
        std::io::ErrorKind::NotFound => "not_found",
        _ => "io_error",
    }
}

fn sql_changeset_error(error: ChangesetError) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(error))
}

#[derive(Debug)]
struct WorkspaceRow {
    root_path: String,
    device_id: u64,
    inode: u64,
    mode: WorkspaceMode,
    registration_epoch: i64,
    mode_epoch: i64,
    git_dir: Option<String>,
    git_common_dir: Option<String>,
}

fn current_git_identity(root: &Path, record: &WorkspaceRow) -> Result<Option<GitIdentity>, ChangesetError> {
    let current = resolve_git_identity(root).map_err(|_| {
        ChangesetError::new(ErrorCode::Conflict, "registered Git worktree identity could not be verified")
    })?;
    let matches = match (record.git_dir.as_deref(), record.git_common_dir.as_deref(), current.as_ref()) {
        (None, None, None) => true,
        (Some(git_dir), Some(common_dir), Some(current)) =>
            git_dir == current.git_dir && common_dir == current.common_dir,
        _ => false,
    };
    if matches { Ok(current) }
    else { Err(ChangesetError::new(ErrorCode::Conflict, "registered Git worktree identity changed")) }
}


fn apply_operation(fs: &WorkspaceFs, item: &PreparedOperation) -> Result<(), WorkspaceFsError> {
    match &item.operation {
        ChangeOperation::Create { path, .. } => fs.create(Path::new(path), item.proposed_content.as_deref().unwrap_or_default().as_bytes()),
        ChangeOperation::Replace { path, .. } => fs.replace(Path::new(path), item.base_sha256.as_deref().unwrap_or_default(), item.proposed_content.as_deref().unwrap_or_default().as_bytes()),
        ChangeOperation::Delete { path, .. } => fs.delete(Path::new(path), item.base_sha256.as_deref().unwrap_or_default()),
        ChangeOperation::Move { from_path, to_path, .. } => fs.move_file(Path::new(from_path), Path::new(to_path), item.base_sha256.as_deref().unwrap_or_default()),
    }
}
fn operation_identity(operation: &ChangeOperation) -> (&'static str, &str, Option<&str>) {
    match operation { ChangeOperation::Create { path, .. } => ("create", path, None), ChangeOperation::Replace { path, .. } => ("replace", path, None), ChangeOperation::Delete { path, .. } => ("delete", path, None), ChangeOperation::Move { from_path, to_path, .. } => ("move", from_path, Some(to_path)) }
}
fn manifest_file(item: &PreparedOperation) -> ManifestFile {
    let (kind, path, destination) = operation_identity(&item.operation);
    ManifestFile { kind: kind.into(), path: path.into(), destination_path: destination.map(str::to_owned), base_sha256: item.base_sha256.clone(), proposed_sha256: item.proposed_sha256.clone(), operation: item.operation.clone() }
}

fn prepared_operation_valid(item: &PreparedOperation) -> bool {
    if item.base_content.as_deref().map(|content| sha256(content.as_bytes())) != item.base_sha256 { return false; }
    if item.proposed_content.as_deref().map(|content| sha256(content.as_bytes())) != item.proposed_sha256 { return false; }
    match &item.operation {
        ChangeOperation::Create { path, content } => item.base_content.is_none() && item.base_sha256.is_none() && item.proposed_content.as_ref() == Some(content) && item.proposed_sha256.is_some() && validate_relative_path(path).is_ok(),
        ChangeOperation::Replace { path, expected_sha256, replacements } => {
            if item.base_sha256.as_deref() != Some(expected_sha256.as_str()) || !validate_relative_path(path).is_ok() { return false; }
            let Some(base) = item.base_content.as_deref() else { return false; };
            let Some(proposed) = item.proposed_content.as_deref() else { return false; };
            let mut rebuilt = base.to_owned();
            for replacement in replacements {
                if replacement.replace_all || replacement.old_text.is_empty() { return false; }
                let mut matches = rebuilt.match_indices(&replacement.old_text);
                if matches.next().is_none() || matches.next().is_some() { return false; }
                rebuilt = rebuilt.replacen(&replacement.old_text, &replacement.new_text, 1);
            }
            rebuilt == proposed
        }
        ChangeOperation::Delete { path, expected_sha256 } => item.base_sha256.as_deref() == Some(expected_sha256.as_str()) && item.proposed_content.is_none() && item.proposed_sha256.is_none() && validate_relative_path(path).is_ok(),
        ChangeOperation::Move { from_path, to_path, expected_sha256 } => item.base_sha256.as_deref() == Some(expected_sha256.as_str()) && item.proposed_sha256 == item.base_sha256 && item.proposed_content == item.base_content && validate_relative_path(from_path).is_ok() && validate_relative_path(to_path).is_ok() && from_path != to_path,
    }
}
fn operation_diff(item: &PreparedOperation) -> String {
    let (_, path, destination) = operation_identity(&item.operation);
    let new_path = destination.unwrap_or(path);
    let old = item.base_content.as_deref().unwrap_or("");
    let new = item.proposed_content.as_deref().unwrap_or("");
    let old_lines = if old.is_empty() { 0 } else { old.lines().count() };
    let new_lines = if new.is_empty() { 0 } else { new.lines().count() };
    let old_name = if item.base_content.is_some() { format!("a/{path}") } else { "/dev/null".into() };
    let new_name = if item.proposed_content.is_some() { format!("b/{new_path}") } else { "/dev/null".into() };
    let mut output = format!("--- {old_name}\n+++ {new_name}\n@@ -1,{old_lines} +1,{new_lines} @@\n");
    append_diff_lines(&mut output, '-', old);
    append_diff_lines(&mut output, '+', new);
    output
}
fn append_diff_lines(output: &mut String, prefix: char, content: &str) {
    for line in content.split_inclusive('\n') {
        output.push(prefix);
        output.push_str(line);
        if !line.ends_with('\n') {
            output.push('\n');
            output.push_str("\\ No newline at end of file\n");
        }
    }
}
fn reserve_path(touched: &mut BTreeSet<String>, path: &str) -> Result<(), ChangesetError> {
    if !touched.insert(path.to_owned()) { return Err(ChangesetError::new(ErrorCode::Conflict, "multiple operations touch the same path")); }
    Ok(())
}
fn validate_relative_path(path: &str) -> Result<(), ChangesetError> {
    let parsed = Path::new(path);
    let parts = path.split('/');
    if path.is_empty() || path.starts_with('/') || path.as_bytes().contains(&0) || path.contains(char::from(92))
        || path.chars().any(char::is_control)
        || parts.clone().any(|part| part.is_empty() || part == "." || part == "..")
        || parsed.is_absolute() || parsed.components().any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ChangesetError::new(ErrorCode::PathEscape, "path must be normalized and workspace-relative"));
    }
    Ok(())
}
fn validate_sha256(value: &str) -> Result<(), ChangesetError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return Err(ChangesetError::new(ErrorCode::InvalidArgument, "expected_sha256 must be 64 lowercase hexadecimal characters"));
    }
    Ok(())
}

fn read_optional(fs: &WorkspaceFs, path: &str, max_file_bytes: u64) -> Result<Option<String>, ChangesetError> {
    match fs.read(Path::new(path), max_file_bytes) {
        Ok(bytes) => {
            if bytes.get(..bytes.len().min(8192)).is_some_and(|prefix| prefix.contains(&0)) { return Err(ChangesetError::new(ErrorCode::InvalidArgument, "binary files cannot be changed")); }
            String::from_utf8(bytes).map(Some).map_err(|_| ChangesetError::new(ErrorCode::InvalidArgument, "binary or invalid UTF-8 files cannot be changed"))
        }
        Err(WorkspaceFsError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(fs_error(error)),
    }
}
fn current_sha256(fs: &WorkspaceFs, path: &str, max_file_bytes: u64) -> Result<Option<String>, ChangesetError> {
    match fs.sha256(Path::new(path), max_file_bytes) {
        Ok(hash) => Ok(Some(hash)),
        Err(WorkspaceFsError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(fs_error(error)),
    }
}
fn read_required(fs: &WorkspaceFs, path: &str, max_file_bytes: u64) -> Result<String, ChangesetError> {
    read_optional(fs, path, max_file_bytes)?.ok_or_else(|| ChangesetError::new(ErrorCode::WorkspaceNotFound, "source file does not exist"))
}
fn sha256(bytes: &[u8]) -> String { hex(&Sha256::digest(bytes)) }
fn hex(bytes: &[u8]) -> String { const HEX: &[u8; 16] = b"0123456789abcdef"; let mut out = String::with_capacity(bytes.len() * 2); for byte in bytes { out.push(HEX[(byte >> 4) as usize] as char); out.push(HEX[(byte & 15) as usize] as char); } out }
fn unix_seconds() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }
fn io_error(error: std::io::Error) -> ChangesetError { ChangesetError::new(ErrorCode::Conflict, error.to_string()) }
fn db_error(error: rusqlite::Error) -> ChangesetError { ChangesetError::new(ErrorCode::Conflict, error.to_string()) }
fn json_error(error: serde_json::Error) -> ChangesetError { ChangesetError::new(ErrorCode::InvalidArgument, error.to_string()) }
fn fs_error(error: WorkspaceFsError) -> ChangesetError {
    let (code, message) = match error {
        WorkspaceFsError::InvalidPath => (ErrorCode::InvalidArgument, "invalid workspace-relative path"),
        WorkspaceFsError::SensitivePath => (ErrorCode::InvalidArgument, "sensitive workspace paths cannot be prepared"),
        WorkspaceFsError::PathEscape => (ErrorCode::SymlinkRefused, "symlink or path escape refused"),
        WorkspaceFsError::Conflict => (ErrorCode::Conflict, "precondition or filesystem identity changed"),
        WorkspaceFsError::AlreadyExists => (ErrorCode::Conflict, "destination already exists"),
        WorkspaceFsError::NotRegularFile => (ErrorCode::InvalidArgument, "source is not a regular file"),
        WorkspaceFsError::TooLarge => (ErrorCode::InvalidArgument, "file exceeds configured limit"),
        WorkspaceFsError::Unsupported => (ErrorCode::Unsupported, "safe mutation containment cannot be proven"),
        WorkspaceFsError::Io(_) => (ErrorCode::Conflict, "filesystem operation failed safely"),
    };
    ChangesetError::new(code, message)
}
fn write_private_payload(path: &Path, payload: &[u8]) -> std::io::Result<()> {
    let mut options = OpenOptions::new(); options.write(true).create_new(true);
    #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt; options.mode(0o600); }
    let mut file = options.open(path)?; file.write_all(payload)?; file.sync_all()?;
    if let Some(parent) = path.parent() { std::fs::File::open(parent)?.sync_all()?; }
    Ok(())
}
#[cfg(unix)]
fn create_private_dir(path: &Path) -> std::io::Result<()> { use std::os::unix::fs::{DirBuilderExt, PermissionsExt}; let mut builder = std::fs::DirBuilder::new(); builder.recursive(true).mode(0o700); builder.create(path)?; std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)) }
#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> std::io::Result<()> { std::fs::create_dir_all(path) }

#[cfg(test)]
mod cleanup_recovery_tests {
    use super::*;
    use parking_lot::Mutex;
    use rusqlite::Connection;
    use tempfile::TempDir;

    #[cfg(unix)]
    #[test]
    fn retry_retires_saved_target_after_unlink_succeeded_but_reported_failure() {
        let temp = TempDir::new().unwrap();
        let pending = temp.path().join("pending");
        let connection = Mutex::new(Connection::open_in_memory().unwrap());
        let service = ChangesetService::build(
            &connection,
            pending.clone(),
            DEFAULT_EXPIRY_SECONDS,
            MAX_FILE_BYTES,
            true,
            false,
        ).unwrap();
        let operation_id = "malformed-operation-id";
        let target_id = "11111111-1111-4111-8111-111111111111";
        let payload_path = pending.join(format!("{target_id}.json"));
        std::fs::write(&payload_path, b"payload").unwrap();
        connection.lock().execute(
            "INSERT INTO changeset_payload_cleanup (operation_id, reason, queued_at)
             VALUES (?1, 'prepare_failed', 1)",
            [operation_id],
        ).unwrap();

        let first = service.recover_malformed_payload_cleanup_with(operation_id, |directory, name| {
            unlink_pending_entry(directory, name)?;
            Err(io::Error::from(io::ErrorKind::Other))
        });

        assert!(first.is_ok());
        assert!(!payload_path.exists());
        let saved_target: Option<String> = connection.lock().query_row(
            "SELECT recovery_target_id FROM changeset_payload_cleanup WHERE operation_id=?1",
            [operation_id],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(saved_target.as_deref(), Some(target_id));

        service.recover_malformed_payload_cleanup(operation_id).unwrap();

        let remaining: i64 = connection.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup WHERE operation_id=?1",
            [operation_id],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(remaining, 0);
    }
}
