use std::fs::Metadata;
use std::path::Path;
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use regex::Regex;
use std::ops::{Deref, DerefMut};
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::WorkspaceMode;
use crate::state::{Result, StateError, schema};
use crate::security::{WorkspaceFs, WorkspaceFsError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceRecord {
    pub id: String,
    pub root_path: String,
    pub device_id: u64,
    pub inode: u64,
    pub mode: WorkspaceMode,
    pub registration_epoch: i64,
    pub mode_epoch: i64,
    pub git_dir: Option<String>,
    pub git_common_dir: Option<String>,
    pub git_head_ref: Option<String>,
    pub git_head_commit: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GitIdentity {
    pub git_dir: String,
    pub common_dir: String,
    pub head_ref: Option<String>,
    pub head_commit: Option<String>,
}
type ExistingWorkspaceRow = (
    String,
    i64,
    i64,
    i64,
    i64,
    String,
    Option<String>,
    Option<String>,
);

struct WorkspaceIdentity {
    root_path: String,
    device_id: u64,
    inode: u64,
    git: Option<GitIdentity>,
}

struct WorkspaceEpochs {
    registration: i64,
    mode: i64,
}

#[derive(Debug, Clone, Default)]
pub struct RegistryOptions {
    pub allow_non_git_roots: bool,
}

enum RegistryConnection<'a> {
    Owned(Connection),
    Borrowed(&'a mut Connection),
}

impl Deref for RegistryConnection<'_> {
    type Target = Connection;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(connection) => connection,
            Self::Borrowed(connection) => connection,
        }
    }
}

impl DerefMut for RegistryConnection<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(connection) => connection,
            Self::Borrowed(connection) => connection,
        }
    }
}

pub struct WorkspaceRegistry<'a> {
    conn: RegistryConnection<'a>,
    options: RegistryOptions,
}

static ID_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z0-9][a-z0-9_-]{0,63}$").unwrap());

impl WorkspaceRegistry<'static> {
    pub fn open(db_path: &Path, options: RegistryOptions) -> Result<Self> {
        if let Some(parent) = db_path.parent() { std::fs::create_dir_all(parent)?; }
        let mut connection = Connection::open(db_path)?;
        schema::initialize(&mut connection)?;
        Ok(Self { conn: RegistryConnection::Owned(connection), options })
    }
}

impl<'a> WorkspaceRegistry<'a> {
    pub fn from_connection(conn: &'a mut Connection, options: RegistryOptions) -> Self {
        Self { conn: RegistryConnection::Borrowed(conn), options }
    }
    
    pub fn list(&self) -> Result<Vec<WorkspaceRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, root_path, device_id, inode, mode, registration_epoch, mode_epoch,
                    git_dir, git_common_dir, git_head_ref, git_head_commit
             FROM workspaces ORDER BY id"
        )?;

        let rows = stmt.query_map([], |row| {
            let mode: String = row.get(4)?;
            Ok(WorkspaceRecord {
                id: row.get(0)?,
                root_path: row.get(1)?,
                device_id: row.get::<_, i64>(2)? as u64,
                inode: row.get::<_, i64>(3)? as u64,
                mode: effective_workspace_mode(&mode),
                registration_epoch: row.get(5)?,
                mode_epoch: row.get(6)?,
                git_dir: row.get(7)?,
                git_common_dir: row.get(8)?,
                git_head_ref: row.get(9)?,
                git_head_commit: row.get(10)?,
            })
        })?;

        let records = rows.collect::<std::result::Result<Vec<_>, _>>().map_err(StateError::from)?;
        drop(stmt);
        Ok(records)
    }

    pub fn register(&mut self, id: &str, root: &Path) -> Result<WorkspaceRecord> {
        self.register_with_mode(id, root, WorkspaceMode::ReadOnly)
    }

    pub fn register_with_mode(&mut self, id: &str, root: &Path, mode: WorkspaceMode) -> Result<WorkspaceRecord> {
        let (root_path, device_id, inode, git) = Self::registration_identity(id, root, &self.options)?;
        let mode_name = mode_name(mode);
        let now = now_millis();
        let tx = self.conn.transaction()?;
        let existing: Option<ExistingWorkspaceRow> = tx.query_row(
            "SELECT root_path, device_id, inode, registration_epoch, mode_epoch, mode, git_dir, git_common_dir
             FROM workspaces WHERE id=?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?)),
        ).optional()?;

        if let Some((existing_root, existing_device, existing_inode, registration_epoch, mode_epoch, existing_mode, git_dir, common_dir)) = existing {
            if existing_root != root_path
                || existing_device as u64 != device_id
                || existing_inode as u64 != inode
                || workspace_mode(&existing_mode) != mode
                || !git_storage_matches(git_dir.as_deref(), common_dir.as_deref(), git.as_ref())
            {
                return Err(registration_conflict(id));
            }
            if mode == WorkspaceMode::ReadWrite {
                ensure_mutation_backend(&root_path, device_id, inode)?;
            }
            tx.execute(
                "UPDATE workspaces SET git_dir=?1, git_common_dir=?2, git_head_ref=?3,
                    git_head_commit=?4, updated_at=?5 WHERE id=?6",
                params![git.as_ref().map(|git| &git.git_dir), git.as_ref().map(|git| &git.common_dir),
                    git.as_ref().and_then(|git| git.head_ref.as_deref()),
                    git.as_ref().and_then(|git| git.head_commit.as_deref()), now, id],
            )?;
            tx.commit()?;
            return Ok(workspace_record(
                id,
                WorkspaceIdentity { root_path, device_id, inode, git },
                mode,
                WorkspaceEpochs { registration: registration_epoch, mode: mode_epoch },
            ));
        }

        Self::check_overlap(&tx, &root_path, device_id, inode, None)?;
        if mode == WorkspaceMode::ReadWrite {
            ensure_mutation_backend(&root_path, device_id, inode)?;
        }
        let epochs: Option<(i64, i64)> = tx.query_row(
            "SELECT registration_epoch, mode_epoch FROM workspace_registration_epochs WHERE id=?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?;
        let (registration_epoch, mode_epoch) = epochs.unwrap_or((1, 1));
        tx.execute(
            "INSERT INTO workspaces (
                id, root_path, device_id, inode, mode, registration_epoch, mode_epoch,
                created_at, updated_at, git_dir, git_common_dir, git_head_ref, git_head_commit
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9, ?10, ?11, ?12)",
            params![id, root_path, device_id as i64, inode as i64, mode_name, registration_epoch, mode_epoch, now,
                git.as_ref().map(|git| git.git_dir.as_str()),
                git.as_ref().map(|git| git.common_dir.as_str()),
                git.as_ref().and_then(|git| git.head_ref.as_deref()),
                git.as_ref().and_then(|git| git.head_commit.as_deref())],
        )?;
        tx.execute(
            "INSERT INTO workspace_registration_epochs (id, registration_epoch, mode_epoch)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET registration_epoch=excluded.registration_epoch,
                 mode_epoch=excluded.mode_epoch",
            params![id, registration_epoch, mode_epoch],
        )?;
        tx.commit()?;
        Ok(workspace_record(
            id,
            WorkspaceIdentity { root_path, device_id, inode, git },
            mode,
            WorkspaceEpochs { registration: registration_epoch, mode: mode_epoch },
        ))
    }

    fn registration_identity(
        id: &str,
        root: &Path,
        options: &RegistryOptions,
    ) -> Result<(String, u64, u64, Option<GitIdentity>)> {
        if !ID_REGEX.is_match(id) { return Err(StateError::InvalidWorkspaceId(id.to_owned())); }
        if !root.is_absolute() { return Err(StateError::InvalidWorkspaceRoot("Path must be absolute".into())); }

        let canonical_root = root.canonicalize().map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                StateError::InvalidWorkspaceRoot("Directory does not exist".into())
            } else { StateError::Io(error) }
        })?;
        let root_path = canonical_root.to_string_lossy().into_owned();
        if root_path == "/" { return Err(StateError::InvalidWorkspaceRoot("Cannot register root directory (/)".into())); }
        if dirs::home_dir().is_some_and(|home| canonical_root == home) {
            return Err(StateError::InvalidWorkspaceRoot("Cannot register home directory".into()));
        }

        let metadata = std::fs::symlink_metadata(&canonical_root)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(StateError::InvalidWorkspaceRoot("Path is not a real directory".into()));
        }
        let (device_id, inode) = file_identity(&metadata)?;
        let git = resolve_git_identity(&canonical_root)?;
        if git.is_none() && !options.allow_non_git_roots { return Err(StateError::NotGitWorktree(root_path)); }
        Ok((root_path, device_id, inode, git))
    }
    fn check_overlap(
        tx: &Transaction,
        new_root: &str,
        device_id: u64,
        inode: u64,
        exclude_id: Option<&str>,
    ) -> Result<()> {
        let mut stmt = tx.prepare("SELECT id, root_path, device_id, inode FROM workspaces")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?,
                row.get::<_, i64>(2)? as u64, row.get::<_, i64>(3)? as u64))
        })?;
        let new_root_slash = format!("{new_root}/");

        for row in rows {
            let (existing_id, existing_root, existing_device, existing_inode) = row?;
            if Some(existing_id.as_str()) == exclude_id { continue; }
            if (existing_device, existing_inode) == (device_id, inode) {
                return Err(StateError::WorkspaceOverlap(format!("Root identity is already registered as '{existing_id}'")));
            }
            let existing_root_slash = format!("{existing_root}/");
            if existing_root == new_root
                || new_root_slash.starts_with(&existing_root_slash)
                || existing_root_slash.starts_with(&new_root_slash)
            {
                return Err(StateError::WorkspaceOverlap(format!("Root overlaps workspace '{existing_id}'")));
            }
        }
        Ok(())
    }

    pub fn set_mode(&mut self, id: &str, mode: WorkspaceMode) -> Result<WorkspaceRecord> {
        let tx = self.conn.transaction()?;
        let existing: Option<ExistingWorkspaceRow> = tx.query_row(
            "SELECT root_path, device_id, inode, registration_epoch, mode_epoch, mode, git_dir, git_common_dir
             FROM workspaces WHERE id=?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?,
                row.get(5)?, row.get(6)?, row.get(7)?)),
        ).optional()?;
        let (root_path, device, inode, registration_epoch, mode_epoch, old_mode, old_git_dir, old_common_dir) =
            existing.ok_or_else(|| StateError::WorkspaceNotFound(id.to_owned()))?;
        let device_id = device as u64;
        let inode = inode as u64;
        verify_root_identity(Path::new(&root_path), device_id, inode)?;
        let git = resolve_git_identity(Path::new(&root_path))?;
        if !git_storage_matches(old_git_dir.as_deref(), old_common_dir.as_deref(), git.as_ref()) {
            return Err(registration_conflict(id));
        }
        if mode == WorkspaceMode::ReadWrite {
            ensure_mutation_backend(&root_path, device_id, inode)?;
        }
        let mode_epoch = if workspace_mode(&old_mode) == mode {
            mode_epoch
        } else {
            mode_epoch.checked_add(1).ok_or_else(|| registration_conflict(id))?
        };
        let now = now_millis();
        tx.execute(
            "UPDATE workspaces SET mode=?1, mode_epoch=?2, updated_at=?3, git_dir=?4,
                git_common_dir=?5, git_head_ref=?6, git_head_commit=?7 WHERE id=?8",
            params![mode_name(mode), mode_epoch, now, git.as_ref().map(|value| value.git_dir.as_str()),
                git.as_ref().map(|value| value.common_dir.as_str()),
                git.as_ref().and_then(|value| value.head_ref.as_deref()),
                git.as_ref().and_then(|value| value.head_commit.as_deref()), id],
        )?;
        tx.execute(
            "INSERT INTO workspace_registration_epochs (id, registration_epoch, mode_epoch)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET mode_epoch=excluded.mode_epoch",
            params![id, registration_epoch, mode_epoch],
        )?;
        tx.commit()?;
        Ok(workspace_record(
            id,
            WorkspaceIdentity { root_path, device_id, inode, git },
            mode,
            WorkspaceEpochs { registration: registration_epoch, mode: mode_epoch },
        ))
    }

    pub fn unregister(&mut self, id: &str) -> Result<()> {
        self.unregister_inner(id, None)
    }

    pub(crate) fn unregister_for_root(&mut self, id: &str, root: &Path) -> Result<()> {
        self.unregister_inner(id, Some(root))
    }

    fn unregister_inner(&mut self, id: &str, expected_root: Option<&Path>) -> Result<()> {
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, i64, i64)> = tx.query_row(
            "SELECT root_path, registration_epoch, mode_epoch FROM workspaces WHERE id=?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).optional()?;
        let (root_path, registration_epoch, mode_epoch) =
            existing.ok_or_else(|| StateError::WorkspaceNotFound(id.to_owned()))?;
        if expected_root.is_some_and(|root| root != Path::new(&root_path)) {
            return Err(registration_conflict(id));
        }
        let tombstone_epoch = registration_epoch.checked_add(1).ok_or_else(|| registration_conflict(id))?;
        crate::changeset::archive_workspace_changesets(&tx, id, now_seconds())?;
        tx.execute(
            "INSERT INTO workspace_registration_epochs (id, registration_epoch, mode_epoch)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET registration_epoch=excluded.registration_epoch,
                 mode_epoch=excluded.mode_epoch",
            params![id, tombstone_epoch, mode_epoch],
        )?;
        tx.execute("DELETE FROM workspaces WHERE id=?1", params![id])?;
        tx.commit()?;
        Ok(())
    }
}

pub(crate) fn resolve_git_identity(root: &Path) -> Result<Option<GitIdentity>> {
    let canonical_root = root.canonicalize()?;
    let repository = match gix::open(&canonical_root) {
        Ok(repository) => repository,
        Err(_) => return Ok(None),
    };
    let Some(work_dir) = repository.workdir() else { return Ok(None); };
    if work_dir.canonicalize()? != canonical_root { return Ok(None); }

    let git_dir = repository.git_dir().canonicalize()?.to_string_lossy().into_owned();
    let common_dir = repository.common_dir().canonicalize()?.to_string_lossy().into_owned();
    let head_ref = repository.head_name()
        .map_err(|_| StateError::InvalidWorkspaceRoot("Git HEAD could not be resolved".into()))?
        .map(|name| name.to_string());
    let head_commit = match repository.head_id() {
        Ok(id) => Some(id.to_string()),
        Err(_) => {
            let unborn = head_ref.is_some()
                && repository.head_ref()
                    .map_err(|_| StateError::InvalidWorkspaceRoot("Git HEAD could not be resolved".into()))?
                    .is_none();
            if unborn { None }
            else { return Err(StateError::InvalidWorkspaceRoot("Git HEAD could not be resolved".into())); }
        }
    };
    Ok(Some(GitIdentity { git_dir, common_dir, head_ref, head_commit }))
}

#[cfg(unix)]
fn file_identity(metadata: &Metadata) -> Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn file_identity(_: &Metadata) -> Result<(u64, u64)> {
    Err(StateError::Unsupported)
}

fn verify_root_identity(root: &Path, expected_device: u64, expected_inode: u64) -> Result<()> {
    let metadata = std::fs::symlink_metadata(root)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink()
        || file_identity(&metadata)? != (expected_device, expected_inode)
    {
        return Err(StateError::WorkspaceOverlap("Registered workspace root identity changed".into()));
    }
    Ok(())
}

fn git_storage_matches(saved_git_dir: Option<&str>, saved_common_dir: Option<&str>, current: Option<&GitIdentity>) -> bool {
    match (saved_git_dir, saved_common_dir, current) {
        (None, None, None) => true,
        (Some(git_dir), Some(common_dir), Some(current)) => git_dir == current.git_dir && common_dir == current.common_dir,
        _ => false,
    }
}

fn workspace_record(
    id: &str,
    identity: WorkspaceIdentity,
    mode: WorkspaceMode,
    epochs: WorkspaceEpochs,
) -> WorkspaceRecord {
    WorkspaceRecord {
        id: id.to_owned(),
        root_path: identity.root_path,
        device_id: identity.device_id,
        inode: identity.inode,
        mode,
        registration_epoch: epochs.registration,
        mode_epoch: epochs.mode,
        git_dir: identity.git.as_ref().map(|value| value.git_dir.clone()),
        git_common_dir: identity.git.as_ref().map(|value| value.common_dir.clone()),
        git_head_ref: identity.git.as_ref().and_then(|value| value.head_ref.clone()),
        git_head_commit: identity.git.and_then(|value| value.head_commit),
    }
}

fn workspace_mode(mode: &str) -> WorkspaceMode {
    if mode == "read_write" { WorkspaceMode::ReadWrite } else { WorkspaceMode::ReadOnly }
}

fn effective_workspace_mode(mode: &str) -> WorkspaceMode {
    match workspace_mode(mode) {
        WorkspaceMode::ReadWrite if !WorkspaceFs::mutation_supported() => WorkspaceMode::ReadOnly,
        mode => mode,
    }
}

fn mode_name(mode: WorkspaceMode) -> &'static str {
    match mode {
        WorkspaceMode::ReadOnly => "read_only",
        WorkspaceMode::ReadWrite => "read_write",
    }
}

fn registration_conflict(id: &str) -> StateError {
    StateError::WorkspaceOverlap(format!("Workspace '{id}' conflicts with its registered identity or mode"))
}

fn now_millis() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().try_into().unwrap_or(i64::MAX)
}
fn now_seconds() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs().try_into().unwrap_or(i64::MAX)
}

fn ensure_mutation_backend(root_path: &str, device_id: u64, inode: u64) -> Result<()> {
    if !WorkspaceFs::mutation_supported() { return Err(StateError::Unsupported); }
    match WorkspaceFs::open_registered(Path::new(root_path), device_id, inode) {
        Ok(_) => Ok(()),
        Err(WorkspaceFsError::Unsupported) => Err(StateError::Unsupported),
        Err(WorkspaceFsError::Conflict) => Err(StateError::WorkspaceOverlap("Registered workspace root identity changed".into())),
        Err(WorkspaceFsError::Io(error)) => Err(StateError::Io(error)),
        Err(_) => Err(StateError::InvalidWorkspaceRoot("Safe mutation capability is unavailable".into())),
    }
}
    
