use thiserror::Error;


#[derive(Debug, Error)]
pub enum StateError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("Database is locked by another process")]
    Busy,
    #[error("Unsupported schema version {0}")]
    UnsupportedSchema(i32),
    #[error("Invalid workspace ID: {0}")]
    InvalidWorkspaceId(String),
    #[error("Invalid workspace root: {0}")]
    InvalidWorkspaceRoot(String),
    #[error("Workspace not found: {0}")]
    WorkspaceNotFound(String),
    #[error("Workspace root overlap: {0}")]
    WorkspaceOverlap(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid vector operation")]
    InvalidVectorOperation,
    #[error("Not a Git worktree: {0}")]
    NotGitWorktree(String),
    #[error("Safe filesystem identity or mutation support is unavailable")]
    Unsupported,
}

pub type Result<T, E = StateError> = std::result::Result<T, E>;

pub mod lock;
pub mod schema;
pub mod registry;

pub use lock::OwnerLock;
pub use registry::{RegistryOptions, WorkspaceRegistry, WorkspaceRecord};
