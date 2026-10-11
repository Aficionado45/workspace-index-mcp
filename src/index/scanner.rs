use crate::security::{WorkspaceFs, WorkspaceFsError};
use ignore::{DirEntry, Walk, WalkBuilder};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct ScannedFile {
    pub relative_path: PathBuf,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExclusionReason {
    Sensitive,
    Unsupported,
    Symlink,
    NonRegular,
    Oversized,
    Binary,
}

impl ExclusionReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sensitive => "sensitive",
            Self::Unsupported => "unsupported",
            Self::Symlink => "symlink",
            Self::NonRegular => "non_regular",
            Self::Oversized => "oversized",
            Self::Binary => "binary",
        }
    }
}

#[derive(Debug)]
pub enum ScanEntry {
    Included(ScannedFile),
    Excluded { relative_path: PathBuf, reason: ExclusionReason },
}

fn excluded(relative_path: PathBuf, reason: ExclusionReason) -> ScanEntry {
    ScanEntry::Excluded { relative_path, reason }
}

fn relative_source_path(root: &Path, path: &Path) -> io::Result<PathBuf> {
    let relative_path = path.strip_prefix(root).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "walker returned a path outside the workspace")
    })?.to_path_buf();
    if relative_path.to_str().is_none() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "source path is not UTF-8"));
    }
    Ok(relative_path)
}

fn scan_read_error(error: WorkspaceFsError) -> io::Error {
    io::Error::other(error)
}

pub fn build_scanner(root: &Path, excluded_data_root: &Path) -> io::Result<Walk> {
    if !root.is_absolute() || !excluded_data_root.is_absolute()
        || root == excluded_data_root || root.starts_with(excluded_data_root)
    {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid scan roots"));
    }
    let excluded_data_root = excluded_data_root.to_path_buf();
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false).require_git(false).ignore(true).git_ignore(true);
    builder.git_global(true).git_exclude(true);
    builder.add_custom_ignore_filename(".workspace-indexignore");
    builder.add_custom_ignore_filename(".augmentignore");
    builder.filter_entry(move |entry| {
        let path = entry.path();
        if path.file_name().is_some_and(|name| name == ".git")
            || path.starts_with(&excluded_data_root)
        {
            return false;
        }
        match entry.file_type() {
            Some(kind) if kind.is_dir() => !is_sensitive(path),
            Some(kind) if kind.is_file() => true,
            Some(kind) if kind.is_symlink() => is_supported_source(path),
            Some(_) => false,
            None => true,
        }
    });
    Ok(builder.build())
}

pub fn read_scanned_file(
    workspace: &WorkspaceFs,
    root: &Path,
    entry: &DirEntry,
    max_file_bytes: u64,
) -> io::Result<ScanEntry> {
    let path = entry.path();
    let relative_path = relative_source_path(root, path)?;
    let file_type = entry.file_type().ok_or_else(|| {
        io::Error::other("walker could not classify a source entry")
    })?;
    if file_type.is_symlink() {
        return Ok(excluded(relative_path, ExclusionReason::Symlink));
    }
    if !file_type.is_file() {
        return Ok(excluded(relative_path, ExclusionReason::NonRegular));
    }
    if is_sensitive(path) {
        return Ok(excluded(relative_path, ExclusionReason::Sensitive));
    }
    if !is_supported_source(path) {
        return Ok(excluded(relative_path, ExclusionReason::Unsupported));
    }
    let bytes = match workspace.read(&relative_path, max_file_bytes) {
        Ok(bytes) => bytes,
        Err(WorkspaceFsError::TooLarge) => {
            return Ok(excluded(relative_path, ExclusionReason::Oversized));
        }
        Err(WorkspaceFsError::PathEscape) => {
            return Ok(excluded(relative_path, ExclusionReason::Symlink));
        }
        Err(WorkspaceFsError::SensitivePath) => {
            return Ok(excluded(relative_path, ExclusionReason::Sensitive));
        }
        Err(error) => return Err(scan_read_error(error)),
    };
    if is_binary(&bytes) {
        return Ok(excluded(relative_path, ExclusionReason::Binary));
    }
    let Ok(content) = String::from_utf8(bytes) else {
        return Ok(excluded(relative_path, ExclusionReason::Binary));
    };
    Ok(ScanEntry::Included(ScannedFile { relative_path, content }))
}
pub fn read_workspace_path(
    workspace: &WorkspaceFs,
    root: &Path,
    relative_path: &Path,
    max_file_bytes: u64,
) -> io::Result<Option<ScanEntry>> {
    let full_path = root.join(relative_path);
    let relative_path = relative_source_path(root, &full_path)?;
    if is_sensitive(&full_path) {
        return Ok(Some(excluded(relative_path, ExclusionReason::Sensitive)));
    }
    if !is_supported_source(&full_path) {
        return Ok(Some(excluded(relative_path, ExclusionReason::Unsupported)));
    }
    let bytes = match workspace.read(&relative_path, max_file_bytes) {
        Ok(bytes) => bytes,
        Err(WorkspaceFsError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(WorkspaceFsError::TooLarge) => {
            return Ok(Some(excluded(relative_path, ExclusionReason::Oversized)));
        }
        Err(WorkspaceFsError::PathEscape) => {
            return Ok(Some(excluded(relative_path, ExclusionReason::Symlink)));
        }
        Err(WorkspaceFsError::NotRegularFile) => {
            return Ok(Some(excluded(relative_path, ExclusionReason::NonRegular)));
        }
        Err(WorkspaceFsError::SensitivePath) => {
            return Ok(Some(excluded(relative_path, ExclusionReason::Sensitive)));
        }
        Err(error) => return Err(scan_read_error(error)),
    };
    if is_binary(&bytes) {
        return Ok(Some(excluded(relative_path, ExclusionReason::Binary)));
    }
    let Ok(content) = String::from_utf8(bytes) else {
        return Ok(Some(excluded(relative_path, ExclusionReason::Binary)));
    };
    Ok(Some(ScanEntry::Included(ScannedFile {
        relative_path,
        content,
    })))
}


pub fn is_supported_source(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "py" | "java" | "js" | "jsx" | "ts" | "tsx" | "go" | "rs" | "rb" | "php"
            | "scala" | "sc" | "c" | "h" | "cpp" | "cc" | "cxx" | "hpp" | "hh" | "cs"
            | "kt" | "kts" | "md" | "markdown" | "yaml" | "yml" | "toml" | "sql"
            | "sh" | "bash" | "zsh"
    )
}

fn is_sensitive(path: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()).is_some_and(|name| {
        name.starts_with(".env") || name.ends_with(".pem")
            || name.starts_with("id_rsa") || name.starts_with("credentials")
    })
}

pub fn is_binary(bytes: &[u8]) -> bool {
    bytes.get(..bytes.len().min(8192))
        .is_some_and(|prefix| prefix.contains(&0))
}
