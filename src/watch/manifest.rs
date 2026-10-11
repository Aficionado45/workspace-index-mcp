use crate::index::scanner::{ScanEntry, build_scanner, read_scanned_file, read_workspace_path};
use crate::security::WorkspaceFs;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub type Manifest = BTreeMap<PathBuf, String>;

pub fn build_manifest(
    workspace: &WorkspaceFs,
    excluded_data_root: &Path,
    max_file_bytes: u64,
) -> io::Result<Manifest> {
    let root = workspace.root_path();
    workspace.verify_registered_root().map_err(io::Error::other)?;
    let mut manifest = Manifest::new();
    for item in build_scanner(root, excluded_data_root)? {
        let entry = item.map_err(|error| io::Error::other(error.to_string()))?;
        let relative_path = entry.path().strip_prefix(root).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "walker returned a path outside the workspace")
        })?;
        if is_ignore_control_file(relative_path) {
            let bytes = workspace.read(relative_path, max_file_bytes).map_err(io::Error::other)?;
            let hash = blake3::hash(&bytes).to_hex().to_string();
            manifest.insert(relative_path.to_path_buf(), format!("control:{hash}"));
            continue;
        }
        match read_scanned_file(workspace, root, &entry, max_file_bytes)? {
            ScanEntry::Included(file) => {
                manifest.insert(
                    file.relative_path,
                    blake3::hash(file.content.as_bytes()).to_hex().to_string(),
                );
            }
            ScanEntry::Excluded { relative_path, reason } => {
                manifest.insert(relative_path, format!("excluded:{}", reason.label()));
            }
        }
    }
    workspace.verify_registered_root().map_err(io::Error::other)?;
    Ok(manifest)
}

pub(crate) fn build_paths_manifest(
    workspace: &WorkspaceFs,
    paths: &[String],
    max_file_bytes: u64,
) -> io::Result<Manifest> {
    let root = workspace.root_path();
    workspace.verify_registered_root().map_err(io::Error::other)?;
    let mut manifest = Manifest::new();
    for path in paths {
        let relative_path = Path::new(path);
        let Some(entry) = read_workspace_path(workspace, root, relative_path, max_file_bytes)? else {
            continue;
        };
        match entry {
            ScanEntry::Included(file) => {
                manifest.insert(
                    file.relative_path,
                    blake3::hash(file.content.as_bytes()).to_hex().to_string(),
                );
            }
            ScanEntry::Excluded { relative_path, reason } => {
                manifest.insert(relative_path, format!("excluded:{}", reason.label()));
            }
        }
    }
    workspace.verify_registered_root().map_err(io::Error::other)?;
    Ok(manifest)
}

fn is_ignore_control_file(relative_path: &Path) -> bool {
    relative_path.file_name().is_some_and(|name| {
        name == ".gitignore" || name == ".workspace-indexignore" || name == ".augmentignore"
    })
}
