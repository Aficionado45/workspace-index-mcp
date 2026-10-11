use crate::model::ErrorCode;
use crate::security::workspace_fs::{WorkspaceFs, WorkspaceFsError};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FileRead {
    pub text: String,
    pub sha256: String,
    pub start_line: usize,
    pub end_line: usize,
    pub total_lines: usize,
    pub truncated: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct FileReadError { pub code: ErrorCode, pub message: String }

pub fn file_read(
    workspace: &WorkspaceFs,
    path: &Path,
    start_line: Option<usize>,
    end_line: Option<usize>,
    expected_sha256: Option<&str>,
    max_file_bytes: u64,
    max_output_bytes: usize,
) -> Result<FileRead, FileReadError> {
    let bytes = workspace.read(path, max_file_bytes).map_err(map_fs_error)?;
    if bytes.get(..bytes.len().min(8192)).is_some_and(|prefix| prefix.contains(&0)) {
        return Err(FileReadError { code: ErrorCode::InvalidArgument, message: "binary files are not readable".into() });
    }
    let sha256 = hex(&sha2::Sha256::digest(&bytes));
    let text = String::from_utf8(bytes).map_err(|_| FileReadError { code: ErrorCode::InvalidArgument, message: "file is not valid UTF-8".into() })?;
    if expected_sha256.is_some_and(|expected| expected != sha256) {
        return Err(FileReadError { code: ErrorCode::Conflict, message: "file SHA-256 does not match expected_sha256".into() });
    }
    let line_count = if text.is_empty() { 0 } else { text.bytes().filter(|byte| *byte == b'\n').count() + usize::from(!text.ends_with('\n')) };
    let start = start_line.unwrap_or(1);
    let end = end_line.unwrap_or(line_count);
    if start == 0 || (line_count > 0 && (start > end || end > line_count)) || (line_count == 0 && (start != 1 || end > 0)) {
        return Err(FileReadError { code: ErrorCode::InvalidArgument, message: "line range is outside the file".into() });
    }
    let selected = if line_count == 0 { String::new() } else {
        text.split_inclusive('\n').skip(start - 1).take(end - start + 1).collect()
    };
    let (text, truncated) = truncate_utf8(selected, max_output_bytes);
    Ok(FileRead { text, sha256, start_line: start, end_line: end, total_lines: line_count, truncated })
}

fn truncate_utf8(mut value: String, max_bytes: usize) -> (String, bool) {
    if value.len() <= max_bytes { return (value, false); }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) { end -= 1; }
    value.truncate(end);
    (value, true)
}
fn map_fs_error(error: WorkspaceFsError) -> FileReadError {
    let (code, message) = match error {
        WorkspaceFsError::InvalidPath => (ErrorCode::InvalidArgument, "path must be normalized and workspace-relative"),
        WorkspaceFsError::SensitivePath => (ErrorCode::InvalidArgument, "sensitive workspace paths are not readable"),
        WorkspaceFsError::PathEscape => (ErrorCode::SymlinkRefused, "path escapes the workspace or crosses a symlink"),
        WorkspaceFsError::Conflict => (ErrorCode::Conflict, "file or parent changed during safe access"),
        WorkspaceFsError::AlreadyExists => (ErrorCode::Conflict, "path already exists"),
        WorkspaceFsError::TooLarge => (ErrorCode::InvalidArgument, "file exceeds configured maximum"),
        WorkspaceFsError::NotRegularFile => (ErrorCode::InvalidArgument, "path is not a regular file"),
        WorkspaceFsError::Unsupported => (ErrorCode::Unsupported, "safe filesystem reads are unavailable on this platform"),
        WorkspaceFsError::Io(_) => (ErrorCode::Conflict, "file could not be read safely"),
    };
    FileReadError { code, message: message.into() }
}
fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes { output.push(HEX[(byte >> 4) as usize] as char); output.push(HEX[(byte & 15) as usize] as char); }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use tempfile::TempDir;

    #[test]
    fn reads_inclusive_line_range_and_checks_hash() {
        let root = TempDir::new().unwrap();
        std::fs::write(root.path().join("code.rs"), "one\ntwo\nthree\n").unwrap();
        let fs = WorkspaceFs::open(root.path()).unwrap();
        let full = file_read(&fs, Path::new("code.rs"), None, None, None, 100, 100).unwrap();
        assert_eq!(full.total_lines, 3);
        assert_eq!(full.start_line, 1);
        assert_eq!(full.end_line, 3);
        let range = file_read(&fs, Path::new("code.rs"), Some(2), Some(2), Some(&full.sha256), 100, 100).unwrap();
        assert_eq!(range.text, "two\n");
        assert_eq!(range.total_lines, 3);
        assert_eq!(file_read(&fs, Path::new("code.rs"), None, None, Some("wrong"), 100, 100).unwrap_err().code, ErrorCode::Conflict);
    }

    #[test]
    fn refuses_symlinks_and_bounds_output() {
        let root = TempDir::new().unwrap();
        std::fs::write(root.path().join("source"), "abcdefghij").unwrap();
        symlink(root.path().join("source"), root.path().join("link")).unwrap();
        let fs = WorkspaceFs::open(root.path()).unwrap();
        assert_eq!(file_read(&fs, Path::new("link"), None, None, None, 100, 100).unwrap_err().code, ErrorCode::SymlinkRefused);
        let read = file_read(&fs, Path::new("source"), None, None, None, 100, 4).unwrap();
        assert_eq!(read.text, "abcd");
        assert!(read.truncated);
    }

    #[test]
    fn refuses_sensitive_path_components_without_returning_secret_content() {
        let root = TempDir::new().unwrap();
        let sentinel = "sensitive sentinel";
        std::fs::create_dir(root.path().join("nested")).unwrap();
        for path in [".env", "nested/credentials.json", "server.pem", "id_rsa"] {
            std::fs::write(root.path().join(path), sentinel).unwrap();
        }
        std::fs::write(root.path().join("src.rs"), "ordinary source").unwrap();
        let fs = WorkspaceFs::open(root.path()).unwrap();

        for path in [".env", "nested/credentials.json", "server.pem", "id_rsa"] {
            let error = file_read(&fs, Path::new(path), None, None, None, 100, 100).unwrap_err();
            assert_eq!(error.code, ErrorCode::InvalidArgument);
            assert!(!error.to_string().contains(sentinel));
        }
        assert_eq!(
            file_read(&fs, Path::new("src.rs"), None, None, None, 100, 100).unwrap().text,
            "ordinary source",
        );
    }
}
