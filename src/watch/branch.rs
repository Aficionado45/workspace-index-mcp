use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchMarker {
    pub git_dir: PathBuf,
    pub head_ref: Option<String>,
    pub commit: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchTransition {
    pub before: Option<BranchMarker>,
    pub after: Option<BranchMarker>,
}

pub fn read_branch_marker(root: &Path) -> io::Result<Option<BranchMarker>> {
    let marker = root.join(".git");
    let metadata = match fs::symlink_metadata(&marker) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let git_dir = if metadata.is_dir() {
        marker.canonicalize()?
    } else if metadata.is_file() {
        read_git_dir_file(root, &marker)?
    } else {
        return Err(invalid_git_metadata());
    };
    let head = fs::read_to_string(git_dir.join("HEAD"))?;
    let (head_ref, commit) = parse_head(&git_dir, &head)?;
    Ok(Some(BranchMarker {
        git_dir,
        head_ref,
        commit,
    }))
}

fn read_git_dir_file(root: &Path, marker: &Path) -> io::Result<PathBuf> {
    let contents = fs::read_to_string(marker)?;
    let target = contents
        .trim()
        .strip_prefix("gitdir:")
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .ok_or_else(invalid_git_metadata)?;
    let target = PathBuf::from(target);
    let target = if target.is_absolute() {
        target
    } else {
        root.join(target)
    };
    target.canonicalize()
}

fn parse_head(git_dir: &Path, head: &str) -> io::Result<(Option<String>, Option<String>)> {
    let head = head.trim();
    if let Some(reference) = head.strip_prefix("ref: ") {
        if !valid_reference(reference) {
            return Err(invalid_git_metadata());
        }
        return Ok((
            Some(reference.to_owned()),
            resolve_ref(git_dir, reference)?,
        ));
    }
    if !valid_object_id(head) {
        return Err(invalid_git_metadata());
    }
    Ok((None, Some(head.to_ascii_lowercase())))
}

fn resolve_ref(git_dir: &Path, reference: &str) -> io::Result<Option<String>> {
    let mut bases = vec![git_dir.to_path_buf()];
    if let Some(common_dir) = read_optional_text(&git_dir.join("commondir"))? {
        let common_dir = common_dir.trim();
        if common_dir.is_empty() {
            return Err(invalid_git_metadata());
        }
        let common_dir = PathBuf::from(common_dir);
        let common_dir = if common_dir.is_absolute() {
            common_dir
        } else {
            git_dir.join(common_dir)
        };
        let common_dir = common_dir.canonicalize()?;
        if !bases.contains(&common_dir) {
            bases.push(common_dir);
        }
    }
    for base in &bases {
        if let Some(value) = read_optional_text(&base.join(reference))? {
            return parse_object_id(&value).map(Some);
        }
    }
    for base in &bases {
        let Some(contents) = read_optional_text(&base.join("packed-refs"))? else {
            continue;
        };
        if let Some(value) = packed_ref_value(&contents, reference)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn packed_ref_value(contents: &str, reference: &str) -> io::Result<Option<String>> {
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') || line.starts_with('^') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let object_id = fields.next().ok_or_else(invalid_git_metadata)?;
        let name = fields.next().ok_or_else(invalid_git_metadata)?;
        if fields.next().is_some() {
            return Err(invalid_git_metadata());
        }
        if name == reference {
            return parse_object_id(object_id).map(Some);
        }
    }
    Ok(None)
}

fn read_optional_text(path: &Path) -> io::Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(contents) => Ok(Some(contents)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn parse_object_id(value: &str) -> io::Result<String> {
    let object_id = value.trim();
    if !valid_object_id(object_id) {
        return Err(invalid_git_metadata());
    }
    Ok(object_id.to_ascii_lowercase())
}

fn valid_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_reference(reference: &str) -> bool {
    reference.starts_with("refs/")
        && !reference.contains("..")
        && !reference.contains("@{")
        && !reference.ends_with('.')
        && !reference.ends_with(".lock")
        && !reference
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ' || byte == b'\\')
        && Path::new(reference).components().all(|component| {
            matches!(component, Component::Normal(_))
        })
        && reference.split('/').all(|part| !part.is_empty())
}

fn invalid_git_metadata() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid Git metadata")
}
