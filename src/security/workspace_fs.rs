use sha2::{Digest, Sha256};
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;


#[derive(Debug, thiserror::Error)]
pub enum WorkspaceFsError {
    #[error("invalid relative path")]
    InvalidPath,
    #[error("sensitive workspace path refused")]
    SensitivePath,
    #[error("symlink or path escape refused")]
    PathEscape,
    #[error("target is not a regular file")]
    NotRegularFile,
    #[error("source changed since it was prepared")]
    Conflict,
    #[error("destination already exists")]
    AlreadyExists,
    #[error("file exceeds configured limit")]
    TooLarge,
    #[error("safe filesystem operation is unsupported; mutation containment cannot be proven")]
    Unsupported,
    #[error("filesystem error: {0}")]
    Io(#[from] io::Error),
}

pub struct WorkspaceFs {
    root: File,
    root_path: PathBuf,
    root_identity: (u64, u64),
}
pub struct ResolvedParent { parent: File, basename: OsString, parent_relative: PathBuf }

impl WorkspaceFs {
    pub fn open(root_path: &Path) -> Result<Self, WorkspaceFsError> {
        let canonical = root_path.canonicalize()?;
        if !root_path.is_absolute() || canonical.as_path() != root_path {
            return Err(WorkspaceFsError::InvalidPath);
        }
        let path_metadata = std::fs::symlink_metadata(root_path)?;
        if !path_metadata.is_dir() || path_metadata.file_type().is_symlink() {
            return Err(WorkspaceFsError::InvalidPath);
        }
        let root = File::from(open_root(root_path)?);
        let root_identity = file_identity(&root.metadata()?)?;
        let workspace = Self { root, root_path: canonical, root_identity };
        workspace.verify_root()?;
        ensure_path_backend(&workspace.root)?;
        Ok(workspace)
    }

    pub fn open_registered(
        root_path: &Path,
        expected_device: u64,
        expected_inode: u64,
    ) -> Result<Self, WorkspaceFsError> {
        let canonical = root_path.canonicalize().map_err(|_| WorkspaceFsError::Conflict)?;
        if canonical.as_path() != root_path { return Err(WorkspaceFsError::Conflict); }
        let metadata = std::fs::symlink_metadata(root_path).map_err(|_| WorkspaceFsError::Conflict)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink()
            || file_identity(&metadata)? != (expected_device, expected_inode)
        {
            return Err(WorkspaceFsError::Conflict);
        }
        let workspace = Self::open(root_path)?;
        if workspace.root_path.as_path() != root_path || workspace.root_identity != (expected_device, expected_inode) {
            return Err(WorkspaceFsError::Conflict);
        }
        workspace.verify_root()?;
        Ok(workspace)
    }


    pub fn root_path(&self) -> &Path { &self.root_path }
    pub fn verify_registered_root(&self) -> Result<(), WorkspaceFsError> {
        self.verify_root()
    }

    pub fn resolve_parent(&self, relative: &Path) -> Result<ResolvedParent, WorkspaceFsError> {
        self.verify_root()?;
        let parts = validated_components(relative)?;
        let (basename, parents) = parts.split_last().ok_or(WorkspaceFsError::InvalidPath)?;
        let mut parent = self.root.try_clone()?;
        for component in parents {
            parent = File::from(open_directory(&parent, component)?);
        }
        let resolved = ResolvedParent {
            parent,
            basename: basename.clone(),
            parent_relative: parents.iter().collect(),
        };
        self.verify_parent(&resolved)?;
        Ok(resolved)
    }

    pub fn read(&self, relative: &Path, max_bytes: u64) -> Result<Vec<u8>, WorkspaceFsError> {
        self.read_with_post_read_hook(relative, max_bytes, || {})
    }

    fn read_with_post_read_hook(
        &self,
        relative: &Path,
        max_bytes: u64,
        after_read: impl FnOnce(),
    ) -> Result<Vec<u8>, WorkspaceFsError> {
        let resolved = self.resolve_parent(relative)?;
        self.verify_parent(&resolved)?;
        let file = File::from(open_regular(&resolved.parent, &resolved.basename)?);
        let metadata = file.metadata()?;
        if !metadata.is_file() { return Err(WorkspaceFsError::NotRegularFile); }
        require_single_link(&metadata)?;
        let identity = file_identity(&metadata)?;
        let bytes = read_limited(file.try_clone()?, max_bytes)?;
        after_read();
        require_single_link(&file.metadata()?)?;
        self.verify_parent(&resolved)?;
        let current = open_regular(&resolved.parent, &resolved.basename)
            .map_err(|_| WorkspaceFsError::Conflict)?;
        if file_identity(&File::from(current).metadata()?)? != identity { return Err(WorkspaceFsError::Conflict); }
        self.verify_parent(&resolved)?;
        Ok(bytes)
    }

    pub fn sha256(&self, relative: &Path, max_bytes: u64) -> Result<String, WorkspaceFsError> {
        Ok(hex(&Sha256::digest(self.read(relative, max_bytes)?)))
    }

    /// Current directory-handle backends cannot prevent concurrent parent relocation during effects.
    pub fn mutation_supported() -> bool { false }

    pub fn create(&self, relative: &Path, _content: &[u8]) -> Result<(), WorkspaceFsError> {
        let parent = self.resolve_parent(relative)?;
        self.verify_parent(&parent)?;
        Err(WorkspaceFsError::Unsupported)
    }

    pub fn replace(&self, relative: &Path, _expected_sha256: &str, _content: &[u8]) -> Result<(), WorkspaceFsError> {
        let _target = self.resolve_target(relative)?;
        Err(WorkspaceFsError::Unsupported)
    }

    pub fn delete(&self, relative: &Path, _expected_sha256: &str) -> Result<(), WorkspaceFsError> {
        let _target = self.resolve_target(relative)?;
        Err(WorkspaceFsError::Unsupported)
    }

    pub fn move_file(&self, from: &Path, to: &Path, _expected_sha256: &str) -> Result<(), WorkspaceFsError> {
        let _source = self.resolve_target(from)?;
        let destination = self.resolve_parent(to)?;
        self.verify_parent(&destination)?;
        Err(WorkspaceFsError::Unsupported)
    }

    fn resolve_target(&self, relative: &Path) -> Result<(ResolvedParent, File), WorkspaceFsError> {
        let parent = self.resolve_parent(relative)?;
        self.verify_parent(&parent)?;
        let target = File::from(open_regular(&parent.parent, &parent.basename)?);
        let metadata = target.metadata()?;
        if !metadata.is_file() { return Err(WorkspaceFsError::NotRegularFile); }
        require_single_link(&metadata)?;
        Ok((parent, target))
    }

    fn verify_parent(&self, parent: &ResolvedParent) -> Result<(), WorkspaceFsError> {
        self.verify_root()?;
        let reopened = if parent.parent_relative.as_os_str().is_empty() {
            self.root.try_clone()?
        } else {
            let parts = validated_components(&parent.parent_relative)?;
            let mut current = self.root.try_clone()?;
            for component in &parts { current = File::from(open_directory(&current, component)?); }
            current
        };
        if file_identity(&reopened.metadata()?)? != file_identity(&parent.parent.metadata()?)? {
            return Err(WorkspaceFsError::Conflict);
        }
        Ok(())
    }

    fn verify_root(&self) -> Result<(), WorkspaceFsError> {
        if file_identity(&self.root.metadata()?)? != self.root_identity {
            return Err(WorkspaceFsError::Conflict);
        }
        let canonical = self.root_path.canonicalize().map_err(|_| WorkspaceFsError::Conflict)?;
        if canonical != self.root_path { return Err(WorkspaceFsError::Conflict); }
        let path = std::fs::symlink_metadata(&self.root_path).map_err(|_| WorkspaceFsError::Conflict)?;
        if !path.is_dir() || path.file_type().is_symlink()
            || file_identity(&path)? != self.root_identity
        {
            return Err(WorkspaceFsError::Conflict);
        }
        Ok(())
    }
}

fn file_identity(metadata: &std::fs::Metadata) -> Result<(u64, u64), WorkspaceFsError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok((metadata.dev(), metadata.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        Err(WorkspaceFsError::Unsupported)
    }
}

fn require_single_link(metadata: &std::fs::Metadata) -> Result<(), WorkspaceFsError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() == 1 { Ok(()) } else { Err(WorkspaceFsError::SensitivePath) }
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        Err(WorkspaceFsError::Unsupported)
    }
}

fn read_limited(file: File, max_bytes: u64) -> Result<Vec<u8>, WorkspaceFsError> {
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes { return Err(WorkspaceFsError::TooLarge); }
    Ok(bytes)
}



fn io_error(error: rustix::io::Errno) -> WorkspaceFsError {
    if error == rustix::io::Errno::LOOP || error == rustix::io::Errno::XDEV {
        WorkspaceFsError::PathEscape
    } else if error == rustix::io::Errno::NOSYS || error == rustix::io::Errno::INVAL {
        WorkspaceFsError::Unsupported
    } else {
        WorkspaceFsError::Io(io::Error::from_raw_os_error(error.raw_os_error()))
    }
}
#[cfg(unix)]
fn validated_components(relative: &Path) -> Result<Vec<OsString>, WorkspaceFsError> {
    let bytes = relative.as_os_str().as_bytes();
    if bytes.is_empty() || bytes.starts_with(b"/") || bytes.contains(&0) {
        return Err(WorkspaceFsError::InvalidPath);
    }
    let parts: Vec<_> = bytes.split(|byte| *byte == b'/').collect();
    if parts.iter().any(|part| part.is_empty() || *part == b"." || *part == b"..") {
        return Err(WorkspaceFsError::InvalidPath);
    }
    let components: Vec<_> = parts.into_iter().map(|part| OsStr::from_bytes(part).to_os_string()).collect();
    if components.iter().any(|component| is_sensitive_component(component)) {
        return Err(WorkspaceFsError::SensitivePath);
    }
    Ok(components)
}

#[cfg(not(unix))]
fn validated_components(relative: &Path) -> Result<Vec<OsString>, WorkspaceFsError> {
    let raw = relative.as_os_str().to_string_lossy();
    if raw.is_empty() || relative.is_absolute() || raw.contains('\0') || raw.contains("//")
        || raw.split(['/', '\\']).any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(WorkspaceFsError::InvalidPath);
    }
    if raw.split(['/', '\\']).any(|part| is_sensitive_component(OsStr::new(part))) {
        return Err(WorkspaceFsError::SensitivePath);
    }
    Err(WorkspaceFsError::Unsupported)
}

fn is_sensitive_component(component: &OsStr) -> bool {
    component.to_str().is_some_and(|name| {
        name.starts_with(".env")
            || name.ends_with(".pem")
            || name.starts_with("id_rsa")
            || name.starts_with("credentials")
    })
}

#[cfg(target_os = "linux")]
fn open_root(path: &Path) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    use rustix::fs::{open, Mode, OFlags};
    open(path, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW, Mode::empty()).map_err(io_error)
}

#[cfg(target_os = "linux")]
fn open_directory(parent: &File, component: &OsStr) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    use rustix::fs::{openat2, Mode, OFlags, ResolveFlags};
    openat2(parent, component, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(), ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS)
        .map_err(io_error)
}

#[cfg(target_os = "linux")]
fn open_regular(parent: &File, basename: &OsStr) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    use rustix::fs::{openat2, Mode, OFlags, ResolveFlags};
    openat2(parent, basename, OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(), ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS)
        .map_err(io_error)
}

#[cfg(target_os = "macos")]
fn open_root(path: &Path) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    use rustix::fs::{open, Mode, OFlags};
    open(path, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW, Mode::empty()).map_err(io_error)
}

#[cfg(target_os = "macos")]
fn open_directory(parent: &File, component: &OsStr) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    use rustix::fs::{openat, Mode, OFlags};
    openat(parent, component, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty()).map_err(io_error)
}

#[cfg(target_os = "macos")]
fn open_regular(parent: &File, basename: &OsStr) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    use rustix::fs::{openat, Mode, OFlags};
    openat(parent, basename, OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty()).map_err(io_error)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_root(_: &Path) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    Err(WorkspaceFsError::Unsupported)
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_directory(_: &File, _: &OsStr) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    Err(WorkspaceFsError::Unsupported)
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_regular(_: &File, _: &OsStr) -> Result<rustix::fd::OwnedFd, WorkspaceFsError> {
    Err(WorkspaceFsError::Unsupported)
}

fn ensure_path_backend(root: &File) -> Result<(), WorkspaceFsError> {
    #[cfg(target_os = "linux")]
    {
        let _probe = open_directory(root, OsStr::new("."))?;
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        let _ = root;
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = root;
        Err(WorkspaceFsError::Unsupported)
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes { out.push(HEX[(byte >> 4) as usize] as char); out.push(HEX[(byte & 0xf) as usize] as char); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use tempfile::TempDir;

    #[test]
    fn reads_regular_files_and_refuses_symlink_components() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("src")).unwrap();
        std::fs::write(temp.path().join("src/lib.rs"), "safe").unwrap();
        std::fs::write(temp.path().join("secret"), "outside").unwrap();
        symlink(temp.path().join("secret"), temp.path().join("src/link.rs")).unwrap();
        let fs = WorkspaceFs::open(temp.path()).unwrap();
        assert_eq!(fs.read(Path::new("src/lib.rs"), 100).unwrap(), b"safe");
        assert!(matches!(fs.read(Path::new("src/link.rs"), 100), Err(WorkspaceFsError::PathEscape)));
        assert!(matches!(fs.read(Path::new("../secret"), 100), Err(WorkspaceFsError::InvalidPath)));
    }


    #[test]
    fn read_rejects_final_basename_replacement_after_open() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("file.txt");
        let displaced = temp.path().join("displaced.txt");
        std::fs::write(&path, "opened content").unwrap();
        let fs = WorkspaceFs::open(temp.path()).unwrap();

        let result = fs.read_with_post_read_hook(Path::new("file.txt"), 100, || {
            std::fs::rename(&path, &displaced).unwrap();
            std::fs::write(&path, "replacement content").unwrap();
        });

        assert!(matches!(result, Err(WorkspaceFsError::Conflict)));
        assert_eq!(std::fs::read_to_string(displaced).unwrap(), "opened content");
        assert_eq!(std::fs::read_to_string(path).unwrap(), "replacement content");
    }
    #[test]
    fn enforces_size_limit_and_hashes_bytes() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("file"), "data").unwrap();
        let fs = WorkspaceFs::open(temp.path()).unwrap();
        assert!(matches!(fs.read(Path::new("file"), 3), Err(WorkspaceFsError::TooLarge)));
        assert_eq!(fs.sha256(Path::new("file"), 4).unwrap(), "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7");
    }

    #[test]
    fn mutations_fail_closed_without_source_effects() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("src")).unwrap();
        std::fs::create_dir(temp.path().join("dst")).unwrap();
        let source = temp.path().join("src/a.rs");
        std::fs::write(&source, "base").unwrap();
        let fs = WorkspaceFs::open(temp.path()).unwrap();
        let expected = hex(&Sha256::digest(b"base"));

        assert!(!WorkspaceFs::mutation_supported());
        assert!(matches!(fs.create(Path::new("src/new.rs"), b"new"), Err(WorkspaceFsError::Unsupported)));
        assert!(matches!(fs.replace(Path::new("src/a.rs"), &expected, b"replacement"), Err(WorkspaceFsError::Unsupported)));
        assert!(matches!(fs.delete(Path::new("src/a.rs"), &expected), Err(WorkspaceFsError::Unsupported)));
        assert!(matches!(fs.move_file(Path::new("src/a.rs"), Path::new("dst/moved.rs"), &expected), Err(WorkspaceFsError::Unsupported)));
        assert_eq!(std::fs::read_to_string(&source).unwrap(), "base");
        assert!(!temp.path().join("src/new.rs").exists());
        assert!(!temp.path().join("dst/moved.rs").exists());
    }

    #[test]
    fn mutation_refuses_symlink_targets_when_backend_is_disabled() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("workspace");
        let outside = temp.path().join("outside.txt");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(&outside, "outside sentinel").unwrap();
        symlink(&outside, root.join("link.txt")).unwrap();
        let fs = WorkspaceFs::open(&root).unwrap();
        let expected = hex(&Sha256::digest(std::fs::read(&outside).unwrap()));

        assert!(matches!(fs.replace(Path::new("link.txt"), &expected, b"overwritten"), Err(WorkspaceFsError::PathEscape)));
        assert!(matches!(fs.delete(Path::new("link.txt"), &expected), Err(WorkspaceFsError::PathEscape)));
        assert!(matches!(fs.move_file(Path::new("link.txt"), Path::new("moved.txt"), &expected), Err(WorkspaceFsError::PathEscape)));
        assert_eq!(hex(&Sha256::digest(std::fs::read(&outside).unwrap())), expected);
    }
    #[test]
    fn registered_root_replacement_invalidates_reads_and_effects() {
        use std::os::unix::fs::MetadataExt;

        let temp = TempDir::new().unwrap();
        let root_path = temp.path().join("workspace");
        std::fs::create_dir(&root_path).unwrap();
        std::fs::write(root_path.join("file"), "registered").unwrap();
        let metadata = std::fs::symlink_metadata(&root_path).unwrap();
        let capability = WorkspaceFs::open_registered(&root_path, metadata.dev(), metadata.ino()).unwrap();
        let replacement = b"replacement-root-sentinel";
        let expected = hex(&Sha256::digest(replacement));

        std::fs::rename(&root_path, temp.path().join("displaced")).unwrap();
        std::fs::create_dir(&root_path).unwrap();
        std::fs::write(root_path.join("file"), replacement).unwrap();

        assert!(matches!(capability.read(Path::new("file"), 100), Err(WorkspaceFsError::Conflict)));
        assert!(matches!(capability.replace(Path::new("file"), &expected, b"changed"), Err(WorkspaceFsError::Conflict)));
        assert!(matches!(
            WorkspaceFs::open_registered(&root_path, metadata.dev(), metadata.ino()),
            Err(WorkspaceFsError::Conflict)
        ));
        assert_eq!(hex(&Sha256::digest(std::fs::read(root_path.join("file")).unwrap())), expected);
    }

    #[test]
    fn registered_root_symlink_replacement_is_not_followed() {
        use std::os::unix::fs::MetadataExt;

        let temp = TempDir::new().unwrap();
        let root = temp.path().join("workspace");
        let displaced = temp.path().join("displaced");
        std::fs::create_dir(&root).unwrap();
        let sentinel = b"registered-root sentinel";
        std::fs::write(root.join("sentinel.txt"), sentinel).unwrap();
        let metadata = std::fs::symlink_metadata(&root).unwrap();
        let capability = WorkspaceFs::open_registered(&root, metadata.dev(), metadata.ino()).unwrap();
        let expected = hex(&Sha256::digest(sentinel));

        std::fs::rename(&root, &displaced).unwrap();
        symlink(&displaced, &root).unwrap();
        assert!(matches!(capability.read(Path::new("sentinel.txt"), 100), Err(WorkspaceFsError::Conflict)));
        assert!(matches!(
            capability.replace(Path::new("sentinel.txt"), &expected, b"overwritten"),
            Err(WorkspaceFsError::Conflict)
        ));
        assert!(matches!(
            WorkspaceFs::open_registered(&root, metadata.dev(), metadata.ino()),
            Err(WorkspaceFsError::Conflict)
        ));
        assert_eq!(hex(&Sha256::digest(std::fs::read(displaced.join("sentinel.txt")).unwrap())), expected);
    }

    #[test]
    fn lexical_path_components_are_not_normalized_away() {
        let temp = TempDir::new().unwrap();
        let fs = WorkspaceFs::open(temp.path()).unwrap();
        for path in ["a//b", "a/./b", "a/../b", "bad\0name"] {
            assert!(matches!(fs.read(Path::new(path), 100), Err(WorkspaceFsError::InvalidPath)));
        }
        assert!(matches!(fs.read(Path::new("/absolute"), 100), Err(WorkspaceFsError::InvalidPath)));
    }

    #[test]
    fn same_inode_changes_are_not_overwritten_deleted_or_moved() {
        use std::os::unix::fs::MetadataExt;

        let temp = TempDir::new().unwrap();
        for name in ["replace.txt", "delete.txt", "move.txt"] {
            std::fs::write(temp.path().join(name), "prepared").unwrap();
        }
        let fs = WorkspaceFs::open(temp.path()).unwrap();
        let prepared_hash = hex(&Sha256::digest(b"prepared"));

        for (name, changed) in [
            ("replace.txt", "concurrent replace"),
            ("delete.txt", "concurrent delete"),
            ("move.txt", "concurrent move"),
        ] {
            let path = temp.path().join(name);
            let inode = std::fs::metadata(&path).unwrap().ino();
            std::fs::write(&path, changed).unwrap();
            assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
            let result = match name {
                "replace.txt" => fs.replace(Path::new(name), &prepared_hash, b"proposed"),
                "delete.txt" => fs.delete(Path::new(name), &prepared_hash),
                "move.txt" => fs.move_file(Path::new(name), Path::new("destination.txt"), &prepared_hash),
                _ => unreachable!(),
            };
            assert!(matches!(result, Err(WorkspaceFsError::Unsupported)));
            assert_eq!(std::fs::read_to_string(&path).unwrap(), changed);
        }
        assert!(!temp.path().join("destination.txt").exists());
        assert!(std::fs::read_dir(temp.path()).unwrap().all(|entry| {
            !entry.unwrap().file_name().to_string_lossy().starts_with(".workspace-index-")
        }));
    }

    #[test]
    fn relocated_parent_never_receives_an_effect_from_the_registered_root() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("workspace");
        let outside = temp.path().join("outside");
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(root.join("nested/source.txt"), "source sentinel").unwrap();
        std::fs::write(outside.join("sentinel.txt"), "outside sentinel").unwrap();
        let fs = WorkspaceFs::open(&root).unwrap();
        let source_hash = hex(&Sha256::digest(b"source sentinel"));
        let outside_hash = hex(&Sha256::digest(b"outside sentinel"));

        assert!(!WorkspaceFs::mutation_supported());
        std::fs::rename(root.join("nested"), outside.join("nested")).unwrap();
        assert!(fs.replace(Path::new("nested/source.txt"), &source_hash, b"changed").is_err());
        assert!(fs.delete(Path::new("nested/source.txt"), &source_hash).is_err());
        assert!(fs.move_file(Path::new("nested/source.txt"), Path::new("moved.txt"), &source_hash).is_err());
        assert_eq!(
            hex(&Sha256::digest(std::fs::read(outside.join("nested/source.txt")).unwrap())),
            source_hash,
        );
        assert_eq!(
            hex(&Sha256::digest(std::fs::read(outside.join("sentinel.txt")).unwrap())),
            outside_hash,
        );
    }

    #[test]
    fn symlink_effects_never_change_outside_sentinel() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("workspace");
        let outside = temp.path().join("outside.txt");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(&outside, "outside sentinel").unwrap();
        symlink(&outside, root.join("link.txt")).unwrap();
        let fs = WorkspaceFs::open(&root).unwrap();
        let expected = hex(&Sha256::digest(std::fs::read(&outside).unwrap()));

        assert!(matches!(fs.replace(Path::new("link.txt"), &expected, b"overwritten"), Err(WorkspaceFsError::PathEscape)));
        assert!(matches!(fs.delete(Path::new("link.txt"), &expected), Err(WorkspaceFsError::PathEscape)));
        assert!(matches!(fs.move_file(Path::new("link.txt"), Path::new("moved.txt"), &expected), Err(WorkspaceFsError::PathEscape)));
        assert_eq!(hex(&Sha256::digest(std::fs::read(&outside).unwrap())), expected);
    }
    #[test]
    fn sensitive_components_are_rejected_by_the_shared_capability_path_guard() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("nested")).unwrap();
        for path in [".env", "nested/credentials.json", "server.pem", "id_rsa"] {
            std::fs::write(temp.path().join(path), "secret sentinel").unwrap();
        }
        std::fs::write(temp.path().join("source.rs"), "source").unwrap();
        let fs = WorkspaceFs::open(temp.path()).unwrap();
        let expected = hex(&Sha256::digest(b"secret sentinel"));

        for path in [".env", "nested/credentials.json", "server.pem", "id_rsa"] {
            assert!(matches!(fs.resolve_parent(Path::new(path)), Err(WorkspaceFsError::SensitivePath)));
            assert!(matches!(fs.read(Path::new(path), 100), Err(WorkspaceFsError::SensitivePath)));
            assert!(matches!(fs.replace(Path::new(path), &expected, b"changed"), Err(WorkspaceFsError::SensitivePath)));
            assert!(matches!(fs.delete(Path::new(path), &expected), Err(WorkspaceFsError::SensitivePath)));
        }
        assert!(matches!(
            fs.create(Path::new("nested/.env.new"), b"secret"),
            Err(WorkspaceFsError::SensitivePath),
        ));
        assert!(matches!(
            fs.move_file(Path::new(".env"), Path::new("moved.txt"), &expected),
            Err(WorkspaceFsError::SensitivePath),
        ));
        assert_eq!(fs.read(Path::new("source.rs"), 100).unwrap(), b"source");
        assert!(!temp.path().join("nested/.env.new").exists());
    }
}
