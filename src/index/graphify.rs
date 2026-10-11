use std::collections::{BTreeMap, HashSet};
#[cfg(target_os = "linux")]
use std::fs::{self, File, OpenOptions};
#[cfg(target_os = "linux")]
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};


#[cfg(target_os = "linux")]
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::io::{BufRead, BufReader, Read};
#[cfg(target_os = "linux")]
use std::process::{Child, Command, Stdio};
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(target_os = "linux")]
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::thread::{self, JoinHandle};
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use uuid::Uuid;

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::config::GraphifyConfig;
#[cfg(target_os = "linux")]
use crate::config::MAX_GRAPHIFY_TIMEOUT_SECONDS;
use crate::security::WorkspaceFs;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn close_range(first: u32, last: u32, flags: std::ffi::c_int) -> std::ffi::c_int;
    fn kill(pid: std::ffi::c_int, signal: std::ffi::c_int) -> std::ffi::c_int;
}
#[cfg(target_os = "linux")]
const CLOSE_RANGE_CLOEXEC: std::ffi::c_int = 4;
#[cfg(target_os = "linux")]
const LINUX_SIGKILL: std::ffi::c_int = 9;


#[cfg(target_os = "linux")]
const MAX_GRAPH_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
#[cfg(target_os = "linux")]
const MAX_GRAPH_STAGING_BYTES: u64 = 128 * 1024 * 1024;
#[cfg(target_os = "linux")]
const MAX_GRAPH_NODES: usize = 100_000;
#[cfg(target_os = "linux")]
const MAX_GRAPH_EDGES: usize = 500_000;
const MAX_GRAPH_ID_BYTES: usize = 1024;
const MAX_GRAPH_LABEL_BYTES: usize = 4096;
const MAX_GRAPH_PATH_BYTES: usize = 4096;
const MAX_GRAPH_KIND_BYTES: usize = 256;
const MAX_GRAPH_RANGE_BYTES: usize = 256;
const MAX_GRAPH_EVIDENCE_BYTES: usize = 1024;
const MAX_GRAPH_RELATION_BYTES: usize = 256;
#[cfg(target_os = "linux")]
const MAX_CAPTURED_PROCESS_OUTPUT: usize = MAX_GRAPH_OUTPUT_BYTES as usize;
#[cfg(target_os = "linux")]
const MAX_GRAPHIFY_EXECUTABLE_BYTES: u64 = 1024 * 1024;
#[cfg(target_os = "linux")]
const MAX_GRAPHIFY_TEMP_BYTES: u64 = 16 * 1024 * 1024;
#[cfg(target_os = "linux")]
const MAX_GRAPH_OUTPUT_ENTRIES: usize = 100_000;
#[cfg(target_os = "linux")]
const MAX_MOUNTINFO_BYTES: u64 = 1024 * 1024;
#[cfg(target_os = "linux")]
const MAX_DIAGNOSTIC_STDERR_BYTES: u64 = 16 * 1024;
#[cfg(target_os = "linux")]
const GRAPHIFY_INPUT_PATH: &str = "/workspace-index/input";
#[cfg(target_os = "linux")]
const GRAPHIFY_OUTPUT_PATH: &str = "/workspace-index/output";
#[cfg(target_os = "linux")]
const GRAPHIFY_RUNTIME_PATH: &str = "/workspace-index/runtime";
#[cfg(target_os = "linux")]
const GRAPHIFY_HOME_PATH: &str = "/workspace-index/home";
#[cfg(target_os = "linux")]
const GRAPHIFY_VENV_PATH: &str = "/workspace-index/venv";
#[cfg(target_os = "linux")]
const GRAPHIFY_PYTHON_PATH: &str = "/workspace-index/venv/bin/python";
#[cfg(target_os = "linux")]
const GRAPHIFY_VENV_CONFIG_PATH: &str = "/workspace-index/venv/pyvenv.cfg";
#[cfg(target_os = "linux")]
const GRAPHIFY_VENV_BIN_PATH: &str = "/workspace-index/venv/bin";
#[cfg(target_os = "linux")]
const GRAPHIFY_RUNNER_NAME: &str = "runner.py";
#[cfg(target_os = "linux")]
const GRAPHIFY_PROGRAM_NAME: &str = "graphify";
#[cfg(target_os = "linux")]
const GRAPHIFY_RUNNER: &str = r#"
import os
import signal
import stat
import subprocess
import sys
import time

program, source, output, output_limit, entry_limit = sys.argv[1:]
output_limit = int(output_limit)
entry_limit = int(entry_limit)
command = [sys.executable, "-I", program, "extract", source, "--code-only", "--no-cluster", "--no-dedup", "--out", output]

def output_size(root):
    pending = [root]
    size = 0
    entries_seen = 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                entries_seen += 1
                if entries_seen > entry_limit:
                    return output_limit + 1
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(entry.path)
                elif stat.S_ISREG(metadata.st_mode):
                    size += metadata.st_size
                    if size >= output_limit:
                        return size
                else:
                    return output_limit + 1
    return size

try:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
    while process.poll() is None:
        if output_size(output) >= output_limit:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            sys.exit(1)
        time.sleep(0.01)
    if process.returncode != 0 or output_size(output) >= output_limit:
        sys.exit(1)
    root_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    graph_dir_fd = os.open("graphify-out", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                           dir_fd=root_fd)
    graph_fd = os.open("graph.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=graph_dir_fd)
    graph = os.fdopen(graph_fd, "rb")
    if not stat.S_ISREG(os.fstat(graph.fileno()).st_mode):
        sys.exit(1)
    total = 0
    while True:
        block = graph.read(65536)
        if not block:
            break
        total += len(block)
        if total > output_limit:
            sys.exit(1)
        sys.stdout.buffer.write(block)
    sys.stdout.buffer.flush()
except (OSError, ValueError):
    sys.exit(1)
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: Option<String>,
    pub kind: Option<String>,
    pub relative_path: String,
    pub range: Option<String>,
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphData {
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub(crate) struct GraphSnapshotMetadata {
    pub workspace_id: String,
    pub registration_epoch: u64,
    pub root_device: u64,
    pub root_inode: u64,
    pub visible_generation: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PublishedGraphData {
    #[serde(flatten)]
    pub metadata: GraphSnapshotMetadata,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub content_hashes: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ExtractedGraph {
    pub data: GraphData,
    pub content_hashes: BTreeMap<String, String>,
}

#[cfg(target_os = "linux")]
type ValidatedGraph = ExtractedGraph;

#[cfg(target_os = "linux")]
#[derive(Debug, Deserialize)]
struct GraphifyOutput {
    nodes: Vec<GraphifyNode>,
    edges: Vec<GraphifyEdge>,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Deserialize)]
struct GraphifyNode {
    id: String,
    source_file: String,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    file_type: Option<String>,
    #[serde(default)]
    source_location: Option<String>,
    #[serde(default, rename = "_origin")]
    origin: Option<String>,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Deserialize)]
struct GraphifyEdge {
    source: String,
    target: String,
    relation: String,
    #[serde(default)]
    source_file: Option<String>,
}


pub fn load_graph_allowlist(
    connection: &Connection,
    workspace_id: &str,
    visible_generation: u64,
) -> Result<BTreeMap<String, String>, String> {
    let mut statement = connection
        .prepare(
            "SELECT relative_path, content_hash FROM files
             WHERE workspace_id = ?1 AND visible_generation = ?2
             ORDER BY relative_path",
        )
        .map_err(|_| graph_unavailable())?;
    let mut rows = statement
        .query(rusqlite::params![workspace_id, visible_generation as i64])
        .map_err(|_| graph_unavailable())?;
    let mut allowlist = BTreeMap::new();
    while let Some(row) = rows.next().map_err(|_| graph_unavailable())? {
        let path: String = row.get(0).map_err(|_| graph_unavailable())?;
        let hash: String = row.get(1).map_err(|_| graph_unavailable())?;
        if !is_normalized_relative_path(&path)
            || !is_blake3_hash(&hash)
            || allowlist.insert(path, hash).is_some()
        {
            return Err(graph_unavailable());
        }
    }
    Ok(allowlist)
}

pub(crate) struct GraphExtractionRequest<'a> {
    pub config: &'a GraphifyConfig,
    pub workspace_id: &'a str,
    pub workspace: &'a WorkspaceFs,
    pub data_dir: &'a Path,
    pub allowlist: &'a BTreeMap<String, String>,
    pub max_file_bytes: u64,
}

#[cfg(target_os = "linux")]
pub(crate) fn extract_graph(request: GraphExtractionRequest<'_>) -> Result<ExtractedGraph, String> {
    let GraphExtractionRequest { config, workspace_id, workspace, data_dir, allowlist, max_file_bytes } = request;
    if !config.enabled || !is_valid_workspace_id(workspace_id) {
        return Err(graph_unavailable());
    }
    let private_data = data_dir.canonicalize().map_err(|_| graph_unavailable())?;
    let sandbox = resolve_graphify_sandbox(config, workspace.root_path(), &private_data)?;
    let scratch = ScratchRoot::create(&private_data)?;
    let staging_root = scratch.0.join("source");
    let runtime_root = scratch.0.join("runtime");
    ensure_private_directory(&staging_root).map_err(|_| graph_unavailable())?;
    ensure_private_directory(&runtime_root).map_err(|_| graph_unavailable())?;
    stage_sources(workspace, &staging_root, allowlist, max_file_bytes)?;
    let bytes = run_graphify(&sandbox, &staging_root, &runtime_root)?;
    let graph = parse_graphify_output(&bytes, allowlist)?;
    verify_graph_sources(workspace, &graph.content_hashes, max_file_bytes)?;
    Ok(graph)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn extract_graph(_request: GraphExtractionRequest<'_>) -> Result<ExtractedGraph, String> {
    Err(graph_unavailable())
}
pub(crate) fn graph_snapshot_path(
    data_dir: &Path,
    metadata: &GraphSnapshotMetadata,
) -> Option<PathBuf> {
    if !is_valid_workspace_id(&metadata.workspace_id) || metadata.registration_epoch == 0 {
        return None;
    }
    Some(data_dir.join("graphs").join(format!(
        "{}_{}_{}_{}_{}_{}",
        metadata.workspace_id.len(),
        metadata.workspace_id,
        metadata.registration_epoch,
        metadata.root_device,
        metadata.root_inode,
        metadata.visible_generation
    )).join("graph.json"))
}

pub(crate) fn remove_registration_graphs(
    data_dir: &Path,
    workspace_id: &str,
    registration_epoch: u64,
) {
    if !is_valid_workspace_id(workspace_id) || registration_epoch == 0 {
        return;
    }
    let graphs_dir = data_dir.join("graphs");
    let Ok(metadata) = std::fs::symlink_metadata(&graphs_dir) else { return; };
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return;
    }
    let prefix = format!("{}_{}_{}_", workspace_id.len(), workspace_id, registration_epoch);
    let legacy_prefix = format!("{workspace_id}_");
    let Ok(entries) = std::fs::read_dir(graphs_dir) else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue; };
        let is_registration_snapshot = name.starts_with(&prefix);
        let is_legacy_snapshot = name.strip_prefix(&legacy_prefix)
            .is_some_and(|generation| generation.parse::<u64>().is_ok());
        if (is_registration_snapshot || is_legacy_snapshot)
            && std::fs::symlink_metadata(&path)
                .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            && std::fs::remove_dir_all(path).is_err()
        {
            eprintln!("Graphify unregistered snapshot cleanup could not remove an old graph");
        }
    }
}

pub(crate) fn remove_graph_snapshot(data_dir: &Path, metadata: &GraphSnapshotMetadata) {
    let graphs_dir = data_dir.join("graphs");
    if !std::fs::symlink_metadata(&graphs_dir)
        .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
    {
        return;
    }
    let Some(snapshot_dir) = graph_snapshot_path(data_dir, metadata)
        .and_then(|path| path.parent().map(Path::to_path_buf))
    else { return; };
    if snapshot_dir.parent() != Some(graphs_dir.as_path()) {
        return;
    }
    if std::fs::symlink_metadata(&snapshot_dir)
        .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
        && std::fs::remove_dir_all(snapshot_dir).is_err()
    {
        eprintln!("Graphify failed snapshot cleanup could not remove the current graph");
    }
}

#[cfg(target_os = "linux")]
fn stage_sources(
    workspace: &WorkspaceFs,
    staging_root: &Path,
    allowlist: &BTreeMap<String, String>,
    max_file_bytes: u64,
) -> Result<(), String> {
    let mut staged_bytes = 0_u64;
    let per_file_limit = max_file_bytes.min(MAX_GRAPH_STAGING_BYTES);
    for (relative_path, expected_hash) in allowlist {
        if !is_normalized_relative_path(relative_path) || !is_blake3_hash(expected_hash) {
            return Err(graph_unavailable());
        }
        let bytes = workspace
            .read(Path::new(relative_path), per_file_limit)
            .map_err(|_| graph_unavailable())?;
        staged_bytes = increase_staged_bytes(staged_bytes, bytes.len() as u64)?;
        if blake3::hash(&bytes).to_hex().as_str() != expected_hash {
            return Err(graph_unavailable());
        }
        write_staged_file(staging_root, relative_path, &bytes)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn increase_staged_bytes(current: u64, added: u64) -> Result<u64, String> {
    current
        .checked_add(added)
        .filter(|total| *total <= MAX_GRAPH_STAGING_BYTES)
        .ok_or_else(graph_unavailable)
}


#[cfg(target_os = "linux")]
fn write_staged_file(root: &Path, relative_path: &str, bytes: &[u8]) -> Result<(), String> {
    let mut path = root.to_path_buf();
    let mut components = relative_path.split('/').peekable();
    while let Some(component) = components.next() {
        path.push(component);
        if components.peek().is_some() {
            ensure_private_directory(&path).map_err(|_| graph_unavailable())?;
        } else {
            write_private_file(&path, bytes).map_err(|_| graph_unavailable())?;
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
struct PythonLauncher {
    real_interpreter: PathBuf,
    runtime_mounts: Vec<(PathBuf, PathBuf)>,
    venv_root: PathBuf,
}

#[cfg(target_os = "linux")]
struct GraphifySandbox {
    bwrap: PathBuf,
    executable: PathBuf,
    launcher: PythonLauncher,
    library_mounts: Vec<(PathBuf, PathBuf)>,
    timeout: Duration,
}

#[cfg(target_os = "linux")]
fn resolve_graphify_sandbox(
    config: &GraphifyConfig,
    workspace_root: &Path,
    data_root: &Path,
) -> Result<GraphifySandbox, String> {
    if !(1..=MAX_GRAPHIFY_TIMEOUT_SECONDS).contains(&config.timeout_seconds) {
        return Err(graph_unavailable());
    }
    let bwrap = resolve_bwrap()?;
    let executable = resolve_graphify_executable(&config.command)?;
    let mut launcher = resolve_python_launcher(&executable)?;
    launcher.runtime_mounts = python_runtime_mounts(&launcher)?;
    let library_mounts = system_library_mounts()?;
    let mount_info = host_mount_info()?;
    validate_private_data_root(workspace_root, data_root, &mount_info)?;
    validate_workspace_isolation(workspace_root, &executable, &launcher, &library_mounts, &mount_info)?;
    validate_data_isolation(data_root, &executable, &launcher, &library_mounts, &mount_info)?;
    Ok(GraphifySandbox {
        bwrap,
        executable,
        launcher,
        library_mounts,
        timeout: Duration::from_secs(config.timeout_seconds),
    })
}

#[cfg(target_os = "linux")]
fn run_graphify(
    sandbox: &GraphifySandbox,
    staging_root: &Path,
    runtime_root: &Path,
) -> Result<Vec<u8>, String> {
    prepare_graphify_runtime(runtime_root, &sandbox.executable)?;
    let arguments = build_sandbox_arguments(
        &sandbox.launcher,
        staging_root,
        runtime_root,
        &sandbox.library_mounts,
        &sandbox.launcher.runtime_mounts,
    );
    execute_sandbox(&sandbox.bwrap, &arguments, sandbox.timeout)
}


#[cfg(target_os = "linux")]
fn resolve_graphify_executable(command: &str) -> Result<PathBuf, String> {
    let path = Path::new(command);
    let candidates = if path.is_absolute() || path.components().count() > 1 {
        vec![path.to_path_buf()]
    } else {
        std::env::var_os("PATH")
            .into_iter()
            .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
            .filter(|directory| !directory.as_os_str().is_empty())
            .map(|directory| directory.join(path))
            .collect()
    };
    candidates
        .into_iter()
        .filter_map(|candidate| candidate.canonicalize().ok())
        .find(|candidate| is_executable_file(candidate))
        .ok_or_else(graph_unavailable)
}

#[cfg(target_os = "linux")]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).is_ok_and(|metadata| {
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
    })
}

#[cfg(target_os = "linux")]
fn resolve_python_launcher(executable: &Path) -> Result<PythonLauncher, String> {
    let shebang = read_shebang(executable)?;
    let mut fields = shebang.split_whitespace();
    let interpreter = PathBuf::from(fields.next().ok_or_else(graph_unavailable)?);
    if !interpreter.is_absolute()
        || !interpreter.file_name().is_some_and(|name| name.to_string_lossy().starts_with("python"))
        || fields.next().is_some_and(|option| !matches!(option, "-E" | "-I" | "-s" | "-u" | "-B"))
        || fields.next().is_some()
    {
        return Err(graph_unavailable());
    }
    let venv_bin = interpreter.parent().ok_or_else(graph_unavailable)?.to_path_buf();
    let venv_root = venv_bin.parent().ok_or_else(graph_unavailable)?.to_path_buf();
    if venv_root.canonicalize().map_err(|_| graph_unavailable())? != venv_root
        || !is_regular_file(&venv_root.join("pyvenv.cfg"))
    {
        return Err(graph_unavailable());
    }
    let real_interpreter = interpreter.canonicalize().map_err(|_| graph_unavailable())?;
    if !is_executable_file(&real_interpreter) {
        return Err(graph_unavailable());
    }
    Ok(PythonLauncher {
        real_interpreter,
        runtime_mounts: Vec::new(),
        venv_root,
    })
}

#[cfg(target_os = "linux")]
fn python_runtime_mounts(launcher: &PythonLauncher) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let executable = launcher.real_interpreter.file_name().and_then(|name| name.to_str());
    let version = executable
        .and_then(|name| name.strip_prefix("python"))
        .filter(|version| version.starts_with(|character: char| character.is_ascii_digit()))
        .ok_or_else(graph_unavailable)?;
    let prefix = launcher.real_interpreter.parent().and_then(Path::parent).ok_or_else(graph_unavailable)?;
    let standard_library = [prefix.join("lib"), prefix.join("lib64")]
        .into_iter()
        .map(|library| library.join(format!("python{version}")))
        .find(|path| path.is_dir())
        .ok_or_else(graph_unavailable)?;
    let site_packages = [
        launcher.venv_root.join("lib").join(format!("python{version}")).join("site-packages"),
        launcher.venv_root.join("lib64").join(format!("python{version}")).join("site-packages"),
    ]
    .into_iter()
    .find(|path| path.is_dir());
    let mut mounts = vec![canonical_runtime_mount(&standard_library)?];
    if let Some(path) = site_packages {
        let source = path.canonicalize().map_err(|_| graph_unavailable())?;
        let relative = path.strip_prefix(&launcher.venv_root).map_err(|_| graph_unavailable())?;
        let destination = Path::new(GRAPHIFY_VENV_PATH).join(relative);
        mounts.push((source, destination));
    }
    mounts.dedup();
    Ok(mounts)
}

#[cfg(target_os = "linux")]
fn canonical_runtime_mount(path: &Path) -> Result<(PathBuf, PathBuf), String> {
    let canonical = path.canonicalize().map_err(|_| graph_unavailable())?;
    Ok((canonical.clone(), canonical))
}

#[cfg(target_os = "linux")]
fn read_shebang(executable: &Path) -> Result<String, String> {
    let mut line = String::new();
    BufReader::new(open_readonly_nofollow(executable)?.take(4096))
        .read_line(&mut line)
        .map_err(|_| graph_unavailable())?;
    let shebang = line.strip_prefix("#!").ok_or_else(graph_unavailable)?;
    Ok(shebang.trim_end_matches(&['\r', '\n'][..]).to_owned())
}

#[cfg(target_os = "linux")]
fn open_readonly_nofollow(path: &Path) -> Result<File, String> {
    use rustix::fs::{open, Mode, OFlags};
    open(
        path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map(File::from)
    .map_err(|_| graph_unavailable())
}

#[cfg(target_os = "linux")]
fn resolve_bwrap() -> Result<PathBuf, String> {
    resolve_bwrap_from(&[Path::new("/usr/bin/bwrap"), Path::new("/bin/bwrap")])
}

#[cfg(target_os = "linux")]
fn resolve_bwrap_from(candidates: &[&Path]) -> Result<PathBuf, String> {
    candidates
        .iter()
        .find_map(|path| validate_bwrap_binary(path).ok())
        .ok_or_else(graph_unavailable)
}

#[cfg(target_os = "linux")]
fn validate_bwrap_binary(path: &Path) -> Result<PathBuf, String> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let path = path.canonicalize().map_err(|_| graph_unavailable())?;
    let metadata = fs::metadata(&path).map_err(|_| graph_unavailable())?;
    if !metadata.is_file()
        || metadata.uid() != 0
        || metadata.permissions().mode() & 0o022 != 0
        || metadata.permissions().mode() & 0o111 == 0
    {
        return Err(graph_unavailable());
    }
    Ok(path)
}

#[cfg(target_os = "linux")]
fn system_library_mounts() -> Result<Vec<(PathBuf, PathBuf)>, String> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let mut mounts = Vec::new();
    for destination in ["/usr/lib", "/lib", "/lib64", "/usr/lib64"] {
        let destination = PathBuf::from(destination);
        let source = match destination.canonicalize() {
            Ok(source) if source.is_dir() => source,
            _ => continue,
        };
        let metadata = fs::metadata(&source).map_err(|_| graph_unavailable())?;
        if metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
            return Err(graph_unavailable());
        }
        if !mounts.contains(&(source.clone(), destination.clone())) {
            mounts.push((source, destination));
        }
    }
    if mounts
        .iter()
        .any(|(_, destination)| destination == Path::new("/usr/lib"))
    {
        Ok(mounts)
    } else {
        Err(graph_unavailable())
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone)]
struct MountInfo {
    device: String,
    root: PathBuf,
    mount_point: PathBuf,
}

#[cfg(target_os = "linux")]
fn validate_workspace_isolation(
    workspace_root: &Path,
    executable: &Path,
    launcher: &PythonLauncher,
    library_mounts: &[(PathBuf, PathBuf)],
    mount_info: &[MountInfo],
) -> Result<(), String> {
    let mounted: Vec<PathBuf> = [
        executable.to_path_buf(),
        launcher.venv_root.join("pyvenv.cfg"),
        launcher.real_interpreter.clone(),
    ]
    .into_iter()
    .chain(library_mounts.iter().map(|(source, _)| source.clone()))
    .chain(launcher.runtime_mounts.iter().map(|(source, _)| source.clone()))
    .collect();
    for source in mounted {
        if paths_overlap(workspace_root, &source)
            || filesystem_paths_overlap(workspace_root, &source, mount_info)?
            || mount_info.iter().any(|entry| entry.mount_point.starts_with(&source))
        {
            return Err(graph_unavailable());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn validate_data_isolation(
    data_root: &Path,
    executable: &Path,
    launcher: &PythonLauncher,
    library_mounts: &[(PathBuf, PathBuf)],
    mount_info: &[MountInfo],
) -> Result<(), String> {
    let mounted: Vec<PathBuf> = [
        executable.to_path_buf(),
        launcher.venv_root.join("pyvenv.cfg"),
        launcher.real_interpreter.clone(),
    ]
    .into_iter()
    .chain(library_mounts.iter().map(|(source, _)| source.clone()))
    .chain(launcher.runtime_mounts.iter().map(|(source, _)| source.clone()))
    .collect();
    for source in mounted {
        if paths_overlap(data_root, &source)
            || filesystem_paths_overlap(data_root, &source, mount_info)?
            || mount_info.iter().any(|entry| entry.mount_point.starts_with(&source))
        {
            return Err(graph_unavailable());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn host_mount_info() -> Result<Vec<MountInfo>, String> {
    let mut contents = Vec::new();
    File::open("/proc/self/mountinfo")
        .map_err(|_| graph_unavailable())?
        .take(MAX_MOUNTINFO_BYTES + 1)
        .read_to_end(&mut contents)
        .map_err(|_| graph_unavailable())?;
    if contents.len() as u64 > MAX_MOUNTINFO_BYTES {
        return Err(graph_unavailable());
    }
    std::str::from_utf8(&contents)
        .map_err(|_| graph_unavailable())?
        .lines()
        .map(parse_mount_info)
        .collect()
}

#[cfg(target_os = "linux")]
fn parse_mount_info(line: &str) -> Result<MountInfo, String> {
    let mut fields = line.split_whitespace();
    fields.next().ok_or_else(graph_unavailable)?;
    fields.next().ok_or_else(graph_unavailable)?;
    let device = fields.next().ok_or_else(graph_unavailable)?.to_owned();
    let root = decode_mount_path(fields.next().ok_or_else(graph_unavailable)?)?;
    let mount_point = decode_mount_path(fields.next().ok_or_else(graph_unavailable)?)?;
    Ok(MountInfo { device, root, mount_point })
}

#[cfg(target_os = "linux")]
fn decode_mount_path(value: &str) -> Result<PathBuf, String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            let digits = bytes.get(index + 1..index + 4).ok_or_else(graph_unavailable)?;
            let digits = std::str::from_utf8(digits).map_err(|_| graph_unavailable())?;
            decoded.push(u8::from_str_radix(digits, 8).map_err(|_| graph_unavailable())?);
            index += 4;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    let decoded = String::from_utf8(decoded).map_err(|_| graph_unavailable())?;
    let path = PathBuf::from(decoded);
    if path.is_absolute() {
        Ok(path)
    } else {
        Err(graph_unavailable())
    }
}

#[cfg(target_os = "linux")]
fn filesystem_path(path: &Path, mount_info: &[MountInfo]) -> Result<(String, PathBuf), String> {
    let mut selected: Option<&MountInfo> = None;
    for entry in mount_info.iter().filter(|entry| path.starts_with(&entry.mount_point)) {
        let Some(current) = selected else {
            selected = Some(entry);
            continue;
        };
        let entry_depth = entry.mount_point.components().count();
        let current_depth = current.mount_point.components().count();
        if entry_depth > current_depth {
            selected = Some(entry);
        } else if entry_depth == current_depth
            && (entry.device != current.device || entry.root != current.root)
        {
            return Err(graph_unavailable());
        }
    }
    let mount = selected.ok_or_else(graph_unavailable)?;
    let suffix = path
        .strip_prefix(&mount.mount_point)
        .map_err(|_| graph_unavailable())?;
    Ok((mount.device.clone(), mount.root.join(suffix)))
}

#[cfg(target_os = "linux")]
fn filesystem_paths_overlap(
    left: &Path,
    right: &Path,
    mount_info: &[MountInfo],
) -> Result<bool, String> {
    let (left_device, left_path) = filesystem_path(left, mount_info)?;
    let (right_device, right_path) = filesystem_path(right, mount_info)?;
    Ok(left_device == right_device && paths_overlap(&left_path, &right_path))
}

#[cfg(target_os = "linux")]
fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

#[cfg(target_os = "linux")]
fn validate_private_data_root(
    workspace_root: &Path,
    data_root: &Path,
    mount_info: &[MountInfo],
) -> Result<(), String> {
    if paths_overlap(workspace_root, data_root)
        || filesystem_paths_overlap(workspace_root, data_root, mount_info)?
        || mount_info
            .iter()
            .any(|entry| entry.mount_point.starts_with(data_root))
    {
        return Err(graph_unavailable());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn prepare_graphify_runtime(runtime_root: &Path, executable: &Path) -> Result<(), String> {
    let bytes = read_bounded_file(executable, MAX_GRAPHIFY_EXECUTABLE_BYTES)?;
    write_executable_file(&runtime_root.join(GRAPHIFY_PROGRAM_NAME), &bytes)
        .map_err(|_| graph_unavailable())?;
    write_executable_file(&runtime_root.join(GRAPHIFY_RUNNER_NAME), GRAPHIFY_RUNNER.as_bytes())
        .map_err(|_| graph_unavailable())
}

#[cfg(target_os = "linux")]
fn read_bounded_file(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let file = open_readonly_nofollow(path)?;
    if !file.metadata().map_err(|_| graph_unavailable())?.is_file() {
        return Err(graph_unavailable());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| graph_unavailable())?;
    if bytes.len() as u64 > limit {
        return Err(graph_unavailable());
    }
    Ok(bytes)
}

#[cfg(target_os = "linux")]
fn write_executable_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o700)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(target_os = "linux")]
fn build_sandbox_arguments(
    launcher: &PythonLauncher,
    staging_root: &Path,
    runtime_root: &Path,
    library_mounts: &[(PathBuf, PathBuf)],
    runtime_mounts: &[(PathBuf, PathBuf)],
) -> Vec<OsString> {
    let mut arguments = vec![
        "--die-with-parent".into(), "--new-session".into(), "--as-pid-1".into(),
        "--unshare-user".into(), "--unshare-pid".into(), "--unshare-net".into(),
        "--unshare-ipc".into(), "--unshare-uts".into(), "--disable-userns".into(),
        "--cap-drop".into(), "CAP_SYS_ADMIN".into(),
    ];
    let mut directories = HashSet::new();
    for directory in [
        Path::new("/proc"), Path::new("/dev"), Path::new("/tmp"),
        Path::new(GRAPHIFY_INPUT_PATH), Path::new(GRAPHIFY_OUTPUT_PATH),
        Path::new(GRAPHIFY_RUNTIME_PATH), Path::new(GRAPHIFY_HOME_PATH),
        Path::new(GRAPHIFY_VENV_BIN_PATH),
    ] {
        push_directory_chain(&mut arguments, directory, &mut directories);
    }
    for (_, destination) in library_mounts {
        push_directory_chain(&mut arguments, destination, &mut directories);
    }
    for (_, destination) in runtime_mounts {
        push_directory_chain(&mut arguments, destination, &mut directories);
    }
    for (source, destination) in library_mounts {
        push_pair(&mut arguments, "--ro-bind", source, destination);
    }
    for (source, destination) in runtime_mounts {
        push_pair(&mut arguments, "--ro-bind", source, destination);
    }
    push_pair(
        &mut arguments,
        "--ro-bind",
        &launcher.real_interpreter,
        Path::new(GRAPHIFY_PYTHON_PATH),
    );
    push_pair(
        &mut arguments,
        "--ro-bind",
        &launcher.venv_root.join("pyvenv.cfg"),
        Path::new(GRAPHIFY_VENV_CONFIG_PATH),
    );
    push_pair(&mut arguments, "--ro-bind", staging_root, Path::new(GRAPHIFY_INPUT_PATH));
    push_pair(&mut arguments, "--ro-bind", runtime_root, Path::new(GRAPHIFY_RUNTIME_PATH));
    push_tmpfs(&mut arguments, GRAPHIFY_OUTPUT_PATH, MAX_GRAPH_OUTPUT_BYTES);
    push_tmpfs(&mut arguments, GRAPHIFY_HOME_PATH, MAX_GRAPHIFY_TEMP_BYTES);
    push_tmpfs(&mut arguments, "/tmp", MAX_GRAPHIFY_TEMP_BYTES);
    push_path_option(&mut arguments, "--proc", Path::new("/proc"));
    push_path_option(&mut arguments, "--dev", Path::new("/dev"));
    push_path_option(&mut arguments, "--remount-ro", Path::new("/"));
    push_path_option(&mut arguments, "--chdir", Path::new(GRAPHIFY_HOME_PATH));
    arguments.extend(["--clearenv".into(), "--setenv".into(), "PATH".into()]);
    arguments.push(GRAPHIFY_VENV_BIN_PATH.into());
    arguments.extend(["--setenv".into(), "HOME".into(), GRAPHIFY_HOME_PATH.into()]);
    arguments.extend(["--setenv".into(), "LANG".into(), "C".into()]);
    arguments.extend(["--setenv".into(), "TMPDIR".into(), "/tmp".into(), "--".into()]);
    arguments.push(GRAPHIFY_PYTHON_PATH.into());
    arguments.extend([
        "-I".into(),
        Path::new(GRAPHIFY_RUNTIME_PATH).join(GRAPHIFY_RUNNER_NAME).into_os_string(),
        Path::new(GRAPHIFY_RUNTIME_PATH).join(GRAPHIFY_PROGRAM_NAME).into_os_string(),
        GRAPHIFY_INPUT_PATH.into(),
        GRAPHIFY_OUTPUT_PATH.into(),
        MAX_GRAPH_OUTPUT_BYTES.to_string().into(),
        MAX_GRAPH_OUTPUT_ENTRIES.to_string().into(),
    ]);
    arguments
}

#[cfg(target_os = "linux")]
fn push_directory_chain(
    arguments: &mut Vec<OsString>,
    path: &Path,
    created: &mut HashSet<PathBuf>,
) {
    let mut current = PathBuf::from("/");
    for component in path.components() {
        if let Component::Normal(part) = component {
            current.push(part);
            if created.insert(current.clone()) {
                arguments.push("--dir".into());
                arguments.push(current.as_os_str().to_os_string());
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn push_pair(arguments: &mut Vec<OsString>, option: &str, source: &Path, destination: &Path) {
    arguments.extend([option.into(), source.as_os_str().to_os_string()]);
    arguments.push(destination.as_os_str().to_os_string());
}

#[cfg(target_os = "linux")]
fn push_path_option(arguments: &mut Vec<OsString>, option: &str, path: &Path) {
    arguments.push(option.into());
    arguments.push(path.as_os_str().to_os_string());
}

#[cfg(target_os = "linux")]
fn push_tmpfs(arguments: &mut Vec<OsString>, destination: &str, limit: u64) {
    arguments.extend(["--size".into(), limit.to_string().into(), "--tmpfs".into(), destination.into()]);
}

#[cfg(target_os = "linux")]
fn execute_sandbox(
    bwrap: &Path,
    arguments: &[OsString],
    timeout: Duration,
) -> Result<Vec<u8>, String> {
    use std::os::unix::process::CommandExt;
    let mut command = Command::new(bwrap);
    command
        .args(arguments)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    // SAFETY: the child needs only standard streams; all other inherited descriptors are closed on exec.
    unsafe {
        command.pre_exec(|| {
            // SAFETY: close_range operates only on the child process descriptor table and preserves stdio.
            let result = close_range(3, u32::MAX, CLOSE_RANGE_CLOEXEC);
            if result == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        });
    }
    let mut child = command.spawn().map_err(|_| graph_unavailable())?;
    let Some(stdout) = child.stdout.take() else {
        stop_child(&mut child);
        return Err(graph_unavailable());
    };
    let Some(stderr) = child.stderr.take() else {
        stop_child(&mut child);
        return Err(graph_unavailable());
    };
    let (capture, overflow) = capture_stream(stdout);
    let stderr_capture = capture_diagnostic_stream(stderr);
    let exit = wait_for_child(&mut child, timeout, &overflow);
    let stderr_bytes = stderr_capture.join().map_err(|_| graph_unavailable())?;
    if exit.is_err() {
        let exit_code = child.try_wait().ok().flatten().and_then(|status| status.code());
        eprintln!(
            "Graphify sandbox child failed (stderr content redacted): exit_code={exit_code:?}, stderr_bytes={stderr_bytes}"
        );
    }
    let bytes = finish_capture(capture)?;
    if overflow.load(Ordering::Acquire) {
        return Err(graph_unavailable());
    }
    exit?;
    Ok(bytes)
}

#[cfg(target_os = "linux")]
fn capture_stream<R: Read + Send + 'static>(
    mut reader: R,
) -> (JoinHandle<Result<Vec<u8>, String>>, Arc<AtomicBool>) {
    let overflow = Arc::new(AtomicBool::new(false));
    let reader_overflow = Arc::clone(&overflow);
    let capture = thread::spawn(move || {
        let mut captured = Vec::with_capacity(64 * 1024);
        let mut buffer = [0_u8; 8192];
        loop {
            let count = reader.read(&mut buffer).map_err(|_| graph_unavailable())?;
            if count == 0 {
                break;
            }
            let remaining = MAX_CAPTURED_PROCESS_OUTPUT.saturating_sub(captured.len());
            if count > remaining {
                reader_overflow.store(true, Ordering::Release);
            }
            captured.extend_from_slice(&buffer[..count.min(remaining)]);
        }
        Ok(captured)
    });
    (capture, overflow)
}

#[cfg(target_os = "linux")]
fn capture_diagnostic_stream<R: Read + Send + 'static>(mut reader: R) -> JoinHandle<u64> {
    thread::spawn(move || {
        let mut buffer = [0_u8; 1024];
        let mut captured = 0_u64;
        loop {
            match reader.read(&mut buffer) {
                Ok(0) | Err(_) => return captured,
                Ok(count) => {
                    captured = captured.saturating_add(count as u64).min(MAX_DIAGNOSTIC_STDERR_BYTES);
                }
            }
        }
    })
}

#[cfg(target_os = "linux")]
fn finish_capture(capture: JoinHandle<Result<Vec<u8>, String>>) -> Result<Vec<u8>, String> {
    capture.join().map_err(|_| graph_unavailable())?
}

#[cfg(target_os = "linux")]
fn wait_for_child(
    child: &mut Child,
    timeout: Duration,
    overflow: &AtomicBool,
) -> Result<(), String> {
    let started = Instant::now();
    loop {
        if overflow.load(Ordering::Acquire) {
            stop_child(child);
            return Err(graph_unavailable());
        }
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(_)) => return Err(graph_unavailable()),
            Ok(None) if started.elapsed() < timeout => thread::sleep(Duration::from_millis(10)),
            Ok(None) | Err(_) => {
                stop_child(child);
                return Err(graph_unavailable());
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn stop_child(child: &mut Child) {
    if let Ok(process_group) = i32::try_from(child.id()) {
        // SAFETY: process_group is the unique group assigned to this bubblewrap child.
        let _ = unsafe { kill(-process_group, LINUX_SIGKILL) };
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(target_os = "linux")]
fn is_regular_file(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
}

#[cfg(target_os = "linux")]
fn parse_graphify_output(
    bytes: &[u8],
    allowlist: &BTreeMap<String, String>,
) -> Result<ValidatedGraph, String> {
    let output: GraphifyOutput =
        serde_json::from_slice(bytes).map_err(|_| graph_unavailable())?;
    if output.nodes.len() > MAX_GRAPH_NODES || output.edges.len() > MAX_GRAPH_EDGES {
        return Err(graph_unavailable());
    }
    validate_graphify_fields(&output)?;
    convert_graphify_output(output, allowlist)
}

#[cfg(target_os = "linux")]
fn validate_graphify_fields(output: &GraphifyOutput) -> Result<(), String> {
    let optional_fits = |value: Option<&str>, limit| value.is_none_or(|text| text.len() <= limit);
    for node in &output.nodes {
        if node.id.len() > MAX_GRAPH_ID_BYTES
            || node.source_file.len() > MAX_GRAPH_PATH_BYTES
            || !optional_fits(node.label.as_deref(), MAX_GRAPH_LABEL_BYTES)
            || !optional_fits(node.file_type.as_deref(), MAX_GRAPH_KIND_BYTES)
            || !optional_fits(node.source_location.as_deref(), MAX_GRAPH_RANGE_BYTES)
            || !optional_fits(node.origin.as_deref(), MAX_GRAPH_EVIDENCE_BYTES)
            || !is_normalized_relative_path(&node.source_file)
        {
            return Err(graph_unavailable());
        }
    }
    for edge in &output.edges {
        if edge.source.len() > MAX_GRAPH_ID_BYTES
            || edge.target.len() > MAX_GRAPH_ID_BYTES
            || edge.relation.len() > MAX_GRAPH_RELATION_BYTES
            || !optional_fits(edge.source_file.as_deref(), MAX_GRAPH_PATH_BYTES)
            || edge.source_file.as_deref().is_some_and(|path| !is_normalized_relative_path(path))
        {
            return Err(graph_unavailable());
        }
    }
    Ok(())
}

pub(crate) fn validate_graph_data(graph: &GraphData) -> bool {
    let mut ids = HashSet::with_capacity(graph.nodes.len());
    graph.nodes.iter().all(|node| {
        node.id.len() <= MAX_GRAPH_ID_BYTES
            && !node.id.trim().is_empty()
            && ids.insert(node.id.as_str())
            && node.relative_path.len() <= MAX_GRAPH_PATH_BYTES
            && is_normalized_relative_path(&node.relative_path)
            && node.label.as_deref().is_none_or(|value| value.len() <= MAX_GRAPH_LABEL_BYTES)
            && node.kind.as_deref().is_none_or(|value| value.len() <= MAX_GRAPH_KIND_BYTES)
            && node.range.as_deref().is_none_or(|value| value.len() <= MAX_GRAPH_RANGE_BYTES)
            && node.evidence.as_deref().is_none_or(|value| value.len() <= MAX_GRAPH_EVIDENCE_BYTES)
    }) && graph.edges.iter().all(|edge| {
        edge.source.len() <= MAX_GRAPH_ID_BYTES
            && edge.target.len() <= MAX_GRAPH_ID_BYTES
            && edge.kind.as_deref().is_none_or(|value| value.len() <= MAX_GRAPH_RELATION_BYTES)
            && ids.contains(edge.source.as_str())
            && ids.contains(edge.target.as_str())
    })
}

#[cfg(target_os = "linux")]
fn convert_graphify_output(
    output: GraphifyOutput,
    allowlist: &BTreeMap<String, String>,
) -> Result<ValidatedGraph, String> {
    let mut node_ids = HashSet::with_capacity(output.nodes.len());
    let mut content_hashes = BTreeMap::new();
    let nodes = output
        .nodes
        .into_iter()
        .map(|node| {
            if node.id.trim().is_empty() || !node_ids.insert(node.id.clone()) {
                return Err(graph_unavailable());
            }
            retain_source_hash(&node.source_file, allowlist, &mut content_hashes)?;
            Ok(GraphNode {
                id: node.id,
                label: node.label,
                kind: node.file_type,
                relative_path: node.source_file,
                range: node.source_location,
                evidence: node.origin,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let edges = output
        .edges
        .into_iter()
        .map(|edge| {
            if edge.source.trim().is_empty()
                || edge.target.trim().is_empty()
                || edge.relation.trim().is_empty()
                || !node_ids.contains(&edge.source)
                || !node_ids.contains(&edge.target)
            {
                return Err(graph_unavailable());
            }
            if let Some(path) = &edge.source_file {
                retain_source_hash(path, allowlist, &mut content_hashes)?;
            }
            Ok(GraphEdge {
                source: edge.source,
                target: edge.target,
                kind: Some(edge.relation),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ValidatedGraph {
        data: GraphData { nodes, edges },
        content_hashes,
    })
}

#[cfg(target_os = "linux")]
fn retain_source_hash(
    path: &str,
    allowlist: &BTreeMap<String, String>,
    content_hashes: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    if !is_normalized_relative_path(path) {
        return Err(graph_unavailable());
    }
    let hash = allowlist.get(path).ok_or_else(graph_unavailable)?;
    content_hashes.insert(path.to_owned(), hash.clone());
    Ok(())
}

#[cfg(target_os = "linux")]
fn verify_graph_sources(
    workspace: &WorkspaceFs,
    content_hashes: &BTreeMap<String, String>,
    max_file_bytes: u64,
) -> Result<(), String> {
    for (relative_path, expected_hash) in content_hashes {
        let bytes = workspace
            .read(Path::new(relative_path), max_file_bytes)
            .map_err(|_| graph_unavailable())?;
        if blake3::hash(&bytes).to_hex().as_str() != expected_hash {
            return Err(graph_unavailable());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub(crate) fn publish_graph(
    data_dir: &Path,
    metadata: &GraphSnapshotMetadata,
    graph: ExtractedGraph,
) -> Result<(), String> {
    if !is_valid_workspace_id(&metadata.workspace_id) || metadata.registration_epoch == 0 {
        return Err(graph_unavailable());
    }
    let graphs_dir = data_dir.join("graphs");
    ensure_private_directory(&graphs_dir).map_err(|_| graph_unavailable())?;
    let graph_file = graph_snapshot_path(data_dir, metadata).ok_or_else(graph_unavailable)?;
    let snapshot_dir = graph_file.parent().ok_or_else(graph_unavailable)?;
    ensure_private_directory(snapshot_dir).map_err(|_| graph_unavailable())?;
    let temporary = snapshot_dir.join(format!(".graph-{}.tmp", Uuid::new_v4()));
    let _temporary_file = TemporaryFile(temporary.clone());
    let published = PublishedGraphData {
        metadata: metadata.clone(),
        nodes: graph.data.nodes,
        edges: graph.data.edges,
        content_hashes: graph.content_hashes,
    };
    let bytes = serde_json::to_vec(&published).map_err(|_| graph_unavailable())?;
    if bytes.len() as u64 > MAX_GRAPH_OUTPUT_BYTES {
        return Err(graph_unavailable());
    }
    write_private_file(&temporary, &bytes).map_err(|_| graph_unavailable())?;
    atomic_replace(&temporary, &graph_file).map_err(|_| graph_unavailable())?;
    if sync_directory(snapshot_dir).is_err() {
        eprintln!("Graphify graph was atomically published; directory sync failed");
    }
    prune_obsolete_graphs(&graphs_dir, metadata);
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn publish_graph(
    _data_dir: &Path,
    _metadata: &GraphSnapshotMetadata,
    _graph: ExtractedGraph,
) -> Result<(), String> {
    Err(graph_unavailable())
}

#[cfg(target_os = "linux")]
fn prune_obsolete_graphs(graphs_dir: &Path, current: &GraphSnapshotMetadata) {
    let prefix = format!(
        "{}_{}_{}_{}_{}_",
        current.workspace_id.len(),
        current.workspace_id,
        current.registration_epoch,
        current.root_device,
        current.root_inode,
    );
    let entries = match fs::read_dir(graphs_dir) {
        Ok(entries) => entries,
        Err(_) => {
            eprintln!("Graphify stale snapshot cleanup could not list graph storage");
            return;
        }
    };
    for entry in entries.flatten() {
        remove_obsolete_graph(&entry, &prefix, current.visible_generation);
    }
}

#[cfg(target_os = "linux")]
fn remove_obsolete_graph(entry: &fs::DirEntry, prefix: &str, current_generation: u64) {
    let name = entry.file_name();
    let Some(name) = name.to_str() else { return; };
    let Some(generation) = name.strip_prefix(prefix).and_then(|value| value.parse::<u64>().ok()) else {
        return;
    };
    if generation >= current_generation {
        return;
    }
    let path = entry.path();
    let Ok(metadata) = fs::symlink_metadata(&path) else { return; };
    if metadata.is_dir() && !metadata.file_type().is_symlink() && fs::remove_dir_all(path).is_err() {
        eprintln!("Graphify stale snapshot cleanup could not remove an old graph");
    }
}

#[cfg(target_os = "linux")]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    #[cfg(windows)]
    if fs::symlink_metadata(destination).is_ok() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "graph snapshot exists"));
    }
    fs::rename(source, destination)
}

#[cfg(target_os = "linux")]
fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(target_os = "linux")]
fn write_private_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}


pub(crate) fn validate_content_hashes(content_hashes: &BTreeMap<String, String>) -> bool {
    content_hashes
        .iter()
        .all(|(path, hash)| is_normalized_relative_path(path) && is_blake3_hash(hash))
}
#[cfg(target_os = "linux")]
fn ensure_private_directory(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            set_private_directory_permissions(path)
        }
        Ok(_) => Err(io::Error::new(io::ErrorKind::InvalidInput, "not a directory")),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            create_private_directory(path)
        }
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "linux")]
fn create_private_directory(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700).create(path)?;
    set_private_directory_permissions(path)
}

#[cfg(target_os = "linux")]
fn set_private_directory_permissions(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

fn is_normalized_relative_path(value: &str) -> bool {
    if value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.contains('\0')
    {
        return false;
    }
    let mut component_count = 0;
    for component in value.split('/') {
        if component.is_empty() || component == "." || component == ".." {
            return false;
        }
        component_count += 1;
    }
    let path = Path::new(value);
    path.components().count() == component_count
        && path.components().all(|component| matches!(component, Component::Normal(_)))
}

fn is_blake3_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_valid_workspace_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 64
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && bytes.iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_' || *byte == b'-'
        })
}

fn graph_unavailable() -> String {
    "GRAPH_UNAVAILABLE".to_owned()
}

#[cfg(target_os = "linux")]
struct ScratchRoot(PathBuf);

#[cfg(target_os = "linux")]
impl ScratchRoot {
    fn create(data_dir: &Path) -> Result<Self, String> {
        let pending = data_dir.join("pending");
        ensure_private_directory(&pending).map_err(|_| graph_unavailable())?;
        for _ in 0..16 {
            let path = pending.join(format!("graphify-{}", Uuid::new_v4()));
            match create_private_directory(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(_) => return Err(graph_unavailable()),
            }
        }
        Err(graph_unavailable())
    }
}

#[cfg(target_os = "linux")]
impl Drop for ScratchRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[cfg(target_os = "linux")]
struct TemporaryFile(PathBuf);

#[cfg(target_os = "linux")]
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
#[cfg(all(test, target_os = "linux"))]
mod sandbox_tests {
    use super::*;

    fn sandbox_arguments() -> Vec<String> {
        let launcher = PythonLauncher {
            real_interpreter: "/usr/bin/python3.14".into(),
            runtime_mounts: vec![(
                PathBuf::from("/usr/lib/python3.14"),
                PathBuf::from("/usr/lib/python3.14"),
            )],
            venv_root: "/opt/graphify".into(),
        };
        let mounts = vec![(PathBuf::from("/usr/lib"), PathBuf::from("/usr/lib"))];
        build_sandbox_arguments(
            &launcher,
            Path::new("/private/staging"),
            Path::new("/private/runtime"),
            &mounts,
            &launcher.runtime_mounts,
        )
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect()
    }

    #[test]
    fn missing_bwrap_is_graph_unavailable_without_fallback() {
        let missing = Path::new("/__workspace_index_missing_bwrap__");
        assert_eq!(resolve_bwrap_from(&[missing]).unwrap_err(), graph_unavailable());
    }

    #[test]
    fn sandbox_argv_is_offline_minimal_and_environment_cleared() {
        let arguments = sandbox_arguments();
        assert!(arguments.contains(&"--unshare-net".into()));
        assert!(arguments.contains(&"--remount-ro".into()));
        assert!(arguments.contains(&"--clearenv".into()));
        assert!(!arguments.contains(&"--share-net".into()));
        assert!(arguments.windows(3).any(|window| {
            window[0] == "--ro-bind"
                && window[1] == "/private/staging"
                && window[2] == GRAPHIFY_INPUT_PATH
        }));
        assert!(arguments.windows(3).any(|window| {
            window[0] == "--ro-bind"
                && window[1] == "/usr/lib/python3.14"
                && window[2] == "/usr/lib/python3.14"
        }));
        assert!(arguments.windows(3).any(|window| {
            window[0] == "--ro-bind"
                && window[1] == "/usr/bin/python3.14"
                && window[2] == GRAPHIFY_PYTHON_PATH
        }));
        assert!(arguments.windows(3).any(|window| {
            window[0] == "--ro-bind"
                && window[1] == "/opt/graphify/pyvenv.cfg"
                && window[2] == GRAPHIFY_VENV_CONFIG_PATH
        }));
        assert!(arguments.contains(&GRAPHIFY_VENV_BIN_PATH.to_owned()));
        assert!(arguments.contains(&MAX_GRAPH_OUTPUT_BYTES.to_string()));
        assert!(arguments.contains(&MAX_GRAPH_OUTPUT_ENTRIES.to_string()));
        assert!(!arguments.iter().any(|argument| argument.contains("/registered/source")));
        assert!(GRAPHIFY_RUNNER.contains("command = [sys.executable, \"-I\", program"));
        assert!(!arguments.iter().any(|argument| {
            argument.starts_with("/opt/graphify/bin")
        }));

        let environment = arguments
            .iter()
            .enumerate()
            .filter(|&(_, argument)| argument == "--setenv")
            .map(|(index, _)| arguments[index + 1].as_str())
            .collect::<Vec<_>>();
        assert_eq!(environment, vec!["PATH", "HOME", "LANG", "TMPDIR"]);
    }

    #[test]
    fn staging_byte_budget_rejects_aggregate_overflow() {
        assert_eq!(
            increase_staged_bytes(MAX_GRAPH_STAGING_BYTES - 1, 1).unwrap(),
            MAX_GRAPH_STAGING_BYTES
        );
        assert!(increase_staged_bytes(MAX_GRAPH_STAGING_BYTES, 1).is_err());
    }

    #[test]
    fn rejects_workspace_subtree_bind_mount_aliases_before_staging() {
        let mount_info = vec![
            MountInfo {
                device: "8:1".into(),
                root: "/".into(),
                mount_point: "/".into(),
            },
            MountInfo {
                device: "8:1".into(),
                root: "/home/source".into(),
                mount_point: "/mnt".into(),
            },
            MountInfo {
                device: "8:1".into(),
                root: "/home/source".into(),
                mount_point: "/usr".into(),
            },
        ];
        let workspace = Path::new("/home/source");
        let data_root = Path::new("/mnt/index");
        assert!(validate_private_data_root(workspace, data_root, &mount_info).is_err());

        let launcher = PythonLauncher {
            real_interpreter: "/usr/bin/python3.14".into(),
            runtime_mounts: vec![
                (
                    PathBuf::from("/usr/lib/python3.14"),
                    PathBuf::from("/usr/lib/python3.14"),
                ),
                (
                    PathBuf::from("/mnt/index/runtime"),
                    PathBuf::from("/mnt/index/runtime"),
                ),
            ],
            venv_root: "/opt/graphify".into(),
        };
        assert!(validate_workspace_isolation(
            workspace,
            Path::new("/opt/graphify/bin/graphify"),
            &launcher,
            &[],
            &mount_info,
        )
        .is_err());
        assert!(validate_data_isolation(
            data_root,
            Path::new("/opt/graphify/bin/graphify"),
            &launcher,
            &[],
            &mount_info,
        )
        .is_err());
    }
}
