use std::fs;
use std::path::PathBuf;
use workspace_index_mcp::config::{Config, GraphifyConfig, Paths};
use workspace_index_mcp::service::WorkspaceService;

#[cfg(target_os = "linux")]
use std::path::Path;
#[cfg(target_os = "linux")]
use std::process::Command;
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
const GRAPHIFY_VENV_BIN_PATH: &str = "/workspace-index/venv/bin";
#[cfg(target_os = "linux")]
const GRAPHIFY_PYTHON_PATH: &str = "/workspace-index/venv/bin/python";
#[cfg(target_os = "linux")]
const GRAPHIFY_VENV_CONFIG_PATH: &str = "/workspace-index/venv/pyvenv.cfg";

fn setup_test_env(name: &str) -> (Paths, Config, PathBuf) {
    let temp = std::env::temp_dir().join(format!("workspace_index_graphify_{}", name));
    let _ = fs::remove_dir_all(&temp);
    fs::create_dir_all(&temp).unwrap();
    
    let root = temp.join("workspace");
    fs::create_dir_all(&root).unwrap();
    
    let paths = Paths {
        config_file: temp.join("config.toml"),
        data_dir: temp.join("data"),
        database: temp.join("data/index.db"),
        vectors_dir: temp.join("data/vectors"),
        pending_dir: temp.join("data/pending"),
    };
    
    fs::create_dir_all(&paths.data_dir).unwrap();
    fs::create_dir_all(&paths.vectors_dir).unwrap();
    fs::create_dir_all(&paths.pending_dir).unwrap();
    
    let config = Config {
        allow_non_git_roots: true,
        graphify: GraphifyConfig {
            enabled: true,
            command: "graphify".to_string(),
            timeout_seconds: 5,
        },
        ..Config::default()
    };
    
    (paths, config, root)
}

#[cfg(target_os = "linux")]
fn install_mock_graphify(
    paths: &Paths,
    name: &str,
    graph_json: &str,
    after_extract: &str,
) -> PathBuf {
    let program = paths
        .data_dir
        .parent()
        .unwrap()
        .join(format!("graphify-{name}"));
    write_mock_graphify(&program, graph_json, after_extract);
    program
}

#[cfg(target_os = "linux")]
fn mock_python(program: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::symlink;

    let venv = program.parent()?.join("graphify-test-venv");
    let bin = venv.join("bin");
    fs::create_dir_all(&bin).ok()?;
    let python = bin.join("python");
    if !python.exists() {
        let system_python = std::env::var_os("PATH")
            .into_iter()
            .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
            .map(|directory| directory.join("python3"))
            .find(|candidate| fs::metadata(candidate).is_ok_and(|metadata| metadata.is_file()))
            .and_then(|candidate| candidate.canonicalize().ok())?;
        symlink(&system_python, &python).ok()?;
        fs::write(
            venv.join("pyvenv.cfg"),
            format!(
                "home = {}\ninclude-system-site-packages = false\n",
                system_python.parent()?.display()
            ),
        )
        .ok()?;
    }
    Some(python)
}
#[cfg(target_os = "linux")]
fn python_launcher_paths(program: &Path) -> Option<(PathBuf, PathBuf)> {
    use std::io::{BufRead, BufReader, Read};

    let mut shebang = String::new();
    BufReader::new(fs::File::open(program).ok()?)
        .take(4096)
        .read_line(&mut shebang)
        .ok()?;
    let interpreter = PathBuf::from(shebang.strip_prefix("#!")?.split_whitespace().next()?);
    if !interpreter.is_absolute() {
        return None;
    }
    let venv_root = interpreter.parent()?.parent()?.to_path_buf();
    Some((interpreter, venv_root))
}

#[cfg(target_os = "linux")]
fn write_mock_graphify(program: &Path, graph_json: &str, after_extract: &str) {
    use std::os::unix::fs::PermissionsExt;

    let python = mock_python(program).unwrap_or_else(|| PathBuf::from("/__workspace_index_missing_python__"));
    let graph_json = serde_json::to_string(graph_json).unwrap();
    let script = format!(
        "#!{} -E\nimport json, os, sys\nout=sys.argv[7]\ntarget=os.path.join(out, 'graphify-out')\nos.makedirs(target, exist_ok=True)\nwith open(os.path.join(target, 'graph.json'), 'w', encoding='utf-8') as f:\n    f.write({graph_json})\n{after_extract}\n",
        python.display()
    );
    fs::write(program, script).unwrap();
    fs::set_permissions(program, fs::Permissions::from_mode(0o700)).unwrap();
}


#[cfg(target_os = "linux")]
fn bwrap_supports_required_namespaces(program: &Path) -> bool {
    use std::collections::HashSet;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let Some(bwrap) = ["/usr/bin/bwrap", "/bin/bwrap"]
        .into_iter()
        .find(|path| Path::new(path).is_file())
    else {
        eprintln!("Graphify test capability probe: bwrap is unavailable");
        return false;
    };
    let Ok(metadata) = fs::metadata(bwrap) else {
        eprintln!("Graphify test capability probe: bwrap metadata is unavailable");
        return false;
    };
    if !metadata.is_file()
        || metadata.uid() != 0
        || metadata.permissions().mode() & 0o022 != 0
        || metadata.permissions().mode() & 0o111 == 0
    {
        eprintln!("Graphify test capability probe: bwrap trust checks failed");
        return false;
    }
    let Some((interpreter, venv_root)) = python_launcher_paths(program) else {
        eprintln!("Graphify test capability probe: Python shebang is unavailable");
        return false;
    };
    let Ok(real_interpreter) = interpreter.canonicalize() else {
        eprintln!("Graphify test capability probe: Python interpreter is unavailable");
        return false;
    };
    let Some(version) = real_interpreter.file_name().and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("python"))
        .filter(|version| version.starts_with(|character: char| character.is_ascii_digit()))
    else {
        eprintln!("Graphify test capability probe: Python interpreter version is unsupported");
        return false;
    };
    let Some(prefix) = real_interpreter.parent().and_then(Path::parent) else {
        eprintln!("Graphify test capability probe: Python prefix is unavailable");
        return false;
    };
    let Some(standard_library) = [prefix.join("lib"), prefix.join("lib64")]
        .into_iter()
        .map(|library| library.join(format!("python{version}")))
        .find(|path| path.is_dir())
        .and_then(|path| path.canonicalize().ok())
    else {
        eprintln!("Graphify test capability probe: Python standard library is unavailable");
        return false;
    };
    let venv_config = venv_root.join("pyvenv.cfg");
    if !venv_config.is_file() {
        return false;
    }
    let site_packages_path = [
        venv_root.join("lib").join(format!("python{version}")).join("site-packages"),
        venv_root.join("lib64").join(format!("python{version}")).join("site-packages"),
    ]
    .into_iter()
    .find(|path| path.is_dir());
    let site_destination = site_packages_path
        .as_ref()
        .and_then(|path| path.strip_prefix(&venv_root).ok())
        .map(|path| Path::new(GRAPHIFY_VENV_PATH).join(path))
        .unwrap_or_else(|| {
            Path::new(GRAPHIFY_VENV_PATH)
                .join("lib")
                .join(format!("python{version}"))
                .join("site-packages")
        });
    let site_packages = site_packages_path.and_then(|path| path.canonicalize().ok());
    let Ok(scratch) = tempfile::tempdir() else { return false; };
    let staging = scratch.path().join("input");
    let runtime = scratch.path().join("runtime");
    if fs::create_dir_all(&staging).is_err()
        || fs::create_dir_all(&runtime).is_err()
        || fs::write(staging.join("probe"), b"input").is_err()
        || fs::write(runtime.join("probe"), b"runtime").is_err()
    {
        return false;
    }

    let mut command = Command::new(bwrap);
    command.args([
        "--die-with-parent", "--new-session", "--as-pid-1",
        "--unshare-user", "--unshare-pid", "--unshare-net",
        "--unshare-ipc", "--unshare-uts", "--disable-userns",
        "--cap-drop", "CAP_SYS_ADMIN",
    ]);
    let mut directories = HashSet::new();
    for directory in [
        Path::new("/usr"), Path::new("/proc"), Path::new("/dev"), Path::new("/tmp"),
        Path::new(GRAPHIFY_INPUT_PATH), Path::new(GRAPHIFY_OUTPUT_PATH),
        Path::new(GRAPHIFY_RUNTIME_PATH), Path::new(GRAPHIFY_HOME_PATH),
        Path::new(GRAPHIFY_VENV_BIN_PATH),
    ] {
        test_directory_chain(&mut command, directory, &mut directories);
    }
    let mut libraries = Vec::new();
    for destination in ["/usr/lib", "/lib", "/lib64", "/usr/lib64"] {
        let Ok(source) = Path::new(destination).canonicalize() else { continue; };
        let Ok(metadata) = fs::metadata(&source) else { return false; };
        if metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
            return false;
        }
        if !libraries.contains(&(source.clone(), PathBuf::from(destination))) {
            libraries.push((source, PathBuf::from(destination)));
        }
    }
    if !libraries.iter().any(|(_, destination)| destination == Path::new("/usr/lib")) {
        return false;
    }
    for (_, destination) in &libraries {
        test_directory_chain(&mut command, destination, &mut directories);
    }
    test_directory_chain(&mut command, &standard_library, &mut directories);
    test_directory_chain(&mut command, &site_destination, &mut directories);
    test_directory_chain(
        &mut command,
        Path::new(GRAPHIFY_PYTHON_PATH).parent().unwrap(),
        &mut directories,
    );
    for (source, destination) in libraries {
        command.arg("--ro-bind").arg(source).arg(destination);
    }
    command.arg("--ro-bind").arg(&standard_library).arg(&standard_library);
    if let Some(site_packages) = site_packages {
        command.arg("--ro-bind").arg(site_packages).arg(&site_destination);
    }
    command.arg("--ro-bind").arg(&real_interpreter).arg(GRAPHIFY_PYTHON_PATH);
    command.arg("--ro-bind").arg(&venv_config).arg(GRAPHIFY_VENV_CONFIG_PATH);
    command.arg("--ro-bind").arg(&staging).arg(GRAPHIFY_INPUT_PATH);
    command.arg("--ro-bind").arg(&runtime).arg(GRAPHIFY_RUNTIME_PATH);
    command.args([
        "--size", "4096", "--tmpfs", GRAPHIFY_OUTPUT_PATH,
        "--size", "4096", "--tmpfs", GRAPHIFY_HOME_PATH,
        "--size", "4096", "--tmpfs", "/tmp",
        "--proc", "/proc", "--dev", "/dev", "--remount-ro", "/",
        "--chdir", GRAPHIFY_HOME_PATH, "--clearenv",
        "--setenv", "PATH", GRAPHIFY_VENV_BIN_PATH,
        "--setenv", "HOME", GRAPHIFY_HOME_PATH,
        "--setenv", "LANG", "C",
        "--setenv", "TMPDIR", "/tmp",
        "--",
        GRAPHIFY_PYTHON_PATH, "-I", "-c",
    ]);
    command.arg(format!(
        r#"import os,sys,encodings
assert sys.prefix == '/workspace-index/venv'
assert {:?} in sys.path
assert {{'PATH','HOME','LANG','TMPDIR'}} <= set(os.environ)
assert not {{'OPENAI_API_KEY','AWS_ACCESS_KEY_ID'}} & set(os.environ)
assert not any(path.startswith('/home/') for path in sys.path)
def denied(path):
    try:
        open(path, 'wb').close()
    except OSError:
        return True
    return False
assert all(denied(path) for path in ['/workspace-index/input/probe','/workspace-index/runtime/probe','/workspace-index/readonly-probe'])
open('/workspace-index/output/probe', 'wb').close()"#,
        site_destination.to_string_lossy(),
    ));
    match command.env_clear().output() {
        Ok(output) => {
            if !output.status.success() {
                eprintln!(
                    "Graphify test capability probe failed: status={}, stderr={}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            output.status.success()
        }
        Err(error) => {
            eprintln!("Graphify test capability probe could not start: {error}");
            false
        }
    }
}

#[cfg(target_os = "linux")]
fn test_directory_chain(
    command: &mut Command,
    path: &Path,
    created: &mut std::collections::HashSet<PathBuf>,
) {
    let mut ancestors = path.ancestors().take_while(|entry| *entry != Path::new("/")).collect::<Vec<_>>();
    ancestors.reverse();
    for ancestor in ancestors {
        if created.insert(ancestor.to_path_buf()) {
            command.arg("--dir").arg(ancestor);
        }
    }
}



#[cfg(target_os = "linux")]
fn published_graph_path(
    paths: &Paths,
    service: &WorkspaceService,
    workspace_id: &str,
    generation: u64,
) -> PathBuf {
    let record = service.list_workspaces().unwrap().into_iter()
        .find(|record| record.id == workspace_id).unwrap();
    paths.data_dir.join("graphs").join(format!(
        "{}_{}_{}_{}_{}_{}",
        workspace_id.len(), workspace_id, record.registration_epoch,
        record.device_id, record.inode, generation,
    )).join("graph.json")
}

fn write_graph_fixture(
    paths: &Paths,
    service: &WorkspaceService,
    workspace_id: &str,
    generation: u64,
    mut graph: serde_json::Value,
) -> PathBuf {
    let record = service.list_workspaces().unwrap().into_iter()
        .find(|record| record.id == workspace_id).unwrap();
    let root = service.workspace_root(workspace_id).unwrap();
    let mut hashes = std::collections::BTreeMap::new();
    for node in graph["nodes"].as_array().into_iter().flatten() {
        let relative_path = node["relative_path"].as_str().unwrap();
        let bytes = fs::read(root.join(relative_path)).unwrap();
        hashes.insert(relative_path.to_owned(), blake3::hash(&bytes).to_hex().to_string());
    }
    graph["workspace_id"] = workspace_id.into();
    graph["registration_epoch"] = (record.registration_epoch as u64).into();
    graph["root_device"] = record.device_id.into();
    graph["root_inode"] = record.inode.into();
    graph["visible_generation"] = generation.into();
    graph["content_hashes"] = serde_json::to_value(hashes).unwrap();
    let path = paths.data_dir.join("graphs").join(format!(
        "{}_{}_{}_{}_{}_{}",
        workspace_id.len(), workspace_id, record.registration_epoch,
        record.device_id, record.inode, generation,
    )).join("graph.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, serde_json::to_vec(&graph).unwrap()).unwrap();
    path
}

#[cfg(target_os = "linux")]
const OBSERVED_GRAPHIFY_OUTPUT: &str = r#"{
    "nodes": [
        {
            "id": "function:foo",
            "label": "foo",
            "source_file": "test.py",
            "file_type": "python",
            "source_location": "L1",
            "_origin": "ast",
            "extra_node_field": "ignored"
        }
    ],
    "edges": [
        {
            "source": "function:foo",
            "target": "function:foo",
            "relation": "calls",
            "source_file": "test.py",
            "weight": 1.0
        }
    ],
    "hyperedges": []
}"#;

#[test]
fn test_graphify_missing_executable() {
    let (paths, mut config, root) = setup_test_env("missing");
    config.graphify.command = "nonexistent_graphify_executable".to_string();
    
    let service = WorkspaceService::open(paths, config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    
    fs::write(root.join("test.py"), "def foo(): pass").unwrap();
    
    // Reconcile with graphify=true
    let report = service.reconcile("ws1", true, true).unwrap();
    assert_eq!(report.updated, 1);
    
    // Read graph
    let (result, warnings) = service.read_graph("ws1", 100, None).unwrap();
    assert!(result.nodes.is_empty());
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, "GRAPH_UNAVAILABLE");

    let warning = service.graphify_warning("ws1").unwrap();
    assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
    assert!(!warning.retryable);
}
#[cfg(not(target_os = "linux"))]
#[test]
fn graphify_is_unavailable_without_the_linux_sandbox() {
    let (paths, config, root) = setup_test_env("unsupported_host");
    let service = WorkspaceService::open(paths, config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    let source_path = root.join("test.py");
    let source = b"def foo(): pass\n";
    fs::write(&source_path, source).unwrap();

    let report = service.reconcile("ws1", true, true).unwrap();
    assert_eq!(report.updated, 1);
    let warning = service.graphify_warning("ws1").unwrap();
    assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
    assert!(!warning.retryable);
    let (page, warnings) = service.read_graph("ws1", 100, None).unwrap();
    assert!(page.nodes.is_empty());
    assert_eq!(warnings[0].code, "GRAPH_UNAVAILABLE");
    assert_eq!(fs::read(source_path).unwrap(), source);
}

#[test]
fn explicitly_disabled_graphify_warns_without_failing_indexing() {
    let (paths, mut config, root) = setup_test_env("disabled");
    config.graphify.enabled = false;
    let service = WorkspaceService::open(paths, config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    fs::write(root.join("test.py"), "def foo(): pass\n").unwrap();

    let report = service.reconcile("ws1", true, true).unwrap();
    assert_eq!(report.updated, 1);
    let warning = service.graphify_warning("ws1").unwrap();
    assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
    assert!(!warning.retryable);
}

#[test]
fn regular_indexing_succeeds_when_graphify_is_disabled() {
    let (paths, mut config, root) = setup_test_env("ordinary_indexing");
    config.graphify.enabled = false;
    let service = WorkspaceService::open(paths, config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    fs::write(root.join("test.py"), "def foo(): pass\n").unwrap();

    let report = service.reconcile("ws1", true, false).unwrap();
    assert_eq!(report.updated, 1);
    assert!(service.graphify_warning("ws1").is_none());
}

#[test]
fn test_graphify_pagination() {
    let (paths, config, root) = setup_test_env("pagination");
    let service = WorkspaceService::open(paths.clone(), config).unwrap();
    
    // We will manually create a mock graph.json to test pagination
    service.register_workspace("ws1", &root).unwrap();
    
    fs::write(root.join("test.py"), "def foo(): pass").unwrap();
    
    let report = service.reconcile("ws1", true, false).unwrap();
    assert_eq!(report.updated, 1);
    
    let mock_graph = serde_json::json!({
        "nodes": [
            {"id": "n1", "relative_path": "test.py"},
            {"id": "n2", "relative_path": "test.py"},
            {"id": "n3", "relative_path": "test.py"}
        ],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"}
        ]
    });
    let generation = service.index_generation("ws1").unwrap();
    write_graph_fixture(&paths, &service, "ws1", generation, mock_graph);
    
    // Test pagination
    let (result1, warnings1) = service.read_graph("ws1", 2, None).unwrap();
    assert!(warnings1.is_empty());
    assert_eq!(result1.nodes.len(), 2);
    assert_eq!(result1.nodes[0].id, "n1");
    assert_eq!(result1.nodes[1].id, "n2");
    assert!(result1.truncated);
    assert!(result1.next_cursor.as_deref().is_some_and(|cursor| !cursor.is_empty()));
    assert_eq!(result1.edges.len(), 1); // Only n1->n2 has both endpoints on this page.

    let (result2, warnings2) = service.read_graph("ws1", 2, result1.next_cursor.as_deref()).unwrap();
    assert!(warnings2.is_empty());
    assert_eq!(result2.nodes.len(), 1);
    assert_eq!(result2.nodes[0].id, "n3");
    assert!(!result2.truncated);
    assert_eq!(result2.next_cursor, None);
    assert!(result2.edges.is_empty()); // n2 is not represented on this page.
}

#[test]
fn graph_cursor_is_stale_when_content_changes_within_same_generation() {
    let (paths, config, root) = setup_test_env("content_cursor");
    let service = WorkspaceService::open(paths.clone(), config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    fs::write(root.join("test.py"), "def foo(): pass").unwrap();
    service.reconcile("ws1", true, false).unwrap();

    let generation = service.index_generation("ws1").unwrap();
    write_graph_fixture(
        &paths,
        &service,
        "ws1",
        generation,
        serde_json::json!({
            "nodes": [
                {"id": "before-a", "relative_path": "test.py"},
                {"id": "before-b", "relative_path": "test.py"}
            ],
            "edges": []
        }),
    );
    let first_page = service.read_graph("ws1", 1, None).unwrap().0;
    let cursor = first_page.next_cursor.unwrap();
    write_graph_fixture(
        &paths,
        &service,
        "ws1",
        generation,
        serde_json::json!({
            "nodes": [
                {"id": "after-a", "relative_path": "test.py"},
                {"id": "after-b", "relative_path": "test.py"}
            ],
            "edges": []
        }),
    );

    let error = service.read_graph("ws1", 1, Some(&cursor)).unwrap_err();
    assert_eq!(error.code, workspace_index_mcp::model::ErrorCode::StaleCursor);
    assert_eq!(service.index_generation("ws1").unwrap(), generation);
}
#[test]
fn unregister_and_reregister_same_id_never_reuses_graph_or_cursor_state() {
    let (paths, config, root_a) = setup_test_env("registration_epoch");
    let service = WorkspaceService::open(paths.clone(), config).unwrap();
    service.register_workspace("ws1", &root_a).unwrap();
    fs::write(root_a.join("test.py"), "def old_root(): pass\n").unwrap();
    service.reconcile("ws1", true, false).unwrap();
    let old_generation = service.index_generation("ws1").unwrap();
    let old_record = service.list_workspaces().unwrap().remove(0);
    let old_path = write_graph_fixture(
        &paths,
        &service,
        "ws1",
        old_generation,
        serde_json::json!({
            "nodes": [
                {"id": "old-a", "relative_path": "test.py"},
                {"id": "old-b", "relative_path": "test.py"}
            ],
            "edges": []
        }),
    );
    let old_bytes = fs::read(&old_path).unwrap();
    let old_cursor = service.read_graph("ws1", 1, None).unwrap().0.next_cursor.unwrap();

    service.unregister_workspace("ws1").unwrap();
    assert!(!old_path.exists());

    let root_b = root_a.parent().unwrap().join("workspace-b");
    fs::create_dir_all(&root_b).unwrap();
    fs::write(root_b.join("test.py"), "def new_root(): pass\n").unwrap();
    service.register_workspace("ws1", &root_b).unwrap();
    service.reconcile("ws1", true, false).unwrap();
    let new_record = service.list_workspaces().unwrap().remove(0);
    assert!(new_record.registration_epoch > old_record.registration_epoch);
    let new_generation = service.index_generation("ws1").unwrap();
    let new_path = published_graph_path(&paths, &service, "ws1", new_generation);
    assert_ne!(new_path, old_path);
    fs::create_dir_all(new_path.parent().unwrap()).unwrap();
    fs::write(&new_path, old_bytes).unwrap();

    assert_eq!(
        service.read_graph("ws1", 1, Some(&old_cursor)).unwrap_err().code,
        workspace_index_mcp::model::ErrorCode::StaleCursor,
    );
    let (page, warnings) = service.read_graph("ws1", 10, None).unwrap();
    assert!(page.nodes.is_empty());
    assert_eq!(warnings[0].code, "GRAPH_UNAVAILABLE");
    assert!(service.index_status("ws1").unwrap().graph_generation.is_none());
}


#[cfg(target_os = "linux")]
#[test]
fn graphify_accepts_observed_no_cluster_shape_and_retains_hashes() {
    use std::os::unix::fs::PermissionsExt;

    let (paths, mut config, root) = setup_test_env("observed_shape");
    let program = install_mock_graphify(&paths, "observed_shape", OBSERVED_GRAPHIFY_OUTPUT, "");
    config.graphify.command = program.to_string_lossy().into_owned();
    let service = WorkspaceService::open(paths.clone(), config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    let source = b"def foo(): pass\n";
    fs::write(root.join("test.py"), source).unwrap();

    let report = service.reconcile("ws1", true, true).unwrap();
    if !bwrap_supports_required_namespaces(&program) {
        let warning = service.graphify_warning("ws1").unwrap();
        assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
        assert!(!warning.retryable);
        assert!(fs::read(root.join("test.py")).unwrap() == source);
        assert!(!published_graph_path(&paths, &service, "ws1", service.index_generation("ws1").unwrap()).exists());
        assert!(fs::read_dir(&paths.pending_dir).unwrap().next().is_none());
        return;
    }
    assert_eq!(report.updated, 1);
    let generation = service.index_generation("ws1").unwrap();
    let (page, warnings) = service.read_graph("ws1", 100, None).unwrap();

    assert!(warnings.is_empty());
    assert_eq!(page.nodes.len(), 1);
    assert_eq!(page.nodes[0].relative_path, "test.py");
    assert_eq!(page.nodes[0].kind.as_deref(), Some("python"));
    assert_eq!(page.nodes[0].range.as_deref(), Some("L1"));
    assert_eq!(page.nodes[0].evidence.as_deref(), Some("ast"));
    assert_eq!(page.edges.len(), 1);
    assert_eq!(page.edges[0].kind.as_deref(), Some("calls"));

    let graph_file = published_graph_path(&paths, &service, "ws1", generation);
    let graph: serde_json::Value =
        serde_json::from_slice(&fs::read(&graph_file).unwrap()).unwrap();
    let expected_hash = blake3::hash(source).to_hex().to_string();
    assert_eq!(
        graph["content_hashes"]["test.py"].as_str(),
        Some(expected_hash.as_str())
    );
    assert_eq!(fs::read(root.join("test.py")).unwrap(), source);
    assert_eq!(
        fs::metadata(graph_file.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(&graph_file).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(fs::read_dir(&paths.pending_dir).unwrap().next().is_none());
    assert!(!root.join("graphify-out").exists());
}
#[cfg(target_os = "linux")]
#[test]
fn installed_graphify_executes_with_only_private_runtime_mounts() {
    let (paths, mut config, root) = setup_test_env("installed_graphify");
    let Some(program) = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .map(|directory| directory.join("graphify"))
        .find(|candidate| fs::metadata(candidate).is_ok_and(|metadata| metadata.is_file()))
        .and_then(|candidate| candidate.canonicalize().ok())
    else {
        config.graphify.command = "__workspace_index_missing_graphify__".to_owned();
        let service = WorkspaceService::open(paths, config).unwrap();
        service.register_workspace("ws1", &root).unwrap();
        fs::write(root.join("test.py"), "def caller(): return callee()\ndef callee(): return 1\n").unwrap();
        assert_eq!(service.reconcile("ws1", true, true).unwrap().updated, 1);
        let warning = service.graphify_warning("ws1").unwrap();
        assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
        assert!(!warning.retryable);
        return;
    };
    config.graphify.command = program.to_string_lossy().into_owned();
    config.graphify.timeout_seconds = 120;
    let backend_available = bwrap_supports_required_namespaces(&program);
    let service = WorkspaceService::open(paths.clone(), config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    let source_path = root.join("test.py");
    let source = b"def caller():\n    return callee()\n\ndef callee():\n    return 1\n";
    fs::write(&source_path, source).unwrap();

    assert_eq!(service.reconcile("ws1", true, true).unwrap().updated, 1);
    if !backend_available {
        let warning = service.graphify_warning("ws1").unwrap();
        assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
        assert!(!warning.retryable);
        assert_eq!(fs::read(&source_path).unwrap(), source);
        return;
    }

    assert!(
        service.graphify_warning("ws1").is_none(),
        "installed Graphify failed despite a successful sandbox capability probe"
    );
    let (page, warnings) = service.read_graph("ws1", 100, None).unwrap();
    assert!(warnings.is_empty());
    assert!(!page.nodes.is_empty());
    assert!(page.nodes.iter().all(|node| node.relative_path == "test.py"));
    assert_eq!(fs::read(&source_path).unwrap(), source);
    assert!(!root.join("graphify-out").exists());
}

#[cfg(target_os = "linux")]
#[test]
fn graphify_rejects_unallowlisted_paths_dangling_endpoints_and_malformed_schema() {
    let cases = [
        (
            "outside_path",
            r#"{"nodes":[{"id":"n1","source_file":"../outside.py"}],"edges":[]}"#,
        ),
        (
            "dangling_endpoint",
            r#"{"nodes":[{"id":"n1","source_file":"test.py"}],"edges":[{"source":"n1","target":"missing","relation":"calls"}]}"#,
        ),
        (
            "malformed_schema",
            r#"{"nodes":[{"id":"n1","source_file":"test.py"}]}"#,
        ),
    ];

    for (name, graph_json) in cases {
        let (paths, mut config, root) = setup_test_env(name);
        let program = install_mock_graphify(&paths, name, graph_json, "");
        config.graphify.command = program.to_string_lossy().into_owned();
        let service = WorkspaceService::open(paths.clone(), config).unwrap();
        service.register_workspace("ws1", &root).unwrap();
        fs::write(root.join("test.py"), "def foo(): pass\n").unwrap();

        service.reconcile("ws1", true, true).unwrap();
        let generation = service.index_generation("ws1").unwrap();
        assert!(!published_graph_path(&paths, &service, "ws1", generation).exists());
        let (page, warnings) = service.read_graph("ws1", 100, None).unwrap();
        assert!(page.nodes.is_empty());
        assert_eq!(warnings[0].code, "GRAPH_UNAVAILABLE");
        assert!(fs::read_dir(&paths.pending_dir).unwrap().next().is_none());
    }
}

#[cfg(target_os = "linux")]
#[test]
fn graphify_child_cannot_modify_registered_source_files() {
    let (paths, mut config, root) = setup_test_env("source_isolation");
    let source_path = root.join("test.py");
    let source = b"def foo(): pass\n";
    fs::write(&source_path, source).unwrap();
    let source_literal = serde_json::to_string(source_path.to_str().unwrap()).unwrap();
    let attempt = format!(
        "target_path = {source_literal}\nos.makedirs(os.path.dirname(target_path), exist_ok=True)\nwith open(target_path, 'w', encoding='utf-8') as source:\n    source.write('child changed sentinel')"
    );
    let program = install_mock_graphify(
        &paths,
        "source_isolation",
        OBSERVED_GRAPHIFY_OUTPUT,
        &attempt,
    );
    config.graphify.command = program.to_string_lossy().into_owned();
    let service = WorkspaceService::open(paths.clone(), config).unwrap();
    service.register_workspace("ws1", &root).unwrap();

    let report = service.reconcile("ws1", true, true).unwrap();
    assert_eq!(report.updated, 1);
    assert_eq!(fs::read(&source_path).unwrap(), source);
    if bwrap_supports_required_namespaces(&program) {
        assert!(service.graphify_warning("ws1").is_none());
        assert_eq!(service.read_graph("ws1", 100, None).unwrap().0.nodes.len(), 1);
    } else {
        let warning = service.graphify_warning("ws1").unwrap();
        assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
        assert!(!warning.retryable);
    }
    assert!(fs::read_dir(&paths.pending_dir).unwrap().next().is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn graphify_failure_preserves_the_last_complete_snapshot() {
    let (paths, mut config, root) = setup_test_env("preserve_snapshot");
    let program = install_mock_graphify(&paths, "preserve_snapshot", OBSERVED_GRAPHIFY_OUTPUT, "");
    config.graphify.command = program.to_string_lossy().into_owned();
    let service = WorkspaceService::open(paths.clone(), config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    let source_path = root.join("test.py");
    let source = b"def foo(): pass\n";
    fs::write(&source_path, source).unwrap();
    service.reconcile("ws1", true, false).unwrap();

    service.reconcile("ws1", false, true).unwrap();
    if !bwrap_supports_required_namespaces(&program) {
        let warning = service.graphify_warning("ws1").unwrap();
        assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
        assert_eq!(fs::read(&source_path).unwrap(), source);
        return;
    }
    let first_generation = service.index_generation("ws1").unwrap();
    let first_graph = published_graph_path(&paths, &service, "ws1", first_generation);
    let first_snapshot = fs::read(&first_graph).unwrap();
    write_mock_graphify(&program, r#"{"nodes":[]}"#, "");

    service.reconcile("ws1", true, true).unwrap();
    let current_generation = service.index_generation("ws1").unwrap();
    let current_graph = published_graph_path(&paths, &service, "ws1", current_generation);
    assert_eq!(fs::read(&first_graph).unwrap(), first_snapshot);
    if current_generation != first_generation {
        assert!(!current_graph.exists());
    }
    assert_eq!(fs::read(&source_path).unwrap(), source);
    assert_eq!(service.graphify_warning("ws1").unwrap().code, "GRAPH_UNAVAILABLE");
    assert!(fs::read_dir(&paths.pending_dir).unwrap().next().is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn graphify_timeout_is_bounded_and_warns_without_failing_indexing() {
    use std::time::{Duration, Instant};

    let (paths, mut config, root) = setup_test_env("timeout");
    let program = install_mock_graphify(
        &paths,
        "timeout",
        OBSERVED_GRAPHIFY_OUTPUT,
        "import time\ntime.sleep(10)",
    );
    config.graphify.command = program.to_string_lossy().into_owned();
    config.graphify.timeout_seconds = 1;
    let service = WorkspaceService::open(paths, config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    let source_path = root.join("test.py");
    fs::write(&source_path, "def foo(): pass\n").unwrap();
    service.reconcile("ws1", true, false).unwrap();

    let started = Instant::now();
    service.reconcile("ws1", false, true).unwrap();
    if bwrap_supports_required_namespaces(&program) {
        assert!(started.elapsed() < Duration::from_secs(5));
    }
    let warning = service.graphify_warning("ws1").unwrap();
    assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
    assert!(!warning.retryable);
    assert_eq!(fs::read(&source_path).unwrap(), b"def foo(): pass\n");
}

#[cfg(target_os = "linux")]
#[test]
fn graphify_output_overflow_is_bounded_in_sandbox() {
    let (paths, mut config, root) = setup_test_env("output_overflow");
    let overflow = "with open(os.path.join(out, 'overflow.bin'), 'wb') as output:\n    chunk = b'x' * (1024 * 1024)\n    for _ in range(65):\n        output.write(chunk)";
    let program = install_mock_graphify(
        &paths,
        "output_overflow",
        OBSERVED_GRAPHIFY_OUTPUT,
        overflow,
    );
    config.graphify.command = program.to_string_lossy().into_owned();
    let service = WorkspaceService::open(paths, config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    let source_path = root.join("test.py");
    let source = b"def foo(): pass\n";
    fs::write(&source_path, source).unwrap();
    service.reconcile("ws1", true, false).unwrap();

    service.reconcile("ws1", false, true).unwrap();
    let warning = service.graphify_warning("ws1").unwrap();
    assert_eq!(warning.code, "GRAPH_UNAVAILABLE");
    assert!(!warning.retryable);
    assert_eq!(fs::read(&source_path).unwrap(), source);
}

#[cfg(target_os = "linux")]
#[test]
fn graphify_child_does_not_inherit_cloud_credentials() {
    if std::env::var_os("GRAPHIFY_CREDENTIAL_TEST_CHILD").is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "graphify_child_does_not_inherit_cloud_credentials",
                "--nocapture",
            ])
            .env("GRAPHIFY_CREDENTIAL_TEST_CHILD", "1")
            .env("OPENAI_API_KEY", "test-openai-credential")
            .env("AWS_ACCESS_KEY_ID", "test-aws-credential")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let (paths, mut config, root) = setup_test_env("credentials");
    let output_script = "payload = {'nodes': [{'id': 'credential-check', 'source_file': 'test.py', 'label': os.environ.get('OPENAI_API_KEY') or os.environ.get('AWS_ACCESS_KEY_ID')}], 'edges': []}\nwith open(os.path.join(out, 'graphify-out', 'graph.json'), 'w', encoding='utf-8') as f:\n    json.dump(payload, f)";
    let program = install_mock_graphify(
        &paths,
        "credentials",
        OBSERVED_GRAPHIFY_OUTPUT,
        output_script,
    );
    config.graphify.command = program.to_string_lossy().into_owned();
    let service = WorkspaceService::open(paths, config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    fs::write(root.join("test.py"), "def foo(): pass\n").unwrap();
    service.reconcile("ws1", true, false).unwrap();
    service.reconcile("ws1", false, true).unwrap();

    if bwrap_supports_required_namespaces(&program) {
        assert!(service.graphify_warning("ws1").is_none());
        let page = service.read_graph("ws1", 100, None).unwrap().0;
        assert_eq!(page.nodes[0].label, None);
    } else {
        assert_eq!(service.graphify_warning("ws1").unwrap().code, "GRAPH_UNAVAILABLE");
    }
}

#[test]
fn graph_resource_pages_stay_within_configured_serialized_byte_limit() {
    let (paths, mut config, root) = setup_test_env("serialized_page_budget");
    let max_tool_bytes = 1_900;
    config.limits.max_tool_bytes = max_tool_bytes;
    config.limits.max_snippet_bytes = 512;
    let service = WorkspaceService::open(paths.clone(), config).unwrap();
    service.register_workspace("ws1", &root).unwrap();
    fs::write(root.join("test.py"), "def foo(): pass\n").unwrap();
    service.reconcile("ws1", true, false).unwrap();
    let generation = service.index_generation("ws1").unwrap();
    let nodes = (0..8).map(|index| serde_json::json!({
        "id": format!("node-{index}-{}", "\\\"".repeat(30)),
        "relative_path": "test.py",
        "label": "\\\"\n".repeat(60),
    })).collect::<Vec<_>>();
    let edges = (0..8).flat_map(|index| (0..12).map(move |edge| serde_json::json!({
        "source": format!("node-{index}-{}", "\\\"".repeat(30)),
        "target": format!("node-{index}-{}", "\\\"".repeat(30)),
        "kind": format!("relation-{edge:02}-{}", "\\\"".repeat(8)),
    }))).collect::<Vec<_>>();
    let graph = serde_json::json!({"nodes": nodes, "edges": edges});
    write_graph_fixture(&paths, &service, "ws1", generation, graph);

    let expected_nodes = (0..8).map(|index| {
        format!("node-{index}-{}", "\\\"".repeat(30))
    }).collect::<std::collections::HashSet<_>>();
    let mut cursor = None;
    let mut seen_nodes = std::collections::HashSet::new();
    let mut page_count = 0;
    loop {
        let (page, warnings) = service.read_graph("ws1", 2, cursor.as_deref()).unwrap();
        assert!(warnings.is_empty());
        let result = workspace_index_mcp::model::ToolResult::Success {
            data: page.clone(),
            warnings,
            meta: workspace_index_mcp::model::Meta { request_id: String::new() },
        };
        assert!(serde_json::to_vec(&result).unwrap().len() <= max_tool_bytes);
        let page_node_ids = page.nodes.iter().map(|node| node.id.clone())
            .collect::<std::collections::HashSet<_>>();
        assert!(page.nodes.iter().all(|node| seen_nodes.insert(node.id.clone())));
        assert!(page.edges.iter().all(|edge| {
            page_node_ids.contains(&edge.source) && page_node_ids.contains(&edge.target)
        }));
        assert_eq!(page.edges_truncated, page.edges.len() < page.nodes.len() * 12);
        assert_eq!(page.truncated, page.nodes_truncated || page.edges_truncated);
        assert!(page.edges_truncated, "the fixture exceeds the per-page edge cap");
        page_count += 1;
        cursor = page.next_cursor;
        if cursor.is_none() { break; }
    }
    assert!(page_count >= 4);
    assert_eq!(seen_nodes, expected_nodes);
}

#[cfg(target_os = "linux")]
#[test]
fn graphify_rejects_each_oversized_string_field() {
    let oversized = "x".repeat(10_000);
    let cases = [
        ("id", serde_json::json!({"nodes":[{"id":oversized.as_str(),"source_file":"test.py"}],"edges":[]})),
        ("label", serde_json::json!({"nodes":[{"id":"n","source_file":"test.py","label":oversized.as_str()}],"edges":[]})),
        ("path", serde_json::json!({"nodes":[{"id":"n","source_file":oversized.as_str()}],"edges":[]})),
        ("kind", serde_json::json!({"nodes":[{"id":"n","source_file":"test.py","file_type":oversized.as_str()}],"edges":[]})),
        ("range", serde_json::json!({"nodes":[{"id":"n","source_file":"test.py","source_location":oversized.as_str()}],"edges":[]})),
        ("evidence", serde_json::json!({"nodes":[{"id":"n","source_file":"test.py","_origin":oversized.as_str()}],"edges":[]})),
        ("source", serde_json::json!({"nodes":[{"id":"n","source_file":"test.py"}],"edges":[{"source":oversized.as_str(),"target":"n","relation":"calls"}]})),
        ("target", serde_json::json!({"nodes":[{"id":"n","source_file":"test.py"}],"edges":[{"source":"n","target":oversized.as_str(),"relation":"calls"}]})),
        ("relation", serde_json::json!({"nodes":[{"id":"n","source_file":"test.py"}],"edges":[{"source":"n","target":"n","relation":oversized.as_str()}]})),
        ("edge_path", serde_json::json!({"nodes":[{"id":"n","source_file":"test.py"}],"edges":[{"source":"n","target":"n","relation":"calls","source_file":oversized.as_str()}]})),
    ];
    for (field, graph) in cases {
        let (paths, mut config, root) = setup_test_env(&format!("oversized_{field}"));
        let program = install_mock_graphify(&paths, field, &graph.to_string(), "");
        config.graphify.command = program.to_string_lossy().into_owned();
        let service = WorkspaceService::open(paths.clone(), config).unwrap();
        service.register_workspace("ws1", &root).unwrap();
        fs::write(root.join("test.py"), "def foo(): pass\n").unwrap();
        service.reconcile("ws1", true, false).unwrap();
        service.reconcile("ws1", false, true).unwrap();
        let generation = service.index_generation("ws1").unwrap();
        assert!(!published_graph_path(&paths, &service, "ws1", generation).exists(), "{field}");
        assert_eq!(service.graphify_warning("ws1").unwrap().code, "GRAPH_UNAVAILABLE");
    }
}