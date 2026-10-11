use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use workspace_index_mcp::config::{Config, Paths};
use workspace_index_mcp::model::WorkspaceMode;
use workspace_index_mcp::retrieval::search::CodeSearchRequest;
use workspace_index_mcp::service::WorkspaceService;
use workspace_index_mcp::tool_registry::{
    ApprovalDecision, TOOL_REGISTRY, policy_yaml, validate_registry_contract,
};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn main() {
    if let Err(error) = run() {
        eprintln!("workspace-index: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let config_override = take_option(&mut args, "--config")?.map(PathBuf::from);
    let data_override = take_option(&mut args, "--data-dir")?.map(PathBuf::from);
    let command = args.first().cloned().unwrap_or_else(|| "help".into());
    if command == "omp-config" { return omp_config(&args[1..]); }
    if matches!(command.as_str(), "help" | "--help" | "-h") { print_help(); return Ok(()); }
    validate_command_args(&command, args.get(1..).unwrap_or_default())?;

    let paths = Paths::resolve(config_override.as_deref(), data_override.as_deref()).map_err(|error| error.to_string())?;
    let config = Config::load(&paths).map_err(|error| error.to_string())?;
    if command == "serve" {
        return tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(|error| error.to_string())?
            .block_on(workspace_index_mcp::mcp::serve_stdio(paths, config));
    }
    let service = WorkspaceService::open(paths, config).map_err(|error| error.to_string())?;
    match command.as_str() {
        "workspace" => workspace_command(&service, &args[1..]),
        "index" => index_command(&service, &args[1..]),
        "search" => search_command(&service, &args[1..]),
        "read" => read_command(&service, &args[1..]),
        "status" => status_command(&service, &args[1..]),
        "doctor" => doctor_command(&service),
        _ => Err(format!("unknown command: {command}")),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SearchArguments {
    query: String,
    workspace_ids: Vec<String>,
    languages: Vec<String>,
    path_globs: Vec<String>,
    limit: usize,
    cursor: Option<String>,
}

fn parse_search_arguments(args: &[String]) -> Result<SearchArguments, String> {
    let parsed = parse_cli_args(args, &["--workspace", "--lang", "--path", "--limit", "--cursor"], &[])?;
    let query = parsed.positional.join(" ");
    if query.is_empty() { return Err("search requires a query".into()); }
    Ok(SearchArguments {
        query,
        workspace_ids: parsed.values.get("--workspace").cloned().unwrap_or_default(),
        languages: parsed.values.get("--lang").cloned().unwrap_or_default(),
        path_globs: parsed.values.get("--path").cloned().unwrap_or_default(),
        limit: one_value(&parsed, "--limit")?.map(parse_usize).transpose()?.unwrap_or(10),
        cursor: one_value(&parsed, "--cursor")?.map(str::to_owned),
    })
}

#[derive(Default)]
struct ParsedArgs {
    values: HashMap<String, Vec<String>>,
    flags: HashSet<String>,
    positional: Vec<String>,
}

fn parse_cli_args(
    args: &[String],
    value_options: &[&str],
    flag_options: &[&str],
) -> Result<ParsedArgs, String> {
    let mut parsed = ParsedArgs::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if value_options.contains(&arg.as_str()) {
            let value = args.get(index + 1).ok_or_else(|| format!("missing value for {arg}"))?;
            if value.starts_with("--") { return Err(format!("missing value for {arg}")); }
            parsed.values.entry(arg.clone()).or_default().push(value.clone());
            index += 2;
        } else if flag_options.contains(&arg.as_str()) {
            if !parsed.flags.insert(arg.clone()) { return Err(format!("duplicate option: {arg}")); }
            index += 1;
        } else if arg.starts_with("--") {
            return Err(format!("unknown option: {arg}"));
        } else {
            parsed.positional.push(arg.clone());
            index += 1;
        }
    }
    Ok(parsed)
}

fn one_value<'a>(parsed: &'a ParsedArgs, name: &str) -> Result<Option<&'a str>, String> {
    match parsed.values.get(name).map(Vec::as_slice) {
        None => Ok(None),
        Some([value]) => Ok(Some(value)),
        Some(_) => Err(format!("option {name} may be supplied only once")),
    }
}

fn require_value<'a>(parsed: &'a ParsedArgs, name: &str, usage: &str) -> Result<&'a str, String> {
    one_value(parsed, name)?.ok_or_else(|| usage.to_owned())
}

fn validate_no_positionals(parsed: &ParsedArgs, command: &str) -> Result<(), String> {
    if let Some(value) = parsed.positional.first() {
        return Err(format!("unexpected argument for {command}: {value}"));
    }
    Ok(())
}

fn validate_command_args(command: &str, args: &[String]) -> Result<(), String> {
    match command {
        "index" => {
            let parsed = parse_cli_args(args, &["--workspace"], &["--full", "--graphify", "--yes"])?;
            validate_no_positionals(&parsed, "index")?;
            one_value(&parsed, "--workspace")?;
        }
        "search" => { parse_search_arguments(args)?; }
        "read" => {
            let parsed = parse_cli_args(args, &["--workspace", "--start-line", "--end-line", "--expected-sha256"], &[])?;
            if parsed.positional.len() != 1 { return Err("read requires exactly one relative path".into()); }
            one_value(&parsed, "--workspace")?;
            one_value(&parsed, "--start-line")?;
            one_value(&parsed, "--end-line")?;
            one_value(&parsed, "--expected-sha256")?;
        }
        "status" => {
            let parsed = parse_cli_args(args, &["--workspace"], &[])?;
            validate_no_positionals(&parsed, "status")?;
            one_value(&parsed, "--workspace")?;
        }
        "doctor" => validate_no_positionals(&parse_cli_args(args, &[], &[])?, "doctor")?,
        "serve" => {
            let parsed = parse_cli_args(args, &[], &["--stdio"])?;
            validate_no_positionals(&parsed, "serve")?;
            if !parsed.flags.contains("--stdio") { return Err("serve requires --stdio".into()); }
        }
        "workspace" => validate_workspace_args(args)?,
        _ => return Err(format!("unknown command: {command}")),
    }
    Ok(())
}

fn validate_workspace_args(args: &[String]) -> Result<(), String> {
    let action = args.first().map(String::as_str).ok_or("workspace requires add|list|set-mode|remove")?;
    let remaining = &args[1..];
    let (values, flags) = match action {
        "list" => (&[][..], &[][..]),
        "add" => (&["--id", "--root"][..], &["--yes"][..]),
        "set-mode" => (&["--id", "--mode"][..], &["--yes"][..]),
        "remove" => (&["--id"][..], &["--yes"][..]),
        _ => return Err(format!("unknown workspace action: {action}")),
    };
    let parsed = parse_cli_args(remaining, values, flags)?;
    validate_no_positionals(&parsed, "workspace")?;
    match action {
        "add" => { require_value(&parsed, "--id", "workspace add requires --id")?; require_value(&parsed, "--root", "workspace add requires --root")?; }
        "set-mode" => { require_value(&parsed, "--id", "workspace set-mode requires --id")?; require_value(&parsed, "--mode", "workspace set-mode requires --mode")?; }
        "remove" => { require_value(&parsed, "--id", "workspace remove requires --id")?; }
        "list" => {}
        _ => return Err(format!("unknown workspace action: {action}")),
    }
    Ok(())
}

fn workspace_command(service: &WorkspaceService, args: &[String]) -> Result<(), String> {
    let Some(action) = args.first().map(String::as_str) else { return Err("workspace requires add|list|set-mode|remove".into()); };
    match action {
        "list" => {
            let records = service.list_workspaces().map_err(|error| error.to_string())?;
            for record in records { println!("{}\t{}\t{}", record.id, mode_name(record.mode), record.root_path); }
            Ok(())
        }
        "add" => {
            let id = option(args, "--id").ok_or("workspace add requires --id")?;
            let root = option(args, "--root").ok_or("workspace add requires --root")?;
            confirm_if_needed(args, "Register workspace")?;
            let record = service.register_workspace(id, Path::new(root)).map_err(|error| error.to_string())?;
            println!("{}\t{}", record.id, mode_name(record.mode));
            Ok(())
        }
        "set-mode" => {
            let id = option(args, "--id").ok_or("workspace set-mode requires --id")?;
            let mode = parse_mode(option(args, "--mode").ok_or("workspace set-mode requires --mode")?)?;
            confirm_if_needed(args, "Change workspace mode")?;
            let record = service.set_workspace_mode(id, mode).map_err(|error| error.to_string())?;
            println!("{}\t{}\t{}", record.id, mode_name(record.mode), record.mode_epoch);
            Ok(())
        }
        "remove" => {
            let id = option(args, "--id").ok_or("workspace remove requires --id")?;
            confirm_if_needed(args, "Unregister workspace")?;
            service.unregister_workspace(id).map_err(|error| error.to_string())?;
            println!("removed {id}");
            Ok(())
        }
        _ => Err(format!("unknown workspace action: {action}")),
    }
}

fn index_command(service: &WorkspaceService, args: &[String]) -> Result<(), String> {
    confirm_if_needed(args, "Reconcile workspace index")?;
    let id = service.resolve_workspace(option(args, "--workspace")).map_err(|error| error.to_string())?;
    let graphify_requested = args.iter().any(|arg| arg == "--graphify");
    let report = service.reconcile(
        &id,
        args.iter().any(|arg| arg == "--full"),
        graphify_requested,
    ).map_err(|error| error.to_string())?;
    let generation = service.index_generation(&id).map_err(|error| error.to_string())?;
    let mut output = json!({
        "workspace_id": id, "generation": generation, "updated": report.updated,
        "unchanged": report.unchanged, "deleted": report.deleted,
        "semantic_degraded": report.degraded_semantic,
    });
    if graphify_requested
        && let Some(warning) = service.graphify_warning(&id)
    {
        output["warnings"] = json!([warning]);
    }
    println!("{}", serde_json::to_string(&output).map_err(|error| error.to_string())?);
    Ok(())
}

fn search_command(service: &WorkspaceService, args: &[String]) -> Result<(), String> {
    let arguments = parse_search_arguments(args)?;
    let ids = service.validate_workspace_ids(arguments.workspace_ids).map_err(|error| error.to_string())?;
    let result = service.code_search(CodeSearchRequest {
        query: arguments.query, query_vector: None, workspace_ids: ids, profile_id: None,
        languages: arguments.languages, path_globs: arguments.path_globs,
        limit: arguments.limit, cursor: arguments.cursor,
    }).map_err(|error| error.to_string())?;
    println!("{}", serde_json::to_string(&result).map_err(|error| error.to_string())?);
    Ok(())
}

fn read_command(service: &WorkspaceService, args: &[String]) -> Result<(), String> {
    let parsed = parse_cli_args(args, &["--workspace", "--start-line", "--end-line", "--expected-sha256"], &[])?;
    let path = parsed.positional.first().ok_or("read requires a relative path")?;
    let workspace = one_value(&parsed, "--workspace")?;
    let id = service.resolve_workspace(workspace).map_err(|error| error.to_string())?;
    let start = one_value(&parsed, "--start-line")?.map(parse_usize).transpose()?;
    let end = one_value(&parsed, "--end-line")?.map(parse_usize).transpose()?;
    let expected_hash = one_value(&parsed, "--expected-sha256")?;
    let result = service.read_file(&id, Path::new(path), start, end, expected_hash)
        .map_err(|error| error.to_string())?;
    println!("{}", serde_json::to_string(&result).map_err(|error| error.to_string())?);
    Ok(())
}

fn status_command(service: &WorkspaceService, args: &[String]) -> Result<(), String> {
    let records = if let Some(id) = option(args, "--workspace") {
        let resolved = service.resolve_workspace(Some(id)).map_err(|error| error.to_string())?;
        service.list_workspaces().map_err(|error| error.to_string())?.into_iter()
            .filter(|record| record.id == resolved).collect()
    } else { service.list_workspaces().map_err(|error| error.to_string())? };
    let values = records.into_iter().map(|record| {
        let status = service.index_status(&record.id).map_err(|error| error.to_string())?;
        Ok(json!({
            "workspace_id": record.id, "mode": mode_name(record.mode),
            "generation": status.visible_generation, "files": status.file_count,
            "chunks": status.chunk_count, "queued_jobs": status.queued_jobs,
        }))
    }).collect::<Result<Vec<_>, String>>()?;
    println!("{}", serde_json::to_string(&values).map_err(|error| error.to_string())?);
    Ok(())
}

fn doctor_command(service: &WorkspaceService) -> Result<(), String> {
    let report = service.doctor_report().map_err(|error| error.to_string())?;
    println!("{}", serde_json::to_string(&report).map_err(|error| error.to_string())?);
    Ok(())
}


fn omp_config(args: &[String]) -> Result<(), String> {
    let parsed = parse_cli_args(args, &["--profile"], &["--check", "--install", "--yes"])?;
    validate_no_positionals(&parsed, "omp-config")?;
    let profile = one_value(&parsed, "--profile")?;
    if let Some(profile) = profile { validate_profile_name(profile)?; }
    if parsed.flags.contains("--check") && parsed.flags.contains("--install") {
        return Err("--check and --install are mutually exclusive".into());
    }
    if parsed.flags.contains("--yes") && !parsed.flags.contains("--install") {
        return Err("--yes requires --install".into());
    }
    validate_registry_contract()?;
    if TOOL_REGISTRY.iter().any(|tool| tool.source_effect && tool.decision != ApprovalDecision::Prompt) {
        return Err("source-effect policy registry is unsafe".into());
    }
    let current_exe = std::env::current_exe().map_err(|error| format!("failed to get current executable path: {error}"))?;
    let exe_path = current_exe.to_string_lossy().into_owned();
    let expected_mcp_entry = json!({"command": exe_path, "args": ["serve", "--stdio"]});
    let is_check = parsed.flags.contains("--check");
    let is_install = parsed.flags.contains("--install");
    if !is_check && !is_install {
        println!("# MCP entry");
        println!("{}", serde_json::to_string_pretty(&json!({"mcpServers":{"workspace-index":expected_mcp_entry}})).map_err(|error| error.to_string())?);
        println!("# tools.approval");
        println!("{}", policy_yaml());
        return Ok(());
    }
    if is_install { confirm_if_needed(args, "Install omp configuration")?; }

    let home = dirs::home_dir().ok_or("home directory unavailable")?;
    let agent_dir = safe_agent_directory(&home, profile, is_install)?;
    let mcp_path = agent_dir.join("mcp.json");
    let config_path = agent_dir.join("config.yml");
    ensure_agent_directory_unchanged(&home, profile, &agent_dir)?;
    let mut mcp_json = read_mcp_json(&mcp_path)?;
    ensure_agent_directory_unchanged(&home, profile, &agent_dir)?;
    inspect_regular_file(&config_path)?;
    ensure_agent_directory_unchanged(&home, profile, &agent_dir)?;
    let mcp_matches = mcp_json.as_object()
        .and_then(|root| root.get("mcpServers"))
        .and_then(Value::as_object)
        .and_then(|servers| servers.get("workspace-index"))
        == Some(&expected_mcp_entry);
    let current_approvals = read_approval_rules(profile)?;
    let approvals_match = approval_rules_match(&current_approvals);
    if is_check {
        if !mcp_matches || !approvals_match {
            return Err(format!("installed MCP config ({mcp_matches}) or approval manifest ({approvals_match}) differs from generated registry"));
        }
        println!("workspace-index omp config is current");
        return Ok(());
    }

    if !mcp_matches {
        ensure_agent_directory_unchanged(&home, profile, &agent_dir)?;
        backup_file(&mcp_path)?;
    }
    if !approvals_match {
        ensure_agent_directory_unchanged(&home, profile, &agent_dir)?;
        backup_file(&config_path)?;
    }
    if !approvals_match {
        ensure_agent_directory_unchanged(&home, profile, &agent_dir)?;
        let approvals = merged_approval_rules(current_approvals);
        let value = serde_json::to_string(&Value::Object(approvals)).map_err(|error| error.to_string())?;
        let mut command = omp_command(profile);
        command.args(["config", "set", "tools.approval", &value]);
        let output = command.output().map_err(|error| format!("failed to run omp config set: {error}"))?;
        if !output.status.success() {
            return Err(format!("omp config set failed: {}", String::from_utf8_lossy(&output.stderr)));
        }
        println!("Updated tools.approval via omp config");
    }
    if !mcp_matches {
        let root = mcp_json.as_object_mut().ok_or("invalid mcp.json structure: root is not an object")?;
        let servers = root.entry("mcpServers").or_insert_with(|| json!({}))
            .as_object_mut().ok_or("invalid mcp.json structure: mcpServers is not an object")?;
        servers.insert("workspace-index".into(), expected_mcp_entry.clone());
        ensure_agent_directory_unchanged(&home, profile, &agent_dir)?;
        atomic_write_json(&mcp_path, &mcp_json)?;
        println!("Updated {}", mcp_path.display());
    }
    if mcp_matches && approvals_match { println!("Configuration is already up to date."); }
    Ok(())
}

fn read_mcp_json(path: &Path) -> Result<Value, String> {
    let Some(_) = inspect_regular_file(path)? else { return Ok(json!({"mcpServers": {}})); };
    let mut file = open_readonly_nofollow(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid {}: {error}", path.display()))?;
    let root = value.as_object_mut().ok_or("invalid mcp.json structure: root is not an object")?;
    if let Some(servers) = root.get("mcpServers") {
        if !servers.is_object() { return Err("invalid mcp.json structure: mcpServers is not an object".into()); }
    } else {
        root.insert("mcpServers".into(), json!({}));
    }
    Ok(value)
}

fn read_approval_rules(profile: Option<&str>) -> Result<serde_json::Map<String, Value>, String> {
    let mut command = omp_command(profile);
    command.args(["config", "get", "tools.approval", "--json"]);
    let output = command.output().map_err(|error| format!("failed to run omp config get: {error}"))?;
    if !output.status.success() {
        return Err(format!("omp config get tools.approval failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let result: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("omp config get tools.approval returned invalid JSON: {error}"))?;
    result.as_object()
        .and_then(|object| object.get("value"))
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| "omp config get tools.approval returned an unexpected JSON shape".into())
}

fn omp_command(profile: Option<&str>) -> Command {
    let mut command = Command::new("omp");
    if let Some(profile) = profile { command.args(["--profile", profile]); }
    command
}

fn approval_rules_match(approvals: &serde_json::Map<String, Value>) -> bool {
    TOOL_REGISTRY.iter().all(|tool| {
        let expected = match tool.decision {
            ApprovalDecision::Allow => "allow",
            ApprovalDecision::Prompt => "prompt",
        };
        approvals.get(tool.omp_name).and_then(Value::as_str) == Some(expected)
    })
}

fn merged_approval_rules(
    mut approvals: serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    for tool in TOOL_REGISTRY {
        let decision = match tool.decision {
            ApprovalDecision::Allow => "allow",
            ApprovalDecision::Prompt => "prompt",
        };
        approvals.insert(tool.omp_name.to_owned(), json!(decision));
    }
    approvals
}

fn safe_agent_directory(home: &Path, profile: Option<&str>, create: bool) -> Result<PathBuf, String> {
    let home_metadata = fs::symlink_metadata(home)
        .map_err(|error| format!("home directory is unavailable: {error}"))?;
    if home_metadata.file_type().is_symlink() || !home_metadata.is_dir() {
        return Err("home directory must not be a symlink".into());
    }
    let mut components = vec![".omp"];
    if profile.is_some() { components.push("profiles"); }
    if let Some(profile) = profile { components.push(profile); }
    components.push("agent");
    let mut current = home.to_path_buf();
    let mut missing = false;
    for component in components {
        current.push(component);
        if missing { continue; }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(format!("omp config directory is not a safe directory: {}", current.display()));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound && create => {
                create_private_directory(&current).map_err(|error| format!("failed to create omp config directory: {error}"))?;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => missing = true,
            Err(error) => return Err(format!("failed to inspect omp config directory: {error}")),
        }
    }
    Ok(current)
}

fn ensure_agent_directory_unchanged(
    home: &Path,
    profile: Option<&str>,
    expected: &Path,
) -> Result<(), String> {
    let actual = safe_agent_directory(home, profile, false)?;
    let metadata = fs::symlink_metadata(&actual)
        .map_err(|error| format!("omp config directory is unavailable: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() || actual != expected {
        return Err("omp config directory changed during operation".into());
    }
    Ok(())

}
#[cfg(unix)]
fn open_readonly_nofollow(path: &Path) -> io::Result<File> {
    use rustix::fs::{open, Mode, OFlags};
    let descriptor = open(path, OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW, Mode::empty())
        .map_err(|error| io::Error::from_raw_os_error(error.raw_os_error()))?;
    Ok(File::from(descriptor))
}

#[cfg(not(unix))]
fn open_readonly_nofollow(path: &Path) -> io::Result<File> {
    File::open(path)
}

fn create_private_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700).create(path)
    }
    #[cfg(not(unix))]
    {
        fs::create_dir(path)
    }
}

fn inspect_regular_file(path: &Path) -> Result<Option<Metadata>, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(format!("omp config path is not a regular file: {}", path.display()))
        }
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("failed to inspect {}: {error}", path.display())),
    }
}

fn backup_file(path: &Path) -> Result<(), String> {
    let Some(metadata) = inspect_regular_file(path)? else { return Ok(()); };
    let backup = backup_path(path);
    let mut source = open_readonly_nofollow(path)
        .map_err(|error| format!("failed to open {} for backup: {error}", path.display()))?;
    let mode = file_mode(&metadata);
    let mut destination = create_new_file(&backup, mode)
        .map_err(|error| format!("failed to create backup {}: {error}", backup.display()))?;
    let result = (|| {
        io::copy(&mut source, &mut destination)?;
        destination.sync_all()?;
        sync_parent(&backup)
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&backup);
        return Err(format!("failed to back up {}: {error}", path.display()));
    }
    Ok(())
}

fn backup_path(path: &Path) -> PathBuf {
    let extension = path.extension().and_then(|value| value.to_str()).unwrap_or_default();
    path.with_extension(if extension.is_empty() { "bak".into() } else { format!("{extension}.bak") })
}

fn atomic_write_json(path: &Path, value: &Value) -> Result<(), String> {
    let parent = path.parent().ok_or("config path has no parent directory")?;
    ensure_safe_directory(parent).map_err(|error| error.to_string())?;
    let old_metadata = inspect_regular_file(path)?;
    let mode = old_metadata.as_ref().map(file_mode).unwrap_or(0o600);
    let file_name = path.file_name().and_then(|name| name.to_str()).ok_or("config file name is invalid")?;
    let (temporary_path, mut temporary) = unique_temp_file(parent, file_name, mode)
        .map_err(|error| format!("failed to create atomic config file: {error}"))?;
    let result = (|| {
        let mut content = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
        content.push(b'\n');
        temporary.write_all(&content)?;
        temporary.sync_all()?;
        drop(temporary);
        ensure_safe_directory(parent).map_err(io::Error::other)?;
        fs::rename(&temporary_path, path)?;
        sync_parent(path)
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary_path);
        return Err(format!("failed to atomically write {}: {error}", path.display()));
    }
    Ok(())
}

fn unique_temp_file(parent: &Path, file_name: &str, mode: u32) -> io::Result<(PathBuf, File)> {
    for _ in 0..32 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".{file_name}.tmp-{}-{sequence}", std::process::id()));
        match create_new_file(&path, mode) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(io::ErrorKind::AlreadyExists, "could not allocate a unique temporary file"))
}

fn create_new_file(path: &Path, mode: u32) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(mode);
    // create_new uses exclusive creation, so an existing symlink is never followed.
    let file = options.open(path)?;
    #[cfg(unix)]
    if let Err(error) = file.set_permissions(fs::Permissions::from_mode(mode)) {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(error);
    }
    Ok(file)
}

#[cfg(unix)]
fn file_mode(metadata: &Metadata) -> u32 { metadata.permissions().mode() & 0o777 }

#[cfg(not(unix))]
fn file_mode(_metadata: &Metadata) -> u32 { 0o600 }

fn ensure_safe_directory(path: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "config parent is not a safe directory"));
    }
    Ok(())
}

fn sync_parent(path: &Path) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    File::open(parent)?.sync_all()
}

fn confirm_if_needed(args: &[String], action: &str) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--yes") { return Ok(()); }
    if !io::stdin().is_terminal() { return Err(format!("{action} requires --yes when stdin is non-interactive")); }
    eprint!("{action}? [y/N] ");
    io::stderr().flush().map_err(|error| error.to_string())?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer).map_err(|error| error.to_string())?;
    if answer.trim().eq_ignore_ascii_case("y") || answer.trim().eq_ignore_ascii_case("yes") { Ok(()) }
    else { Err("cancelled".into()) }
}

fn parse_mode(value: &str) -> Result<WorkspaceMode, String> {
    match value {
        "read_only" => Ok(WorkspaceMode::ReadOnly),
        "read_write" => Ok(WorkspaceMode::ReadWrite),
        _ => Err("mode must be read_only or read_write".into()),
    }
}

fn mode_name(mode: WorkspaceMode) -> &'static str {
    match mode { WorkspaceMode::ReadOnly => "read_only", WorkspaceMode::ReadWrite => "read_write" }
}

fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|arg| arg == name)
        .and_then(|index| args.get(index + 1)).map(String::as_str)
}

fn take_option(args: &mut Vec<String>, name: &str) -> Result<Option<String>, String> {
    let matches = args.iter().enumerate().filter_map(|(index, arg)| (arg == name).then_some(index)).collect::<Vec<_>>();
    if matches.len() > 1 { return Err(format!("option {name} may be supplied only once")); }
    let Some(index) = matches.first().copied() else { return Ok(None); };
    let value = args.get(index + 1).ok_or_else(|| format!("missing value for {name}"))?;
    if value.starts_with("--") { return Err(format!("missing value for {name}")); }
    let value = value.clone();
    args.drain(index..=index + 1);
    Ok(Some(value))
}

fn parse_usize(value: &str) -> Result<usize, String> {
    value.parse().map_err(|_| format!("invalid integer: {value}"))
}

fn validate_profile_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 64
        || !name.bytes().enumerate().all(|(index, byte)| {
            if index == 0 { byte.is_ascii_lowercase() || byte.is_ascii_digit() }
            else { byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-') }
        })
    {
        return Err("profile name must match [a-z0-9][a-z0-9_-]{0,63}".into());
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        "workspace-index serve --stdio | workspace add --id ID --root PATH --yes | workspace list | workspace set-mode --id ID --mode MODE --yes | workspace remove --id ID --yes | index [--workspace ID] [--full] [--graphify] [--yes] | search QUERY [--workspace ID ...] [--lang LANG ...] [--path GLOB ...] [--limit N] [--cursor TOKEN] | read PATH [--workspace ID] [--start-line N] [--end-line N] [--expected-sha256 HASH] | status [--workspace ID] | doctor | omp-config [--profile NAME] [--check] | omp-config --install --yes [--profile NAME]\nPath filters are relative-path globs; repeated --path filters match when any pattern matches."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_path_options_are_preserved_for_or_filtering() {
        let args = [
            "needle".to_owned(),
            "--path".to_owned(),
            "src/**".to_owned(),
            "--path".to_owned(),
            "tests/**".to_owned(),
            "--path".to_owned(),
            "src/**".to_owned(),
        ];
        let parsed = parse_search_arguments(&args).expect("valid repeated path filters");
        assert_eq!(parsed.path_globs, vec!["src/**", "tests/**", "src/**"]);
    }

    #[test]
    fn command_parsers_reject_unknown_options() {
        let args = ["needle".to_owned(), "--unknown".to_owned()];
        assert!(parse_search_arguments(&args).unwrap_err().contains("unknown option"));
    }
}
