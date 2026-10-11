use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use tempfile::TempDir;
use workspace_index_mcp::tool_registry::{ApprovalDecision, TOOL_REGISTRY};

fn binary() -> &'static str { env!("CARGO_BIN_EXE_workspace-index") }

fn fake_omp(root: &Path) -> PathBuf {
    let directory = root.join("bin");
    std::fs::create_dir_all(&directory).unwrap();
    let executable = directory.join("omp");
    std::fs::write(
        &executable,
        "#!/bin/sh\n\
         if [ \"$1\" = --profile ]; then\n\
           [ -n \"$2\" ] || exit 9\n\
           shift 2\n\
         fi\n\
         if [ \"$1\" = config ] && [ \"$2\" = get ]; then\n\
           [ \"$OMP_GET_FAIL\" = 1 ] && exit 7\n\
           printf '%s' \"$OMP_APPROVAL_JSON\"\n\
           exit 0\n\
         fi\n\
         if [ \"$1\" = config ] && [ \"$2\" = set ]; then\n\
           printf '%s' \"$*\" > \"$OMP_SET_LOG\"\n\
           exit 0\n\
         fi\n\
         exit 9\n",
    ).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    directory
}

fn approvals() -> serde_json::Value {
    let values = TOOL_REGISTRY.iter().map(|tool| {
        let value = match tool.decision {
            ApprovalDecision::Allow => "allow",
            ApprovalDecision::Prompt => "prompt",
        };
        (tool.omp_name.to_owned(), serde_json::json!(value))
    }).collect::<serde_json::Map<_, _>>();
    serde_json::json!({"value": values})
}

fn run_omp(
    home: &Path,
    bin: &Path,
    args: &[&str],
    approval_json: &str,
    get_fail: bool,
) -> Output {
    Command::new(binary())
        .args(args)
        .env("HOME", home)
        .env("PATH", bin)
        .env("OMP_APPROVAL_JSON", approval_json)
        .env("OMP_GET_FAIL", if get_fail { "1" } else { "0" })
        .env("OMP_SET_LOG", home.join("omp-set.log"))
        .output()
        .unwrap()
}

fn test_home(temp: &TempDir) -> PathBuf {
    let home = temp.path().join("home");
    std::fs::create_dir(&home).unwrap();
    home
}

#[test]
fn omp_config_check_is_static_and_validates_installed_files() {
    let temp = TempDir::new().unwrap();
    let home = test_home(&temp);
    let bin = fake_omp(temp.path());
    let data = temp.path().join("state");
    let args = ["omp-config", "--profile", "dev", "--check", "--data-dir"];
    let missing = Command::new(binary())
        .args(args)
        .arg(&data)
        .env("HOME", &home)
        .env("PATH", &bin)
        .env("OMP_APPROVAL_JSON", approvals().to_string())
        .output()
        .unwrap();
    assert!(!missing.status.success());
    assert!(!data.exists());

    let agent = home.join(".omp/profiles/dev/agent");
    std::fs::create_dir_all(&agent).unwrap();
    let mcp = serde_json::to_string_pretty(&serde_json::json!({
        "mcpServers": {
            "workspace-index": {"command": binary(), "args": ["serve", "--stdio"]}
        },
        "unrelated": {"preserved": true}
    })).unwrap() + "\n";
    std::fs::write(agent.join("mcp.json"), mcp).unwrap();
    std::fs::write(agent.join("config.yml"), workspace_index_mcp::tool_registry::policy_yaml()).unwrap();
    let check = run_omp(&home, &bin, &["omp-config", "--profile", "dev", "--check"], &approvals().to_string(), false);
    assert!(check.status.success(), "{}", String::from_utf8_lossy(&check.stderr));
    assert!(!data.exists());

    let unsafe_profile = run_omp(&home, &bin, &["omp-config", "--profile", "../outside"], "{}", false);
    assert!(!unsafe_profile.status.success());
}


#[test]
fn non_object_mcp_json_and_mcp_servers_fail_without_replacement() {
    for invalid in [&b"[]"[..], &b"null"[..], &b"{\"mcpServers\":[]}"[..]] {
        let temp = TempDir::new().unwrap();
        let home = test_home(&temp);
        let bin = fake_omp(temp.path());
        let agent = home.join(".omp/agent");
        std::fs::create_dir_all(&agent).unwrap();
        let mcp_path = agent.join("mcp.json");
        let config_path = agent.join("config.yml");
        let settings = b"tools:\n  approval:\n    unrelated: deny\n";
        std::fs::write(&mcp_path, invalid).unwrap();
        std::fs::write(&config_path, settings).unwrap();

        let output = run_omp(&home, &bin, &["omp-config", "--install", "--yes"], &approvals().to_string(), false);
        assert!(!output.status.success());
        assert_eq!(std::fs::read(&mcp_path).unwrap(), invalid);
        assert_eq!(std::fs::read(&config_path).unwrap(), settings);
    }
}
#[test]
fn malformed_mcp_json_fails_without_mutating_original_files() {
    let temp = TempDir::new().unwrap();
    let home = test_home(&temp);
    let bin = fake_omp(temp.path());
    let agent = home.join(".omp/agent");
    std::fs::create_dir_all(&agent).unwrap();
    let mcp_path = agent.join("mcp.json");
    let config_path = agent.join("config.yml");
    let malformed = b"{\"mcpServers\":";
    let settings = b"tools:\n  approval:\n    unrelated: deny\n";
    std::fs::write(&mcp_path, malformed).unwrap();
    std::fs::write(&config_path, settings).unwrap();

    let output = run_omp(&home, &bin, &["omp-config", "--install", "--yes"], &approvals().to_string(), false);
    assert!(!output.status.success());
    assert_eq!(std::fs::read(mcp_path).unwrap(), malformed);
    assert_eq!(std::fs::read(config_path).unwrap(), settings);
}

#[test]
fn failed_or_malformed_approval_read_leaves_mcp_and_settings_unchanged() {
    for (approval_json, get_fail) in [("{\"value\":[]}", false), ("", true)] {
        let temp = TempDir::new().unwrap();
        let home = test_home(&temp);
        let bin = fake_omp(temp.path());
        let agent = home.join(".omp/agent");
        std::fs::create_dir_all(&agent).unwrap();
        let mcp_path = agent.join("mcp.json");
        let config_path = agent.join("config.yml");
        let mcp_before = b"{\"mcpServers\":{\"other\":{\"command\":\"keep\"}}}\n";
        let settings_before = b"tools:\n  approval:\n    unrelated: deny\n";
        std::fs::write(&mcp_path, mcp_before).unwrap();
        std::fs::write(&config_path, settings_before).unwrap();

        let output = run_omp(&home, &bin, &["omp-config", "--install", "--yes"], approval_json, get_fail);
        assert!(!output.status.success());
        assert_eq!(std::fs::read(mcp_path).unwrap(), mcp_before);
        assert_eq!(std::fs::read(config_path).unwrap(), settings_before);
    }
}

#[test]
fn omp_install_preserves_unrelated_servers_and_approval_rules() {
    let temp = TempDir::new().unwrap();
    let home = test_home(&temp);
    let bin = fake_omp(temp.path());
    let agent = home.join(".omp/agent");
    std::fs::create_dir_all(&agent).unwrap();
    let mcp_path = agent.join("mcp.json");
    let config_path = agent.join("config.yml");
    std::fs::write(&mcp_path, "{\"mcpServers\":{\"other\":{\"command\":\"keep\"}},\"unrelated\":{\"preserved\":true}}\n").unwrap();
    std::fs::write(&config_path, "tools:\n  approval:\n    unrelated: deny\n").unwrap();

    let current_approvals = "{\"value\":{\"unrelated\":\"deny\"}}";
    let install = run_omp(&home, &bin, &["omp-config", "--install", "--yes"], current_approvals, false);
    assert!(install.status.success(), "{}", String::from_utf8_lossy(&install.stderr));
    let mcp: serde_json::Value = serde_json::from_slice(&std::fs::read(&mcp_path).unwrap()).unwrap();
    assert_eq!(
        mcp["mcpServers"]["workspace-index"],
        serde_json::json!({"command": binary(), "args": ["serve", "--stdio"]})
    );
    assert_eq!(mcp["mcpServers"]["other"]["command"], "keep");
    assert_eq!(mcp["unrelated"]["preserved"], true);
    assert_eq!(
        std::fs::read(mcp_path.with_extension("json.bak")).unwrap(),
        b"{\"mcpServers\":{\"other\":{\"command\":\"keep\"}},\"unrelated\":{\"preserved\":true}}\n"
    );
    assert!(config_path.with_extension("yml.bak").exists());
    let set_args = std::fs::read_to_string(home.join("omp-set.log")).unwrap();
    assert!(set_args.contains("unrelated"));
    assert!(set_args.contains("deny"));
}

#[test]
fn backup_collision_fails_without_overwriting_backup_or_config() {
    let temp = TempDir::new().unwrap();
    let home = test_home(&temp);
    let bin = fake_omp(temp.path());
    let agent = home.join(".omp/agent");
    std::fs::create_dir_all(&agent).unwrap();
    let mcp_path = agent.join("mcp.json");
    let backup_path = mcp_path.with_extension("json.bak");
    let config_path = agent.join("config.yml");
    let mcp_before = b"{\"mcpServers\":{\"other\":{}}}\n";
    let backup_before = b"prior backup\n";
    let config_before = b"tools:\n  approval:\n    unrelated: deny\n";
    std::fs::write(&mcp_path, mcp_before).unwrap();
    std::fs::write(&backup_path, backup_before).unwrap();
    std::fs::write(&config_path, config_before).unwrap();

    let output = run_omp(&home, &bin, &["omp-config", "--install", "--yes"], "{\"value\":{\"unrelated\":\"deny\"}}", false);
    assert!(!output.status.success());
    assert_eq!(std::fs::read(mcp_path).unwrap(), mcp_before);
    assert_eq!(std::fs::read(backup_path).unwrap(), backup_before);
    assert_eq!(std::fs::read(config_path).unwrap(), config_before);
}

#[cfg(unix)]
#[test]
fn omp_config_files_use_restrictive_mode_and_preserve_existing_mode() {
    use std::os::unix::fs::PermissionsExt;

    let temp = TempDir::new().unwrap();
    let home = test_home(&temp);
    let bin = fake_omp(temp.path());
    let agent = home.join(".omp/agent");
    std::fs::create_dir_all(&agent).unwrap();
    let mcp_path = agent.join("mcp.json");
    let install = run_omp(&home, &bin, &["omp-config", "--install", "--yes"], &approvals().to_string(), false);
    assert!(install.status.success(), "{}", String::from_utf8_lossy(&install.stderr));
    assert_eq!(std::fs::metadata(&mcp_path).unwrap().permissions().mode() & 0o777, 0o600);

    std::fs::set_permissions(&mcp_path, std::fs::Permissions::from_mode(0o640)).unwrap();
    let second = run_omp(&home, &bin, &["omp-config", "--install", "--yes"], &approvals().to_string(), false);
    assert!(second.status.success(), "{}", String::from_utf8_lossy(&second.stderr));
    assert_eq!(std::fs::metadata(&mcp_path).unwrap().permissions().mode() & 0o777, 0o640);
}

#[cfg(unix)]
#[test]
fn profile_symlink_is_rejected_before_config_is_written() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().unwrap();
    let home = test_home(&temp);
    let bin = fake_omp(temp.path());
    let omp = home.join(".omp");
    std::fs::create_dir(&omp).unwrap();
    let profiles = omp.join("profiles");
    std::fs::create_dir(&profiles).unwrap();
    let outside = temp.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    symlink(&outside, profiles.join("dev")).unwrap();

    let output = run_omp(&home, &bin, &["omp-config", "--profile", "dev", "--install", "--yes"], &approvals().to_string(), false);
    assert!(!output.status.success());
    assert!(!outside.join("agent/mcp.json").exists());
}

#[test]
fn unknown_command_flags_fail_clearly() {
    let temp = TempDir::new().unwrap();
    let home = test_home(&temp);
    let bin = fake_omp(temp.path());
    for args in [
        vec!["index", "--unknown"],
        vec!["read", "source.rs", "--unknown"],
        vec!["workspace", "list", "--unknown"],
        vec!["omp-config", "--unknown"],
    ] {
        let output = run_omp(&home, &bin, &args, "{}", false);
        assert!(!output.status.success(), "accepted arguments: {args:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("unknown option"));
    }
}

#[test]
fn noninteractive_workspace_mutation_requires_yes_and_rejects_unsafe_write_mode() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("repo");
    std::fs::create_dir(&root).unwrap();
    let git = Command::new("git").args(["init", "--initial-branch=main"]).arg(&root).output().unwrap();
    assert!(git.status.success());
    let data = temp.path().join("state");
    let denied = Command::new(binary()).args(["workspace", "add", "--id", "repo", "--root"]).arg(&root).arg("--data-dir").arg(&data).stdin(Stdio::null()).output().unwrap();
    assert!(!denied.status.success());
    assert!(String::from_utf8_lossy(&denied.stderr).contains("requires --yes"));

    let added = Command::new(binary()).args(["workspace", "add", "--id", "repo", "--root"]).arg(&root).arg("--data-dir").arg(&data).arg("--yes").output().unwrap();
    assert!(added.status.success(), "{}", String::from_utf8_lossy(&added.stderr));
    let listed = Command::new(binary()).arg("workspace").arg("list").arg("--data-dir").arg(&data).output().unwrap();
    assert!(listed.status.success());
    assert!(String::from_utf8_lossy(&listed.stdout).contains("repo\tread_only"));

    let mode = Command::new(binary()).args(["workspace", "set-mode", "--id", "repo", "--mode", "read_write", "--yes"]).arg("--data-dir").arg(&data).output().unwrap();
    assert!(!mode.status.success());
    let still_read_only = Command::new(binary()).arg("workspace").arg("list").arg("--data-dir").arg(&data).output().unwrap();
    assert!(still_read_only.status.success());
    assert!(String::from_utf8_lossy(&still_read_only.stdout).contains("repo\tread_only"));
    let removed = Command::new(binary()).args(["workspace", "remove", "--id", "repo", "--yes"]).arg("--data-dir").arg(&data).output().unwrap();
    assert!(removed.status.success());
}
