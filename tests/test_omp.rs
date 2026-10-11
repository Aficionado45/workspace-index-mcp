use std::process::Command;
use tempfile::TempDir;

#[test]
fn omp_config_get_returns_installed_tool_approval_policy() {
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir(&home).unwrap();
    let agent = home.join(".omp/profiles/dev/agent");
    std::fs::create_dir_all(&agent).unwrap();
    let mcp = serde_json::to_string_pretty(&serde_json::json!({
        "mcpServers": {
            "workspace-index": {
                "command": env!("CARGO_BIN_EXE_workspace-index"),
                "args": ["serve", "--stdio"]
            }
        }
    })).unwrap() + "\n";
    std::fs::write(agent.join("mcp.json"), mcp).unwrap();
    std::fs::write(agent.join("config.yml"), workspace_index_mcp::tool_registry::policy_yaml()).unwrap();

    let output = Command::new("omp")
        .env("HOME", &home)
        .args(["--profile", "dev", "config", "get", "tools.approval", "--json"])
        .output()
        .expect("omp config get starts");
    assert!(
        output.status.success(),
        "omp config get failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("omp returns approval JSON");
    assert_eq!(result["key"], "tools.approval");
    assert_eq!(result["value"]["mcp__workspace_index_changeset_apply"], "prompt");
    assert_eq!(result["value"]["mcp__workspace_index_file_read"], "allow");
}
