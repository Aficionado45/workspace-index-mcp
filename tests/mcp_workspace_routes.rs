use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{ClientHandler, ServiceExt};
use rmcp::model::{CallToolRequestParams, ClientConfig, ProtocolVersion};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use tempfile::TempDir;

#[derive(Clone)]
struct Client;
impl ClientHandler for Client { fn get_info(&self) -> ClientConfig { ClientConfig::default().with_protocol_version(ProtocolVersion::V_2025_11_25) } }

async fn call_json(client: &rmcp::service::RunningService<rmcp::RoleClient, Client>, name: &'static str, args: Value) -> Value {
    let arguments = args.as_object().cloned().unwrap();
    client.call_tool(CallToolRequestParams::new(name).with_arguments(arguments)).await.unwrap().structured_content.unwrap()
}

#[tokio::test]
async fn explicit_workspace_read_and_search_never_cross_route() {
    let temp = TempDir::new().unwrap();
    let data = temp.path().join("data");
    let config = temp.path().join("config.toml");
    std::fs::write(&config, "").unwrap();
    let repo_a = temp.path().join("repo-a");
    let repo_b = temp.path().join("repo-b");
    std::fs::create_dir(&repo_a).unwrap();
    std::fs::create_dir(&repo_b).unwrap();
    std::fs::write(repo_a.join("lib.rs"), "fn unrelated() {}\n").unwrap();
    std::fs::write(repo_b.join("lib.rs"), "fn alpha_function() { println!(\"alpha\"); }\n").unwrap();
    for root in [&repo_a, &repo_b] {
        assert!(Command::new("git").args(["init", "--initial-branch=main"]).arg(root).output().unwrap().status.success());
    }
    for (id, root) in [("alpha", &repo_a), ("beta", &repo_b)] {
        let output = Command::new(env!("CARGO_BIN_EXE_workspace-index")).args(["workspace", "add", "--id", id, "--root"]).arg(root).arg("--yes").arg("--config").arg(&config).arg("--data-dir").arg(&data).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    }

    let command = tokio::process::Command::new(env!("CARGO_BIN_EXE_workspace-index")).configure(|command| {
        command.arg("serve").arg("--stdio").arg("--config").arg(&config).arg("--data-dir").arg(&data).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    });
    let client = Client.serve(TokioChildProcess::new(command).unwrap()).await.unwrap();

    let read = call_json(&client, "file_read", json!({"workspace_id":"beta", "path":"lib.rs", "start_line":1, "end_line":1})).await;
    assert_eq!(read["ok"], true);
    assert!(read["data"]["text"].as_str().unwrap().contains("alpha_function"));
    assert_eq!(read["data"]["total_lines"], 1);
    assert_eq!(read["data"]["sha256"].as_str().unwrap().len(), 64);

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut found = false;
    while tokio::time::Instant::now() < deadline {
        let search = call_json(&client, "code_search", json!({"query":"alpha_function", "workspace_ids":["beta"], "limit":10})).await;
        if search["data"]["items"].as_array().is_some_and(|items| items.iter().any(|item| item["workspace_id"] == "beta" && item["relative_path"] == "lib.rs")) {
            let items = search["data"]["items"].as_array().unwrap();
            assert!(items.iter().all(|item| item["workspace_id"] == "beta"));
            found = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(found, "indexed beta source did not become searchable");
    let unknown = call_json(&client, "file_read", json!({"workspace_id":"missing", "path":"lib.rs"})).await;
    assert_eq!(unknown["ok"], false);
    assert_eq!(unknown["error"]["code"], "WORKSPACE_NOT_FOUND");

    let ambiguous = call_json(&client, "index_status", json!({})).await;
    assert_eq!(ambiguous["ok"], false);
    assert_eq!(ambiguous["error"]["code"], "WORKSPACE_REQUIRED");

    client.cancel().await.unwrap();
}
