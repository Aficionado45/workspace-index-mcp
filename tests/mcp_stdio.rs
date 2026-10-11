use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{ClientHandler, ServiceExt};
use rmcp::model::{
    CallToolRequestParams, ClientConfig, GetPromptRequestParams, GetPromptResponse,
    ProtocolVersion, ReadResourceRequestParams, ResourceContents, ResourceUpdatedNotificationParam,
    SubscribeRequestParams, UnsubscribeRequestParams,
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::process::Stdio;
use std::time::Duration;
use tempfile::TempDir;
use workspace_index_mcp::tool_registry::{ApprovalDecision, TOOL_REGISTRY, policy_yaml};

#[derive(Clone)]
struct LegacyClient;
impl ClientHandler for LegacyClient {
    fn get_info(&self) -> ClientConfig { ClientConfig::default().with_protocol_version(ProtocolVersion::V_2025_11_25) }
}
#[derive(Clone)]
struct SubscriptionClient {
    updates: tokio::sync::mpsc::UnboundedSender<String>,
}

impl ClientHandler for SubscriptionClient {
    fn get_info(&self) -> ClientConfig {
        ClientConfig::default().with_protocol_version(ProtocolVersion::V_2025_11_25)
    }

    async fn on_resource_updated(
        &self,
        params: ResourceUpdatedNotificationParam,
        _context: rmcp::service::NotificationContext<rmcp::RoleClient>,
    ) {
        let _ = self.updates.send(params.uri);
    }
}
async fn await_within<T>(future: impl std::future::Future<Output = T>, label: &str) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .unwrap_or_else(|_| panic!("{label} timed out"))
}

async fn read_resource_json(
    client: &rmcp::service::RunningService<rmcp::RoleClient, SubscriptionClient>,
    uri: &str,
) -> Value {
    let response = await_within(
        client.read_resource(ReadResourceRequestParams::new(uri)),
        "read_resource",
    ).await.unwrap();
    let text = match &response.contents[0] {
        ResourceContents::TextResourceContents { text, .. } => text,
        _ => panic!("expected text resource"),
    };
    serde_json::from_str(text).unwrap()
}

async fn call_tool_json(
    client: &rmcp::service::RunningService<rmcp::RoleClient, SubscriptionClient>,
    name: &'static str,
    arguments: Value,
) -> Value {
    await_within(
        client.call_tool(CallToolRequestParams::new(name)
            .with_arguments(arguments.as_object().unwrap().clone())),
        "call_tool",
    ).await.unwrap().structured_content.unwrap()
}

fn prompt_text(response: GetPromptResponse) -> String {
    let result = match response {
        GetPromptResponse::Complete(result) => result,
        _ => panic!("onboarding prompt unexpectedly requested interactive input"),
    };
    match &result.messages[0].content {
        rmcp::model::ContentBlock::Text(content) => content.text.clone(),
        _ => panic!("onboarding prompt did not return text"),
    }
}

#[tokio::test]
async fn stdio_negotiates_legacy_checks_registry_schemas_and_routes_prompt_arguments() {
    let temp = TempDir::new().unwrap();
    let config = temp.path().join("config.toml");
    let data = temp.path().join("data");
    let command = tokio::process::Command::new(env!("CARGO_BIN_EXE_workspace-index")).configure(|command| {
        command.arg("serve").arg("--stdio").arg("--config").arg(&config).arg("--data-dir").arg(&data).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    });
    let transport = TokioChildProcess::new(command).unwrap();
    let client = LegacyClient.serve(transport).await.unwrap();
    assert_eq!(client.peer_info().unwrap().protocol_version, ProtocolVersion::V_2025_11_25);

    let listed = client.list_tools(None).await.unwrap();
    let actual_names = listed.tools.iter().map(|tool| tool.name.to_string()).collect::<BTreeSet<_>>();
    let expected_names = TOOL_REGISTRY.iter().map(|tool| tool.registered_name.to_owned()).collect::<BTreeSet<_>>();
    assert_eq!(actual_names, expected_names);
    assert_eq!(listed.tools.len(), TOOL_REGISTRY.len());
    let expected_approvals = vec![
        ("workspace_list", "mcp__workspace_index_workspace_list", false, ApprovalDecision::Allow),
        ("workspace_register", "mcp__workspace_index_workspace_register", false, ApprovalDecision::Prompt),
        ("workspace_set_mode", "mcp__workspace_index_workspace_set_mode", false, ApprovalDecision::Prompt),
        ("workspace_unregister", "mcp__workspace_index_workspace_unregister", false, ApprovalDecision::Prompt),
        ("index_status", "mcp__workspace_index_index_status", false, ApprovalDecision::Allow),
        ("index_reconcile", "mcp__workspace_index_index_reconcile", false, ApprovalDecision::Allow),
        ("code_search", "mcp__workspace_index_code_search", false, ApprovalDecision::Allow),
        ("symbol_search", "mcp__workspace_index_symbol_search", false, ApprovalDecision::Allow),
        ("file_read", "mcp__workspace_index_file_read", false, ApprovalDecision::Allow),
        ("changeset_prepare", "mcp__workspace_index_changeset_prepare", false, ApprovalDecision::Allow),
        ("changeset_apply", "mcp__workspace_index_changeset_apply", true, ApprovalDecision::Prompt),
        ("changeset_status", "mcp__workspace_index_changeset_status", false, ApprovalDecision::Allow),
        ("changeset_discard", "mcp__workspace_index_changeset_discard", false, ApprovalDecision::Allow),
    ];
    let actual_approvals = TOOL_REGISTRY.iter()
        .map(|spec| (spec.name, spec.omp_name, spec.source_effect, spec.decision))
        .collect::<Vec<_>>();
    assert_eq!(actual_approvals, expected_approvals);
    let policy = policy_yaml();
    for spec in TOOL_REGISTRY {
        let tool = listed.tools.iter().find(|tool| tool.name == spec.registered_name).unwrap();
        let schema: Value = serde_json::to_value(tool.input_schema.as_ref()).unwrap();
        assert_eq!(schema, (spec.input_schema)(), "input schema for {}", spec.name);
        let decision = match spec.decision {
            ApprovalDecision::Allow => "allow",
            ApprovalDecision::Prompt => "prompt",
        };
        assert!(policy.lines().any(|line| line.trim() == format!("{}: {decision}", spec.omp_name)));
        if spec.source_effect { assert_eq!(spec.decision, ApprovalDecision::Prompt); }
    }
    assert_eq!(TOOL_REGISTRY.iter().filter(|spec| spec.source_effect).count(), 1);

    let resources = client.list_resources(None).await.unwrap();
    assert!(resources.resources.iter().any(|resource| resource.uri == "workspace-index://workspaces"));
    let templates = client.list_resource_templates(None).await.unwrap();
    assert!(templates.resource_templates.iter().any(|resource| resource.uri_template == "workspace-index://workspace/{id}/manifest"));
    let prompts = client.list_prompts(None).await.unwrap();
    let onboarding = prompts.prompts.iter().find(|prompt| prompt.name == "onboard-workspace").unwrap();
    assert!(onboarding.arguments.as_ref().unwrap().iter().any(|argument| {
        argument.name == "workspace_id" && !argument.required.unwrap_or(false)
    }));

    let mut prompt_args = serde_json::Map::new();
    prompt_args.insert("workspace_id".into(), json!("selected-workspace"));
    let explicit = client.get_prompt_once(
        GetPromptRequestParams::new("onboard-workspace").with_arguments(prompt_args),
    ).await.unwrap();
    assert!(prompt_text(explicit).contains("selected-workspace"));
    let omitted = client.get_prompt_once(GetPromptRequestParams::new("onboard-workspace")).await.unwrap();
    assert!(prompt_text(omitted).contains("require the user to select an explicit workspace_id"));

    let response = client.call_tool(CallToolRequestParams::new("workspace_list")).await.unwrap();
    let text = response.content[0].as_text().unwrap().text.clone();
    let text_json: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(response.structured_content, Some(text_json.clone()));
    assert_eq!(text_json["ok"], true);
    assert_eq!(text_json["data"]["truncated"], false);

    client.cancel().await.unwrap();
}

#[tokio::test]
#[allow(deprecated)]
async fn stdio_resource_subscriptions_update_unsubscribe_and_enforce_capacity() {
    let temp = TempDir::new().unwrap();
    let config = temp.path().join("config.toml");
    let data = temp.path().join("data");
    let root = temp.path().join("repo");
    std::fs::write(&config, "allow_non_git_roots = true\n").unwrap();
    std::fs::create_dir(&root).unwrap();
    let command = tokio::process::Command::new(env!("CARGO_BIN_EXE_workspace-index")).configure(|command| {
        command.arg("serve").arg("--stdio").arg("--config").arg(&config).arg("--data-dir").arg(&data).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    });
    let (updates_tx, mut updates_rx) = tokio::sync::mpsc::unbounded_channel();
    let client = tokio::time::timeout(
        Duration::from_secs(10),
        SubscriptionClient { updates: updates_tx }.serve(TokioChildProcess::new(command).unwrap()),
    ).await.expect("client initialize timed out").unwrap();

    let registration = call_tool_json(
        &client,
        "workspace_register",
        json!({"workspace_id":"updates","root":root.to_string_lossy()}),
    ).await;
    assert_eq!(registration["ok"], true);

    let baseline = call_tool_json(
        &client,
        "index_reconcile",
        json!({"workspace_id":"updates","full":true}),
    ).await;
    assert_eq!(baseline["ok"], true);


    let status_uri = "workspace-index://workspace/updates/status";
    let ready_deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let before = loop {
        let status = read_resource_json(&client, status_uri).await;
        let visible = status["visible_generation"].as_u64().unwrap();
        let building = status["building_generation"].as_u64().unwrap();
        if status["dirty"] == false
            && visible > 0
            && building == visible
            && status["queued_jobs"].as_u64() == Some(0)
        {
            break status;
        }
        assert!(tokio::time::Instant::now() < ready_deadline, "initial watcher did not settle");
        tokio::task::yield_now().await;
    };

    let invalid = await_within(client.subscribe(SubscribeRequestParams::new(
        "workspace-index://workspace/updates/unsupported",
    )), "subscribe invalid resource").await;
    assert!(invalid.is_err());
    await_within(
        client.subscribe(SubscribeRequestParams::new(status_uri)),
        "subscribe status",
    ).await.unwrap();


    let changed = call_tool_json(
        &client,
        "index_reconcile",
        json!({"workspace_id":"updates","full":true}),
    ).await;
    assert_eq!(changed["ok"], true);

    let updated_uri = tokio::time::timeout(Duration::from_secs(5), updates_rx.recv())
        .await
        .expect("resource update was not notified")
        .expect("notification channel closed");
    assert_eq!(updated_uri, status_uri);

    let after = read_resource_json(&client, status_uri).await;
    assert!(
        after["visible_generation"].as_u64().unwrap()
            > before["visible_generation"].as_u64().unwrap(),
    );

    await_within(
        client.unsubscribe(UnsubscribeRequestParams::new(status_uri)),
        "unsubscribe status",
    ).await.unwrap();
    while let Ok(uri) = updates_rx.try_recv() {
        assert_eq!(uri, status_uri);
    }

    let final_reconcile = call_tool_json(
        &client,
        "index_reconcile",
        json!({"workspace_id":"updates","full":true}),
    ).await;
    assert_eq!(final_reconcile["ok"], true);

    assert!(
        tokio::time::timeout(Duration::from_millis(1500), updates_rx.recv())
            .await
            .is_err(),
        "unsubscribed status resource emitted a later update",
    );
    await_within(client.cancel(), "client cancel").await.unwrap();
}
