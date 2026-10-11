use crate::changeset::ChangeOperation;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalDecision { Allow, Prompt }

#[derive(Clone, Copy, Debug)]
pub struct ToolSpec {
    pub name: &'static str,
    pub registered_name: &'static str,
    pub omp_name: &'static str,
    pub source_effect: bool,
    pub decision: ApprovalDecision,
    pub input_schema: fn() -> Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WorkspaceIdArgs { pub workspace_id: Option<String> }
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WorkspaceRegisterArgs { pub workspace_id: String, pub root: String }
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetModeArgs { pub workspace_id: String, pub mode: crate::model::WorkspaceMode }
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReconcileArgs { pub workspace_id: Option<String>, pub full: Option<bool>, pub graphify: Option<bool> }
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchArgs {
    pub query: String,
    pub workspace_ids: Option<Vec<String>>,
    pub languages: Option<Vec<String>>,
    pub path_glob: Option<String>,
    pub limit: Option<usize>,
    pub cursor: Option<String>,
}
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SymbolSearchArgs {
    pub query: String,
    pub workspace_ids: Option<Vec<String>>,
    pub direction: Option<String>,
    pub limit: Option<usize>,
    pub cursor: Option<String>,
}
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FileReadArgs {
    pub workspace_id: Option<String>,
    pub path: String,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub expected_sha256: Option<String>,
}
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChangesetPrepareArgs { pub workspace_id: String, pub operations: Vec<ChangeOperation> }
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChangesetApplyArgs { pub operation_id: String, pub digest: String }
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChangesetIdArgs { pub operation_id: String }
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OnboardWorkspaceArgs { pub workspace_id: Option<String> }

fn empty_input_schema() -> Value {
    let schema = rmcp::handler::server::common::schema_for_empty_input();
    serde_json::to_value(schema.as_ref()).expect("empty MCP input schema must serialize")
}
fn input_schema<T: JsonSchema + std::any::Any>() -> Value {
    let schema = rmcp::handler::server::tool::schema_for_input::<T>()
        .expect("registry argument types must produce valid MCP input schemas");
    serde_json::to_value(schema.as_ref()).expect("MCP input schemas must serialize")
}

macro_rules! tool {
    ($name:literal, $omp:literal, $effect:literal, $decision:expr, $args:ty) => {
        ToolSpec {
            name: $name,
            registered_name: $name,
            omp_name: $omp,
            source_effect: $effect,
            decision: $decision,
            input_schema: input_schema::<$args>,
        }
    };
}

pub const TOOL_REGISTRY: &[ToolSpec] = &[
    ToolSpec {
        name: "workspace_list",
        registered_name: "workspace_list",
        omp_name: "mcp__workspace_index_workspace_list",
        source_effect: false,
        decision: ApprovalDecision::Allow,
        input_schema: empty_input_schema,
    },
    tool!("workspace_register", "mcp__workspace_index_workspace_register", false, ApprovalDecision::Prompt, WorkspaceRegisterArgs),
    tool!("workspace_set_mode", "mcp__workspace_index_workspace_set_mode", false, ApprovalDecision::Prompt, SetModeArgs),
    tool!("workspace_unregister", "mcp__workspace_index_workspace_unregister", false, ApprovalDecision::Prompt, WorkspaceIdArgs),
    tool!("index_status", "mcp__workspace_index_index_status", false, ApprovalDecision::Allow, WorkspaceIdArgs),
    tool!("index_reconcile", "mcp__workspace_index_index_reconcile", false, ApprovalDecision::Allow, ReconcileArgs),
    tool!("code_search", "mcp__workspace_index_code_search", false, ApprovalDecision::Allow, SearchArgs),
    tool!("symbol_search", "mcp__workspace_index_symbol_search", false, ApprovalDecision::Allow, SymbolSearchArgs),
    tool!("file_read", "mcp__workspace_index_file_read", false, ApprovalDecision::Allow, FileReadArgs),
    tool!("changeset_prepare", "mcp__workspace_index_changeset_prepare", false, ApprovalDecision::Allow, ChangesetPrepareArgs),
    tool!("changeset_apply", "mcp__workspace_index_changeset_apply", true, ApprovalDecision::Prompt, ChangesetApplyArgs),
    tool!("changeset_status", "mcp__workspace_index_changeset_status", false, ApprovalDecision::Allow, ChangesetIdArgs),
    tool!("changeset_discard", "mcp__workspace_index_changeset_discard", false, ApprovalDecision::Allow, ChangesetApplyArgs),
];

pub fn validate_registry_contract() -> Result<(), String> {
    let mut names = BTreeMap::new();
    let mut omp_names = BTreeMap::new();
    let mut source_effects = 0;
    for spec in TOOL_REGISTRY {
        if names.insert(spec.registered_name, ()).is_some()
            || omp_names.insert(spec.omp_name, ()).is_some()
        {
            return Err("tool registry contains duplicate names".into());
        }
        if (spec.input_schema)().get("type").and_then(Value::as_str) != Some("object") {
            return Err(format!("tool {} has an invalid input schema", spec.name));
        }
        if spec.source_effect {
            source_effects += 1;
            if spec.decision != ApprovalDecision::Prompt {
                return Err(format!("source-effect tool {} is not prompt-gated", spec.name));
            }
        }
    }
    if source_effects != 1 {
        return Err("tool registry must define exactly one source-effect tool".into());
    }
    Ok(())
}

pub fn validate_tool_definitions(tools: &[rmcp::model::Tool]) -> Result<(), String> {
    validate_registry_contract()?;
    if tools.len() != TOOL_REGISTRY.len() {
        return Err("RMCP tool count does not match the approval registry".into());
    }
    let mut actual = BTreeMap::new();
    for tool in tools {
        if actual.insert(tool.name.as_ref(), tool).is_some() {
            return Err(format!("RMCP registered duplicate tool {}", tool.name));
        }
    }
    for spec in TOOL_REGISTRY {
        let tool = actual.remove(spec.registered_name)
            .ok_or_else(|| format!("RMCP omitted registered tool {}", spec.registered_name))?;
        let actual_schema = serde_json::to_value(tool.input_schema.as_ref())
            .map_err(|_| format!("RMCP schema for {} is invalid", spec.name))?;
        if actual_schema != (spec.input_schema)() {
            return Err(format!("RMCP input schema for {} differs from the static registry", spec.name));
        }
    }
    if !actual.is_empty() {
        return Err("RMCP registered tools that are absent from the static registry".into());
    }
    Ok(())
}

pub fn policy_yaml() -> String {
    let mut output = String::from("tools:\n  approval:\n");
    for tool in TOOL_REGISTRY {
        let decision = match tool.decision {
            ApprovalDecision::Allow => "allow",
            ApprovalDecision::Prompt => "prompt",
        };
        output.push_str(&format!("    {}: {}\n", tool.omp_name, decision));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_source_effect_requires_prompt_approval() {
        assert_eq!(
            TOOL_REGISTRY.iter().filter(|tool| tool.source_effect).map(|tool| tool.name).collect::<Vec<_>>(),
            vec!["changeset_apply"],
        );
        assert!(TOOL_REGISTRY.iter().filter(|tool| tool.source_effect).all(|tool| {
            tool.decision == ApprovalDecision::Prompt
        }));
        validate_registry_contract().expect("static registry contract");
    }
}
