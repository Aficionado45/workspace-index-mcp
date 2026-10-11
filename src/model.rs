use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::de::Deserializer;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceMode { ReadOnly, ReadWrite }

#[derive(Clone, Copy, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexState { Unindexed, Reconciling, Ready, Degraded }

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct Warning { pub code: String, pub message: String, pub retryable: bool }

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct Meta { pub request_id: String }

#[derive(Clone, Debug, PartialEq)]
pub enum ToolResult<T> {
    Success {
        data: T,
        warnings: Vec<Warning>,
        meta: Meta,
    },
    Failure {
        error: ToolError,
        meta: Meta,
    },
}


#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
pub struct ToolError { pub code: ErrorCode, pub message: String, pub retryable: bool, pub details: Value }

#[derive(Clone, Copy, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode { InvalidArgument, WorkspaceRequired, WorkspaceNotFound, ReadOnly, PathEscape, SymlinkRefused, Conflict, Expired, Busy, StaleCursor, EmbeddingUnavailable, Unsupported }

impl Meta {
    pub fn new() -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}
impl Default for Meta {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ToolResult<T> {
    pub fn success(data: T) -> Self {
        Self::Success {
            data,
            warnings: Vec::new(),
            meta: Meta::new(),
        }
    }

    pub fn failure(code: ErrorCode, message: impl Into<String>, retryable: bool) -> Self {
        Self::Failure {
            error: ToolError {
                code,
                message: message.into(),
                retryable,
                details: Value::Null,
            },
            meta: Meta::new(),
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    pub fn data(&self) -> Option<&T> {
        match self {
            Self::Success { data, .. } => Some(data),
            Self::Failure { .. } => None,
        }
    }

    pub fn into_data(self) -> Option<T> {
        match self {
            Self::Success { data, .. } => Some(data),
            Self::Failure { .. } => None,
        }
    }

    pub fn warnings(&self) -> &[Warning] {
        match self {
            Self::Success { warnings, .. } => warnings,
            Self::Failure { .. } => &[],
        }
    }

    pub fn with_warnings(self, new_warnings: Vec<Warning>) -> Self {
        match self {
            Self::Success { data, meta, .. } => Self::Success {
                data,
                warnings: new_warnings,
                meta,
            },
            Self::Failure { error, meta } => Self::Failure { error, meta },
        }
    }

    pub fn with_warning(self, warning: Warning) -> Self {
        match self {
            Self::Success {
                data,
                mut warnings,
                meta,
            } => {
                warnings.push(warning);
                Self::Success {
                    data,
                    warnings,
                    meta,
                }
            }
            Self::Failure { error, meta } => Self::Failure { error, meta },
        }
    }

    pub fn error(&self) -> Option<&ToolError> {
        match self {
            Self::Success { .. } => None,
            Self::Failure { error, .. } => Some(error),
        }
    }
    pub fn into_error(self) -> Option<ToolError> {
        match self {
            Self::Success { .. } => None,
            Self::Failure { error, .. } => Some(error),
        }
    }


    pub fn meta(&self) -> &Meta {
        match self {
            Self::Success { meta, .. } | Self::Failure { meta, .. } => meta,
        }
    }
}

impl<T: Serialize> Serialize for ToolResult<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("ToolResult", 4)?;
        match self {
            Self::Success {
                data,
                warnings,
                meta,
            } => {
                state.serialize_field("ok", &true)?;
                state.serialize_field("data", data)?;
                state.serialize_field("warnings", warnings)?;
                state.serialize_field("meta", meta)?;
            }
            Self::Failure { error, meta } => {
                state.serialize_field("ok", &false)?;
                state.serialize_field("warnings", &[] as &[Warning])?;
                state.serialize_field("error", error)?;
                state.serialize_field("meta", meta)?;
            }
        }
        state.end()
    }
}

#[derive(Default)]
enum Presence<T> {
    #[default]
    Missing,
    Present(T),
}


fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Presence<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Presence::Present)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, bound(deserialize = "T: Deserialize<'de>"))]
struct RawToolResult<T> {
    ok: bool,
    #[serde(default = "Presence::default", deserialize_with = "deserialize_present")]
    data: Presence<T>,
    warnings: Vec<Warning>,
    #[serde(default = "Presence::default", deserialize_with = "deserialize_present")]
    error: Presence<ToolError>,
    meta: Meta,
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ToolResult<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawToolResult::<T>::deserialize(deserializer)?;
        match (raw.ok, raw.data, raw.warnings, raw.error) {
            (true, Presence::Present(data), warnings, Presence::Missing) => Ok(Self::Success {
                data,
                warnings,
                meta: raw.meta,
            }),
            (false, Presence::Missing, warnings, Presence::Present(error)) if warnings.is_empty() => {
                Ok(Self::Failure {
                    error,
                    meta: raw.meta,
                })
            }
            _ => Err(<D::Error as serde::de::Error>::custom(
                "invalid ToolResult shape for success or failure",
            )),
        }
    }
}

impl<T: JsonSchema> JsonSchema for ToolResult<T> {
    fn schema_name() -> Cow<'static, str> {
        format!("ToolResult_{}", T::schema_name()).into()
    }

    fn schema_id() -> Cow<'static, str> {
        format!("{}<{}>", module_path!(), T::schema_id()).into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "ok": { "const": true },
                        "data": generator.subschema_for::<T>(),
                        "warnings": {
                            "type": "array",
                            "items": generator.subschema_for::<Warning>()
                        },
                        "meta": generator.subschema_for::<Meta>()
                    },
                    "required": ["ok", "data", "warnings", "meta"],
                    "additionalProperties": false
                },
                {
                    "type": "object",
                    "properties": {
                        "ok": { "const": false },
                        "warnings": { "type": "array", "maxItems": 0 },
                        "error": generator.subschema_for::<ToolError>(),
                        "meta": generator.subschema_for::<Meta>()
                    },
                    "required": ["ok", "warnings", "error", "meta"],
                    "additionalProperties": false
                }
            ]
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct Page<T> { pub items: Vec<T>, pub truncated: bool, pub next_cursor: Option<String> }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn result_serialization_preserves_the_boolean_discriminator_and_round_trips() {
        let success = ToolResult::success(Page::<String> {
            items: vec![],
            truncated: false,
            next_cursor: None,
        });
        let value = serde_json::to_value(&success).unwrap();
        assert_eq!(value["ok"], true);
        assert!(value.get("data").is_some());
        assert_eq!(value["warnings"], json!([]));
        assert!(value.get("error").is_none());
        assert_eq!(value["data"]["truncated"], false);
        assert!(value["meta"]["request_id"].is_string());
        let decoded: ToolResult<Page<String>> = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, success);

        let failure: ToolResult<()> = ToolResult::failure(ErrorCode::Busy, "locked", true);
        let value = serde_json::to_value(&failure).unwrap();
        assert_eq!(value["ok"], false);
        assert!(value.get("data").is_none());
        assert_eq!(value["warnings"], json!([]));
        assert_eq!(value["error"]["code"], "BUSY");
        assert!(value["meta"]["request_id"].is_string());
        let decoded: ToolResult<()> = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, failure);
    }

    #[test]
    fn result_deserialization_rejects_invalid_success_and_failure_shapes() {
        let meta = json!({"request_id": "request"});
        let error = json!({
            "code": "BUSY",
            "message": "locked",
            "retryable": true,
            "details": null
        });
        let warning = json!([{
            "code": "WARNING",
            "message": "not allowed on failure",
            "retryable": false
        }]);
        let invalid = [
            json!({"ok": true, "warnings": [], "meta": meta.clone()}),
            json!({"ok": true, "data": null, "warnings": [], "error": error.clone(), "meta": meta.clone()}),
            json!({"ok": false, "data": null, "warnings": [], "error": error.clone(), "meta": meta.clone()}),
            json!({"ok": false, "warnings": warning.clone(), "error": error.clone(), "meta": meta.clone()}),
            json!({"ok": false, "warnings": [], "meta": meta.clone()}),
        ];
        for value in invalid {
            assert!(
                serde_json::from_value::<ToolResult<Value>>(value).is_err(),
                "invalid ToolResult shape was accepted"
            );
        }
    }

    #[test]
    fn result_json_schema_matches_the_closed_wire_variants() {
        let schema = serde_json::to_value(schemars::schema_for!(ToolResult<String>)).unwrap();
        assert_eq!(schema["oneOf"][0]["properties"]["ok"]["const"], true);
        assert_eq!(schema["oneOf"][1]["properties"]["ok"]["const"], false);
        assert_eq!(schema["oneOf"][1]["properties"]["warnings"]["maxItems"], 0);
        assert_eq!(schema["oneOf"][1]["required"], json!(["ok", "warnings", "error", "meta"]));
    }
}
