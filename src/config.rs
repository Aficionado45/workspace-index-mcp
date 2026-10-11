use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("configuration I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid TOML configuration: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("invalid embedding configuration: {0}")]
    Embedding(String),
    #[error("invalid Graphify configuration: {0}")]
    Graphify(String),
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub enum EmbeddingProviderKind {
    #[serde(rename = "fastembed")]
    FastEmbed,
    #[serde(rename = "ollama")]
    Ollama,
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

impl EmbeddingProviderKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FastEmbed => "fastembed",
            Self::Ollama => "ollama",
            Self::OpenAiCompatible => "openai_compatible",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingMetric {
    #[default]
    Cosine,
    L2,
    InnerProduct,
}

impl EmbeddingMetric {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::L2 => "l2",
            Self::InnerProduct => "inner_product",
        }
    }
}

pub const MAX_REMOTE_BATCH_SIZE: usize = 32;
pub const MAX_EMBEDDING_DIMENSION: usize = 16_384;
pub const MAX_EMBEDDING_TIMEOUT_SECONDS: u64 = 30;
pub const MAX_RETRY_DELAYS: usize = 2;
pub const MAX_RETRY_DELAY_MS: u64 = 1_000;
pub const MAX_GRAPHIFY_TIMEOUT_SECONDS: u64 = 120;


#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub allow_non_git_roots: bool,
    pub embedding: EmbeddingConfig,
    pub limits: Limits,
    pub reconciliation_interval_seconds: u64,
    pub graphify: GraphifyConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub allow_remote: bool,
    pub endpoint: Option<String>,
    pub api_key_env: Option<String>,
    pub revision: Option<String>,
    pub dimension: Option<usize>,
    pub metric: EmbeddingMetric,
    pub query_template_version: String,
    pub document_template_version: String,
    pub batch_size: usize,
    pub timeout_seconds: u64,
    pub retry_delays_ms: Vec<u64>,
}

impl EmbeddingConfig {
    pub fn provider_kind(&self) -> Result<EmbeddingProviderKind, ConfigError> {
        match self.provider.as_str() {
            "fastembed" => Ok(EmbeddingProviderKind::FastEmbed),
            "ollama" => Ok(EmbeddingProviderKind::Ollama),
            "openai_compatible" => Ok(EmbeddingProviderKind::OpenAiCompatible),
            _ => Err(ConfigError::Embedding(
                "provider must be fastembed, ollama, or openai_compatible".into(),
            )),
        }
    }

    pub fn supported_dimension(&self) -> Result<usize, ConfigError> {
        let known = match (self.provider_kind()?, self.model.as_str()) {
            (EmbeddingProviderKind::FastEmbed, "BAAI/bge-small-en-v1.5") => Some(384),
            (EmbeddingProviderKind::Ollama, "nomic-embed-text") => Some(768),
            (EmbeddingProviderKind::Ollama, "mxbai-embed-large" | "bge-m3") => Some(1024),
            (EmbeddingProviderKind::Ollama, "all-minilm") => Some(384),
            (EmbeddingProviderKind::Ollama, "embeddinggemma") => Some(768),
            (EmbeddingProviderKind::OpenAiCompatible, "text-embedding-3-small") => Some(1536),
            (EmbeddingProviderKind::OpenAiCompatible, "text-embedding-3-large") => Some(3072),
            (EmbeddingProviderKind::OpenAiCompatible, "text-embedding-ada-002") => Some(1536),
            _ => None,
        };
        let dimension = match (known, self.dimension) {
            (Some(expected), Some(actual)) if expected != actual => {
                return Err(ConfigError::Embedding(
                    "configured dimension does not match the known model dimension".into(),
                ));
            }
            (Some(expected), _) => expected,
            (None, Some(explicit))
                if self.provider_kind()? != EmbeddingProviderKind::FastEmbed =>
            {
                explicit
            }
            (None, Some(_)) => {
                return Err(ConfigError::Embedding(
                    "unsupported fastembed model".into(),
                ));
            }
            (None, None) => {
                return Err(ConfigError::Embedding(
                    "model dimension is unknown; set embedding.dimension explicitly".into(),
                ));
            }
        };
        if dimension == 0 || dimension > MAX_EMBEDDING_DIMENSION {
            return Err(ConfigError::Embedding(
                "dimension must be between 1 and 16384".into(),
            ));
        }
        Ok(dimension)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let provider = self.provider_kind()?;
        self.supported_dimension()?;
        if self.model.trim().is_empty()
            || self.model.len() > 512
            || self.model.chars().any(char::is_control)
            || self.revision.as_ref().is_some_and(|value| {
                value.trim().is_empty() || value.len() > 512 || value.chars().any(char::is_control)
            })
        {
            return Err(ConfigError::Embedding(
                "model or revision is invalid".into(),
            ));
        }
        if self.batch_size == 0 || self.batch_size > MAX_REMOTE_BATCH_SIZE {
            return Err(ConfigError::Embedding(
                "batch_size must be between 1 and 32".into(),
            ));
        }
        if self.timeout_seconds == 0 || self.timeout_seconds > MAX_EMBEDDING_TIMEOUT_SECONDS {
            return Err(ConfigError::Embedding(
                "timeout_seconds must be between 1 and 30".into(),
            ));
        }
        if self.retry_delays_ms.len() > MAX_RETRY_DELAYS
            || self.retry_delays_ms.iter().any(|delay| *delay > MAX_RETRY_DELAY_MS)
        {
            return Err(ConfigError::Embedding(
                "retry_delays_ms must contain at most two delays, each no greater than 1000 ms".into(),
            ));
        }
        if self.query_template_version.trim().is_empty()
            || self.query_template_version.len() > 128
            || self.document_template_version.trim().is_empty()
            || self.document_template_version.len() > 128
        {
            return Err(ConfigError::Embedding(
                "query and document template versions are required".into(),
            ));
        }
        if self.api_key_env.as_ref().is_some_and(|name| {
            name.is_empty() || name.contains('=') || name.chars().any(char::is_control)
        }) {
            return Err(ConfigError::Embedding(
                "api_key_env must be a valid environment variable name".into(),
            ));
        }
        if provider != EmbeddingProviderKind::FastEmbed {
            if !self.allow_remote {
                return Err(ConfigError::Embedding(
                    "remote embedding requires allow_remote = true".into(),
                ));
            }
            if self.endpoint.as_deref().is_none_or(|endpoint| endpoint.trim().is_empty()) {
                return Err(ConfigError::Embedding(
                    "remote embedding endpoint is required".into(),
                ));
            }
        }
        Ok(())
    }
}


#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Limits {
    pub max_file_bytes: u64,
    pub max_results: usize,
    pub max_snippet_bytes: usize,
    pub max_tool_bytes: usize,
    pub changeset_expiry_seconds: u64,
    pub watcher_queue_paths: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct GraphifyConfig {
    pub enabled: bool,
    pub command: String,
    pub timeout_seconds: u64,
}
impl GraphifyConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.command.trim().is_empty()
            || self.command.len() > 4096
            || self.command.chars().any(char::is_control)
        {
            return Err(ConfigError::Graphify("command is invalid".into()));
        }
        if self.timeout_seconds == 0 || self.timeout_seconds > MAX_GRAPHIFY_TIMEOUT_SECONDS {
            return Err(ConfigError::Graphify(
                "timeout_seconds must be between 1 and 120".into(),
            ));
        }
        Ok(())
    }
}



impl Default for Config {
    fn default() -> Self {
        Self {
            allow_non_git_roots: false,
            embedding: EmbeddingConfig::default(),
            limits: Limits::default(),
            reconciliation_interval_seconds: 300,
            graphify: GraphifyConfig::default(),
        }
    }
}
impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "fastembed".into(),
            model: "BAAI/bge-small-en-v1.5".into(),
            allow_remote: false,
            endpoint: None,
            api_key_env: None,
            revision: None,
            dimension: None,
            metric: EmbeddingMetric::Cosine,
            query_template_version: "query-v1".into(),
            document_template_version: "document-v1".into(),
            batch_size: 32,
            timeout_seconds: 30,
            retry_delays_ms: vec![250, 1000],
        }
    }
}
impl Default for Limits {
    fn default() -> Self {
        Self { max_file_bytes: 1_048_576, max_results: 100, max_snippet_bytes: 16_384, max_tool_bytes: 524_288, changeset_expiry_seconds: 600, watcher_queue_paths: 10_000 }
    }
}
impl Default for GraphifyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            command: "graphify".into(),
            timeout_seconds: 120,
        }
    }
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paths { pub config_file: PathBuf, pub data_dir: PathBuf, pub database: PathBuf, pub vectors_dir: PathBuf, pub pending_dir: PathBuf }

impl Paths {
    pub fn resolve(config_override: Option<&Path>, data_override: Option<&Path>) -> Result<Self, ConfigError> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let config_home = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".config"));
        let data_home = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".local/share"));
        let config_file = config_override.map(Path::to_path_buf).unwrap_or_else(|| config_home.join("workspace-index-mcp/config.toml"));
        let data_dir = data_override.map(Path::to_path_buf).unwrap_or_else(|| data_home.join("workspace-index-mcp"));
        Ok(Self { config_file, database: data_dir.join("index.db"), vectors_dir: data_dir.join("vectors"), pending_dir: data_dir.join("pending"), data_dir })
    }

    pub fn create_private_dirs(&self) -> Result<(), ConfigError> {
        create_private_dir(&self.data_dir)?;
        create_private_dir(&self.vectors_dir)?;
        create_private_dir(&self.pending_dir)?;
        Ok(())
    }
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> Result<(), std::io::Error> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true).mode(0o700);
    builder.create(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
}
#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> Result<(), std::io::Error> { std::fs::create_dir_all(path) }

impl Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.embedding.validate()?;
        self.graphify.validate()
    }

    pub fn load(paths: &Paths) -> Result<Self, ConfigError> {
        let config = if paths.config_file.exists() {
            let contents = std::fs::read_to_string(&paths.config_file)?;
            toml::from_str(&contents)?
        } else {
            Self::default()
        };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn overrides_and_defaults_resolve_isolated_paths() {
        let temp = TempDir::new().unwrap();
        let config = temp.path().join("custom.toml");
        let data = temp.path().join("private");
        let paths = Paths::resolve(Some(&config), Some(&data)).unwrap();
        assert_eq!(paths.database, data.join("index.db"));
        assert_eq!(paths.config_file, config);
        assert_eq!(Config::load(&paths).unwrap().limits.max_file_bytes, 1_048_576);
    }

    #[cfg(unix)]
    #[test]
    fn state_directories_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let temp = TempDir::new().unwrap();
        let paths = Paths::resolve(None, Some(&temp.path().join("state"))).unwrap();
        paths.create_private_dirs().unwrap();
        assert_eq!(std::fs::metadata(&paths.data_dir).unwrap().permissions().mode() & 0o777, 0o700);
        assert_eq!(std::fs::metadata(&paths.pending_dir).unwrap().permissions().mode() & 0o777, 0o700);
    }
    
    #[test]
    fn provider_literals_deserialize_and_resolve_to_typed_kinds() {
        let providers = [
            ("fastembed", EmbeddingProviderKind::FastEmbed),
            ("ollama", EmbeddingProviderKind::Ollama),
            ("openai_compatible", EmbeddingProviderKind::OpenAiCompatible),
        ];
        for (literal, expected) in providers {
            let config: EmbeddingConfig =
                toml::from_str(&format!("provider = \"{literal}\"")).unwrap();
            assert_eq!(config.provider_kind().unwrap(), expected);
        }
    }

    #[test]
    fn unknown_provider_and_model_never_receive_a_default_dimension() {
        let mut config = EmbeddingConfig { provider: "unrecognized".into(), ..Default::default() };
        assert!(config.provider_kind().is_err());

        config.provider = "ollama".into();
        config.model = "unknown-model".into();
        assert!(config.supported_dimension().is_err());
        config.dimension = Some(768);
        assert_eq!(config.supported_dimension().unwrap(), 768);
    }

    #[test]
    fn known_models_have_provider_specific_dimensions() {
        let mut config = EmbeddingConfig::default();
        assert_eq!(config.supported_dimension().unwrap(), 384);

        config.provider = "ollama".into();
        config.model = "nomic-embed-text".into();
        assert_eq!(config.supported_dimension().unwrap(), 768);

        config.provider = "openai_compatible".into();
        config.model = "text-embedding-3-small".into();
        assert_eq!(config.supported_dimension().unwrap(), 1536);
    }

    #[test]
    fn remote_limits_and_retry_delays_fail_validation() {
        let mut config = EmbeddingConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.timeout_seconds, 30);
        assert_eq!(config.retry_delays_ms, vec![250, 1000]);
        config.validate().unwrap();

        config.timeout_seconds = 31;
        assert!(config.validate().unwrap_err().to_string().contains("1 and 30"));
        config.timeout_seconds = 30;
        config.retry_delays_ms = vec![250, 1000, 250];
        assert!(config.validate().unwrap_err().to_string().contains("at most two"));
        config.retry_delays_ms = vec![1001];
        assert!(config.validate().unwrap_err().to_string().contains("1000 ms"));
        config.retry_delays_ms = vec![250, 1000];
        config.batch_size = 33;
        assert!(config.validate().unwrap_err().to_string().contains("1 and 32"));
    }

    #[test]
    fn graphify_command_and_timeout_are_bounded() {
        let mut graphify = GraphifyConfig::default();
        assert!(graphify.validate().is_ok());
        graphify.timeout_seconds = 0;
        assert!(graphify.validate().is_err());
        graphify.timeout_seconds = 121;
        assert!(graphify.validate().is_err());
        graphify.timeout_seconds = 120;
        graphify.command = "   ".into();
        assert!(graphify.validate().is_err());
    }

    #[test]
    fn loading_unknown_provider_literals_fails_configuration() {
        let temp = TempDir::new().unwrap();
        let config_file = temp.path().join("config.toml");
        std::fs::write(&config_file, "[embedding]\nprovider = \"unknown\"").unwrap();
        let paths = Paths::resolve(Some(&config_file), Some(&temp.path().join("data"))).unwrap();
        assert!(Config::load(&paths).is_err());
    }
}
