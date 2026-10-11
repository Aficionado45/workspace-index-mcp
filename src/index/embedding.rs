use std::path::PathBuf;
use std::sync::Mutex;
use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};
use crate::index::adapter::EmbeddingProvider;

const FASTEMBED_IMPLEMENTATION_ID: &str = "fastembed-rs-7.1.1/workspace-index-adapter-v2";
const FASTEMBED_MODEL_REVISION: &str =
    "Xenova/bge-small-en-v1.5@ea104dacec62c0de699686887e3f920caeb4f3e3";
const FASTEMBED_MODEL_ARTIFACT: &str = "onnx/model.onnx";

fn fastembed_fingerprint() -> String {
    format!(
        "fastembed:{FASTEMBED_IMPLEMENTATION_ID}:{FASTEMBED_MODEL_REVISION}/{FASTEMBED_MODEL_ARTIFACT}:sha256-unavailable",
    )
}

pub struct LocalEmbedder {
    model: Mutex<TextEmbedding>,
    fingerprint: String,
    dimension: usize,
    batch_size: usize,
}

impl LocalEmbedder {
    /// Creates a new local embedding provider using BAAI/bge-small-en-v1.5.
    /// Downloads the ONNX model and tokenizer to the provided XDG cache directory on first run.
    pub fn new(cache_dir: PathBuf) -> Result<Self, String> {
        Self::with_batch_size(cache_dir, 32)
    }

    pub fn with_batch_size(cache_dir: PathBuf, batch_size: usize) -> Result<Self, String> {
        if batch_size == 0 {
            return Err("Embedding batch size must be greater than zero".to_string());
        }
        let model_type = EmbeddingModel::BGESmallENV15;
        let options = TextInitOptions::new(model_type.clone())
            .with_cache_dir(cache_dir)
            .with_show_download_progress(false);
        let model = TextEmbedding::try_new(options)
            .map_err(|e| format!("Failed to initialize local embedding model: {}", e))?;
        // FastEmbed's model metadata does not expose an authoritative artifact checksum.
        let dimension = 384;
        let fingerprint = fastembed_fingerprint();
        Ok(Self { model: Mutex::new(model), fingerprint, dimension, batch_size })
    }
}

impl EmbeddingProvider for LocalEmbedder {
    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = self.model.lock().map_err(|_| "Embedding model lock poisoned".to_string())?;
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size) {
            vectors.extend(model.embed(batch, None)
                .map_err(|e| format!("Embedding inference failed: {}", e))?);
        }
        Ok(vectors)
    }
}
use crate::config::{
    ConfigError, EmbeddingConfig, EmbeddingProviderKind, MAX_REMOTE_BATCH_SIZE,
};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::time::Duration;
use thiserror::Error;

const MAX_REMOTE_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_REMOTE_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const REMOTE_IMPLEMENTATION_VERSION: &str = "remote-http-v2";

#[derive(Debug, Error)]
pub enum RemoteEmbeddingError {
    #[error(transparent)]
    Configuration(#[from] ConfigError),
    #[error("remote embedding requires allow_remote = true")]
    RemoteDisabled,
    #[error("RemoteEmbedder requires ollama or openai_compatible")]
    UnsupportedProvider,
    #[error("remote endpoint is invalid or disallowed")]
    InvalidEndpoint,
    #[error("configured embedding credential is missing or invalid")]
    CredentialUnavailable,
    #[error("remote embedding request failed")]
    Transport,
    #[error("remote embedding provider returned HTTP status {status}")]
    HttpStatus { status: u16 },
    #[error("remote embedding response exceeded the hard size limit")]
    ResponseTooLarge,
    #[error("remote embedding request exceeds the 8 MiB hard size limit")]
    RequestTooLarge,
    #[error("remote embedding request could not be serialized")]
    RequestSerialization,
    #[error("remote embedding response is invalid")]
    InvalidResponse,
    #[error("embedding batch exceeds the configured limit")]
    BatchTooLarge,
}

pub struct RemoteEmbedder {
    endpoint: String,
    provider: EmbeddingProviderKind,
    model: String,
    api_key: Option<String>,
    fingerprint: String,
    dimension: usize,
    batch_size: usize,
    retry_delays: Vec<u64>,
    agent: ureq::Agent,
}

impl RemoteEmbedder {
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self, RemoteEmbeddingError> {
        let provider = validate_remote_config(config)?;
        let api_key = configured_credential(config)?;
        Self::build_validated(config, provider, api_key)
    }
    /// Returns the validated, normalized endpoint used for remote requests.
    pub fn normalized_endpoint(&self) -> &str {
        &self.endpoint
    }


    fn build_validated(
        config: &EmbeddingConfig,
        provider: EmbeddingProviderKind,
        api_key: Option<String>,
    ) -> Result<Self, RemoteEmbeddingError> {
        let endpoint = normalize_endpoint(
            config.endpoint.as_deref().ok_or(RemoteEmbeddingError::InvalidEndpoint)?,
            provider,
            config.allow_remote,
        )?;
        let dimension = config.supported_dimension()?;
        let model = config.model.trim().to_owned();
        let revision = config.revision.as_deref().unwrap_or_default().trim();
        let fingerprint =
            provider_fingerprint(config, provider, &endpoint, &model, revision, dimension);
        let timeout = Duration::from_secs(config.timeout_seconds);
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .max_redirects(0)
            .http_status_as_error(false)
            .build()
            .into();
        Ok(Self {
            endpoint,
            provider,
            model,
            api_key,
            fingerprint,
            dimension,
            batch_size: config.batch_size,
            retry_delays: config.retry_delays_ms.clone(),
            agent,
        })
    }

    fn embed_batch_checked(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, RemoteEmbeddingError> {
        if texts.len() > self.batch_size || texts.len() > MAX_REMOTE_BATCH_SIZE {
            return Err(RemoteEmbeddingError::BatchTooLarge);
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut retries = 0;
        loop {
            match self.do_request(texts) {
                Err(RemoteEmbeddingError::HttpStatus { status })
                    if retryable_status(status) && retries < self.retry_delays.len() =>
                {
                    std::thread::sleep(Duration::from_millis(self.retry_delays[retries]));
                    retries += 1;
                }
                result => return result,
            }
        }
    }

    fn do_request(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RemoteEmbeddingError> {
        let payload = serde_json::json!({ "model": self.model, "input": texts });
        let body = serde_json::to_vec(&payload)
            .map_err(|_| RemoteEmbeddingError::RequestSerialization)?;
        if body.len() > MAX_REMOTE_REQUEST_BYTES {
            return Err(RemoteEmbeddingError::RequestTooLarge);
        }
        let mut request = self.agent.post(&self.endpoint).header("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", &format!("Bearer {key}"));
        }
        let response = request
            .send(body.as_slice())
            .map_err(|_| RemoteEmbeddingError::Transport)?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(RemoteEmbeddingError::HttpStatus { status });
        }
        let mut response_body = response.into_body();
        let mut reader = response_body
            .as_reader()
            .take((MAX_REMOTE_RESPONSE_BYTES + 1) as u64);
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|_| RemoteEmbeddingError::Transport)?;
        if bytes.len() > MAX_REMOTE_RESPONSE_BYTES {
            return Err(RemoteEmbeddingError::ResponseTooLarge);
        }
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|_| RemoteEmbeddingError::InvalidResponse)?;
        match self.provider {
            EmbeddingProviderKind::Ollama => {
                parse_ollama_vectors(&json, texts.len(), self.dimension)
            }
            EmbeddingProviderKind::OpenAiCompatible => {
                parse_openai_vectors(&json, texts.len(), self.dimension)
            }
            EmbeddingProviderKind::FastEmbed => Err(RemoteEmbeddingError::UnsupportedProvider),
        }
    }
}

impl EmbeddingProvider for RemoteEmbedder {
    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.embed_batch_checked(texts)
            .map_err(|error| error.to_string())
    }
}

fn validate_remote_config(
    config: &EmbeddingConfig,
) -> Result<EmbeddingProviderKind, RemoteEmbeddingError> {
    let provider = config.provider_kind()?;
    if provider == EmbeddingProviderKind::FastEmbed {
        return Err(RemoteEmbeddingError::UnsupportedProvider);
    }
    if !config.allow_remote {
        return Err(RemoteEmbeddingError::RemoteDisabled);
    }
    config.validate()?;
    Ok(provider)
}

fn configured_credential(config: &EmbeddingConfig) -> Result<Option<String>, RemoteEmbeddingError> {
    resolve_credential(config.api_key_env.as_deref(), |name| std::env::var(name))
}

fn resolve_credential(
    name: Option<&str>,
    read_value: impl FnOnce(&str) -> Result<String, std::env::VarError>,
) -> Result<Option<String>, RemoteEmbeddingError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let value = read_value(name).map_err(|_| RemoteEmbeddingError::CredentialUnavailable)?;
    validate_credential(&value).map(Some)
}

fn validate_credential(value: &str) -> Result<String, RemoteEmbeddingError> {
    let value = value.trim();
    if value.is_empty() || !value.is_ascii() || value.chars().any(char::is_control) {
        return Err(RemoteEmbeddingError::CredentialUnavailable);
    }
    Ok(value.to_owned())
}

fn retryable_status(status: u16) -> bool {
    status == 429 || (500..=599).contains(&status)
}

fn parse_ollama_vectors(
    json: &serde_json::Value,
    expected_count: usize,
    dimension: usize,
) -> Result<Vec<Vec<f32>>, RemoteEmbeddingError> {
    let embeddings = json
        .get("embeddings")
        .and_then(serde_json::Value::as_array)
        .ok_or(RemoteEmbeddingError::InvalidResponse)?;
    if embeddings.len() != expected_count {
        return Err(RemoteEmbeddingError::InvalidResponse);
    }
    embeddings
        .iter()
        .map(|value| parse_vector(value, dimension))
        .collect()
}

fn parse_openai_vectors(
    json: &serde_json::Value,
    expected_count: usize,
    dimension: usize,
) -> Result<Vec<Vec<f32>>, RemoteEmbeddingError> {
    let data = json
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or(RemoteEmbeddingError::InvalidResponse)?;
    if data.len() != expected_count {
        return Err(RemoteEmbeddingError::InvalidResponse);
    }
    let has_indices = data.iter().any(|item| item.get("index").is_some());
    if !has_indices {
        return data
            .iter()
            .map(|item| {
                parse_vector(
                    item.get("embedding")
                        .ok_or(RemoteEmbeddingError::InvalidResponse)?,
                    dimension,
                )
            })
            .collect();
    }
    let mut ordered = vec![None; expected_count];
    for item in data {
        let index = item
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|index| *index < expected_count)
            .ok_or(RemoteEmbeddingError::InvalidResponse)?;
        if ordered[index].is_some() {
            return Err(RemoteEmbeddingError::InvalidResponse);
        }
        ordered[index] = Some(
            item.get("embedding")
                .ok_or(RemoteEmbeddingError::InvalidResponse)?,
        );
    }
    ordered
        .into_iter()
        .map(|value| parse_vector(value.ok_or(RemoteEmbeddingError::InvalidResponse)?, dimension))
        .collect()
}

fn parse_vector(
    value: &serde_json::Value,
    dimension: usize,
) -> Result<Vec<f32>, RemoteEmbeddingError> {
    let values = value
        .as_array()
        .filter(|values| values.len() == dimension)
        .ok_or(RemoteEmbeddingError::InvalidResponse)?;
    values
        .iter()
        .map(|value| {
            let number = value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or(RemoteEmbeddingError::InvalidResponse)? as f32;
            number
                .is_finite()
                .then_some(number)
                .ok_or(RemoteEmbeddingError::InvalidResponse)
        })
        .collect()
}

fn provider_fingerprint(
    config: &EmbeddingConfig,
    provider: EmbeddingProviderKind,
    endpoint: &str,
    model: &str,
    revision: &str,
    dimension: usize,
) -> String {
    let mut hasher = Sha256::new();
    let dimension = dimension.to_string();
    for value in [
        REMOTE_IMPLEMENTATION_VERSION,
        provider.as_str(),
        endpoint,
        model,
        revision,
        dimension.as_str(),
        config.metric.as_str(),
        config.query_template_version.as_str(),
        config.document_template_version.as_str(),
    ] {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
    }
    let digest = hasher.finalize();
    let mut fingerprint = String::from("remote-embedding-v2:");
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(fingerprint, "{byte:02x}");
    }
    fingerprint
}

fn normalize_endpoint(
    endpoint: &str,
    provider: EmbeddingProviderKind,
    allow_remote: bool,
) -> Result<String, RemoteEmbeddingError> {
    if !allow_remote {
        return Err(RemoteEmbeddingError::RemoteDisabled);
    }
    let endpoint = endpoint.trim();
    if endpoint.is_empty() || endpoint.len() > 2048 || endpoint.chars().any(char::is_control) {
        return Err(RemoteEmbeddingError::InvalidEndpoint);
    }
    let (scheme, rest) = endpoint
        .split_once("://")
        .ok_or(RemoteEmbeddingError::InvalidEndpoint)?;
    let scheme = scheme.to_ascii_lowercase();
    if !matches!(scheme.as_str(), "http" | "https")
        || rest.contains('?')
        || rest.contains('#')
        || rest.contains('@')
    {
        return Err(RemoteEmbeddingError::InvalidEndpoint);
    }
    let authority_end = rest.find('/').unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let base_path = &rest[authority_end..];
    let (authority, local) = normalize_authority(authority, &scheme)?;
    if scheme == "http" && !(provider == EmbeddingProviderKind::Ollama && local) {
        return Err(RemoteEmbeddingError::InvalidEndpoint);
    }
    let path = normalize_path(base_path)?;
    let route = match provider {
        EmbeddingProviderKind::Ollama => "/api/embed",
        EmbeddingProviderKind::OpenAiCompatible => "/embeddings",
        EmbeddingProviderKind::FastEmbed => return Err(RemoteEmbeddingError::UnsupportedProvider),
    };
    let path = if path.ends_with(route) {
        path
    } else if provider == EmbeddingProviderKind::Ollama && path.ends_with("/api") {
        format!("{path}/embed")
    } else {
        format!("{path}{route}")
    };
    Ok(format!("{scheme}://{authority}{path}"))
}

fn normalize_authority(
    authority: &str,
    scheme: &str,
) -> Result<(String, bool), RemoteEmbeddingError> {
    if authority.is_empty() || authority.contains('@') {
        return Err(RemoteEmbeddingError::InvalidEndpoint);
    }
    let (host, port, local) = if let Some(bracketed) = authority.strip_prefix('[') {
        let (address, suffix) = bracketed
            .split_once(']')
            .ok_or(RemoteEmbeddingError::InvalidEndpoint)?;
        let ip = address
            .parse::<Ipv6Addr>()
            .map_err(|_| RemoteEmbeddingError::InvalidEndpoint)?;
        let port = parse_port(suffix)?;
        (format!("[{ip}]"), port, ip.is_loopback())
    } else {
        if authority.matches(':').count() > 1 {
            return Err(RemoteEmbeddingError::InvalidEndpoint);
        }
        let (host, port) = match authority.split_once(':') {
            Some((host, port)) => (host, Some(parse_port_value(port)?)),
            None => (authority, None),
        };
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        let ip = host.parse::<Ipv4Addr>().ok();
        let local = ip.is_some_and(|address| address.is_loopback()) || host == "localhost";
        if ip.is_none() && !valid_dns_name(&host) {
            return Err(RemoteEmbeddingError::InvalidEndpoint);
        }
        (host, port, local)
    };
    let port = port.filter(|port| !((*port == 80 && scheme == "http") || (*port == 443 && scheme == "https")));
    let authority = port.map_or(host.clone(), |port| format!("{host}:{port}"));
    Ok((authority, local))
}

fn parse_port(suffix: &str) -> Result<Option<u16>, RemoteEmbeddingError> {
    if suffix.is_empty() {
        return Ok(None);
    }
    let port = suffix
        .strip_prefix(':')
        .filter(|port| !port.is_empty())
        .ok_or(RemoteEmbeddingError::InvalidEndpoint)?;
    Ok(Some(parse_port_value(port)?))
}

fn parse_port_value(port: &str) -> Result<u16, RemoteEmbeddingError> {
    let port = port
        .parse::<u16>()
        .map_err(|_| RemoteEmbeddingError::InvalidEndpoint)?;
    if port == 0 {
        return Err(RemoteEmbeddingError::InvalidEndpoint);
    }
    Ok(port)
}

fn valid_dns_name(host: &str) -> bool {
    !host.is_empty()
        && host.len() <= 253
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                && label.as_bytes()[0] != b'-'
                && label.as_bytes()[label.len() - 1] != b'-'
        })
}

fn normalize_path(path: &str) -> Result<String, RemoteEmbeddingError> {
    let mut segments = Vec::new();
    for segment in path.split('/').filter(|segment| !segment.is_empty()) {
        if segment == "." || segment == ".."
            || !segment.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~')
            })
        {
            return Err(RemoteEmbeddingError::InvalidEndpoint);
        }
        segments.push(segment);
    }
    if segments.is_empty() {
        return Ok(String::new());
    }
    Ok(format!("/{}", segments.join("/")))
}

#[cfg(test)]
#[path = "embedding_tests.rs"]
mod provider_tests;

#[cfg(test)]
mod fingerprint_tests {
    use super::fastembed_fingerprint;

    #[test]
    fn fastembed_fingerprint_binds_adapter_model_revision_and_artifact() {
        let fingerprint = fastembed_fingerprint();
        assert!(fingerprint.contains("fastembed-rs-7.1.1/workspace-index-adapter-v2"));
        assert!(fingerprint.contains(
            "Xenova/bge-small-en-v1.5@ea104dacec62c0de699686887e3f920caeb4f3e3"
        ));
        assert!(fingerprint.contains("onnx/model.onnx"));
        assert!(fingerprint.ends_with("sha256-unavailable"));
    }
}
