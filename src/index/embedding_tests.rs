use super::*;

#[cfg(test)]
mod remote_tests {
    use super::*;
    use crate::config::{EmbeddingConfig, EmbeddingMetric, EmbeddingProviderKind};
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc::{self, Receiver};
    use std::thread::JoinHandle;
    use std::time::Duration;

    struct MockReply {
        status: u16,
        body: Vec<u8>,
    }

    struct MockServer {
        endpoint: String,
        requests: Receiver<String>,
        thread: JoinHandle<()>,
    }

    impl MockServer {
        fn start(replies: Vec<MockReply>) -> Self {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            listener.set_nonblocking(true).unwrap();
            let address = listener.local_addr().unwrap();
            let (sender, requests) = mpsc::channel();
            let thread = std::thread::spawn(move || {
                for reply in replies {
                    let Some(mut stream) = accept_request(&listener) else { break };
                    let request = read_request(&mut stream);
                    let _ = sender.send(request);
                    let reason = if (200..300).contains(&reply.status) { "OK" } else { "Error" };
                    let headers = format!(
                        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        reply.status, reason, reply.body.len()
                    );
                    let _ = stream.write_all(headers.as_bytes());
                    let _ = stream.write_all(&reply.body);
                }
            });
            Self { endpoint: format!("http://{address}"), requests, thread }
        }

        fn finish(self) -> Vec<String> {
            let _ = self.thread.join();
            self.requests.try_iter().collect()
        }
    }

    fn accept_request(listener: &TcpListener) -> Option<TcpStream> {
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            match listener.accept() {
                Ok((stream, _)) => return Some(stream),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock
                    && std::time::Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => return None,
            }
        }
    }

    fn read_request(stream: &mut TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buffer = [0; 4096];
        let mut content_length = None;
        while let Ok(count) = stream.read(&mut buffer) {
            if count == 0 { break; }
            bytes.extend_from_slice(&buffer[..count]);
            let Some(body_start) = bytes.windows(4).position(|part| part == b"\r\n\r\n") else {
                continue;
            };
            if content_length.is_none() {
                let headers = String::from_utf8_lossy(&bytes[..body_start]);
                content_length = headers.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok()).flatten()
                });
            }
            if bytes.len() >= body_start + 4 + content_length.unwrap_or(0) { break; }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn reply(status: u16, body: &str) -> MockReply {
        MockReply { status, body: body.as_bytes().to_vec() }
    }

    fn test_embedder(
        provider: EmbeddingProviderKind,
        endpoint: String,
        dimension: usize,
        batch_size: usize,
        retry_delays: Vec<u64>,
    ) -> RemoteEmbedder {
        RemoteEmbedder {
            endpoint, provider, model: "test-model".into(), api_key: Some("mock-secret".into()),
            fingerprint: "test".into(), dimension, batch_size, retry_delays,
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(2)))
                .max_redirects(0)
                .http_status_as_error(false)
                .build().into(),
        }
    }

    fn config(provider: &str, endpoint: &str, model: &str) -> EmbeddingConfig {
        EmbeddingConfig {
            provider: provider.into(),
            model: model.into(),
            allow_remote: true,
            endpoint: Some(endpoint.into()),
            ..Default::default()
        }
    }
    #[test]
    fn provider_constructor_rejects_timeout_and_retry_schedule_overrides() {
        let mut settings = config("ollama", "http://localhost:11434", "nomic-embed-text");
        settings.timeout_seconds = 31;
        let timeout_error = RemoteEmbedder::from_config(&settings)
            .err().expect("timeout cap").to_string();
        assert!(timeout_error.contains("timeout_seconds must be between 1 and 30"));

        settings.timeout_seconds = 30;
        settings.retry_delays_ms = vec![250, 1000, 250];
        let retry_error = RemoteEmbedder::from_config(&settings)
            .err().expect("retry schedule cap").to_string();
        assert!(retry_error.contains("at most two delays"));
        assert!(!retry_error.contains("mock-secret"));
    }

    #[test]
    fn oversized_single_query_body_is_rejected_before_http_send() {
        let server = MockServer::start(vec![reply(200, r#"{"embeddings":[[1.0,2.0]]}"#)]);
        let embedder = test_embedder(
            EmbeddingProviderKind::Ollama, format!("{}/api/embed", server.endpoint), 2, 1, vec![],
        );
        let query = "q".repeat(MAX_REMOTE_REQUEST_BYTES);
        assert!(matches!(
            embedder.embed_batch_checked(&[query]),
            Err(RemoteEmbeddingError::RequestTooLarge)
        ));
        assert!(server.finish().is_empty(), "oversized query was sent over HTTP");
    }

    #[test]
    fn oversized_document_batch_body_is_rejected_before_http_send() {
        let server = MockServer::start(vec![reply(200, r#"{"embeddings":[[1.0,2.0]]}"#)]);
        let embedder = test_embedder(
            EmbeddingProviderKind::Ollama, format!("{}/api/embed", server.endpoint), 2, 1, vec![],
        );
        let document = "d".repeat(MAX_REMOTE_REQUEST_BYTES);
        assert!(matches!(
            embedder.embed_batch_checked(&[document]),
            Err(RemoteEmbeddingError::RequestTooLarge)
        ));
        assert!(server.finish().is_empty(), "oversized document batch was sent over HTTP");
    }


    #[test]
    fn remote_provider_reports_configured_batch_limit() {
        let embedder = test_embedder(
            EmbeddingProviderKind::Ollama, "http://127.0.0.1:1/api/embed".into(), 2, 7, vec![],
        );
        assert_eq!(embedder.batch_size(), 7);
        assert!(matches!(embedder.embed_batch_checked(&vec!["text".into(); 8]), Err(RemoteEmbeddingError::BatchTooLarge)));
    }

    #[test]
    fn ollama_uses_api_embed_request_shape_and_validates_vectors() {
        let server = MockServer::start(vec![reply(200, r#"{"embeddings":[[1.0,2.0],[3.0,4.0]]}"#)]);
        let embedder = test_embedder(
            EmbeddingProviderKind::Ollama, format!("{}/api/embed", server.endpoint), 2, 2, vec![],
        );
        let texts = vec!["alpha".into(), "beta".into()];
        assert_eq!(embedder.embed_batch(&texts).unwrap(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let requests = server.finish();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /api/embed HTTP/1.1"));
        let body = requests[0].split_once("\r\n\r\n").unwrap().1;
        let json: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["input"], serde_json::json!(["alpha", "beta"]));
    }

    #[test]
    fn openai_compatible_uses_embeddings_shape_and_orders_vectors() {
        let server = MockServer::start(vec![reply(200, r#"{"data":[{"index":1,"embedding":[3.0,4.0]},{"index":0,"embedding":[1.0,2.0]}]}"#)]);
        let embedder = test_embedder(
            EmbeddingProviderKind::OpenAiCompatible, format!("{}/embeddings", server.endpoint), 2, 2, vec![],
        );
        let vectors = embedder.embed_batch(&["alpha".into(), "beta".into()]).unwrap();
        assert_eq!(vectors, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let requests = server.finish();
        assert!(requests[0].starts_with("POST /embeddings HTTP/1.1"));
    }

    #[test]
    fn invalid_remote_vector_batches_are_rejected() {
        assert!(parse_ollama_vectors(
            &serde_json::json!({"embeddings": [[1.0, 2.0]]}), 2, 2,
        ).is_err());
        assert!(parse_ollama_vectors(
            &serde_json::json!({"embeddings": [[1.0]]}), 1, 2,
        ).is_err());
        assert!(parse_ollama_vectors(
            &serde_json::json!({"embeddings": [[1.0, "NaN"]]}), 1, 2,
        ).is_err());
    }

    #[test]
    fn remote_constructor_enforces_opt_in_and_normalizes_provider_paths() {
        let mut disabled = config("ollama", "http://localhost:11434", "nomic-embed-text");
        disabled.allow_remote = false;
        assert!(RemoteEmbedder::from_config(&disabled).is_err());

        let ollama = RemoteEmbedder::from_config(&config(
            "ollama", "http://localhost:11434", "nomic-embed-text",
        )).unwrap();
        assert_eq!(ollama.normalized_endpoint(), "http://localhost:11434/api/embed");

        let openai = RemoteEmbedder::from_config(&config(
            "openai_compatible", "https://api.example/v1", "text-embedding-3-small",
        )).unwrap();
        assert_eq!(openai.normalized_endpoint(), "https://api.example/v1/embeddings");
        assert_eq!(openai.batch_size(), EmbeddingConfig::default().batch_size);
    }

    #[test]
    fn remote_retries_only_retryable_status_codes() {
        let server = MockServer::start(vec![
            reply(503, "unavailable"),
            reply(200, r#"{"embeddings":[[1.0,2.0]]}"#),
        ]);
        let embedder = test_embedder(
            EmbeddingProviderKind::Ollama, format!("{}/api/embed", server.endpoint), 2, 2,
            vec![0],
        );
        assert_eq!(embedder.embed_batch(&["alpha".into()]).unwrap(), vec![vec![1.0, 2.0]]);
        assert_eq!(server.finish().len(), 2);
    }

    #[test]
    fn provider_config_fixture_retains_metric_defaults() {
        let settings = config("ollama", "http://localhost:11434", "nomic-embed-text");
        assert_eq!(settings.metric, EmbeddingMetric::Cosine);
    }

    #[test]
    fn non_retryable_http_status_is_not_retried() {
        let server = MockServer::start(vec![reply(400, "bad request"), reply(200, "{}")]);
        let embedder = test_embedder(
            EmbeddingProviderKind::Ollama, format!("{}/api/embed", server.endpoint), 2, 2, vec![0],
        );
        assert!(matches!(
            embedder.embed_batch(&["alpha".into()]),
            Err(error) if error.contains("400")
        ));
        assert_eq!(server.finish().len(), 1);
    }

    #[test]
    fn configured_but_missing_credential_fails_without_exposing_value() {
        let result = resolve_credential(Some("WORKSPACE_INDEX_MISSING_TEST_KEY"), |_| {
            Err(std::env::VarError::NotPresent)
        });
        let error = result.unwrap_err().to_string();
        assert!(!error.contains("WORKSPACE_INDEX_MISSING_TEST_KEY"));
    }

    #[test]
    fn oversized_http_response_is_rejected() {
        let oversized_body = vec![b'x'; MAX_REMOTE_RESPONSE_BYTES + 1];
        let server = MockServer::start(vec![MockReply { status: 200, body: oversized_body }]);
        let embedder = test_embedder(
            EmbeddingProviderKind::Ollama, format!("{}/api/embed", server.endpoint), 2, 1, vec![],
        );
        assert!(embedder.embed_batch(&["alpha".into()]).is_err());
        assert_eq!(server.finish().len(), 1);
    }

    #[test]
    fn endpoint_validation_rejects_non_local_http_for_ollama() {
        assert!(RemoteEmbedder::from_config(&config(
            "ollama", "http://example.com:11434", "nomic-embed-text",
        )).is_err());
    }
}
