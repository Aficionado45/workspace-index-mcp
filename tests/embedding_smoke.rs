use tempfile::TempDir;
use workspace_index_mcp::index::{EmbeddingProvider, LocalEmbedder};

#[test]
#[ignore = "downloads the default ONNX model on first run"]
fn default_local_model_embeds_queries() {
    let cache = TempDir::new().unwrap();
    let provider = LocalEmbedder::new(cache.path().to_path_buf()).unwrap();
    assert_eq!(provider.dimension(), 384);
    let vectors = provider.embed_batch(&["fn query() { search_workspace(); }".to_owned()]).unwrap();
    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), 384);
    assert!(vectors[0].iter().all(|value| value.is_finite()));
}
