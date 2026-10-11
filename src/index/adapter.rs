pub trait LanguageAdapter {
    fn extract_symbols(&self, source: &str) -> Vec<String>;
}

pub trait EmbeddingProvider {
    fn fingerprint(&self) -> String;
    fn dimension(&self) -> usize;
    fn batch_size(&self) -> usize {
        32
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

pub struct UnavailableEmbeddingProvider;

impl EmbeddingProvider for UnavailableEmbeddingProvider {
    fn fingerprint(&self) -> String {
        "unavailable".to_string()
    }
    
    fn dimension(&self) -> usize {
        0
    }
    
    fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Err("Semantic embeddings unavailable".to_string())
    }
}
