use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
/// Cache manifest for a vector index snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheManifest {
    pub workspace_id: String,
    pub profile_id: String,
    pub dimension: usize,
    pub metric: String,
    pub generation: u64,
    #[serde(default)]
    pub outbox_count: u64,
    #[serde(default)]
    pub outbox_max_id: u64,
    pub count: u64,
    pub checksum: String,
}

#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("Failed to initialize vector index: {0}")]
    Initialization(String),
    #[error("Failed to add vector: {0}")]
    Add(String),
    #[error("Failed to remove vector: {0}")]
    Remove(String),
    #[error("Failed to search vectors: {0}")]
    Search(String),
    #[error("Failed to save vector index: {0}")]
    Save(String),
    #[error("Failed to load vector index: {0}")]
    Load(String),
    #[error("Manifest mismatch: expected {expected:?} but found {found:?}")]
    ManifestMismatch {
        expected: Box<CacheManifest>,
        found: Box<CacheManifest>,
    },
    #[error("Manifest error: {0}")]
    Manifest(String),
    #[error("Invalid embedding data: {0}")]
    InvalidEmbedding(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = VectorError> = std::result::Result<T, E>;
pub(crate) trait SnapshotDurability {
    fn sync_file(&self, file: &File) -> std::io::Result<()>;
    fn sync_directory(&self, path: &Path) -> std::io::Result<()>;
}

struct FileSystemSnapshotDurability;

impl SnapshotDurability for FileSystemSnapshotDurability {
    fn sync_file(&self, file: &File) -> std::io::Result<()> {
        file.sync_all()
    }

    fn sync_directory(&self, path: &Path) -> std::io::Result<()> {
        File::open(path)?.sync_all()
    }
}

fn validate_vector_byte_length(bytes: &[u8], dimension: usize) -> Result<(), VectorError> {
    let valid_length = dimension
        .checked_mul(std::mem::size_of::<f32>())
        .is_some_and(|length| dimension > 0 && bytes.len() == length);
    if !valid_length {
        return Err(VectorError::InvalidEmbedding(format!(
            "expected exactly {} bytes for dimension {dimension}, got {}",
            dimension.saturating_mul(std::mem::size_of::<f32>()),
            bytes.len()
        )));
    }
    Ok(())
}

pub(crate) fn validate_vector_bytes(bytes: &[u8], dimension: usize) -> Result<(), VectorError> {
    validate_vector_byte_length(bytes, dimension)?;
    for chunk in bytes.as_chunks::<{ std::mem::size_of::<f32>() }>().0 {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(VectorError::InvalidEmbedding(
                "values must all be finite".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn decode_vector_bytes(bytes: &[u8], dimension: usize) -> Result<Vec<f32>> {
    validate_vector_byte_length(bytes, dimension)?;
    let vector: Vec<f32> = bytes
        .as_chunks::<{ std::mem::size_of::<f32>() }>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(VectorError::InvalidEmbedding(
            "values must all be finite".to_owned(),
        ));
    }
    Ok(vector)
}

/// Adapter for the usearch vector index.
pub struct VectorIndex {
    index: Index,
    pub manifest: CacheManifest,
}

impl VectorIndex {
    /// Creates a new VectorIndex.
    pub fn new(manifest: CacheManifest) -> Result<Self> {
        if manifest.dimension == 0 {
            return Err(VectorError::Initialization(
                "Vector dimension must be positive".to_owned(),
            ));
        }
        let metric = match manifest.metric.as_str() {
            "cosine" => MetricKind::Cos,
            "l2" | "l2sq" => MetricKind::L2sq,
            "ip" => MetricKind::IP,
            _ => return Err(VectorError::Initialization(format!("Unsupported metric: {}", manifest.metric))),
        };

        let options = IndexOptions {
            dimensions: manifest.dimension,
            metric,
            quantization: ScalarKind::F32,
            ..Default::default()
        };

        let index = Index::new(&options)
            .map_err(|e| VectorError::Initialization(e.to_string()))?;

        Ok(Self { index, manifest })
    }

    /// Adds a vector with a given u64 ID.
    pub fn add(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.manifest.dimension {
            return Err(VectorError::Add(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.manifest.dimension,
                vector.len()
            )));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(VectorError::Add("Vector values must be finite".to_owned()));
        }

        if self.index.size() >= self.index.capacity() {
            let next_capacity = self.index.capacity().max(64).saturating_mul(2);
            self.index.reserve(next_capacity)
                .map_err(|e| VectorError::Add(e.to_string()))?;
        }
        self.index.add(id, vector)
            .map_err(|e| VectorError::Add(e.to_string()))
    }

    /// Removes a vector by its u64 ID.
    pub fn remove(&self, id: u64) -> Result<()> {
        self.index.remove(id)
            .map(|_| ())
            .map_err(|e| VectorError::Remove(e.to_string()))
    }

    /// Searches for the nearest neighbors to a query vector.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(u64, f32)>> {
        if query.len() != self.manifest.dimension {
            return Err(VectorError::Search(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.manifest.dimension,
                query.len()
            )));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(VectorError::Search("Query values must be finite".to_owned()));
        }
        let matches = self.index.search(query, limit)
            .map_err(|error| VectorError::Search(error.to_string()))?;
        let results = matches.keys.into_iter().zip(matches.distances).collect();
        Ok(results)
    }

    /// Returns the number of vectors in the index.
    pub fn size(&self) -> usize {
        self.index.size()
    }

    /// Saves index and manifest through synced temporary files and atomic renames.
    pub fn save(&mut self, base_path: &Path) -> Result<()> {
        self.save_with_durability(base_path, &FileSystemSnapshotDurability)
    }

    pub(crate) fn save_with_durability(
        &mut self,
        base_path: &Path,
        durability: &impl SnapshotDurability,
    ) -> Result<()> {
        let parent = base_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let nonce = uuid::Uuid::new_v4();
        let temp_index_path = base_path.with_extension(format!("usearch.tmp.{nonce}"));
        let temp_manifest_path = base_path.with_extension(format!("json.tmp.{nonce}"));
        let index_path = base_path.with_extension("usearch");
        let manifest_path = base_path.with_extension("json");

        let result = (|| {
            self.manifest.count = self.size() as u64;
            let temp_index = temp_index_path
                .to_str()
                .ok_or_else(|| VectorError::Save("snapshot path is not valid UTF-8".to_owned()))?;
            self.index.save(temp_index)
                .map_err(|error| VectorError::Save(error.to_string()))?;

            let mut index_file = File::open(&temp_index_path)?;
            durability.sync_file(&index_file)?;
            index_file.seek(SeekFrom::Start(0))?;
            let mut hasher = blake3::Hasher::new();
            std::io::copy(&mut index_file, &mut hasher)?;
            self.manifest.checksum = hasher.finalize().to_hex().to_string();
            drop(index_file);

            let manifest_json = serde_json::to_vec_pretty(&self.manifest)
                .map_err(|error| VectorError::Manifest(error.to_string()))?;
            let mut manifest_file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_manifest_path)?;
            manifest_file.write_all(&manifest_json)?;
            durability.sync_file(&manifest_file)?;
            drop(manifest_file);

            fs::rename(&temp_index_path, &index_path)?;
            fs::rename(&temp_manifest_path, &manifest_path)?;
            durability.sync_directory(parent)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temp_index_path);
            let _ = fs::remove_file(&temp_manifest_path);
        }
        result
    }

    pub(crate) fn sync_snapshot(
        base_path: &Path,
        durability: &impl SnapshotDurability,
    ) -> Result<()> {
        let parent = base_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        for path in [
            base_path.with_extension("usearch"),
            base_path.with_extension("json"),
        ] {
            let file = File::open(path)?;
            durability.sync_file(&file)?;
        }
        durability.sync_directory(parent)?;
        Ok(())
    }

    /// Loads an index and manifest from the given base path, validating against expected manifest.
    pub fn load(base_path: &Path, expected_manifest: &CacheManifest) -> Result<Self> {
        let manifest_path = base_path.with_extension("json");
        let index_path = base_path.with_extension("usearch");
        
        if !manifest_path.exists() || !index_path.exists() {
            return Err(VectorError::Load("Missing index or manifest file".to_string()));
        }
        
        let manifest_json = fs::read_to_string(&manifest_path)?;
        let manifest: CacheManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| VectorError::Manifest(e.to_string()))?;
            
        if manifest.workspace_id != expected_manifest.workspace_id ||
           manifest.profile_id != expected_manifest.profile_id ||
           manifest.dimension != expected_manifest.dimension ||
           manifest.metric != expected_manifest.metric ||
           manifest.generation != expected_manifest.generation {
            return Err(VectorError::ManifestMismatch {
                expected: Box::new(expected_manifest.clone()),
                found: Box::new(manifest),
            });
        }
        
        // Validate checksum
        let mut hasher = blake3::Hasher::new();
        let mut file = fs::File::open(&index_path)?;
        std::io::copy(&mut file, &mut hasher)?;
        let actual_checksum = hasher.finalize().to_hex().to_string();
        
        if manifest.checksum != actual_checksum {
            return Err(VectorError::Load(format!(
                "Checksum mismatch: expected {}, got {}",
                manifest.checksum, actual_checksum
            )));
        }
        
        if !expected_manifest.checksum.is_empty() && manifest.checksum != expected_manifest.checksum {
            return Err(VectorError::ManifestMismatch {
                expected: Box::new(expected_manifest.clone()),
                found: Box::new(manifest),
            });
        }
        
        let vector_index = Self::new(manifest)?;
        
        let index_path = index_path
            .to_str()
            .ok_or_else(|| VectorError::Load("snapshot path is not valid UTF-8".to_owned()))?;
        vector_index.index.load(index_path)
            .map_err(|error| VectorError::Load(error.to_string()))?;
            
        if vector_index.size() as u64 != vector_index.manifest.count {
            return Err(VectorError::Load(format!(
                "Index size mismatch: expected {}, got {}",
                vector_index.manifest.count,
                vector_index.size()
            )));
        }
        
        Ok(vector_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_manifest() -> CacheManifest {
        CacheManifest {
            workspace_id: "test-workspace".to_string(),
            profile_id: "test-profile".to_string(),
            dimension: 3,
            metric: "cosine".to_string(),
            generation: 1,
            outbox_count: 0,
            outbox_max_id: 0,
            count: 0,
            checksum: "test-checksum".to_string(),
        }
    }

    #[test]
    fn test_vector_index_add_search() {
        let manifest = create_test_manifest();
        let index = VectorIndex::new(manifest).unwrap();
        
        index.add(1, &[1.0, 0.0, 0.0]).unwrap();
        index.add(2, &[0.0, 1.0, 0.0]).unwrap();
        index.add(3, &[0.0, 0.0, 1.0]).unwrap();
        
        assert_eq!(index.size(), 3);
        
        let results = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // ID 1 should be closest
    }

    #[test]
    fn test_vector_index_remove() {
        let manifest = create_test_manifest();
        let index = VectorIndex::new(manifest).unwrap();
        
        index.add(1, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.size(), 1);
        
        index.remove(1).unwrap();
        assert_eq!(index.size(), 0);
    }

    #[test]
    fn test_vector_index_dimension_mismatch() {
        let manifest = create_test_manifest();
        let index = VectorIndex::new(manifest).unwrap();
        
        let result = index.add(1, &[1.0, 0.0]); // 2D instead of 3D
        assert!(matches!(result, Err(VectorError::Add(_))));
        
        let result = index.search(&[1.0, 0.0], 1); // 2D instead of 3D
        assert!(matches!(result, Err(VectorError::Search(_))));
    }

    #[test]
    fn test_vector_index_save_load() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("test_index");
        
        let manifest = create_test_manifest();
        let mut index = VectorIndex::new(manifest.clone()).unwrap();
        
        index.add(1, &[1.0, 0.0, 0.0]).unwrap();
        index.add(2, &[0.0, 1.0, 0.0]).unwrap();
        
        index.save(&base_path).unwrap();
        
        // Load with correct expected manifest
        let expected_manifest = CacheManifest {
            count: 2,
            checksum: String::new(),
            ..manifest.clone()
        };
        let loaded_index = VectorIndex::load(&base_path, &expected_manifest).unwrap();
        
        assert_eq!(loaded_index.size(), 2);
        let results = loaded_index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_vector_index_load_manifest_mismatch() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("test_index");
        
        let manifest = create_test_manifest();
        let mut index = VectorIndex::new(manifest.clone()).unwrap();
        index.add(1, &[1.0, 0.0, 0.0]).unwrap();
        index.save(&base_path).unwrap();
        
        // Load with wrong generation
        let mut wrong_manifest = manifest.clone();
        wrong_manifest.generation = 2;
        wrong_manifest.count = 1;
        
        let result = VectorIndex::load(&base_path, &wrong_manifest);
        assert!(matches!(result, Err(VectorError::ManifestMismatch { .. })));
    }
    #[test]
    fn save_propagates_file_and_directory_sync_failures_and_cleans_temps() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct FailAtSync {
            call_to_fail: usize,
            calls: AtomicUsize,
        }

        impl SnapshotDurability for FailAtSync {
            fn sync_file(&self, file: &File) -> std::io::Result<()> {
                self.fail_if_requested()?;
                file.sync_all()
            }

            fn sync_directory(&self, path: &Path) -> std::io::Result<()> {
                self.fail_if_requested()?;
                File::open(path)?.sync_all()
            }
        }

        impl FailAtSync {
            fn fail_if_requested(&self) -> std::io::Result<()> {
                let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
                if call == self.call_to_fail {
                    return Err(std::io::Error::other("injected sync failure"));
                }
                Ok(())
            }
        }

        for call_to_fail in 1..=3 {
            let dir = tempdir().unwrap();
            let base_path = dir.path().join("test_index");
            let mut index = VectorIndex::new(create_test_manifest()).unwrap();
            index.add(1, &[1.0, 0.0, 0.0]).unwrap();
            let durability = FailAtSync {
                call_to_fail,
                calls: AtomicUsize::new(0),
            };

            assert!(index.save_with_durability(&base_path, &durability).is_err());
            assert!(fs::read_dir(dir.path()).unwrap().all(|entry| {
                !entry.unwrap().file_name().to_string_lossy().contains(".tmp.")
            }));
        }
    }

    #[test]
    fn decodes_only_exact_finite_little_endian_vectors() {
        let bytes = [1.0_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat();
        assert_eq!(decode_vector_bytes(&bytes, 2).unwrap(), vec![1.0, -2.5]);
        assert!(decode_vector_bytes(&[0, 0, 0, 0, 9], 1).is_err());
        assert!(decode_vector_bytes(&[0, 0, 0], 1).is_err());
        assert!(decode_vector_bytes(&f32::NAN.to_le_bytes(), 1).is_err());
        assert!(decode_vector_bytes(&f32::INFINITY.to_le_bytes(), 1).is_err());
    }
}
