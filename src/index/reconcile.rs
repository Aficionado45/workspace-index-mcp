use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use codeindex_tree_sitter_langs::{LanguageRegistry, parse_source as codeindex_parse_source};
use rusqlite::{Connection, Transaction};

use crate::index::adapter::EmbeddingProvider;
use crate::index::chunker::{
    Chunk, SyntaxRegion, PARSER_VERSION, STORED_CHUNKER_VERSION, hash_content, syntax_chunker,
};
use crate::index::pipeline::IndexReport;
use crate::index::scanner::{build_scanner, read_scanned_file};
use crate::index::tree_sitter::{ParseResult, SourceLanguage, SymbolKind, SymbolRecord, parse_source};
use crate::security::WorkspaceFs;
use crate::state::{Result, StateError};

#[derive(Default)]
pub(crate) struct WorkspaceSnapshot {
    files: HashMap<String, ExistingFile>,
}

struct ExistingFile {
    content_hash: String,
    parser_version: i64,
    chunker_version: i64,
    chunk_count: usize,
    embedding_counts: HashMap<String, usize>,
    vector_ids: HashMap<String, Vec<String>>,
}

struct DependencyRecord {
    name: String,
    evidence: &'static str,
    byte_range: Range<usize>,
    line_range: Range<usize>,
}

struct PreparedFile {
    relative_path: String,
    content_hash: String,
    size_bytes: usize,
    language: &'static str,
    symbols: Vec<SymbolRecord>,
    dependencies: Vec<DependencyRecord>,
    chunks: Vec<Chunk>,
    embeddings: HashMap<String, Option<Vec<Vec<f32>>>>,
    old_vector_ids: HashMap<String, Vec<String>>,
}

pub(crate) struct ProfileProvider<'a> {
    pub(crate) profile_id: &'a str,
    pub(crate) provider: &'a dyn EmbeddingProvider,
}

pub(crate) struct WorkspacePreparation<'a> {
    pub(crate) workspace_id: &'a str,
    pub(crate) workspace: &'a WorkspaceFs,
    pub(crate) excluded_data_root: &'a Path,
    pub(crate) profiles: &'a [ProfileProvider<'a>],
    pub(crate) generation: u64,
    pub(crate) max_file_bytes: u64,
    pub(crate) force_all: bool,
    pub(crate) force_paths: &'a [String],
    pub(crate) snapshot: WorkspaceSnapshot,
}
pub(crate) struct WorkspacePlan {
    workspace_id: String,
    profile_ids: Vec<String>,
    generation: u64,
    target_only: bool,
    unchanged_paths: Vec<String>,
    updated_files: Vec<PreparedFile>,
    pub(crate) covered_paths: HashSet<String>,
    pub(crate) degraded_profile_ids: HashSet<String>,
    excluded_files: Vec<(String, HashMap<String, Vec<String>>)>,
    deleted_files: Vec<(String, HashMap<String, Vec<String>>)>,
    pub(crate) report: IndexReport,
}
pub(crate) fn existing_snapshot(
    connection: &Connection,
    workspace_id: &str,
    _profile_id: &str,
) -> Result<WorkspaceSnapshot> {
    existing_snapshot_for_profiles(connection, workspace_id, &[])
}

pub(crate) fn existing_snapshot_for_profiles(
    connection: &Connection,
    workspace_id: &str,
    _profile_ids: &[String],
) -> Result<WorkspaceSnapshot> {
    let mut statement = connection.prepare(
        "SELECT f.relative_path, f.content_hash, f.parser_version, f.chunker_version,
                (SELECT count(*) FROM chunks c
                 WHERE c.workspace_id=f.workspace_id AND c.relative_path=f.relative_path),
                e.profile_id, e.vector_id
         FROM files f
         LEFT JOIN chunk_embeddings e
           ON e.workspace_id=f.workspace_id AND e.relative_path=f.relative_path
         WHERE f.workspace_id=?1 ORDER BY f.relative_path, e.profile_id, e.chunk_index",
    )?;
    let rows = statement.query_map([workspace_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, Option<String>>(6)?,
        ))
    })?;
    let mut files: HashMap<String, ExistingFile> = HashMap::new();
    for row in rows {
        let (path, content_hash, parser_version, chunker_version, chunk_count, profile_id, vector_id) = row?;
        let file = files.entry(path).or_insert_with(|| ExistingFile {
            content_hash,
            parser_version,
            chunker_version,
            chunk_count: chunk_count.max(0) as usize,
            embedding_counts: HashMap::new(),
            vector_ids: HashMap::new(),
        });
        if let (Some(profile_id), Some(vector_id)) = (profile_id, vector_id) {
            *file.embedding_counts.entry(profile_id.clone()).or_default() += 1;
            file.vector_ids.entry(profile_id).or_default().push(vector_id);
        }
    }
    Ok(WorkspaceSnapshot { files })
}



pub(crate) fn prepare_workspace_profiles(
    request: WorkspacePreparation<'_>,
) -> Result<WorkspacePlan> {
    let WorkspacePreparation {
        workspace_id,
        workspace,
        excluded_data_root,
        profiles,
        generation,
        max_file_bytes,
        force_all,
        force_paths,
        snapshot,
    } = request;
    if profiles.is_empty() || profiles.iter().any(|profile| profile.profile_id.is_empty()) {
        return Err(StateError::InvalidVectorOperation);
    }
    let root = workspace.root_path();
    workspace.verify_registered_root()
        .map_err(|error| StateError::Io(std::io::Error::other(error)))?;
    let forced: HashSet<&str> = force_paths.iter().map(String::as_str).collect();
    let mut report = IndexReport::default();
    let mut degraded_profile_ids = HashSet::new();
    let mut seen_files = HashSet::new();
    let mut excluded_paths = HashSet::new();
    let mut unchanged_paths = Vec::new();
    let mut updated_files = Vec::new();
    if !force_all && !force_paths.is_empty() {
        return prepare_targeted_workspace(
            workspace_id,
            workspace,
            profiles,
            generation,
            max_file_bytes,
            force_paths,
            snapshot,
        );
    }
    let parser_registry = LanguageRegistry::new();

    for item in build_scanner(root, excluded_data_root)? {
        let entry = item.map_err(|error| StateError::Io(std::io::Error::other(error.to_string())))?;
        match read_scanned_file(workspace, root, &entry, max_file_bytes)? {
            crate::index::scanner::ScanEntry::Excluded { relative_path, reason: _ } => {
                let relative_path = relative_path.to_str().ok_or_else(|| {
                    StateError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "source path is not UTF-8"))
                })?;
                excluded_paths.insert(relative_path.to_owned());
            }
            crate::index::scanner::ScanEntry::Included(scanned) => {
                let path = scanned.relative_path;
                let relative_path = path.to_str().ok_or_else(|| {
                    StateError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "source path is not UTF-8"))
                })?.to_owned();
                seen_files.insert(relative_path.clone());
                let content = scanned.content;
                let content_hash = hash_content(&content);
                let force = force_all || forced.contains(relative_path.as_str());
                let current = snapshot.files.get(&relative_path);
                let has_profile_coverage = current.is_some_and(|file| profiles.iter().all(|profile| {
                    file.embedding_counts.get(profile.profile_id).copied().unwrap_or_default()
                        == file.chunk_count
                }));
                if !force && has_profile_coverage && current.is_some_and(|file| {
                    file.content_hash == content_hash
                        && file.parser_version == PARSER_VERSION
                        && file.chunker_version == STORED_CHUNKER_VERSION
                }) {
                    unchanged_paths.push(relative_path);
                    report.unchanged += 1;
                    continue;
                }
                let old_vector_ids = current.map(|file| file.vector_ids.clone()).unwrap_or_default();
                let file = prepare_file(FilePreparation {
                    path: &path,
                    relative_path,
                    content,
                    content_hash,
                    old_vector_ids,
                    profiles,
                    parser_registry: &parser_registry,
                    degraded_profile_ids: &mut degraded_profile_ids,
                });
                updated_files.push(file);
                report.updated += 1;
            }
        }
    }
    workspace.verify_registered_root()
        .map_err(|error| StateError::Io(std::io::Error::other(error)))?;
    let mut covered_paths = seen_files.union(&excluded_paths).cloned().collect::<HashSet<_>>();
    let mut excluded_files = Vec::new();
    let mut deleted_files = Vec::new();
    for (path, file) in snapshot.files {
        if seen_files.contains(&path) {
            continue;
        }
        if excluded_paths.contains(&path) {
            covered_paths.insert(path.clone());
            excluded_files.push((path, file.vector_ids));
            continue;
        }
        match workspace.read(Path::new(&path), max_file_bytes) {
            Ok(_) | Err(crate::security::workspace_fs::WorkspaceFsError::TooLarge)
            | Err(crate::security::workspace_fs::WorkspaceFsError::PathEscape)
            | Err(crate::security::workspace_fs::WorkspaceFsError::NotRegularFile) => {
                covered_paths.insert(path.clone());
                excluded_files.push((path, file.vector_ids));
            }
            Err(crate::security::workspace_fs::WorkspaceFsError::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                covered_paths.insert(path.clone());
                deleted_files.push((path, file.vector_ids));
            }
            Err(error) => return Err(StateError::Io(std::io::Error::other(error))),
        }
    }
    report.excluded = excluded_files.len();
    report.deleted = deleted_files.len();
    let degraded_semantic = !degraded_profile_ids.is_empty();
    Ok(WorkspacePlan {
        workspace_id: workspace_id.to_owned(),
        profile_ids: profiles.iter().map(|profile| profile.profile_id.to_owned()).collect(),
        generation,
        target_only: false,
        unchanged_paths,
        updated_files,
        covered_paths,
        degraded_profile_ids,
        excluded_files,
        deleted_files,
        report: IndexReport { degraded_semantic, ..report },
    })
}
fn prepare_targeted_workspace(
    workspace_id: &str,
    workspace: &WorkspaceFs,
    profiles: &[ProfileProvider<'_>],
    generation: u64,
    max_file_bytes: u64,
    force_paths: &[String],
    snapshot: WorkspaceSnapshot,
) -> Result<WorkspacePlan> {
    let parser_registry = LanguageRegistry::new();
    let mut paths = force_paths.to_vec();
    paths.sort();
    paths.dedup();
    let mut report = IndexReport::default();
    let mut degraded_profile_ids = HashSet::new();
    let mut unchanged_paths = Vec::new();
    let mut updated_files = Vec::new();
    let mut excluded_files = Vec::new();
    let mut deleted_files = Vec::new();
    let mut covered_paths = HashSet::with_capacity(paths.len());
    for relative_path in paths {
        prepare_targeted_path(
            TargetedPathPreparation {
                workspace,
                profiles,
                max_file_bytes,
                relative_path,
                snapshot: &snapshot,
                parser_registry: &parser_registry,
                degraded_profile_ids: &mut degraded_profile_ids,
                unchanged_paths: &mut unchanged_paths,
                updated_files: &mut updated_files,
                excluded_files: &mut excluded_files,
                deleted_files: &mut deleted_files,
                covered_paths: &mut covered_paths,
                report: &mut report,
            },
        )?;
    }
    workspace.verify_registered_root()
        .map_err(|error| StateError::Io(std::io::Error::other(error)))?;
    let degraded_semantic = !degraded_profile_ids.is_empty();
    Ok(WorkspacePlan {
        workspace_id: workspace_id.to_owned(),
        profile_ids: profiles.iter().map(|profile| profile.profile_id.to_owned()).collect(),
        generation,
        target_only: true,
        unchanged_paths,
        updated_files,
        covered_paths,
        degraded_profile_ids,
        excluded_files,
        deleted_files,
        report: IndexReport { degraded_semantic, ..report },
    })
}

struct TargetedPathPreparation<'a> {
    workspace: &'a WorkspaceFs,
    profiles: &'a [ProfileProvider<'a>],
    max_file_bytes: u64,
    relative_path: String,
    snapshot: &'a WorkspaceSnapshot,
    parser_registry: &'a LanguageRegistry,
    degraded_profile_ids: &'a mut HashSet<String>,
    unchanged_paths: &'a mut Vec<String>,
    updated_files: &'a mut Vec<PreparedFile>,
    excluded_files: &'a mut Vec<(String, HashMap<String, Vec<String>>)>,
    deleted_files: &'a mut Vec<(String, HashMap<String, Vec<String>>)>,
    covered_paths: &'a mut HashSet<String>,
    report: &'a mut IndexReport,
}

fn prepare_targeted_path(input: TargetedPathPreparation<'_>) -> Result<()> {
    let TargetedPathPreparation {
        workspace,
        profiles,
        max_file_bytes,
        relative_path,
        snapshot,
        parser_registry,
        degraded_profile_ids,
        unchanged_paths,
        updated_files,
        excluded_files,
        deleted_files,
        covered_paths,
        report,
    } = input;
    let path = PathBuf::from(relative_path.as_str());
    let entry = crate::index::scanner::read_workspace_path(
        workspace,
        workspace.root_path(),
        &path,
        max_file_bytes,
    ).map_err(StateError::Io)?;
    covered_paths.insert(relative_path.clone());
    match entry {
        Some(crate::index::scanner::ScanEntry::Included(scanned)) => {
            let content_hash = hash_content(&scanned.content);
            let current = snapshot.files.get(&relative_path);
            let has_profile_coverage = current.is_some_and(|file| profiles.iter().all(|profile| {
                file.embedding_counts.get(profile.profile_id).copied().unwrap_or_default()
                    == file.chunk_count
            }));
            if has_profile_coverage && current.is_some_and(|file| {
                file.content_hash == content_hash
                    && file.parser_version == PARSER_VERSION
                    && file.chunker_version == STORED_CHUNKER_VERSION
            }) {
                unchanged_paths.push(relative_path);
                report.unchanged += 1;
                return Ok(());
            }
            let old_vector_ids = current.map(|file| file.vector_ids.clone()).unwrap_or_default();
            updated_files.push(prepare_file(FilePreparation {
                path: &path,
                relative_path,
                content: scanned.content,
                content_hash,
                old_vector_ids,
                profiles,
                parser_registry,
                degraded_profile_ids,
            }));
            report.updated += 1;
        }
        Some(crate::index::scanner::ScanEntry::Excluded { .. }) => {
            if let Some(file) = snapshot.files.get(&relative_path) {
                excluded_files.push((relative_path, file.vector_ids.clone()));
                report.excluded += 1;
            }
        }
        None => {
            if let Some(file) = snapshot.files.get(&relative_path) {
                deleted_files.push((relative_path, file.vector_ids.clone()));
                report.deleted += 1;
            }
        }
    }
    Ok(())
}

struct FilePreparation<'a> {
    path: &'a Path,
    relative_path: String,
    content: String,
    content_hash: String,
    old_vector_ids: HashMap<String, Vec<String>>,
    profiles: &'a [ProfileProvider<'a>],
    parser_registry: &'a LanguageRegistry,
    degraded_profile_ids: &'a mut HashSet<String>,
}
fn prepare_file(input: FilePreparation<'_>) -> PreparedFile {
    let FilePreparation {
        path,
        relative_path,
        content,
        content_hash,
        old_vector_ids,
        profiles,
        parser_registry,
        degraded_profile_ids,
    } = input;
    let language = detect_language(path);
    let language_label = language_label(path);
    let (symbols, dependencies, regions) = extract_records(path, &content, language, parser_registry);
    let chunks = syntax_chunker(&content, &regions, &relative_path, language_label);
    let embedding_texts = chunks.iter().map(|chunk| chunk.embedding_text.clone()).collect::<Vec<_>>();
    let embeddings = profiles.iter().map(|profile| {
        let batch_size = profile.provider.batch_size().max(1);
        let vectors = if embedding_texts.is_empty() {
            Some(Vec::new())
        } else {
            let mut completed = Vec::with_capacity(chunks.len());
            let mut valid = true;
            for batch in embedding_texts.chunks(batch_size) {
                match profile.provider.embed_batch(batch) {
                    Ok(vectors) if valid_embedding_batch(&vectors, batch.len(), profile.provider.dimension()) => {
                        completed.extend(vectors);
                    }
                    _ => {
                        valid = false;
                        break;
                    }
                }
            }
            if valid { Some(completed) } else {
                degraded_profile_ids.insert(profile.profile_id.to_owned());
                None
            }
        };
        (profile.profile_id.to_owned(), vectors)
    }).collect();
    PreparedFile {
        relative_path,
        content_hash,
        size_bytes: content.len(),
        language: language_label,
        symbols,
        dependencies,
        chunks,
        embeddings,
        old_vector_ids,
    }
}

fn extract_records(
    path: &Path,
    content: &str,
    language: Option<SourceLanguage>,
    parser_registry: &LanguageRegistry,
) -> (Vec<SymbolRecord>, Vec<DependencyRecord>, Vec<SyntaxRegion>) {
    let local_parse = language.and_then(|language| parse_source(content, language).ok());
    let extension = path.extension().and_then(|value| value.to_str())
        .map(|value| format!(".{}", value.to_ascii_lowercase()));
    if let Some(plugin) = extension.as_deref().and_then(|value| parser_registry.plugin_for_extension(value))
        && let Ok(tree) = codeindex_parse_source(plugin, content.as_bytes())
    {
        return extract_upstream_records(plugin, &tree, content, local_parse.as_ref());
    }
    let Some(parsed) = local_parse else { return (Vec::new(), Vec::new(), Vec::new()) };
    extract_local_records(content, parsed)
}

fn extract_upstream_records(
    plugin: &dyn codeindex_tree_sitter_langs::LanguagePlugin,
    tree: &tree_sitter::Tree,
    content: &str,
    local_parse: Option<&ParseResult>,
) -> (Vec<SymbolRecord>, Vec<DependencyRecord>, Vec<SyntaxRegion>) {
    let upstream = plugin.extract_regions(tree, content.as_bytes());
    let mut symbols = Vec::new();
    let mut regions = Vec::new();
    let mut region_map = vec![None; upstream.len()];
    for (index, region) in upstream.iter().enumerate() {
        let Some(byte_range) = normalized_range(content, region.start_byte as usize, region.end_byte as usize) else {
            continue;
        };
        let parent_name = region.parent_index.and_then(|parent| upstream.get(parent))
            .map(|parent| parent.name.clone());
        let doc_comment = local_parse.and_then(|parsed| {
            doc_comment_for_region(parsed, &region.name, &byte_range)
        });
        let new_index = regions.len();
        region_map[index] = Some(new_index);
        symbols.push(SymbolRecord {
            name: region.name.clone(),
            kind: symbol_kind_from_upstream(region.kind),
            parent_name: parent_name.clone(),
            signature: Some(region.signature.clone()),
            doc_comment: doc_comment.clone(),
            line_range: inclusive_line_range(content, &byte_range),
            byte_range: byte_range.clone(),
        });
        regions.push(SyntaxRegion {
            start_byte: byte_range.start,
            end_byte: byte_range.end,
            parent_index: None,
            parent_name,
            signature: Some(region.signature.clone()),
            doc_comment,
        });
    }
    for (index, region) in upstream.iter().enumerate() {
        let Some(new_index) = region_map[index] else { continue };
        regions[new_index].parent_index = region.parent_index.and_then(|parent| region_map.get(parent).copied().flatten());
    }
    let dependencies = plugin.extract_dependencies(tree, content.as_bytes(), &upstream).into_iter()
        .filter_map(|dependency| {
            let region = upstream.get(dependency.source_region_index)?;
            let range = normalized_range(content, region.start_byte as usize, region.end_byte as usize)?;
            Some(DependencyRecord {
                name: dependency.target_symbol,
                evidence: "source_region",
                line_range: inclusive_line_range(content, &range),
                byte_range: range,
            })
        }).collect();
    (symbols, dependencies, regions)
}

fn extract_local_records(
    content: &str,
    parsed: ParseResult,
) -> (Vec<SymbolRecord>, Vec<DependencyRecord>, Vec<SyntaxRegion>) {
    let symbols = parsed.symbols.into_iter()
        .filter_map(|symbol| normalize_symbol(content, symbol)).collect::<Vec<_>>();
    let regions = syntax_regions_from_symbols(&symbols);
    let dependencies = parsed.imports.into_iter()
        .filter_map(|record| normalized_dependency(content, record.name, "import", record.byte_range))
        .chain(parsed.calls.into_iter()
            .filter_map(|record| normalized_dependency(content, record.name, "call_candidate", record.byte_range)))
        .collect();
    (symbols, dependencies, regions)
}

fn normalized_dependency(
    content: &str,
    name: String,
    evidence: &'static str,
    range: Range<usize>,
) -> Option<DependencyRecord> {
    let byte_range = normalized_range(content, range.start, range.end)?;
    Some(DependencyRecord {
        name,
        evidence,
        line_range: inclusive_line_range(content, &byte_range),
        byte_range,
    })
}

fn normalize_symbol(content: &str, mut symbol: SymbolRecord) -> Option<SymbolRecord> {
    let range = normalized_range(content, symbol.byte_range.start, symbol.byte_range.end)?;
    symbol.byte_range = range.clone();
    symbol.line_range = inclusive_line_range(content, &range);
    Some(symbol)
}

fn syntax_regions_from_symbols(symbols: &[SymbolRecord]) -> Vec<SyntaxRegion> {
    symbols.iter().enumerate().map(|(index, symbol)| {
        let parent_index = symbol.parent_name.as_ref().and_then(|parent_name| {
            symbols[..index].iter().rposition(|candidate| {
                candidate.name.as_str() == parent_name.as_str()
                    && candidate.byte_range.start <= symbol.byte_range.start
                    && candidate.byte_range.end >= symbol.byte_range.end
            })
        });
        SyntaxRegion {
            start_byte: symbol.byte_range.start,
            end_byte: symbol.byte_range.end,
            parent_index,
            parent_name: symbol.parent_name.clone(),
            signature: symbol.signature.clone(),
            doc_comment: symbol.doc_comment.clone(),
        }
    }).collect()
}

fn doc_comment_for_region(parsed: &ParseResult, name: &str, range: &Range<usize>) -> Option<String> {
    parsed.symbols.iter().find(|symbol| {
        symbol.name == name && symbol.byte_range.start <= range.start && symbol.byte_range.end >= range.end
    }).and_then(|symbol| symbol.doc_comment.clone())
}

fn normalized_range(content: &str, start: usize, end: usize) -> Option<Range<usize>> {
    let range = start.min(content.len())..end.min(content.len());
    (range.start < range.end && content.is_char_boundary(range.start) && content.is_char_boundary(range.end))
        .then_some(range)
}

fn inclusive_line_range(content: &str, range: &Range<usize>) -> Range<usize> {
    let bytes = content.as_bytes();
    let start = bytes[..range.start].iter().filter(|byte| **byte == b'\n').count() + 1;
    let end = bytes[..range.end.saturating_sub(1)].iter().filter(|byte| **byte == b'\n').count() + 1;
    start..end
}

fn valid_embedding_batch(vectors: &[Vec<f32>], expected_count: usize, dimension: usize) -> bool {
    dimension > 0 && vectors.len() == expected_count
        && vectors.iter().all(|vector| vector.len() == dimension && vector.iter().all(|value| value.is_finite()))
}

fn symbol_kind_from_upstream(kind: codeindex_tree_sitter_langs::RegionKind) -> SymbolKind {
    match kind {
        codeindex_tree_sitter_langs::RegionKind::Function => SymbolKind::Function,
        codeindex_tree_sitter_langs::RegionKind::Method => SymbolKind::Method,
        codeindex_tree_sitter_langs::RegionKind::Class => SymbolKind::Class,
        codeindex_tree_sitter_langs::RegionKind::Interface => SymbolKind::Interface,
        codeindex_tree_sitter_langs::RegionKind::Module => SymbolKind::Module,
        codeindex_tree_sitter_langs::RegionKind::Struct
        | codeindex_tree_sitter_langs::RegionKind::ImplBlock
        | codeindex_tree_sitter_langs::RegionKind::Enum => SymbolKind::Type,
    }
}

fn detect_language(path: &Path) -> Option<SourceLanguage> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "py" => Some(SourceLanguage::Python), "java" => Some(SourceLanguage::Java),
        "js" | "jsx" => Some(SourceLanguage::JavaScript), "ts" => Some(SourceLanguage::TypeScript),
        "tsx" => Some(SourceLanguage::Tsx), "go" => Some(SourceLanguage::Go), "rs" => Some(SourceLanguage::Rust),
        "rb" => Some(SourceLanguage::Ruby), "php" => Some(SourceLanguage::Php), "scala" | "sc" => Some(SourceLanguage::Scala),
        "c" | "h" => Some(SourceLanguage::C), "cpp" | "cc" | "cxx" | "hpp" | "hh" => Some(SourceLanguage::Cpp),
        "cs" => Some(SourceLanguage::CSharp), "kt" | "kts" => Some(SourceLanguage::Kotlin), _ => None,
    }
}

fn language_label(path: &Path) -> &'static str {
    let Some(extension) = path.extension().and_then(|value| value.to_str()) else { return "unknown" };
    match extension.to_ascii_lowercase().as_str() {
        "py" => "python", "java" => "java", "js" | "jsx" => "javascript",
        "ts" => "typescript", "tsx" => "tsx", "go" => "go", "rs" => "rust",
        "rb" => "ruby", "php" => "php", "scala" | "sc" => "scala", "c" => "c",
        "h" => "c", "cpp" | "cc" | "cxx" | "hpp" | "hh" => "cpp", "cs" => "csharp",
        "kt" | "kts" => "kotlin", "md" | "markdown" => "markdown", "yaml" | "yml" => "yaml",
        "toml" => "toml", "sql" => "sql", "sh" | "bash" | "zsh" => "shell", _ => "unknown",
    }
}

pub(crate) fn persist_workspace(transaction: &Transaction<'_>, plan: &WorkspacePlan) -> Result<()> {
    if plan.target_only {
        transaction.execute(
            "UPDATE files SET visible_generation=?1, file_generation=?1 WHERE workspace_id=?2",
            rusqlite::params![plan.generation as i64, plan.workspace_id],
        )?;
        transaction.execute(
            "UPDATE chunk_embeddings SET generation=?1 WHERE workspace_id=?2",
            rusqlite::params![plan.generation as i64, plan.workspace_id],
        )?;
    }
    for path in &plan.unchanged_paths {
        transaction.execute(
            "UPDATE files SET visible_generation=?1, file_generation=?1 WHERE workspace_id=?2 AND relative_path=?3",
            rusqlite::params![plan.generation as i64, plan.workspace_id, path],
        )?;
        for profile_id in &plan.profile_ids {
            transaction.execute(
                "UPDATE chunk_embeddings SET generation=?1
                 WHERE workspace_id=?2 AND relative_path=?3 AND profile_id=?4",
                rusqlite::params![plan.generation as i64, plan.workspace_id, path, profile_id],
            )?;
        }
    }
    for file in &plan.updated_files {
        persist_updated_file(transaction, plan, file)?;
    }
    for (path, vector_ids) in plan.excluded_files.iter().chain(&plan.deleted_files) {
        for profile_id in &plan.profile_ids {
            if let Some(ids) = vector_ids.get(profile_id) {
                enqueue_deletes(transaction, plan, profile_id, ids)?;
            }
        }
        transaction.execute(
            "DELETE FROM files WHERE workspace_id=?1 AND relative_path=?2",
            rusqlite::params![plan.workspace_id, path],
        )?;
    }
    let desired_profile = plan.profile_ids.last().ok_or(StateError::InvalidVectorOperation)?;
    let updated_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    transaction.execute(
        "INSERT INTO workspace_index_state (
            workspace_id, generation, updated_at, building_generation, visible_generation, dirty,
            health, desired_profile_id, active_profile_id
         ) VALUES (?1, ?2, ?3, ?2, ?2, 0, 'degraded', ?4, NULL)
         ON CONFLICT(workspace_id) DO UPDATE SET
            generation=MAX(workspace_index_state.generation, excluded.generation),
            building_generation=MAX(workspace_index_state.building_generation, excluded.building_generation),
            visible_generation=MAX(workspace_index_state.visible_generation, excluded.visible_generation),
            dirty=CASE WHEN workspace_index_state.building_generation <= excluded.building_generation THEN 0
                       ELSE workspace_index_state.dirty END,
            health=CASE WHEN workspace_index_state.building_generation <= excluded.building_generation THEN 'degraded'
                        ELSE workspace_index_state.health END,
            desired_profile_id=COALESCE(workspace_index_state.desired_profile_id, excluded.desired_profile_id),
            updated_at=excluded.updated_at",
        rusqlite::params![plan.workspace_id, plan.generation as i64, updated_at, desired_profile],
    )?;
    Ok(())
}

fn persist_updated_file(transaction: &Transaction<'_>, plan: &WorkspacePlan, file: &PreparedFile) -> Result<()> {
    for profile_id in &plan.profile_ids {
        if let Some(vector_ids) = file.old_vector_ids.get(profile_id) {
            enqueue_deletes(transaction, plan, profile_id, vector_ids)?;
        }
    }
    transaction.execute("DELETE FROM files WHERE workspace_id=?1 AND relative_path=?2", rusqlite::params![plan.workspace_id, file.relative_path])?;
    transaction.execute(
        "INSERT INTO files (workspace_id, relative_path, content_hash, size_bytes, visible_generation, file_generation, parser_version, chunker_version)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6, ?7)",
        rusqlite::params![
            plan.workspace_id, file.relative_path, file.content_hash, file.size_bytes as i64,
            plan.generation as i64, PARSER_VERSION, STORED_CHUNKER_VERSION,
        ],
    )?;
    persist_symbols(transaction, plan, file)?;
    persist_dependencies(transaction, plan, file)?;
    persist_chunks(transaction, plan, file)?;
    persist_embeddings(transaction, plan, file)
}

fn enqueue_deletes(
    transaction: &Transaction<'_>,
    plan: &WorkspacePlan,
    profile_id: &str,
    vector_ids: &[String],
) -> Result<()> {
    for vector_id in vector_ids {
        transaction.execute(
            "INSERT INTO vector_ops (workspace_id, profile_id, generation, op_type, vector_id, embedding) VALUES (?1, ?2, ?3, 'delete', ?4, NULL)",
            rusqlite::params![plan.workspace_id, profile_id, plan.generation as i64, vector_id],
        )?;
    }
    Ok(())
}


fn persist_symbols(transaction: &Transaction<'_>, plan: &WorkspacePlan, file: &PreparedFile) -> Result<()> {
    let mut statement = transaction.prepare(
        "INSERT INTO symbols (workspace_id, relative_path, symbol_index, name, kind, parent_name, signature, doc_comment, byte_start, byte_end, start_line, end_line)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
    )?;
    for (index, symbol) in file.symbols.iter().enumerate() {
        statement.execute(rusqlite::params![
            plan.workspace_id, file.relative_path, index as i64, symbol.name, symbol_kind(symbol.kind), symbol.parent_name,
            symbol.signature, symbol.doc_comment, symbol.byte_range.start as i64, symbol.byte_range.end as i64,
            symbol.line_range.start as i64, symbol.line_range.end as i64,
        ])?;
    }
    Ok(())
}

fn persist_dependencies(transaction: &Transaction<'_>, plan: &WorkspacePlan, file: &PreparedFile) -> Result<()> {
    let mut statement = transaction.prepare(
        "INSERT INTO dependency_edges (workspace_id, relative_path, edge_index, evidence, name, source_symbol, byte_start, byte_end, start_line, end_line)
         VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?7, ?8, ?9)",
    )?;
    for (edge_index, dependency) in file.dependencies.iter().enumerate() {
        persist_dependency(&mut statement, plan, file, edge_index, dependency)?;
    }
    Ok(())
}

fn persist_dependency(
    statement: &mut rusqlite::Statement<'_>,
    plan: &WorkspacePlan,
    file: &PreparedFile,
    edge_index: usize,
    dependency: &DependencyRecord,
) -> Result<()> {
    statement.execute(rusqlite::params![
        plan.workspace_id, file.relative_path, edge_index as i64, dependency.evidence, dependency.name,
        dependency.byte_range.start as i64, dependency.byte_range.end as i64,
        dependency.line_range.start as i64, dependency.line_range.end as i64,
    ])?;
    Ok(())
}

fn persist_chunks(transaction: &Transaction<'_>, plan: &WorkspacePlan, file: &PreparedFile) -> Result<()> {
    let mut statement = transaction.prepare(
        "INSERT INTO chunks (workspace_id, relative_path, chunk_index, start_line, end_line, content, symbol_context, language, chunk_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, ?7, ?8)",
    )?;
    let language = file.language;
    for chunk in &file.chunks {
        statement.execute(rusqlite::params![
            plan.workspace_id, file.relative_path, chunk.index as i64, chunk.start_line as i64, chunk.end_line as i64,
            chunk.content, language, hash_content(&chunk.content),
        ])?;
    }
    Ok(())
}

fn persist_embeddings(transaction: &Transaction<'_>, plan: &WorkspacePlan, file: &PreparedFile) -> Result<()> {
    let mut statement = transaction.prepare(
        "INSERT INTO chunk_embeddings (workspace_id, relative_path, chunk_index, profile_id, generation, vector_id, embedding)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;
    for profile_id in &plan.profile_ids {
        let Some(Some(embeddings)) = file.embeddings.get(profile_id) else { continue };
        for (index, embedding) in embeddings.iter().enumerate() {
            let vector_id = format!("{}#{}", file.relative_path, index);
            let bytes = embedding.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            statement.execute(rusqlite::params![
                plan.workspace_id, file.relative_path, index as i64, profile_id,
                plan.generation as i64, vector_id, bytes,
            ])?;
            transaction.execute(
                "INSERT INTO vector_ops (workspace_id, profile_id, generation, op_type, vector_id, embedding)
                 VALUES (?1, ?2, ?3, 'add', ?4, ?5)",
                rusqlite::params![plan.workspace_id, profile_id, plan.generation as i64, vector_id, bytes],
            )?;
        }
    }
    Ok(())
}

fn symbol_kind(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Function => "function", SymbolKind::Method => "method", SymbolKind::Class => "class",
        SymbolKind::Interface => "interface", SymbolKind::Type => "type", SymbolKind::Module => "module",
    }
}

