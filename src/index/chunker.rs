use std::ops::Range;

pub(crate) const PARSER_VERSION: i64 = 2;
pub(crate) const CHUNKER_VERSION: i64 = 2;
pub(crate) const EMBEDDING_TEMPLATE_VERSION: i64 = 2;
pub(crate) const STORED_CHUNKER_VERSION: i64 = CHUNKER_VERSION * 100 + EMBEDDING_TEMPLATE_VERSION;
pub const MAX_ESTIMATED_TOKENS: usize = 1_200;
const WINDOW_LINES: usize = 120;
const WINDOW_OVERLAP: usize = 20;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Chunk {
    pub index: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub embedding_text: String,
}

#[derive(Debug, Clone)]
pub struct SyntaxRegion {
    pub start_byte: usize,
    pub end_byte: usize,
    pub parent_index: Option<usize>,
    pub parent_name: Option<String>,
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
}

pub fn line_chunker(source: &str, max_lines: usize, overlap_lines: usize) -> Vec<Chunk> {
    let spans = line_spans(source, 0..source.len());
    let max_lines = max_lines.max(1);
    let step = max_lines.saturating_sub(overlap_lines).max(1);
    let mut chunks = Vec::new();
    let mut first = 0;
    while first < spans.len() {
        let last = (first + max_lines).min(spans.len());
        chunks.push(make_line_chunk(source, &spans[first..last], chunks.len()));
        if last == spans.len() { break; }
        first += step;
    }
    chunks
}

pub fn syntax_chunker(
    source: &str,
    regions: &[SyntaxRegion],
    relative_path: &str,
    language: &str,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut roots = regions.iter().enumerate()
        .filter(|(_, region)| region.parent_index.is_none() && region_range(source, region).is_some())
        .map(|(index, _)| index).collect::<Vec<_>>();
    roots.sort_by_key(|index| regions[*index].start_byte);
    let mut cursor = 0;
    for index in roots {
        let range = region_range(source, &regions[index]).expect("validated syntax region");
        if range.start < cursor { continue; }
        emit_line_windows(source, cursor..range.start, relative_path, language, None, &mut chunks);
        emit_region(source, regions, index, relative_path, language, &mut chunks);
        cursor = range.end;
    }
    emit_line_windows(source, cursor..source.len(), relative_path, language, None, &mut chunks);
    if chunks.is_empty() && regions.is_empty() {
        emit_line_windows(source, 0..source.len(), relative_path, language, None, &mut chunks);
    }
    for (index, chunk) in chunks.iter_mut().enumerate() { chunk.index = index; }
    chunks
}

fn emit_region(
    source: &str,
    regions: &[SyntaxRegion],
    index: usize,
    path: &str,
    language: &str,
    output: &mut Vec<Chunk>,
) {
    let region = &regions[index];
    let Some(range) = region_range(source, region) else { return };
    let header = embedding_header(path, language, Some(region));
    if estimated_tokens(&header, &source[range.clone()]) <= MAX_ESTIMATED_TOKENS {
        emit_budgeted_range(source, range, path, language, Some(region), output);
        return;
    }
    let mut children = regions.iter().enumerate()
        .filter(|(_, child)| child.parent_index == Some(index))
        .filter_map(|(child_index, child)| region_range(source, child).map(|range| (child_index, range)))
        .filter(|(_, child)| child.start >= range.start && child.end <= range.end)
        .collect::<Vec<_>>();
    children.sort_by_key(|(_, child)| child.start);
    if children.is_empty() {
        emit_line_windows(source, range, path, language, Some(region), output);
        return;
    }
    let mut cursor = range.start;
    for (child_index, child_range) in children {
        if child_range.start < cursor { continue; }
        emit_line_windows(source, cursor..child_range.start, path, language, Some(region), output);
        emit_region(source, regions, child_index, path, language, output);
        cursor = child_range.end;
    }
    emit_line_windows(source, cursor..range.end, path, language, Some(region), output);
}

fn emit_line_windows(
    source: &str,
    range: Range<usize>,
    path: &str,
    language: &str,
    context: Option<&SyntaxRegion>,
    output: &mut Vec<Chunk>,
) {
    if range.start >= range.end || source.get(range.clone()).is_none() { return; }
    let spans = line_spans(source, range);
    let step = WINDOW_LINES - WINDOW_OVERLAP;
    let mut first = 0;
    while first < spans.len() {
        let last = (first + WINDOW_LINES).min(spans.len());
        emit_budgeted_range(source, spans[first].start..spans[last - 1].end, path, language, context, output);
        if last == spans.len() { break; }
        first += step;
    }
}

fn emit_budgeted_range(
    source: &str,
    range: Range<usize>,
    path: &str,
    language: &str,
    context: Option<&SyntaxRegion>,
    output: &mut Vec<Chunk>,
) {
    let Some(content) = source.get(range.clone()) else { return };
    if content.trim().is_empty() { return; }
    let header = embedding_header(path, language, context);
    let remaining = MAX_ESTIMATED_TOKENS * 4 - header.chars().count();
    let mut start = range.start;
    let mut chars = 0;
    for (offset, _) in content.char_indices() {
        if chars == remaining {
            push_chunk(source, start..range.start + offset, &header, output);
            start = range.start + offset;
            chars = 0;
        }
        chars += 1;
    }
    push_chunk(source, start..range.end, &header, output);
}

fn push_chunk(source: &str, range: Range<usize>, header: &str, output: &mut Vec<Chunk>) {
    let Some(content) = source.get(range.clone()) else { return };
    if content.trim().is_empty() { return; }
    let (start_line, end_line) = line_range(source, &range);
    let content = content.to_owned();
    output.push(Chunk {
        index: output.len(),
        start_line,
        end_line,
        embedding_text: format!("{header}{content}"),
        content,
    });
}

fn line_spans(source: &str, range: Range<usize>) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut offset = range.start;
    for part in source[range].split_inclusive('\n') {
        let end = offset + part.len();
        spans.push(offset..end);
        offset = end;
    }
    spans
}

fn make_line_chunk(source: &str, spans: &[Range<usize>], index: usize) -> Chunk {
    let range = spans.first().map(|first| first.start).unwrap_or(0)
        ..spans.last().map(|last| last.end).unwrap_or(0);
    let content = source[range.clone()].to_owned();
    let (start_line, end_line) = line_range(source, &range);
    Chunk { index, start_line, end_line, embedding_text: content.clone(), content }
}

fn region_range(source: &str, region: &SyntaxRegion) -> Option<Range<usize>> {
    let range = region.start_byte.min(source.len())..region.end_byte.min(source.len());
    (range.start < range.end && source.is_char_boundary(range.start) && source.is_char_boundary(range.end))
        .then_some(range)
}

fn line_range(source: &str, range: &Range<usize>) -> (usize, usize) {
    let bytes = source.as_bytes();
    let start = bytes[..range.start].iter().filter(|byte| **byte == b'\n').count() + 1;
    let end_offset = range.end.saturating_sub(1);
    let end = bytes[..end_offset].iter().filter(|byte| **byte == b'\n').count() + 1;
    (start, end)
}

fn embedding_header(path: &str, language: &str, region: Option<&SyntaxRegion>) -> String {
    let field = |value: Option<&str>, max| value.unwrap_or("").chars().take(max).collect::<String>();
    let parent = region.and_then(|value| value.parent_name.as_deref());
    let signature = region.and_then(|value| value.signature.as_deref());
    let doc = region.and_then(|value| value.doc_comment.as_deref());
    format!(
        "path: {}\nlanguage: {}\nparent: {}\nsignature: {}\ndoc: {}\nsource:\n",
        field(Some(path), 256), field(Some(language), 64), field(parent, 256),
        field(signature, 512), field(doc, 512),
    )
}

fn estimated_tokens(header: &str, content: &str) -> usize {
    (header.chars().count() + content.chars().count()).div_ceil(4)
}

pub fn hash_content(content: &str) -> String {
    blake3::hash(content.as_bytes()).to_hex().to_string()
}
