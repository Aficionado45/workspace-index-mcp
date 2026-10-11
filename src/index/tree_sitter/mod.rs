use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLanguage {
    Python, Java, JavaScript, TypeScript, Tsx, Go, Rust, Ruby,
    Php, Scala, C, Cpp, CSharp, Kotlin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Interface,
    Type,
    Module,
}

#[derive(Debug, Clone)]
pub struct SymbolRecord {
    pub name: String,
    pub kind: SymbolKind,
    pub parent_name: Option<String>,
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
    pub byte_range: Range<usize>,
    pub line_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct ImportRecord {
    pub name: String,
    pub byte_range: Range<usize>,
    pub line_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct CallRecord {
    pub name: String,
    pub byte_range: Range<usize>,
    pub line_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub symbols: Vec<SymbolRecord>,
    pub imports: Vec<ImportRecord>,
    pub calls: Vec<CallRecord>,
    pub has_error: bool,
}

mod languages;
mod parser;

pub use parser::parse_source;
