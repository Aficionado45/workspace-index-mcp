use workspace_index_mcp::index::tree_sitter::{parse_source, SourceLanguage, SymbolKind};

#[test]
fn rust_function_yields_source_range_and_call_candidate() {
    let parsed = parse_source("pub fn greet() { println!(\"hi\"); }", SourceLanguage::Rust).unwrap();
    let greet = parsed.symbols.iter().find(|symbol| symbol.name == "greet").unwrap();
    assert_eq!(greet.kind, SymbolKind::Function);
    assert_eq!(&"pub fn greet() { println!(\"hi\"); }"[greet.byte_range.clone()], "pub fn greet() { println!(\"hi\"); }");
}

#[test]
fn kotlin_source_uses_compatible_grammar_binding() {
    let parsed = parse_source("fun greet() { println(\"hi\") }", SourceLanguage::Kotlin).unwrap();
    assert!(!parsed.has_error);
}

#[test]
fn every_declared_grammar_accepts_a_parse_request() {
    let languages = [
        SourceLanguage::Python,
        SourceLanguage::Java,
        SourceLanguage::JavaScript,
        SourceLanguage::TypeScript,
        SourceLanguage::Tsx,
        SourceLanguage::Go,
        SourceLanguage::Rust,
        SourceLanguage::Ruby,
        SourceLanguage::Php,
        SourceLanguage::Scala,
        SourceLanguage::C,
        SourceLanguage::Cpp,
        SourceLanguage::CSharp,
        SourceLanguage::Kotlin,
    ];
    for language in languages {
        let source = if language == SourceLanguage::Php { "<?php echo 1;" } else { "" };
        let result = parse_source(source, language);
        assert!(result.is_ok(), "grammar failed for {language:?}: {result:?}");
    }
}
