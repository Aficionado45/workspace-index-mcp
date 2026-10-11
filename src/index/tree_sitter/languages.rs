use super::{SourceLanguage, SymbolKind};

#[derive(Clone, Copy)]
pub(super) struct DeclarationRule {
    pub(super) node: &'static str,
    pub(super) kind: SymbolKind,
    pub(super) method: bool,
}

pub(super) struct GrammarConfig {
    pub(super) declarations: &'static [DeclarationRule],
    pub(super) method_containers: &'static [&'static str],
    pub(super) scope_nodes: &'static [(&'static str, &'static str)],
    pub(super) function_binding_nodes: &'static [&'static str],
    pub(super) function_value_nodes: &'static [&'static str],
    pub(super) import_nodes: &'static [&'static str],
    pub(super) call_nodes: &'static [&'static str],
    pub(super) call_name_fields: &'static [&'static str],
    pub(super) import_call_names: &'static [&'static str],
}

const fn decl(node: &'static str, kind: SymbolKind) -> DeclarationRule {
    DeclarationRule { node, kind, method: false }
}

const fn method(node: &'static str) -> DeclarationRule {
    DeclarationRule { node, kind: SymbolKind::Method, method: true }
}

const EMPTY: &[&str] = &[];
const EMPTY_SCOPE: &[(&str, &str)] = &[];
const COMMON_CALL_FIELDS: &[&str] = &["function", "callee", "name", "method"];
const JS_FUNCTION_VALUES: &[&str] = &["arrow_function", "function_expression", "generator_function"];

const PYTHON_DECLS: &[DeclarationRule] = &[decl("function_definition", SymbolKind::Function), decl("class_definition", SymbolKind::Class)];
const JAVA_DECLS: &[DeclarationRule] = &[
    decl("class_declaration", SymbolKind::Class), decl("record_declaration", SymbolKind::Class),
    decl("interface_declaration", SymbolKind::Interface), decl("annotation_type_declaration", SymbolKind::Interface),
    decl("enum_declaration", SymbolKind::Type), method("method_declaration"), method("constructor_declaration"),
];
const JS_DECLS: &[DeclarationRule] = &[decl("function_declaration", SymbolKind::Function), decl("generator_function_declaration", SymbolKind::Function), decl("class_declaration", SymbolKind::Class), method("method_definition")];
const TS_DECLS: &[DeclarationRule] = &[
    decl("function_declaration", SymbolKind::Function), decl("generator_function_declaration", SymbolKind::Function),
    decl("class_declaration", SymbolKind::Class), decl("abstract_class_declaration", SymbolKind::Class),
    decl("interface_declaration", SymbolKind::Interface), decl("type_alias_declaration", SymbolKind::Type),
    decl("enum_declaration", SymbolKind::Type), method("method_definition"), method("method_signature"),
    method("abstract_method_signature"),
];
const GO_DECLS: &[DeclarationRule] = &[
    decl("function_declaration", SymbolKind::Function), method("method_declaration"),
    decl("type_spec", SymbolKind::Type), decl("package_clause", SymbolKind::Module),
];
const GO_SCOPES: &[(&str, &str)] = &[("type_declaration", "type")];
const RUST_DECLS: &[DeclarationRule] = &[
    decl("function_item", SymbolKind::Function), decl("struct_item", SymbolKind::Type), decl("enum_item", SymbolKind::Type),
    decl("union_item", SymbolKind::Type), decl("trait_item", SymbolKind::Interface), decl("type_item", SymbolKind::Type),
    decl("mod_item", SymbolKind::Module),
];
const RUST_SCOPES: &[(&str, &str)] = &[("impl_item", "type")];
const RUBY_DECLS: &[DeclarationRule] = &[
    decl("class", SymbolKind::Class), decl("module", SymbolKind::Module), method("method"), method("singleton_method"),
];
const PHP_DECLS: &[DeclarationRule] = &[
    decl("function_definition", SymbolKind::Function), decl("class_declaration", SymbolKind::Class),
    decl("interface_declaration", SymbolKind::Interface), decl("trait_declaration", SymbolKind::Type),
    decl("enum_declaration", SymbolKind::Type), method("method_declaration"),
    decl("namespace_definition", SymbolKind::Module),
];
const SCALA_DECLS: &[DeclarationRule] = &[
    decl("function_definition", SymbolKind::Function), decl("function_declaration", SymbolKind::Function),
    decl("class_definition", SymbolKind::Class), decl("trait_definition", SymbolKind::Interface),
    decl("object_definition", SymbolKind::Module), decl("type_definition", SymbolKind::Type), decl("enum_definition", SymbolKind::Type),
];
const C_DECLS: &[DeclarationRule] = &[
    decl("function_definition", SymbolKind::Function), decl("struct_specifier", SymbolKind::Type),
    decl("union_specifier", SymbolKind::Type), decl("enum_specifier", SymbolKind::Type), decl("type_definition", SymbolKind::Type),
];
const CPP_DECLS: &[DeclarationRule] = &[
    decl("function_definition", SymbolKind::Function), decl("class_specifier", SymbolKind::Class),
    decl("struct_specifier", SymbolKind::Type), decl("union_specifier", SymbolKind::Type), decl("enum_specifier", SymbolKind::Type),
    decl("type_definition", SymbolKind::Type), decl("namespace_definition", SymbolKind::Module),
];
const CSHARP_DECLS: &[DeclarationRule] = &[
    decl("class_declaration", SymbolKind::Class), decl("record_declaration", SymbolKind::Class),
    decl("struct_declaration", SymbolKind::Type), decl("interface_declaration", SymbolKind::Interface),
    decl("enum_declaration", SymbolKind::Type), decl("delegate_declaration", SymbolKind::Type),
    decl("namespace_declaration", SymbolKind::Module), decl("file_scoped_namespace_declaration", SymbolKind::Module),
    method("method_declaration"), method("constructor_declaration"),
    method("destructor_declaration"), method("operator_declaration"),
    method("local_function_statement"),
];
const KOTLIN_DECLS: &[DeclarationRule] = &[
    decl("function_declaration", SymbolKind::Function), decl("class_declaration", SymbolKind::Class),
    decl("object_declaration", SymbolKind::Module), decl("type_alias", SymbolKind::Type),
];
pub(super) fn grammar(language: SourceLanguage) -> tree_sitter::Language {
    match language {
        SourceLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        SourceLanguage::Java => tree_sitter_java::LANGUAGE.into(),
        SourceLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        SourceLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        SourceLanguage::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        SourceLanguage::Go => tree_sitter_go::LANGUAGE.into(),
        SourceLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
        SourceLanguage::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        SourceLanguage::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        SourceLanguage::Scala => tree_sitter_scala::LANGUAGE.into(),
        SourceLanguage::C => tree_sitter_c::LANGUAGE.into(),
        SourceLanguage::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        SourceLanguage::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        SourceLanguage::Kotlin => tree_sitter_kotlin_sg::LANGUAGE.into(),
    }
}

pub(super) fn config(language: SourceLanguage) -> GrammarConfig {
    match language {
        SourceLanguage::Python => GrammarConfig {
            declarations: PYTHON_DECLS,
            method_containers: &["class_definition"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["import_statement", "import_from_statement"],
            call_nodes: &["call"],
            call_name_fields: &["function", "name"],
            import_call_names: &["__import__"],
        },
        SourceLanguage::Java => GrammarConfig {
            declarations: JAVA_DECLS,
            method_containers: &["class_declaration", "record_declaration", "interface_declaration", "enum_declaration", "annotation_type_declaration"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["import_declaration"],
            call_nodes: &["method_invocation", "object_creation_expression", "explicit_constructor_invocation"],
            call_name_fields: &["name", "type", "constructor", "object", "function"],
            import_call_names: EMPTY,
        },
        SourceLanguage::JavaScript => GrammarConfig {
            declarations: JS_DECLS,
            method_containers: &["class_declaration", "class_body"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: &["variable_declarator"],
            function_value_nodes: JS_FUNCTION_VALUES,
            import_nodes: &["import_statement"],
            call_nodes: &["call_expression", "new_expression"],
            call_name_fields: COMMON_CALL_FIELDS,
            import_call_names: EMPTY,
        },
        SourceLanguage::TypeScript | SourceLanguage::Tsx => GrammarConfig {
            declarations: TS_DECLS,
            method_containers: &["class_declaration", "abstract_class_declaration", "class_body", "interface_declaration"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: &["variable_declarator"],
            function_value_nodes: JS_FUNCTION_VALUES,
            import_nodes: &["import_statement"],
            call_nodes: &["call_expression", "new_expression"],
            call_name_fields: COMMON_CALL_FIELDS,
            import_call_names: EMPTY,
        },
        SourceLanguage::Go => GrammarConfig {
            declarations: GO_DECLS,
            method_containers: EMPTY,
            scope_nodes: GO_SCOPES,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["import_declaration"],
            call_nodes: &["call_expression"],
            call_name_fields: &["function", "name"],
            import_call_names: EMPTY,
        },
        SourceLanguage::Rust => GrammarConfig {
            declarations: RUST_DECLS,
            method_containers: &["impl_item", "trait_item"],
            scope_nodes: RUST_SCOPES,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["use_declaration", "extern_crate_declaration"],
            call_nodes: &["call_expression", "macro_invocation"],
            call_name_fields: &["function", "macro", "name"],
            import_call_names: EMPTY,
        },
        SourceLanguage::Ruby => GrammarConfig {
            declarations: RUBY_DECLS,
            method_containers: &["class", "module", "singleton_class"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: EMPTY,
            call_nodes: &["call", "method_call"],
            call_name_fields: &["method", "name"],
            import_call_names: &["require", "require_relative", "load"],
        },
        SourceLanguage::Php => GrammarConfig {
            declarations: PHP_DECLS,
            method_containers: &["class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["namespace_use_declaration", "use_declaration"],
            call_nodes: &["function_call_expression", "member_call_expression", "scoped_call_expression", "object_creation_expression"],
            call_name_fields: &["function", "name", "type", "object"],
            import_call_names: EMPTY,
        },
        SourceLanguage::Scala => GrammarConfig {
            declarations: SCALA_DECLS,
            method_containers: &["class_definition", "trait_definition", "object_definition", "template_body"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["import_declaration"],
            call_nodes: &["call_expression", "infix_expression"],
            call_name_fields: &["function", "name", "operator"],
            import_call_names: EMPTY,
        },
        SourceLanguage::C => GrammarConfig {
            declarations: C_DECLS,
            method_containers: EMPTY,
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["preproc_include"],
            call_nodes: &["call_expression"],
            call_name_fields: &["function", "name"],
            import_call_names: EMPTY,
        },
        SourceLanguage::Cpp => GrammarConfig {
            declarations: CPP_DECLS,
            method_containers: &["class_specifier", "struct_specifier"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["preproc_include"],
            call_nodes: &["call_expression"],
            call_name_fields: &["function", "name"],
            import_call_names: EMPTY,
        },
        SourceLanguage::CSharp => GrammarConfig {
            declarations: CSHARP_DECLS,
            method_containers: &["class_declaration", "record_declaration", "struct_declaration", "interface_declaration", "enum_declaration"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["using_directive", "using_static_directive", "global_using_directive"],
            call_nodes: &["invocation_expression", "object_creation_expression", "implicit_object_creation_expression"],
            call_name_fields: &["function", "name", "type"],
            import_call_names: EMPTY,
        },
        SourceLanguage::Kotlin => GrammarConfig {
            declarations: KOTLIN_DECLS,
            method_containers: &["class_declaration", "object_declaration", "class_body"],
            scope_nodes: EMPTY_SCOPE,
            function_binding_nodes: EMPTY,
            function_value_nodes: EMPTY,
            import_nodes: &["import_header"],
            call_nodes: &["call_expression"],
            call_name_fields: &["callee", "function", "name"],
            import_call_names: EMPTY,
        },
    }
}
