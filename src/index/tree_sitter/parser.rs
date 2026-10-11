use tree_sitter::{Node, Parser};
use super::{CallRecord, ImportRecord, ParseResult, SourceLanguage, SymbolKind, SymbolRecord};
use super::languages::{config, grammar, GrammarConfig};

pub fn parse_source(source: &str, language: SourceLanguage) -> Result<ParseResult, &'static str> {
    let mut parser = Parser::new();
    parser.set_language(&grammar(language)).map_err(|_| "Failed to set language grammar")?;
    
    let tree = parser.parse(source, None).ok_or("Failed to parse source")?;
    let root = tree.root_node();
    let has_error = root.has_error();
    
    let cfg = config(language);
    let mut symbols = Vec::new();
    let mut imports = Vec::new();
    let mut calls = Vec::new();
    

    let mut stack = vec![(root, None::<String>)];
    
    while let Some((node, parent_name)) = stack.pop() {
        let kind = node.kind();
        let mut next_parent_name = parent_name.clone();
        
        // Match declarations
        if let Some(rule) = cfg.declarations.iter().find(|r| r.node == kind)
            && let Some(name) = extract_name(node, source)
        {
            let actual_kind = if rule.method && parent_name.is_none() && !is_method_container(node.parent(), &cfg) {
                SymbolKind::Function
            } else {
                rule.kind
            };
            
            symbols.push(SymbolRecord {
                name: name.clone(),
                kind: actual_kind,
                parent_name: parent_name.clone(),
                signature: extract_signature(node, source),
                doc_comment: extract_doc_comment(node, source),
                byte_range: node.byte_range(),
                line_range: (node.start_position().row + 1)..(node.end_position().row + 1),
            });
            
            next_parent_name = Some(name);
        }
        
        // Match imports
        if cfg.import_nodes.contains(&kind)
            && let Some(text) = extract_import_text(node, source)
        {
            imports.push(ImportRecord {
                name: text,
                byte_range: node.byte_range(),
                line_range: (node.start_position().row + 1)..(node.end_position().row + 1),
            });
        }
        
        // Match calls
        if cfg.call_nodes.contains(&kind) {
            if let Some(name) = extract_call_name(node, source, &cfg) {
                calls.push(CallRecord {
                    name,
                    byte_range: node.byte_range(),
                    line_range: (node.start_position().row + 1)..(node.end_position().row + 1),
                });
            }
        } else if cfg.import_call_names.contains(&kind)
            && let Some(name) = extract_call_name(node, source, &cfg)
        {
            calls.push(CallRecord {
                name: name.clone(),
                byte_range: node.byte_range(),
                line_range: (node.start_position().row + 1)..(node.end_position().row + 1),
            });
            imports.push(ImportRecord {
                name,
                byte_range: node.byte_range(),
                line_range: (node.start_position().row + 1)..(node.end_position().row + 1),
            });
        }
        
        // Match function bindings (JS/TS var = () => {})
        if cfg.function_binding_nodes.contains(&kind)
            && let Some(name) = extract_name(node, source)
            && has_function_value(node, &cfg)
        {
            symbols.push(SymbolRecord {
                name: name.clone(),
                kind: SymbolKind::Function,
                parent_name: parent_name.clone(),
                signature: extract_signature(node, source),
                doc_comment: extract_doc_comment(node, source),
                byte_range: node.byte_range(),
                line_range: (node.start_position().row + 1)..(node.end_position().row + 1),
            });
            next_parent_name = Some(name);
        }
        
        // Handle scope nodes (e.g. Go/Rust impl blocks)
        for (scope_node, name_field) in cfg.scope_nodes {
            if kind == *scope_node
                && let Some(name_node) = node.child_by_field_name(name_field)
                && let Ok(name) = name_node.utf8_text(source.as_bytes())
            {
                next_parent_name = Some(name.to_string());
            }
        }
        
        // Push children in reverse order so they are processed in source order
        let mut children = Vec::new();
        let mut child_cursor = node.walk();
        for child in node.children(&mut child_cursor) {
            children.push(child);
        }
        for child in children.into_iter().rev() {
            stack.push((child, next_parent_name.clone()));
        }
    }
    
    Ok(ParseResult {
        symbols,
        imports,
        calls,
        has_error,
    })
}

fn extract_name(node: Node, source: &str) -> Option<String> {
    if let Some(name_node) = node.child_by_field_name("name") {
        return name_node.utf8_text(source.as_bytes()).ok().map(String::from);
    }
    if let Some(name_node) = node.child_by_field_name("declarator") {
        if let Some(id_node) = name_node.child_by_field_name("name") {
            return id_node.utf8_text(source.as_bytes()).ok().map(String::from);
        }
        return name_node.utf8_text(source.as_bytes()).ok().map(String::from);
    }
    None
}

fn extract_signature(node: Node, source: &str) -> Option<String> {
    let mut sig_end = node.start_byte();
    let mut found_params = false;
    
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if kind == "parameters" || kind == "formal_parameters" || kind == "parameter_list" {
            sig_end = child.end_byte();
            found_params = true;
        } else if found_params && (kind == "return_type" || kind == "type_annotation" || kind == "result") {
            sig_end = child.end_byte();
        } else if kind == "block" || kind == "statement_block" || kind == "function_body" {
            break;
        }
    }
    
    if sig_end > node.start_byte() {
        let sig_bytes = &source.as_bytes()[node.start_byte()..sig_end];
        if let Ok(sig) = std::str::from_utf8(sig_bytes) {
            return Some(sig.trim().to_string());
        }
    }
    None
}

fn extract_doc_comment(node: Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut comments = Vec::new();
    
    while let Some(sibling) = prev {
        if sibling.kind() == "comment" || sibling.kind() == "line_comment" || sibling.kind() == "block_comment" {
            if let Ok(text) = sibling.utf8_text(source.as_bytes()) {
                comments.push(text.to_string());
            }
            prev = sibling.prev_sibling();
        } else {
            break;
        }
    }
    
    if comments.is_empty() {
        // Python/Ruby docstrings are often the first child expression
        let mut cursor = node.walk();
        if let Some(body) = node.children(&mut cursor).find(|c| c.kind() == "block" || c.kind() == "body")
            && let Some(first) = body.child(0)
            && first.kind() == "expression_statement"
            && let Some(string_node) = first.child(0)
            && string_node.kind() == "string"
            && let Ok(text) = string_node.utf8_text(source.as_bytes())
        {
            return Some(text.to_string());
        }
        return None;
    }
    
    comments.reverse();
    Some(comments.join("\n"))
}

fn extract_import_text(node: Node, source: &str) -> Option<String> {
    node.utf8_text(source.as_bytes()).ok().map(|s| s.trim().to_string())
}

fn extract_call_name(node: Node, source: &str, cfg: &GrammarConfig) -> Option<String> {
    for field in cfg.call_name_fields {
        if let Some(name_node) = node.child_by_field_name(field)
            && let Ok(text) = name_node.utf8_text(source.as_bytes())
        {
            return Some(text.to_string());
        }
    }
    
    // Fallback for languages without named call fields
    if let Some(first) = node.child(0)
        && matches!(first.kind(), "identifier" | "scoped_identifier" | "field_identifier")
        && let Ok(text) = first.utf8_text(source.as_bytes())
    {
        return Some(text.to_string());
    }
    None
}

fn is_method_container(parent: Option<Node>, cfg: &GrammarConfig) -> bool {
    if let Some(p) = parent {
        cfg.method_containers.contains(&p.kind())
    } else {
        false
    }
}

fn has_function_value(node: Node, cfg: &GrammarConfig) -> bool {
    if let Some(value) = node.child_by_field_name("value") {
        cfg.function_value_nodes.contains(&value.kind())
    } else {
        false
    }
}
