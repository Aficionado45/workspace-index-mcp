use std::path::Path;
use workspace_index_mcp::index::scanner::build_scanner;

#[test]
fn test_scanner() {
    let root = Path::new("/tmp/workspace");
    let excluded = Path::new("/tmp/workspace/.private-index");
    std::fs::create_dir_all(excluded).unwrap();
    std::fs::write(root.join("source.rs"), "fn visible() {}").unwrap();
    std::fs::write(excluded.join("private.rs"), "fn hidden_state() {}").unwrap();
    
    let scanner = build_scanner(root, excluded).unwrap();
    for entry in scanner {
        println!("{:?}", entry);
    }
}
