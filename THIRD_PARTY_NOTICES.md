# Third-party notices

No source code from third-party repositories was copied into this project. Direct Rust dependencies and their declared licenses are listed below; transitive dependencies and exact versions are recorded in `Cargo.lock`. License identifiers were taken from `cargo metadata` for the locked dependency graph.

| Dependency | License |
|---|---|
| `rmcp` | Apache-2.0 |
| `fastembed` | Apache-2.0 |
| `rusqlite` | MIT |
| `usearch` | Apache-2.0 |
| `rustix` | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| `notify` | CC0-1.0 |
| `tree-sitter` and declared language grammars | MIT |
| `codeindex-tree-sitter-langs` (MIT; pinned Git revision `1bf886863362edec0d2eae6653e23aaafe5316ae`) | MIT |
| `serde`, `serde_json`, `schemars`, `tokio`, `uuid`, `sha2`, `hmac`, `blake3`, `fs2`, `ignore`, `parking_lot`, `toml`, `regex`, `dirs`, `thiserror` | MIT and/or Apache-2.0, as declared by each crate in `Cargo.lock` |

The default `BAAI/bge-small-en-v1.5` model is downloaded at runtime by FastEmbed and is not included in this repository. Its model-artifact license and terms are supplied by the upstream model publisher and must be reviewed by the deployer before distribution.
