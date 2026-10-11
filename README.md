# workspace-index-mcp

A standalone Rust binary that exposes a local code index through MCP stdio and a CLI. It supports explicit workspace registration, lexical/symbol/semantic retrieval, citations, freshness checks, and optional code-graph extraction. This checkout is at `~/workspace-index-mcp`, separate from the `ai-newsfeed` repository; it does not provide cloud-scale or proprietary Augment parity, and it does not automatically register arbitrary workspaces.
Architecture overview: see [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements and support

- Rust toolchain with Cargo and a C toolchain for native dependencies.
- Linux and macOS implementations support registered-root reading, indexing, and search; this run verified Linux only, so macOS runtime behavior remains unverified. Other platforms fail closed for these filesystem operations. **Source mutations are disabled on every platform**; `changeset_apply` returns `UNSUPPORTED` before any source effect, even after approval or `read_write` mode is selected.
- Git worktrees by default. Non-Git roots can be opted into for read/indexing with `allow_non_git_roots = true`; this does not relax path security.
- The default FastEmbed model is downloaded on first use and needs local disk space and network access for that download. The ignored local model smoke test also downloads/loads the model.
- Optional Graphify runs only on Linux, inside bubblewrap (`bwrap`) namespaces. If bubblewrap or the required sandbox support is unavailable, requested extraction fails closed with `GRAPH_UNAVAILABLE`; normal indexing and search remain available.

## Build, run, and install

Run commands from `workspace-index-mcp/`:

```sh
cargo build --release
./target/release/workspace-index --help
./target/release/workspace-index serve --stdio
```

Install the locked package under the user's local Cargo root:

```sh
cargo install --path . --locked --root "$HOME/.local"
"$HOME/.local/bin/workspace-index" --help
```

The installed absolute executable path is used by the omp installer. The MCP server uses rmcp stdio and MCP protocol version `2025-11-25`; stdout is reserved for protocol frames and diagnostics go to stderr.

## Configuration and durable state

By default, configuration is read from `${XDG_CONFIG_HOME:-~/.config}/workspace-index-mcp/config.toml`. State is under `${XDG_DATA_HOME:-~/.local/share}/workspace-index-mcp/`: `index.db` (SQLite), `vectors/` (rebuildable semantic snapshots), `pending/` (private active changeset payloads), and optional `graphs/` snapshots. Cursors are process-local and are not persisted there. SQLite schema migrations are append-only through version 10. Unregistered changesets retain status, per-file hashes/paths, and audit evidence without source text; payload cleanup is queued durably and retried on startup or `doctor`. Orphan recovery scans at most 10,000 pending-directory entries; overflow or unprovable malformed records remain visible as unresolved in `doctor`. State directories and private payloads are owner-only on Unix. Use `--config PATH` and `--data-dir PATH` to isolate or override these locations.

Minimal defaults are local FastEmbed (`BAAI/bge-small-en-v1.5`), remote providers disabled, Git roots required, five-minute reconciliation, 1 MiB source-file limit, 100 search results maximum, 16 KiB snippets, 512 KiB tool output, and 10-minute prepared-change expiry. Full limits and provider options are declared in `src/config.rs`; an absent configuration file uses these defaults. Example optional settings:

```toml
allow_non_git_roots = false
reconciliation_interval_seconds = 300

[embedding]
provider = "fastembed" # fastembed, ollama, openai_compatible
model = "BAAI/bge-small-en-v1.5"
allow_remote = false
# endpoint = "http://127.0.0.1:11434"
# api_key_env = "WORKSPACE_INDEX_API_KEY"
# revision = "model-revision"
# dimension = 768 # required for remote models without a known dimension
batch_size = 32
timeout_seconds = 30
retry_delays_ms = [250, 1000]

[graphify]
enabled = false
command = "graphify"
timeout_seconds = 120
```

## Register and index workspaces

Registration is explicit, and workspace roots must be existing absolute, non-overlapping directories. Git is required unless opted out. IDs follow `[a-z0-9][a-z0-9_-]{0,63}`. Registration defaults to `read_only`.

```sh
workspace-index workspace add --id engine --root /absolute/path/to/engine --yes
workspace-index workspace add --id tools --root /absolute/path/to/tools --yes
workspace-index workspace list
workspace-index index --workspace engine --full --yes
workspace-index index --workspace tools --yes
workspace-index status --workspace engine
workspace-index doctor
```

Omit `--yes` for an interactive confirmation; non-interactive state-changing CLI commands require it. A running stdio server retains the exclusive data-directory owner lock, so a separate CLI process opening the same state can receive `BUSY`. Watcher stop waits at most two seconds; if the worker has not exited, unregister returns `BUSY` and keeps the workspace registered. Shutdown signals watchers with a shared two-second bound. Every watcher stop queues a durable full reconciliation before releasing its backend, including stops with no locally observed pending work; if queuing fails, watcher status remains dirty with the failure visible. Synchronous provider/filesystem calls cannot be forcibly cancelled; timed-out worker handles remain tracked, and new work for that root/service is refused until they finish. Use the owning MCP session for state commands while its server is running.

Index scanning honors Git ignore files, `.workspace-indexignore`, and `.augmentignore`; excludes `.git`, the data directory, symlinks, unsupported/binary/oversized files, and sensitive-name defaults. Scanner source reads, watcher manifests/reconciliation, retrieval freshness checks, and file reads use the registered-root capability boundary. Parsers include tree-sitter adapters for Python, Java, JavaScript, TypeScript/TSX, Go, Rust, Ruby, PHP, Scala, C/C++, C#, and Kotlin; Markdown/YAML/TOML/SQL/shell use fallback chunking.

## CLI and MCP workflows

CLI provides `workspace add|list|set-mode|remove`, `index`, `search`, `read`, `status`, `doctor`, `serve --stdio`, and `omp-config`. For example:

```sh
workspace-index search "connection pool" --workspace engine --lang rust --path 'src/**' --limit 10
workspace-index read src/lib.rs --workspace engine --start-line 1 --end-line 80
workspace-index index --workspace engine --graphify --yes
```

Search accepts repeatable `--workspace`, `--lang`, and `--path` filters; path globs are relative and multiple patterns match if any matches. `--cursor` continues a page. With one registered workspace, a CLI/MCP request may omit its ID where supported; with multiple workspaces the target must be explicit. MCP exposes the registry's 13 tools, resources, resource subscriptions/updates, and `onboard-workspace` prompt. MCP `changeset_prepare` creates a private proposal and complete bounded diff without touching source. `changeset_status` and `changeset_discard` inspect/discard prepared operations; after unregister, status returns the terminal archived journal and apply/discard are refused. `changeset_apply` remains prompt-gated by policy but currently fails closed before source effects.

## Retrieval, citations, and freshness

Search combines lexical, symbol, and eligible semantic results. Search citations include a BLAKE3 content hash and a separate full-file SHA-256; use the SHA-256 as `file_read.expected_sha256` to check that a cited file still matches. File reads and freshness verification are capability-scoped and reject path escapes and symlinks.

A stale candidate is withheld and durably queued. Search may refresh up to three stale files and retry once within the ten-second deadline. Refresh/retry runs on one tracked worker per service; if synchronous filesystem/provider work outlives the deadline, it may continue in the background and further refresh work is denied until it finishes. `INDEX_DEGRADED` reports unresolved freshness. Branch/HEAD changes gate old-generation search and graph results until a new visible generation is published. Semantic results require current profile/generation coverage; lexical and symbol retrieval can continue with a degraded warning when semantic state is unavailable.

Cursors are authenticated and refer to an in-process frozen ranked candidate set, capped at 1,000 candidates and retained for ten minutes. Generation/profile changes, process restart, cache eviction, or expiry make a cursor stale (`STALE_CURSOR`). A capped result set declares `truncated: true` and ends without a continuation cursor.

## Embedding privacy and Graphify

FastEmbed runs locally. Ollama and OpenAI-compatible providers require a configured endpoint and `embedding.allow_remote = true`; even a local HTTP endpoint requires this opt-in. API credentials are read only from the configured environment-variable name. Remote requests send query/chunk text to the chosen service; do not enable remote embeddings for sensitive source unless that disclosure is acceptable. Provider/model/configuration mismatches fail validation rather than silently selecting another model.

Graphify is optional and not on ordinary indexing/search's critical path. On Linux, enabling it or passing `index --graphify` stages only indexed code in private storage and invokes explicit code-only extraction inside bubblewrap with constrained mounts/environment and a bounded timeout. Publication/pruning rechecks generation, registration epoch, and root identity under the per-root lock; graph paths, cache entries, and cursors are registration-bound, and source hashes are revalidated on reads. Failed current-generation extraction returns no graph nodes. Source workspaces are not used as Graphify output locations. No sandbox means no extraction; Graphify is unavailable on non-Linux platforms in this build.

## omp integration

The static `workspace-index omp-config` prints the MCP entry and policy; `--profile NAME` targets the named profile. `--check` compares the installed MCP entry and owned approval rules. Install with explicit consent:

```sh
workspace-index omp-config --install --yes
workspace-index omp-config --check
workspace-index omp-config --install --yes --profile NAME
workspace-index omp-config --check --profile NAME
```

Installation merges the `workspace-index` server into `mcp.json` and the owned tool decisions into omp's `tools.approval` setting, keeps unrelated entries, writes backups with `.bak` suffixes for files it changes, and uses the absolute current executable path. It invokes the external `omp config set` command for approval rules. The install assumes intentional user config writes and is **not race-proof** against concurrent external config mutation. For a disposable install/check, set `HOME` (and, if desired, XDG variables) to a temporary directory and ensure `omp` is available to that environment; the user's actual default config was not modified during the recorded isolated installer exercise.

Registration, mode changes, removal, and apply are prompt-gated. Approval is not a mutation capability: source mutations remain disabled. Interactive tool-approval E2E through the actual omp UI is unverified. Headless agents can search/read/prepare; prompt-gated calls fail closed when there is no interactive approval.

## Verification and release limitations

Run checks from this package directory when the final implementation is ready:

```sh
cargo test --workspace
cargo test --test embedding_smoke -- --ignored
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build --release
```

Recorded evidence (not a fresh run for this documentation change):

| Check | Evidence |
| --- | --- |
| `cargo test --workspace` | Passed 263 tests across 17 suites with 1 ignored after request-bound cursor fixes. |
| Strict Clippy | Passed: `cargo clippy --workspace --all-targets --all-features -- -D warnings`. |
| Formatting | `cargo fmt --all -- --check` reported extensive rustfmt diffs across the package; no whole-tree reformat was applied. |
| Release build/install | `cargo build --release` and `cargo install --path . --locked --root "$HOME/.local"` passed; installed `--help` worked with minimal PATH from `/tmp` and `/var/tmp`, and an isolated `doctor` run returned `ok:true`, schema version 10, and zero workspaces/cleanup items. |
| Ignored FastEmbed local model smoke | Passed 1/1 in 10.12s; requires model download/cache. |
| Graphify private-staging smoke | Latest Linux bubblewrap suite passed 16/16, including the installed Graphify CLI. |
| Retrieval benchmark | See the experiment table below. Three final 100-sample FTS-first runs met the local 750ms synthetic p95 target; retrieval quality is not measured by this corpus. |
| Retrieval quality | Qrels and retrieval-quality thresholds are unverified; the local synthetic latency result is not a relevance or production-performance guarantee. |
| Platform/write proof | Linux registered-root reads/index/search were exercised. A macOS implementation exists, but macOS runtime behavior and mutation proof are unverified. Mutations remain disabled on every platform. |
| omp interactive approval | Actual interactive tool approval E2E is unverified. The installer was exercised with an isolated temporary HOME; the actual user default omp configuration was not modified. |

All runs use 250,000 chunks, 10 workspaces, 32-dimensional synthetic vectors, top-10 results, 10 warmups, and the same query with one lexical hit per workspace on an Intel i7-9750H/EndeavourOS host. Every warmup/sample validates a successful response with exactly 10 hits. The original mismatched source/chunk corpus run is excluded.

| Variant | Samples | p50 (ms) | p95 (ms) | Result |
| --- | ---: | ---: | ---: | --- |
| Corrected corpus before v10/FTS-first changes | 100 | 27,179.706 | 28,776.046 | Missed 750ms |
| v10 numeric-ID index diagnostic | 10 | 15,009.033 | 15,926.123 | Diagnostic; missed |
| Semantic-ID batch diagnostic | 10 | 14,549.871 | 16,142.016 | Diagnostic; no measured p95 gain |
| Candidate-hydration batch trial (reverted) | 10 | 14,799.655 | 15,971.310 | Neutral; reverted |
| FTS-first query plan with v10 index | 100 | 145.075 | 176.183 | Met 750ms |
| Repeat FTS-first query plan | 100 | 149.566 | 169.146 | Met 750ms |
| FTS-first after full cursor-parameter binding | 100 | 155.510 | 178.048 | Met 750ms |

Do not interpret these checks as full release readiness. No retrieval-quality threshold or production performance guarantee is claimed; the only latency guarantee exercised here is the local synthetic p95 target above. macOS runtime/mutation proof remains unverified, and source writes are disabled on all platforms.
