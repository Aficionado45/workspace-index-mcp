use rusqlite::Connection;
use crate::state::Result;

pub const CURRENT_SCHEMA_VERSION: i32 = 10;

pub fn initialize(conn: &mut Connection) -> Result<()> {
    // Enable WAL and foreign keys
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "synchronous", "FULL")?;

    let tx = conn.transaction()?;
    
    // Check current version
    let user_version: i32 = tx.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    
    if user_version > CURRENT_SCHEMA_VERSION {
        return Err(crate::state::StateError::UnsupportedSchema(user_version));
    }
    
    if user_version < 1 {
        // V1 Schema: Workspaces and base persistence
        tx.execute_batch(
            r#"
            CREATE TABLE workspaces (
                id TEXT PRIMARY KEY,
                root_path TEXT NOT NULL,
                device_id INTEGER NOT NULL,
                inode INTEGER NOT NULL,
                mode TEXT NOT NULL,
                registration_epoch INTEGER NOT NULL,
                mode_epoch INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            
            CREATE UNIQUE INDEX idx_workspaces_root ON workspaces(root_path);
            
            -- HNSW profiles
            CREATE TABLE embedding_profiles (
                id TEXT PRIMARY KEY,
                provider_kind TEXT NOT NULL,
                endpoint TEXT,
                model TEXT NOT NULL,
                revision TEXT,
                dimension INTEGER NOT NULL,
                metric TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            
            -- Files
            CREATE TABLE files (
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                relative_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                visible_generation INTEGER NOT NULL,
                PRIMARY KEY (workspace_id, relative_path)
            );
            
            -- Chunks
            CREATE TABLE chunks (
                workspace_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                symbol_context TEXT,
                FOREIGN KEY (workspace_id, relative_path) REFERENCES files(workspace_id, relative_path) ON DELETE CASCADE,
                PRIMARY KEY (workspace_id, relative_path, chunk_index)
            );
            
            -- FTS
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                symbol_context,
                content='chunks',
                content_rowid='rowid'
            );
            
            -- Vector ops outbox
            CREATE TABLE vector_ops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
                generation INTEGER NOT NULL,
                op_type TEXT NOT NULL,
                vector_id TEXT NOT NULL,
                embedding BLOB
            );
            
            -- Changesets
            CREATE TABLE pending_changesets (
                operation_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                digest TEXT NOT NULL,
                state TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );
            "#
        )?;
        
        tx.pragma_update(None, "user_version", 1)?;
    }

    if user_version < 2 {
        // V2: durable file generations, parser metadata, semantic embeddings,
        // dependency candidates, and trigger-maintained lexical search.
        tx.execute_batch(
            r#"
            ALTER TABLE files ADD COLUMN file_generation INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE files ADD COLUMN parser_version INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE files ADD COLUMN chunker_version INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE chunks ADD COLUMN language TEXT;
            ALTER TABLE chunks ADD COLUMN chunk_hash TEXT;

            CREATE TABLE workspace_index_state (
                workspace_id TEXT PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
                generation INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0
            );

            INSERT INTO workspace_index_state (workspace_id, generation, updated_at)
            SELECT w.id,
                   MAX(
                       COALESCE((SELECT MAX(f.visible_generation) FROM files f WHERE f.workspace_id = w.id), 0),
                       COALESCE((SELECT MAX(v.generation) FROM vector_ops v WHERE v.workspace_id = w.id), 0)
                   ),
                   0
            FROM workspaces w;

            CREATE TABLE chunk_embeddings (
                workspace_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                profile_id TEXT NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
                generation INTEGER NOT NULL,
                vector_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (workspace_id, relative_path, chunk_index, profile_id),
                FOREIGN KEY (workspace_id, relative_path, chunk_index)
                    REFERENCES chunks(workspace_id, relative_path, chunk_index) ON DELETE CASCADE
            );

            CREATE INDEX idx_chunk_embeddings_profile
                ON chunk_embeddings(workspace_id, profile_id, generation);
            CREATE INDEX idx_vector_ops_generation
                ON vector_ops(workspace_id, profile_id, generation, id);

            CREATE TABLE symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                symbol_index INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                parent_name TEXT,
                signature TEXT,
                doc_comment TEXT,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                FOREIGN KEY (workspace_id, relative_path)
                    REFERENCES files(workspace_id, relative_path) ON DELETE CASCADE,
                UNIQUE (workspace_id, relative_path, symbol_index)
            );

            CREATE INDEX idx_symbols_name ON symbols(workspace_id, name);

            CREATE TABLE dependency_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                edge_index INTEGER NOT NULL,
                evidence TEXT NOT NULL,
                name TEXT NOT NULL,
                source_symbol TEXT,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                FOREIGN KEY (workspace_id, relative_path)
                    REFERENCES files(workspace_id, relative_path) ON DELETE CASCADE,
                UNIQUE (workspace_id, relative_path, edge_index)
            );

            CREATE INDEX idx_dependency_edges_name
                ON dependency_edges(workspace_id, name, evidence);

            CREATE TRIGGER chunks_fts_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, symbol_context)
                VALUES (new.rowid, new.content, new.symbol_context);
            END;

            CREATE TRIGGER chunks_fts_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_context)
                VALUES ('delete', old.rowid, old.content, old.symbol_context);
            END;

            CREATE TRIGGER chunks_fts_au AFTER UPDATE OF content, symbol_context ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_context)
                VALUES ('delete', old.rowid, old.content, old.symbol_context);
                INSERT INTO chunks_fts(rowid, content, symbol_context)
                VALUES (new.rowid, new.content, new.symbol_context);
            END;

            INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild');
            "#
        )?;

        tx.pragma_update(None, "user_version", 2)?;
    }
    
    if user_version < 3 {
        tx.execute_batch(
            r#"
            ALTER TABLE pending_changesets ADD COLUMN registration_epoch INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE pending_changesets ADD COLUMN mode_epoch INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE pending_changesets ADD COLUMN root_device INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE pending_changesets ADD COLUMN root_inode INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE pending_changesets ADD COLUMN head_ref TEXT;
            ALTER TABLE pending_changesets ADD COLUMN head_commit TEXT;
            ALTER TABLE pending_changesets ADD COLUMN payload_path TEXT;

            CREATE TABLE changeset_files (
                operation_id TEXT NOT NULL REFERENCES pending_changesets(operation_id) ON DELETE CASCADE,
                file_index INTEGER NOT NULL,
                op_kind TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                destination_path TEXT,
                base_sha256 TEXT,
                proposed_sha256 TEXT,
                observed_sha256 TEXT,
                effect_state TEXT NOT NULL DEFAULT 'pending',
                intent_at INTEGER,
                observed_at INTEGER,
                PRIMARY KEY (operation_id, file_index)
            );

            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                operation_id TEXT,
                event_type TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX idx_audit_workspace_time ON audit_events(workspace_id, created_at);
            CREATE INDEX idx_changeset_files_state ON changeset_files(operation_id, effect_state);
            CREATE TABLE index_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                relative_path TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 1,
                state TEXT NOT NULL DEFAULT 'queued',
                created_at INTEGER NOT NULL,
                UNIQUE (workspace_id, relative_path, state)
            );
            "#
        )?;
        tx.pragma_update(None, "user_version", 3)?;
    }
    if user_version < 4 {
        tx.execute_batch(
            r#"
            ALTER TABLE workspaces ADD COLUMN git_dir TEXT;
            ALTER TABLE workspaces ADD COLUMN git_common_dir TEXT;
            ALTER TABLE workspaces ADD COLUMN git_head_ref TEXT;
            ALTER TABLE workspaces ADD COLUMN git_head_commit TEXT;

            ALTER TABLE workspace_index_state
                ADD COLUMN building_generation INTEGER NOT NULL DEFAULT 0
                CHECK (building_generation >= 0);
            ALTER TABLE workspace_index_state
                ADD COLUMN visible_generation INTEGER NOT NULL DEFAULT 0
                CHECK (visible_generation >= 0 AND visible_generation <= building_generation);
            ALTER TABLE workspace_index_state
                ADD COLUMN dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0, 1));
            ALTER TABLE workspace_index_state
                ADD COLUMN health TEXT NOT NULL DEFAULT 'unknown';
            ALTER TABLE workspace_index_state
                ADD COLUMN desired_profile_id TEXT REFERENCES embedding_profiles(id) ON DELETE SET NULL;
            ALTER TABLE workspace_index_state
                ADD COLUMN active_profile_id TEXT REFERENCES embedding_profiles(id) ON DELETE SET NULL;

            UPDATE workspace_index_state
            SET generation = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM files f
                        WHERE f.workspace_id = workspace_index_state.workspace_id
                    ) THEN MAX(0, COALESCE((
                        SELECT MAX(f.visible_generation) FROM files f
                        WHERE f.workspace_id = workspace_index_state.workspace_id
                    ), 0))
                    ELSE MAX(generation, 0)
                END,
                visible_generation = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM files f
                        WHERE f.workspace_id = workspace_index_state.workspace_id
                    ) THEN MAX(0, COALESCE((
                        SELECT MAX(f.visible_generation) FROM files f
                        WHERE f.workspace_id = workspace_index_state.workspace_id
                    ), 0))
                    ELSE MAX(generation, 0)
                END,
                building_generation = MAX(
                    0,
                    generation,
                    COALESCE((
                        SELECT MAX(f.visible_generation) FROM files f
                        WHERE f.workspace_id = workspace_index_state.workspace_id
                    ), 0),
                    COALESCE((
                        SELECT MAX(v.generation) FROM vector_ops v
                        WHERE v.workspace_id = workspace_index_state.workspace_id
                    ), 0)
                ),
                dirty = 1,
                health = 'reconcile_required',
                desired_profile_id = (
                    SELECT id FROM embedding_profiles
                    WHERE lower(state) = 'active'
                    ORDER BY created_at DESC, id LIMIT 1
                ),
                active_profile_id = (
                    SELECT id FROM embedding_profiles
                    WHERE lower(state) = 'active'
                    ORDER BY created_at DESC, id LIMIT 1
                );

            CREATE TABLE profile_applied_generations (
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
                applied_generation INTEGER NOT NULL DEFAULT 0 CHECK (applied_generation >= 0),
                backfill_required INTEGER NOT NULL DEFAULT 0 CHECK (backfill_required IN (0, 1)),
                updated_at INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (workspace_id, profile_id)
            );

            INSERT INTO profile_applied_generations (
                workspace_id, profile_id, applied_generation, backfill_required, updated_at
            )
            SELECT pairs.workspace_id,
                   pairs.profile_id,
                   0,
                   CASE
                       WHEN EXISTS (
                           SELECT 1 FROM chunks c
                           WHERE c.workspace_id = pairs.workspace_id
                       ) OR EXISTS (
                           SELECT 1 FROM chunk_embeddings e
                           WHERE e.workspace_id = pairs.workspace_id
                             AND e.profile_id = pairs.profile_id
                       ) OR EXISTS (
                           SELECT 1 FROM vector_ops v
                           WHERE v.workspace_id = pairs.workspace_id
                             AND v.profile_id = pairs.profile_id
                       ) THEN 1
                       ELSE 0
                   END,
                   0
            FROM (
                SELECT workspace_id, profile_id FROM chunk_embeddings
                UNION
                SELECT workspace_id, profile_id FROM vector_ops
                UNION
                SELECT workspace_id, active_profile_id FROM workspace_index_state
                WHERE active_profile_id IS NOT NULL
                UNION
                SELECT workspace_id, desired_profile_id FROM workspace_index_state
                WHERE desired_profile_id IS NOT NULL
            ) AS pairs;

            CREATE TABLE stable_vector_ids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
                vector_id TEXT NOT NULL,
                UNIQUE (workspace_id, profile_id, vector_id)
            );

            INSERT INTO stable_vector_ids (workspace_id, profile_id, vector_id)
            SELECT workspace_id, profile_id, vector_id FROM chunk_embeddings
            UNION
            SELECT workspace_id, profile_id, vector_id FROM vector_ops;

            ALTER TABLE chunk_embeddings
                ADD COLUMN numeric_vector_id INTEGER
                REFERENCES stable_vector_ids(id) ON DELETE SET NULL;
            ALTER TABLE vector_ops
                ADD COLUMN numeric_vector_id INTEGER
                REFERENCES stable_vector_ids(id) ON DELETE SET NULL;

            UPDATE chunk_embeddings
            SET numeric_vector_id = (
                SELECT id FROM stable_vector_ids
                WHERE stable_vector_ids.workspace_id = chunk_embeddings.workspace_id
                  AND stable_vector_ids.profile_id = chunk_embeddings.profile_id
                  AND stable_vector_ids.vector_id = chunk_embeddings.vector_id
            );
            UPDATE vector_ops
            SET numeric_vector_id = (
                SELECT id FROM stable_vector_ids
                WHERE stable_vector_ids.workspace_id = vector_ops.workspace_id
                  AND stable_vector_ids.profile_id = vector_ops.profile_id
                  AND stable_vector_ids.vector_id = vector_ops.vector_id
            );

            CREATE TRIGGER chunk_embeddings_numeric_vector_id_ai
            AFTER INSERT ON chunk_embeddings BEGIN
                INSERT OR IGNORE INTO stable_vector_ids (workspace_id, profile_id, vector_id)
                VALUES (NEW.workspace_id, NEW.profile_id, NEW.vector_id);
                UPDATE chunk_embeddings
                SET numeric_vector_id = (
                    SELECT id FROM stable_vector_ids
                    WHERE workspace_id = NEW.workspace_id
                      AND profile_id = NEW.profile_id
                      AND vector_id = NEW.vector_id
                )
                WHERE workspace_id = NEW.workspace_id
                  AND relative_path = NEW.relative_path
                  AND chunk_index = NEW.chunk_index
                  AND profile_id = NEW.profile_id;
            END;

            CREATE TRIGGER chunk_embeddings_numeric_vector_id_au
            AFTER UPDATE OF workspace_id, profile_id, vector_id ON chunk_embeddings BEGIN
                INSERT OR IGNORE INTO stable_vector_ids (workspace_id, profile_id, vector_id)
                VALUES (NEW.workspace_id, NEW.profile_id, NEW.vector_id);
                UPDATE chunk_embeddings
                SET numeric_vector_id = (
                    SELECT id FROM stable_vector_ids
                    WHERE workspace_id = NEW.workspace_id
                      AND profile_id = NEW.profile_id
                      AND vector_id = NEW.vector_id
                )
                WHERE workspace_id = NEW.workspace_id
                  AND relative_path = NEW.relative_path
                  AND chunk_index = NEW.chunk_index
                  AND profile_id = NEW.profile_id;
            END;

            CREATE TRIGGER vector_ops_numeric_vector_id_ai
            AFTER INSERT ON vector_ops BEGIN
                INSERT OR IGNORE INTO stable_vector_ids (workspace_id, profile_id, vector_id)
                VALUES (NEW.workspace_id, NEW.profile_id, NEW.vector_id);
                UPDATE vector_ops
                SET numeric_vector_id = (
                    SELECT id FROM stable_vector_ids
                    WHERE workspace_id = NEW.workspace_id
                      AND profile_id = NEW.profile_id
                      AND vector_id = NEW.vector_id
                )
                WHERE id = NEW.id;
            END;

            CREATE TRIGGER vector_ops_numeric_vector_id_au
            AFTER UPDATE OF workspace_id, profile_id, vector_id ON vector_ops BEGIN
                INSERT OR IGNORE INTO stable_vector_ids (workspace_id, profile_id, vector_id)
                VALUES (NEW.workspace_id, NEW.profile_id, NEW.vector_id);
                UPDATE vector_ops
                SET numeric_vector_id = (
                    SELECT id FROM stable_vector_ids
                    WHERE workspace_id = NEW.workspace_id
                      AND profile_id = NEW.profile_id
                      AND vector_id = NEW.vector_id
                )
                WHERE id = NEW.id;
            END;

            INSERT OR IGNORE INTO index_jobs (
                workspace_id, relative_path, priority, state, created_at
            )
            SELECT workspace_id, relative_path, 0, 'queued', 0 FROM files;

            CREATE TRIGGER workspace_index_generations_monotonic
            BEFORE UPDATE OF building_generation, visible_generation ON workspace_index_state
            WHEN NEW.building_generation < OLD.building_generation
              OR NEW.visible_generation < OLD.visible_generation
            BEGIN
                SELECT RAISE(ABORT, 'workspace generation watermarks cannot decrease');
            END;

            CREATE INDEX idx_profile_applied_generation
                ON profile_applied_generations(workspace_id, profile_id, applied_generation);
            CREATE INDEX idx_index_jobs_queued
                ON index_jobs(workspace_id, state, priority, id);
            "#
        )?;
        tx.pragma_update(None, "user_version", 4)?;
    }
    if user_version < 5 {
        tx.execute_batch(
            r#"
            CREATE TABLE workspace_registration_epochs (
                id TEXT PRIMARY KEY,
                registration_epoch INTEGER NOT NULL CHECK (registration_epoch >= 1),
                mode_epoch INTEGER NOT NULL CHECK (mode_epoch >= 1)
            );

            INSERT INTO workspace_registration_epochs (id, registration_epoch, mode_epoch)
            SELECT id, registration_epoch, mode_epoch FROM workspaces;
            "#,
        )?;
        tx.pragma_update(None, "user_version", 5)?;
    }
    if user_version < 6 {
        tx.execute_batch(
            r#"
            ALTER TABLE embedding_profiles
                ADD COLUMN api_key_env TEXT;
            ALTER TABLE embedding_profiles
                ADD COLUMN query_template_version TEXT NOT NULL DEFAULT 'query-v1';
            ALTER TABLE embedding_profiles
                ADD COLUMN document_template_version TEXT NOT NULL DEFAULT 'document-v1';
            ALTER TABLE embedding_profiles
                ADD COLUMN batch_size INTEGER NOT NULL DEFAULT 32;
            ALTER TABLE embedding_profiles
                ADD COLUMN timeout_seconds INTEGER NOT NULL DEFAULT 30;
            ALTER TABLE embedding_profiles
                ADD COLUMN retry_delays_ms TEXT NOT NULL DEFAULT '[250,1000]';
            UPDATE embedding_profiles
            SET revision=NULL
            WHERE id='local-fastembed'
              AND provider_kind='fastembed'
              AND model='BAAI/bge-small-en-v1.5';

            CREATE TRIGGER embedding_profiles_immutable_identity
            BEFORE UPDATE OF id, provider_kind, endpoint, model, revision, dimension, metric,
                             api_key_env, query_template_version, document_template_version,
                             batch_size, timeout_seconds, retry_delays_ms
            ON embedding_profiles
            WHEN NEW.id IS NOT OLD.id
              OR NEW.provider_kind IS NOT OLD.provider_kind
              OR NEW.endpoint IS NOT OLD.endpoint
              OR NEW.model IS NOT OLD.model
              OR NEW.revision IS NOT OLD.revision
              OR NEW.dimension IS NOT OLD.dimension
              OR NEW.metric IS NOT OLD.metric
              OR NEW.api_key_env IS NOT OLD.api_key_env
              OR NEW.query_template_version IS NOT OLD.query_template_version
              OR NEW.document_template_version IS NOT OLD.document_template_version
              OR NEW.batch_size IS NOT OLD.batch_size
              OR NEW.timeout_seconds IS NOT OLD.timeout_seconds
              OR NEW.retry_delays_ms IS NOT OLD.retry_delays_ms
            BEGIN
                SELECT RAISE(ABORT, 'embedding profile identity is immutable');
            END;
            "#
        )?;
        tx.pragma_update(None, "user_version", 6)?;
    }
    if user_version < 7 {
        tx.execute_batch(
            r#"
            ALTER TABLE workspace_index_state
                ADD COLUMN branch_transition INTEGER NOT NULL DEFAULT 0
                CHECK (branch_transition IN (0, 1));
            "#
        )?;
        tx.pragma_update(None, "user_version", 7)?;
    }
    if user_version < 8 {
        tx.execute_batch(
            r#"
            CREATE TABLE unregistered_changesets (
                operation_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                digest TEXT NOT NULL,
                state TEXT NOT NULL,
                archive_json TEXT NOT NULL,
                archived_at INTEGER NOT NULL
            );

            CREATE TRIGGER unregistered_changesets_no_update
            BEFORE UPDATE ON unregistered_changesets
            BEGIN
                SELECT RAISE(ABORT, 'unregistered changeset archives are append-only');
            END;

            CREATE TRIGGER unregistered_changesets_no_delete
            BEFORE DELETE ON unregistered_changesets
            BEGIN
                SELECT RAISE(ABORT, 'unregistered changeset archives are append-only');
            END;

            CREATE TABLE changeset_payload_cleanup (
                operation_id TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                queued_at INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );
            CREATE INDEX idx_changeset_payload_cleanup_queued
                ON changeset_payload_cleanup(queued_at, operation_id);
            "#
        )?;
        tx.pragma_update(None, "user_version", 8)?;
    }
    if user_version < 9 {
        tx.execute_batch(
            r#"
            ALTER TABLE changeset_payload_cleanup
                ADD COLUMN recovery_target_id TEXT;
            "#
        )?;
        tx.pragma_update(None, "user_version", 9)?;
    }
    if user_version < 10 {
        tx.execute_batch(
            r#"
            CREATE INDEX idx_chunk_embeddings_semantic_lookup
                ON chunk_embeddings(workspace_id, profile_id, numeric_vector_id, generation);
            "#
        )?;
        tx.pragma_update(None, "user_version", 10)?;
    }

    tx.commit()?;
    
    Ok(())
}
