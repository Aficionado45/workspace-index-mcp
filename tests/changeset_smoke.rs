use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tempfile::TempDir;
use workspace_index_mcp::changeset::{ChangeOperation, ChangesetService, Replacement};
use workspace_index_mcp::model::ErrorCode;
use workspace_index_mcp::state::schema;

fn hash(value: &[u8]) -> String { Sha256::digest(value).iter().map(|byte| format!("{byte:02x}")).collect() }

use parking_lot::Mutex;

fn setup() -> (TempDir, Mutex<Connection>, PathBuf) {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("repo");
    std::fs::create_dir(&root).unwrap();
    let db_path = temp.path().join("index.db");
    let mut db = Connection::open(db_path).unwrap();
    schema::initialize(&mut db).unwrap();
    let metadata = std::fs::metadata(&root).unwrap();
    #[cfg(unix)] {
        use std::os::unix::fs::MetadataExt;
        db.execute("INSERT INTO workspaces (id, root_path, device_id, inode, mode, registration_epoch, mode_epoch, created_at, updated_at) VALUES ('repo', ?1, ?2, ?3, 'read_write', 1, 1, 0, 0)", params![root.to_string_lossy(), metadata.dev() as i64, metadata.ino() as i64]).unwrap();
    }
    let pending = temp.path().join("pending");
    (temp, Mutex::new(db), pending)
}

#[test]
fn prepare_does_not_write_and_apply_is_explicitly_unsupported() {
    let (_temp, db, pending) = setup();
    let service = ChangesetService::new(&db, &pending).unwrap();
    let prepared = service.prepare("repo", vec![ChangeOperation::Create {
        path: "new.rs".into(), content: "fn ready() {}\n".into(),
    }]).unwrap();
    let workspace_file = pending.parent().unwrap().join("repo/new.rs");
    assert!(!workspace_file.exists());
    assert!(prepared.unified_diff.contains("+++ b/new.rs"));
    assert_eq!(service.apply(&prepared.operation_id, &prepared.digest).unwrap_err().code, ErrorCode::Unsupported);
    assert_eq!(service.status(&prepared.operation_id).unwrap().state, "Prepared");
    assert!(!workspace_file.exists());
}

#[test]
fn status_expires_prepared_changeset_and_cleans_payload_in_live_service() {
    let (_temp, db, pending) = setup();
    let service = ChangesetService::new(&db, &pending).unwrap();
    let prepared = service.prepare("repo", vec![ChangeOperation::Create {
        path: "new.rs".into(), content: "fn ready() {}\n".into(),
    }]).unwrap();
    let payload = pending.join(format!("{}.json", prepared.operation_id));
    assert!(payload.exists());

    db.lock().execute(
        "UPDATE pending_changesets SET expires_at=0 WHERE operation_id=?1",
        [&prepared.operation_id],
    ).unwrap();

    let status = service.status(&prepared.operation_id).unwrap();

    assert_eq!(status.state, "Expired");
    assert_eq!(service.apply(&prepared.operation_id, &prepared.digest).unwrap().state, "Expired");
    assert!(!pending.parent().unwrap().join("repo/new.rs").exists());
    assert!(!payload.exists());
    let queued: i64 = db.lock().query_row(
        "SELECT count(*) FROM changeset_payload_cleanup WHERE operation_id=?1",
        [&prepared.operation_id],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(queued, 0);
}

#[test]
fn replace_delete_and_move_remain_unchanged_without_a_safe_backend() {
    let (temp, db, pending) = setup();
    let root = temp.path().join("repo");
    std::fs::create_dir(root.join("src")).unwrap();
    std::fs::create_dir(root.join("dst")).unwrap();
    std::fs::write(root.join("src/a.rs"), "one\ntwo\n").unwrap();
    let expected = hash(b"one\ntwo\n");
    let service = ChangesetService::new(&db, &pending).unwrap();

    let replace = service.prepare("repo", vec![ChangeOperation::Replace {
        path: "src/a.rs".into(), expected_sha256: expected.clone(),
        replacements: vec![Replacement { old_text: "two".into(), new_text: "TWO".into(), replace_all: false }],
    }]).unwrap();
    assert_eq!(service.apply(&replace.operation_id, &replace.digest).unwrap_err().code, ErrorCode::Unsupported);
    assert_eq!(std::fs::read_to_string(root.join("src/a.rs")).unwrap(), "one\ntwo\n");

    let moved = service.prepare("repo", vec![ChangeOperation::Move {
        from_path: "src/a.rs".into(), to_path: "dst/a.rs".into(), expected_sha256: expected.clone(),
    }]).unwrap();
    assert_eq!(service.apply(&moved.operation_id, &moved.digest).unwrap_err().code, ErrorCode::Unsupported);
    assert_eq!(std::fs::read_to_string(root.join("src/a.rs")).unwrap(), "one\ntwo\n");
    assert!(!root.join("dst/a.rs").exists());

    let delete = service.prepare("repo", vec![ChangeOperation::Delete {
        path: "src/a.rs".into(), expected_sha256: expected,
    }]).unwrap();
    assert_eq!(service.apply(&delete.operation_id, &delete.digest).unwrap_err().code, ErrorCode::Unsupported);
    assert_eq!(std::fs::read_to_string(root.join("src/a.rs")).unwrap(), "one\ntwo\n");
}

#[test]
fn create_and_move_require_existing_destination_parent() {
    let (temp, db, pending) = setup();
    let root = temp.path().join("repo");
    std::fs::write(root.join("source.txt"), "source").unwrap();
    let service = ChangesetService::new(&db, &pending).unwrap();

    assert!(service.prepare("repo", vec![ChangeOperation::Create {
        path: "missing/new.txt".into(), content: "new".into(),
    }]).is_err());
    assert!(service.prepare("repo", vec![ChangeOperation::Move {
        from_path: "source.txt".into(), to_path: "missing/moved.txt".into(),
        expected_sha256: hash(b"source"),
    }]).is_err());
    assert!(!root.join("missing").exists());
    assert_eq!(std::fs::read_to_string(root.join("source.txt")).unwrap(), "source");
}


#[test]
fn prepare_refuses_wrong_preimage_duplicate_old_text_and_symlink() {
    let (temp, db, pending) = setup();
    let root = temp.path().join("repo");
    std::fs::write(root.join("code.rs"), "foo foo").unwrap();
    std::fs::write(root.join("outside"), "secret").unwrap();
    #[cfg(unix)] std::os::unix::fs::symlink(root.join("outside"), root.join("link.rs")).unwrap();
    let service = ChangesetService::new(&db, &pending).unwrap();
    assert!(service.prepare("repo", vec![ChangeOperation::Replace { path: "code.rs".into(), expected_sha256: "wrong".into(), replacements: vec![] }]).is_err());
    assert!(service.prepare("repo", vec![ChangeOperation::Replace { path: "code.rs".into(), expected_sha256: hash(b"foo foo"), replacements: vec![Replacement { old_text: "foo".into(), new_text: "bar".into(), replace_all: false }] }]).is_err());
    assert!(service.prepare("repo", vec![ChangeOperation::Delete { path: "link.rs".into(), expected_sha256: hash(b"secret") }]).is_err());
    assert_eq!(std::fs::read_to_string(root.join("outside")).unwrap(), "secret");
}

#[test]
fn prepare_refuses_sensitive_path_components_without_exposing_payloads() {
    let (temp, db, pending) = setup();
    let root = temp.path().join("repo");
    let sentinel = "private content sentinel";
    std::fs::create_dir(root.join("nested")).unwrap();
    for path in [".env", "nested/credentials.json", "server.pem", "id_rsa"] {
        std::fs::write(root.join(path), sentinel).unwrap();
    }
    let service = ChangesetService::new(&db, &pending).unwrap();

    for path in [".env", "nested/credentials.json", "server.pem", "id_rsa"] {
        let error = service.prepare("repo", vec![ChangeOperation::Replace {
            path: path.into(),
            expected_sha256: hash(sentinel.as_bytes()),
            replacements: vec![Replacement {
                old_text: sentinel.into(),
                new_text: "replacement".into(),
                replace_all: false,
            }],
        }]).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        assert!(!error.to_string().contains(sentinel));
        assert_eq!(std::fs::read_to_string(root.join(path)).unwrap(), sentinel);
    }
    assert!(std::fs::read_dir(pending).unwrap().next().is_none());
}

#[cfg(unix)]
#[test]
fn prepare_refuses_hard_link_alias_of_sensitive_file() {
    let (temp, db, pending) = setup();
    let root = temp.path().join("repo");
    std::fs::create_dir_all(root.join("src")).unwrap();
    let secret = b"private hard-link content";
    std::fs::write(root.join(".env"), secret).unwrap();
    std::fs::hard_link(root.join(".env"), root.join("src/config.rs")).unwrap();
    let service = ChangesetService::new(&db, &pending).unwrap();

    let error = service.prepare("repo", vec![ChangeOperation::Replace {
        path: "src/config.rs".into(),
        expected_sha256: hash(secret),
        replacements: vec![Replacement {
            old_text: "private".into(),
            new_text: "public".into(),
            replace_all: false,
        }],
    }]).unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(!error.to_string().contains("private hard-link content"));
    assert_eq!(std::fs::read(root.join(".env")).unwrap(), secret);
    assert!(std::fs::read_dir(pending).unwrap().next().is_none());
}

#[test]
fn prepare_rejects_control_characters_in_paths_before_approval_output() {
    let (_temp, db, pending) = setup();
    let service = ChangesetService::new(&db, &pending).unwrap();

    for path in ["line\nbreak.rs", "tab\tname.rs", "escape\u{1b}name.rs", "delete\u{7f}.rs"] {
        let error = service.prepare("repo", vec![ChangeOperation::Create {
            path: path.into(),
            content: "payload".into(),
        }]).unwrap_err();
        assert_eq!(error.code, ErrorCode::PathEscape);
        assert!(!error.to_string().contains(path));
    }
    assert!(std::fs::read_dir(pending).unwrap().next().is_none());
}

#[test]
fn approval_diff_marks_missing_final_newline() {
    let (_temp, db, pending) = setup();
    let root = pending.parent().unwrap().join("repo");
    std::fs::write(root.join("no-newline.rs"), "old").unwrap();
    let service = ChangesetService::new(&db, &pending).unwrap();
    let prepared = service.prepare("repo", vec![ChangeOperation::Replace {
        path: "no-newline.rs".into(),
        expected_sha256: hash(b"old"),
        replacements: vec![Replacement {
            old_text: "old".into(),
            new_text: "new".into(),
            replace_all: false,
        }],
    }]).unwrap();

    assert!(prepared.unified_diff.contains("-old\n\\ No newline at end of file\n"));
    assert!(prepared.unified_diff.contains("+new\n\\ No newline at end of file\n"));
    assert_eq!(std::fs::read(root.join("no-newline.rs")).unwrap(), b"old");
}

#[test]
fn changed_payload_is_rejected_before_any_source_effect() {
    let (_temp, db, pending) = setup();
    let service = ChangesetService::new(&db, &pending).unwrap();
    let prepared = service.prepare("repo", vec![ChangeOperation::Create { path: "safe.rs".into(), content: "safe".into() }]).unwrap();
    std::fs::write(pending.join(format!("{}.json", prepared.operation_id)), b"tampered").unwrap();
    assert!(service.apply(&prepared.operation_id, &prepared.digest).is_err());
    assert!(!pending.parent().unwrap().join("repo/safe.rs").exists());
}

#[test]
fn apply_rejects_read_only_workspace_and_escape_paths() {
    let (temp, db, pending) = setup();
    let prepared = {
        let service = ChangesetService::new(&db, &pending).unwrap();
        assert!(service.prepare("repo", vec![ChangeOperation::Create { path: "../outside".into(), content: "x".into() }]).is_err());
        service.prepare("repo", vec![ChangeOperation::Create { path: "new.rs".into(), content: "x".into() }]).unwrap()
    };
    db.lock().execute("UPDATE workspaces SET mode = 'read_only', mode_epoch = mode_epoch + 1 WHERE id='repo'", []).unwrap();
    let service = ChangesetService::new(&db, &pending).unwrap();
    assert!(service.apply(&prepared.operation_id, &prepared.digest).is_err());
    assert!(!temp.path().join("repo/new.rs").exists());
}

#[test]
fn startup_recovery_classifies_effects_without_replaying_them() {
    let (temp, db, pending) = setup();
    let prepared = {
        let service = ChangesetService::new(&db, &pending).unwrap();
        service.prepare("repo", vec![ChangeOperation::Create {
            path: "after-crash.rs".into(), content: "effect".into(),
        }]).unwrap()
    };
    let root = temp.path().join("repo");
    std::fs::write(root.join("after-crash.rs"), "effect").unwrap();
    db.lock().execute("UPDATE pending_changesets SET state='Applying' WHERE operation_id=?1", [&prepared.operation_id]).unwrap();
    db.lock().execute("UPDATE changeset_files SET effect_state='intent' WHERE operation_id=?1", [&prepared.operation_id]).unwrap();
    let service = ChangesetService::new(&db, &pending).unwrap();
    let recovered = service.status(&prepared.operation_id).unwrap();
    assert_eq!(recovered.state, "Applied");
    assert_eq!(recovered.files[0].state, "applied");
    assert_eq!(std::fs::read_to_string(root.join("after-crash.rs")).unwrap(), "effect");
}

fn enqueue_malformed_cleanup(db: &Mutex<Connection>) {
    db.lock().execute(
        "INSERT INTO changeset_payload_cleanup (operation_id, reason, queued_at)
         VALUES ('malformed-operation-id', 'prepare_failed', 1)",
        [],
    ).unwrap();
}

#[test]
fn malformed_cleanup_recovers_only_a_unique_unreferenced_uuid_payload() {
    let (_temp, db, pending) = setup();
    std::fs::create_dir(&pending).unwrap();
    enqueue_malformed_cleanup(&db);
    let orphan = pending.join("11111111-1111-4111-8111-111111111111.json");
    std::fs::write(&orphan, b"orphan").unwrap();

    let service = ChangesetService::new(&db, &pending).unwrap();

    assert!(!orphan.exists());
    let rows: i64 = db.lock().query_row(
        "SELECT count(*) FROM changeset_payload_cleanup WHERE operation_id='malformed-operation-id'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(rows, 0);
    drop(service);
}

#[test]
fn malformed_cleanup_preserves_all_candidates_when_orphan_scan_is_ambiguous() {
    let (_temp, db, pending) = setup();
    std::fs::create_dir(&pending).unwrap();
    enqueue_malformed_cleanup(&db);
    let paths = [
        pending.join("11111111-1111-4111-8111-111111111111.json"),
        pending.join("22222222-2222-4222-8222-222222222222.json"),
    ];
    for path in &paths { std::fs::write(path, b"orphan").unwrap(); }

    let service = ChangesetService::new(&db, &pending).unwrap();

    assert!(paths.iter().all(|path| path.exists()));
    let (target, error): (Option<String>, Option<String>) = db.lock().query_row(
        "SELECT recovery_target_id, last_error FROM changeset_payload_cleanup
         WHERE operation_id='malformed-operation-id'",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ).unwrap();
    assert_eq!(target, None);
    assert_eq!(error.as_deref(), Some("recovery_ambiguous_orphans"));
    drop(service);
}

#[test]
fn malformed_cleanup_respects_active_queued_terminal_and_archive_references() {
    let ids = [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        "33333333-3333-4333-8333-333333333333",
        "44444444-4444-4444-8444-444444444444",
    ];
    for (index, id) in ids.into_iter().enumerate() {
        let (_temp, db, pending) = setup();
        std::fs::create_dir(&pending).unwrap();
        enqueue_malformed_cleanup(&db);
        let payload = pending.join(format!("{id}.json"));
        std::fs::write(&payload, b"payload").unwrap();
        match index {
            0 => {
                db.lock().execute(
                    "INSERT INTO pending_changesets (operation_id, workspace_id, digest, state, expires_at, created_at, payload_path)
                     VALUES (?1, 'repo', 'd', 'Prepared', 4_102_444_800, 1, ?2)",
                    params![id, payload.to_string_lossy()],
                ).unwrap();
            }
            1 => {
                db.lock().execute(
                    "INSERT INTO changeset_payload_cleanup (operation_id, reason, queued_at)
                    VALUES (?1, 'terminal', 2)",
                    [id],
                ).unwrap();
            }
            2 => {
                db.lock().execute(
                    "INSERT INTO pending_changesets (operation_id, workspace_id, digest, state, expires_at, created_at, payload_path)
                     VALUES (?1, 'repo', 'd', 'Applied', 99, 1, ?2)",
                    params![id, payload.to_string_lossy()],
                ).unwrap();
            }
            3 => {
                db.lock().execute(
                    "INSERT INTO unregistered_changesets (operation_id, workspace_id, digest, state, archive_json, archived_at)
                     VALUES (?1, 'repo', 'd', 'Applied', '{}', 1)",
                    [id],
                ).unwrap();
            }
            _ => unreachable!(),
        }

        let service = ChangesetService::new(&db, &pending).unwrap();
        let malformed_error = db.lock().query_row(
            "SELECT last_error FROM changeset_payload_cleanup
             WHERE operation_id='malformed-operation-id'",
            [],
            |row| row.get::<_, Option<String>>(0),
        ).optional().unwrap().flatten();
        let malformed_remaining: i64 = db.lock().query_row(
            "SELECT count(*) FROM changeset_payload_cleanup
             WHERE operation_id='malformed-operation-id'",
            [],
            |row| row.get(0),
        ).unwrap();

        match index {
            0 => {
                assert!(payload.exists(), "active Prepared payload is protected");
                assert_eq!(malformed_error.as_deref(), Some("recovery_no_orphan_found"));
                assert_eq!(malformed_remaining, 1, "unrelated malformed row remains unresolved");
            }
            1 | 2 => {
                assert!(!payload.exists(), "queued/terminal cleanup removes its payload");
                assert_eq!(malformed_error.as_deref(), Some("recovery_no_orphan_found"));
                assert_eq!(malformed_remaining, 1, "unrelated malformed row remains unresolved");
            }
            3 => {
                assert!(!payload.exists(), "archive-only payload is recovered as an orphan");
                assert_eq!(malformed_error, None, "malformed cleanup row retires after orphan recovery");
                assert_eq!(malformed_remaining, 0);
            }
            _ => unreachable!(),
        }
        drop(service);
    }
}
#[test]
fn uncertain_effects_require_fresh_expected_hashes() {
    let (temp, db, pending) = setup();
    let root = temp.path().join("repo");
    std::fs::write(root.join("uncertain.rs"), "base").unwrap();
    let base_hash = hash(b"base");
    
    let prepared1 = {
        let service = ChangesetService::new(&db, &pending).unwrap();
        service.prepare("repo", vec![ChangeOperation::Replace {
            path: "uncertain.rs".into(), expected_sha256: base_hash.clone(),
            replacements: vec![Replacement { old_text: "base".into(), new_text: "new".into(), replace_all: false }],
        }]).unwrap()
    };
    
    // Create a new prepared changeset BEFORE the uncertain effect
    let prepared2 = {
        let service = ChangesetService::new(&db, &pending).unwrap();
        service.prepare("repo", vec![ChangeOperation::Replace {
            path: "uncertain.rs".into(), expected_sha256: base_hash.clone(),
            replacements: vec![Replacement { old_text: "base".into(), new_text: "newer".into(), replace_all: false }],
        }]).unwrap()
    };
    
    // Simulate an uncertain effect for prepared1
    db.lock().execute("UPDATE pending_changesets SET state='Applied' WHERE operation_id=?1", [&prepared1.operation_id]).unwrap();
    db.lock().execute("UPDATE changeset_files SET effect_state='unknown', intent_at=9999999999 WHERE operation_id=?1", [&prepared1.operation_id]).unwrap();
    
    let service = ChangesetService::new(&db, &pending).unwrap();
    let result = service.apply(&prepared2.operation_id, &prepared2.digest);
    assert!(result.is_err());
    assert!(result.as_ref().unwrap_err().to_string().contains("uncertain effects"), "Error was: {:?}", result);
}
