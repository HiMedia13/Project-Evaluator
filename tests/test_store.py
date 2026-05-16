from project_evaluator.db.store import Store


def test_init_is_idempotent(tmp_path):
    db = tmp_path / "e.db"
    Store(db).init_db()
    Store(db).init_db()  # second call must not raise
    s = Store(db)
    assert s.schema_version() == 1


def test_record_and_get_last_evaluation(tmp_path):
    s = Store(tmp_path / "e.db")
    s.init_db()
    repo_id = s.upsert_repo("https://x/y.git", "y", "/ws/y")
    eval_id = s.create_evaluation(repo_id, "abc123", "looks fine", "openai:gpt-5.4")
    s.add_file_evaluation(eval_id, repo_id, "src/a.py", "h1", "ok", "clean", "python")
    s.add_tech_finding(eval_id, "FastAPI", "fits", "idiomatic", "no", "well scoped", "src/a.py:1")

    last = s.get_last_evaluation("https://x/y.git")
    assert last["commit_sha"] == "abc123"
    assert last["overall_summary"] == "looks fine"
    assert last["files"][0]["file_path"] == "src/a.py"
    assert last["tech_findings"][0]["technology"] == "FastAPI"


def test_get_last_evaluation_unknown_repo_returns_none(tmp_path):
    s = Store(tmp_path / "e.db")
    s.init_db()
    assert s.get_last_evaluation("https://nope") is None


def test_upsert_repo_is_stable_on_url(tmp_path):
    s = Store(tmp_path / "e.db")
    s.init_db()
    a = s.upsert_repo("https://x/y.git", "y", "/ws/y")
    b = s.upsert_repo("https://x/y.git", "y", "/ws/y2")
    assert a == b
