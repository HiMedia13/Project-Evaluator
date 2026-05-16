from project_evaluator.config import Config
from project_evaluator.tools import repo as R


def test_workspace_path_is_stable(tmp_path):
    a = R.workspace_path_for("https://x/y.git", tmp_path)
    b = R.workspace_path_for("https://x/y.git/", tmp_path)
    assert a == b


def test_clone_then_update(tmp_path, origin_repo, monkeypatch):
    monkeypatch.setenv("EVALUATOR_DATA_DIR", str(tmp_path / "data"))
    cfg = Config.from_env()
    cfg.ensure_dirs()

    r1 = R.clone_or_update_repo(origin_repo["url"], cfg.workspaces_dir)
    assert r1["fresh_clone"] is True
    assert r1["commit_sha"] == origin_repo["sha1"]

    # add a second commit upstream
    (origin_repo["path"] / "main.py").write_text("print('hi')\n")
    (origin_repo["path"] / "extra.py").write_text("x = 1\n")
    origin_repo["repo"].index.add(["main.py", "extra.py"])
    c2 = origin_repo["repo"].index.commit("c2")

    r2 = R.clone_or_update_repo(origin_repo["url"], cfg.workspaces_dir)
    assert r2["fresh_clone"] is False
    assert r2["commit_sha"] == c2.hexsha

    diff = R.diff_since(r2["path"], origin_repo["sha1"])
    assert "main.py" in diff["modified"]
    assert "extra.py" in diff["added"]
    assert diff["deleted"] == []


def test_build_project_map_skips_and_caps(tmp_path, origin_repo):
    m = R.build_project_map(
        origin_repo["path"],
        skip_dirs={"node_modules", ".git"},
        max_files=4000,
        max_file_bytes=500_000,
    )
    assert m["languages"].get(".py") == 1
    assert m["total_files"] == 2  # main.py, README.md ; node_modules skipped
    assert "junk.js" not in m["tree_excerpt"]
    assert "main.py" in m["entrypoints"]
