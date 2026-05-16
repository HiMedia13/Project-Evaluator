import pytest
from git import Repo


@pytest.fixture
def origin_repo(tmp_path):
    """A local git repo used as a clone source (no network)."""
    src = tmp_path / "origin"
    src.mkdir()
    repo = Repo.init(src)
    repo.config_writer().set_value("user", "email", "t@t.t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    (src / "main.py").write_text("print('hello')\n")
    (src / "README.md").write_text("# demo\n")
    (src / "node_modules").mkdir()
    (src / "node_modules" / "junk.js").write_text("x\n")
    repo.index.add(["main.py", "README.md", "node_modules/junk.js"])
    c1 = repo.index.commit("init")
    return {"path": src, "url": str(src), "sha1": c1.hexsha, "repo": repo}
