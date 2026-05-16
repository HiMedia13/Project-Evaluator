# project-evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conversational CLI TUI agent that clones a git repo and evaluates the *quality of its technology decisions* (purpose-fit, idiomatic use, over/under-engineering), with SQLite-backed incremental re-evaluation and an always-on verification critic.

**Architecture:** A `deepagents` (LangGraph) main agent orchestrates four subagents (`recon`, `dependency-stack`, `tech-fit`, `evaluation-critic`). UI-agnostic core (config, db, tools, subagents, agent) with a thin `rich` REPL client. LangSmith tracing on by default via env injection. The repo is never executed — static read + LLM reasoning + optional web search only.

**Tech Stack:** Python 3.11+, deepagents, langchain-openai, langsmith, GitPython, rich, pydantic, httpx, pytest. Build backend: hatchling, src layout.

**Spec:** `docs/superpowers/specs/2026-05-16-project-evaluator-agent-design.md`. Branch: `feat/evaluator-design`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Packaging, deps, console scripts `pe`/`project-evaluator`, pytest config |
| `src/project_evaluator/__init__.py` | Package marker, version |
| `src/project_evaluator/config.py` | Env-driven config dataclass; LangSmith env injection |
| `src/project_evaluator/db/schema.sql` | SQLite schema (spec §5.2) + `schema_version` |
| `src/project_evaluator/db/store.py` | Low-level SQLite store: init + CRUD |
| `src/project_evaluator/tools/repo.py` | Git clone/update (cached workspace), diff, project-map builder |
| `src/project_evaluator/tools/manifests.py` | Manifest parsers → dependency inventory |
| `src/project_evaluator/tools/citation_check.py` | Layer-0 deterministic evidence verification (0 tokens) |
| `src/project_evaluator/tools/usage.py` | Per-subagent token aggregation + rich table |
| `src/project_evaluator/tools/persistence.py` | Agent-facing tool closures wrapping store + repo |
| `src/project_evaluator/tools/websearch.py` | Best-practice web search (Tavily, graceful disable) |
| `src/project_evaluator/subagents/recon.py` | `recon` subagent dict builder |
| `src/project_evaluator/subagents/dependency_stack.py` | `dependency-stack` subagent dict builder |
| `src/project_evaluator/subagents/tech_fit.py` | `tech-fit` subagent dict builder (evidence contract) |
| `src/project_evaluator/subagents/evaluation_critic.py` | `evaluation-critic` subagent dict builder (always run) |
| `src/project_evaluator/agent.py` | `build_deep_agent(config)` wiring; checkpointer |
| `src/project_evaluator/cli.py` | `rich` streaming REPL; report + usage rendering; `main()` |
| `tests/...` | Mirrors source; local git fixtures, no network |

Dependency order (each task yields working, tested software): scaffolding → config → db → repo → manifests → citation_check → usage → persistence → websearch → 4 subagents → agent → cli → integration.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/project_evaluator/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_package.py`

- [ ] **Step 1: Write the failing test**

`tests/test_package.py`:
```python
def test_package_version():
    import project_evaluator
    assert project_evaluator.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'project_evaluator'`

- [ ] **Step 3: Create the package and packaging files**

`pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "project-evaluator"
version = "0.1.0"
description = "Conversational agent that evaluates the technology decisions of a git repo"
requires-python = ">=3.11"
dependencies = [
    "deepagents",
    "langchain-openai",
    "langsmith",
    "GitPython",
    "rich",
    "pydantic>=2",
    "httpx",
]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
pe = "project_evaluator.cli:main"
project-evaluator = "project_evaluator.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/project_evaluator"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["live: tests that call real LLM/network (deselected by default)"]
addopts = "-m 'not live'"
```

`src/project_evaluator/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/project_evaluator/__init__.py tests/__init__.py tests/test_package.py
git commit -m "build: project scaffolding (hatchling, src layout, pytest)"
```

---

## Task 2: Config

**Files:**
- Create: `src/project_evaluator/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import logging
from pathlib import Path
from project_evaluator.config import Config


def test_defaults(tmp_path, monkeypatch):
    for var in ("EVALUATOR_MODEL", "EVALUATOR_DATA_DIR", "EVALUATOR_MAX_FILES",
                "LANGSMITH_TRACING", "LANGSMITH_PROJECT", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EVALUATOR_DATA_DIR", str(tmp_path))
    cfg = Config.from_env()
    assert cfg.model == "openai:gpt-5.4"
    assert cfg.db_path == tmp_path / "evaluator.db"
    assert cfg.workspaces_dir == tmp_path / "workspaces"
    assert cfg.max_files == 4000
    assert ".git" in cfg.skip_dirs and "node_modules" in cfg.skip_dirs
    assert cfg.langsmith_tracing is True
    assert cfg.langsmith_project == "project-evaluator"


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALUATOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EVALUATOR_MODEL", "openai:gpt-5.4-mini")
    monkeypatch.setenv("EVALUATOR_MAX_FILES", "10")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    cfg = Config.from_env()
    assert cfg.model == "openai:gpt-5.4-mini"
    assert cfg.max_files == 10
    assert cfg.langsmith_tracing is False


def test_apply_observability_warns_without_key(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("EVALUATOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    cfg = Config.from_env()
    with caplog.at_level(logging.WARNING):
        cfg.apply_observability_env()
    assert "LANGSMITH_API_KEY" in caplog.text
    import os
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_apply_observability_sets_env_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALUATOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    import os
    cfg = Config.from_env()
    cfg.apply_observability_env()
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "project-evaluator"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'project_evaluator.config'`

- [ ] **Step 3: Write minimal implementation**

`src/project_evaluator/config.py`:
```python
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", ".venv", "venv",
    "__pycache__", ".mypy_cache", ".pytest_cache", "target",
    ".idea", ".gradle", "vendor", ".next", "out",
}


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    model: str
    data_dir: Path
    max_files: int
    max_file_bytes: int
    skip_dirs: frozenset[str]
    langsmith_tracing: bool
    langsmith_project: str
    tavily_api_key: str | None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "evaluator.db"

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"

    @classmethod
    def from_env(cls) -> "Config":
        data_dir = Path(
            os.environ.get("EVALUATOR_DATA_DIR")
            or (Path.home() / ".project-evaluator")
        ).expanduser()
        return cls(
            model=os.environ.get("EVALUATOR_MODEL", "openai:gpt-5.4"),
            data_dir=data_dir,
            max_files=int(os.environ.get("EVALUATOR_MAX_FILES", "4000")),
            max_file_bytes=int(os.environ.get("EVALUATOR_MAX_FILE_BYTES", "500000")),
            skip_dirs=frozenset(DEFAULT_SKIP_DIRS),
            langsmith_tracing=_bool(os.environ.get("LANGSMITH_TRACING"), True),
            langsmith_project=os.environ.get("LANGSMITH_PROJECT", "project-evaluator"),
            tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

    def apply_observability_env(self) -> None:
        """Inject LangSmith env. Default on; degrade gracefully without key."""
        if self.langsmith_tracing and not os.environ.get("LANGSMITH_API_KEY"):
            logger.warning(
                "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is unset; "
                "disabling tracing for this run."
            )
            os.environ["LANGSMITH_TRACING"] = "false"
            return
        os.environ["LANGSMITH_TRACING"] = "true" if self.langsmith_tracing else "false"
        if self.langsmith_tracing:
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/config.py tests/test_config.py
git commit -m "feat: env-driven Config with LangSmith env injection"
```

---

## Task 3: SQLite store

**Files:**
- Create: `src/project_evaluator/db/__init__.py` (empty)
- Create: `src/project_evaluator/db/schema.sql`
- Create: `src/project_evaluator/db/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'project_evaluator.db.store'`

- [ ] **Step 3: Write schema and store**

`src/project_evaluator/db/__init__.py`: empty file.

`src/project_evaluator/db/schema.sql`:
```sql
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS repos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT UNIQUE NOT NULL,
  name TEXT,
  workspace_path TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  repo_id INTEGER NOT NULL REFERENCES repos(id),
  commit_sha TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  overall_summary TEXT,
  model TEXT
);

CREATE TABLE IF NOT EXISTS file_evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
  repo_id INTEGER NOT NULL REFERENCES repos(id),
  file_path TEXT NOT NULL,
  content_hash TEXT,
  verdict TEXT,
  notes TEXT,
  tech_tags TEXT
);

CREATE TABLE IF NOT EXISTS tech_findings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
  technology TEXT,
  purpose_fit TEXT,
  correctness TEXT,
  overengineering TEXT,
  rationale TEXT,
  evidence TEXT
);
```

`src/project_evaluator/db/store.py`:
```python
from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

SCHEMA_VERSION = 1


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        schema = resources.files("project_evaluator.db").joinpath("schema.sql").read_text()
        with self._connect() as conn:
            conn.executescript(schema)
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

    def schema_version(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            return row["version"] if row else 0

    def upsert_repo(self, url: str, name: str, workspace_path: str) -> int:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO repos (url, name, workspace_path) VALUES (?, ?, ?) "
                "ON CONFLICT(url) DO UPDATE SET name=excluded.name, "
                "workspace_path=excluded.workspace_path, updated_at=datetime('now')",
                (url, name, workspace_path),
            )
            return conn.execute("SELECT id FROM repos WHERE url = ?", (url,)).fetchone()["id"]

    def create_evaluation(self, repo_id: int, commit_sha: str, summary: str, model: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO evaluations (repo_id, commit_sha, overall_summary, model) "
                "VALUES (?, ?, ?, ?)",
                (repo_id, commit_sha, summary, model),
            )
            return cur.lastrowid

    def add_file_evaluation(self, evaluation_id: int, repo_id: int, file_path: str,
                            content_hash: str, verdict: str, notes: str, tech_tags: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO file_evaluations (evaluation_id, repo_id, file_path, "
                "content_hash, verdict, notes, tech_tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (evaluation_id, repo_id, file_path, content_hash, verdict, notes, tech_tags),
            )

    def add_tech_finding(self, evaluation_id: int, technology: str, purpose_fit: str,
                         correctness: str, overengineering: str, rationale: str,
                         evidence: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tech_findings (evaluation_id, technology, purpose_fit, "
                "correctness, overengineering, rationale, evidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (evaluation_id, technology, purpose_fit, correctness,
                 overengineering, rationale, evidence),
            )

    def get_last_evaluation(self, url: str) -> dict | None:
        with self._connect() as conn:
            repo = conn.execute("SELECT id FROM repos WHERE url = ?", (url,)).fetchone()
            if repo is None:
                return None
            ev = conn.execute(
                "SELECT * FROM evaluations WHERE repo_id = ? ORDER BY id DESC LIMIT 1",
                (repo["id"],),
            ).fetchone()
            if ev is None:
                return None
            files = conn.execute(
                "SELECT file_path, content_hash, verdict, notes, tech_tags "
                "FROM file_evaluations WHERE evaluation_id = ?", (ev["id"],),
            ).fetchall()
            findings = conn.execute(
                "SELECT technology, purpose_fit, correctness, overengineering, "
                "rationale, evidence FROM tech_findings WHERE evaluation_id = ?",
                (ev["id"],),
            ).fetchall()
            return {
                "evaluation_id": ev["id"],
                "commit_sha": ev["commit_sha"],
                "overall_summary": ev["overall_summary"],
                "model": ev["model"],
                "files": [dict(r) for r in files],
                "tech_findings": [dict(r) for r in findings],
            }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/db tests/test_store.py
git commit -m "feat: SQLite store with versioned schema (spec 5.2)"
```

---

## Task 4: Git repo tools + project map

**Files:**
- Create: `src/project_evaluator/tools/__init__.py` (empty)
- Create: `src/project_evaluator/tools/repo.py`
- Create: `tests/conftest.py`
- Create: `tests/test_repo.py`

- [ ] **Step 1: Write the failing test**

`tests/conftest.py`:
```python
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
```

`tests/test_repo.py`:
```python
from project_evaluator.config import Config
from project_evaluator.tools import repo as R


def _cfg(tmp_path):
    return Config.from_env_for_test(tmp_path)


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
```

Add to `src/project_evaluator/config.py` a test helper (Step 3 includes it):

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_repo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'project_evaluator.tools.repo'`

- [ ] **Step 3: Write implementation**

Append to `src/project_evaluator/config.py`:
```python
    @classmethod
    def from_env_for_test(cls, data_dir) -> "Config":
        """Construct a Config rooted at an explicit data dir (tests only)."""
        import os
        os.environ["EVALUATOR_DATA_DIR"] = str(data_dir)
        return cls.from_env()
```

`src/project_evaluator/tools/__init__.py`: empty file.

`src/project_evaluator/tools/repo.py`:
```python
from __future__ import annotations

import hashlib
from pathlib import Path

from git import Repo
from git.exc import GitCommandError

ENTRYPOINT_NAMES = {
    "main.py", "app.py", "__main__.py", "manage.py",
    "index.js", "main.js", "server.js", "main.go", "main.rs",
}


def _normalize_url(url: str) -> str:
    u = url.strip().rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    return u.lower()


def workspace_path_for(url: str, workspaces_dir: Path) -> Path:
    digest = hashlib.sha1(_normalize_url(url).encode()).hexdigest()[:16]
    return Path(workspaces_dir) / digest


class RepoAccessError(RuntimeError):
    """Clone/fetch failed (bad URL, private repo, network)."""


def clone_or_update_repo(url: str, workspaces_dir: Path, ref: str | None = None) -> dict:
    path = workspace_path_for(url, workspaces_dir)
    fresh = not path.exists()
    try:
        if fresh:
            repo = Repo.clone_from(url, path)
        else:
            repo = Repo(path)
            repo.remotes.origin.fetch()
            target = ref or repo.remotes.origin.refs[repo.active_branch.name].name
            repo.git.reset("--hard", target if ref else "origin/HEAD")
        if ref:
            repo.git.checkout(ref)
    except GitCommandError as e:
        raise RepoAccessError(
            f"Could not access '{url}'. If private, configure git credentials. ({e})"
        ) from e
    return {"path": path, "commit_sha": repo.head.commit.hexsha, "fresh_clone": fresh}


def diff_since(repo_path: Path, base_sha: str) -> dict:
    repo = Repo(repo_path)
    out = repo.git.diff("--name-status", base_sha, "HEAD")
    added, modified, deleted = [], [], []
    for line in (l for l in out.splitlines() if l.strip()):
        status, _, name = line.partition("\t")
        name = name.strip()
        if status.startswith("A"):
            added.append(name)
        elif status.startswith("D"):
            deleted.append(name)
        else:
            modified.append(name)
    return {"added": added, "modified": modified, "deleted": deleted}


def build_project_map(repo_path: Path, *, skip_dirs: set[str],
                      max_files: int, max_file_bytes: int) -> dict:
    repo_path = Path(repo_path)
    languages: dict[str, int] = {}
    entrypoints: list[str] = []
    tree_lines: list[str] = []
    total = 0
    skipped = 0
    for p in sorted(repo_path.rglob("*")):
        rel = p.relative_to(repo_path)
        if any(part in skip_dirs for part in rel.parts):
            skipped += 1
            continue
        if p.is_dir():
            continue
        if total >= max_files:
            skipped += 1
            continue
        total += 1
        ext = p.suffix.lower()
        if ext:
            languages[ext] = languages.get(ext, 0) + 1
        if p.name in ENTRYPOINT_NAMES:
            entrypoints.append(str(rel).replace("\\", "/"))
        if len(tree_lines) < 200:
            tree_lines.append(str(rel).replace("\\", "/"))
    build_systems = []
    for marker, label in {
        "package.json": "npm", "pyproject.toml": "python",
        "requirements.txt": "pip", "go.mod": "go-modules",
        "Cargo.toml": "cargo", "pom.xml": "maven", "build.gradle": "gradle",
    }.items():
        if (repo_path / marker).exists():
            build_systems.append(label)
    return {
        "languages": languages,
        "total_files": total,
        "skipped_files": skipped,
        "build_systems": build_systems,
        "entrypoints": entrypoints,
        "tree_excerpt": "\n".join(tree_lines),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_repo.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/tools/__init__.py src/project_evaluator/tools/repo.py src/project_evaluator/config.py tests/conftest.py tests/test_repo.py
git commit -m "feat: git clone/update cache, diff, project-map builder"
```

---

## Task 5: Manifest parsers

**Files:**
- Create: `src/project_evaluator/tools/manifests.py`
- Create: `tests/test_manifests.py`

- [ ] **Step 1: Write the failing test**

`tests/test_manifests.py`:
```python
from project_evaluator.tools.manifests import parse_manifests


def test_parses_package_json_and_requirements(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"react": "^18.0.0"}, '
        '"devDependencies": {"vite": "5.0.0"}}'
    )
    (tmp_path / "requirements.txt").write_text("fastapi==0.110.0\nhttpx>=0.27\n# c\n")
    inv = parse_manifests(tmp_path)
    by_eco = {m["ecosystem"]: m for m in inv}
    assert by_eco["npm"]["dependencies"]["react"] == "^18.0.0"
    assert by_eco["npm"]["dependencies"]["vite"] == "5.0.0"
    assert by_eco["pip"]["dependencies"]["fastapi"] == "0.110.0"
    assert "httpx" in by_eco["pip"]["dependencies"]


def test_parses_pyproject_pep621(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\ndependencies=["pydantic>=2", "rich"]\n'
    )
    inv = parse_manifests(tmp_path)
    py = [m for m in inv if m["ecosystem"] == "pyproject"][0]
    assert "pydantic" in py["dependencies"]
    assert "rich" in py["dependencies"]


def test_empty_dir_returns_empty(tmp_path):
    assert parse_manifests(tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_manifests.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'project_evaluator.tools.manifests'`

- [ ] **Step 3: Write implementation**

`src/project_evaluator/tools/manifests.py`:
```python
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*([=<>!~]=?[^#;]+)?")


def _parse_requirements(text: str) -> dict[str, str]:
    deps: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        m = _REQ_RE.match(line)
        if m:
            name = m.group(1)
            spec = (m.group(2) or "").strip()
            deps[name] = spec.lstrip("=").strip() if spec.startswith("==") else (spec or "*")
    return deps


def parse_manifests(repo_path: Path) -> list[dict]:
    repo_path = Path(repo_path)
    out: list[dict] = []

    pkg = repo_path / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            out.append({"ecosystem": "npm", "file": "package.json", "dependencies": deps})
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    req = repo_path / "requirements.txt"
    if req.exists():
        out.append({
            "ecosystem": "pip", "file": "requirements.txt",
            "dependencies": _parse_requirements(req.read_text(encoding="utf-8", errors="ignore")),
        })

    pyp = repo_path / "pyproject.toml"
    if pyp.exists():
        try:
            data = tomllib.loads(pyp.read_text(encoding="utf-8"))
            deps: dict[str, str] = {}
            for item in data.get("project", {}).get("dependencies", []):
                m = _REQ_RE.match(item)
                if m:
                    deps[m.group(1)] = (m.group(2) or "*").strip()
            poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            for k, v in poetry.items():
                if k.lower() != "python":
                    deps[k] = v if isinstance(v, str) else "*"
            out.append({"ecosystem": "pyproject", "file": "pyproject.toml",
                        "dependencies": deps})
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            pass

    for fname, eco in (("go.mod", "go-modules"), ("Cargo.toml", "cargo")):
        f = repo_path / fname
        if f.exists():
            out.append({"ecosystem": eco, "file": fname,
                        "dependencies": {}, "raw": f.read_text(encoding="utf-8", errors="ignore")[:4000]})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_manifests.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/tools/manifests.py tests/test_manifests.py
git commit -m "feat: manifest parsers (npm, pip, pyproject, go, cargo)"
```

---

## Task 6: Layer-0 citation check

**Files:**
- Create: `src/project_evaluator/tools/citation_check.py`
- Create: `tests/test_citation_check.py`

- [ ] **Step 1: Write the failing test**

`tests/test_citation_check.py`:
```python
from project_evaluator.tools.citation_check import Evidence, check_evidence, check_evidence_list


def test_valid_citation_ok(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    ev = Evidence(claim="defines y", file_path="a.py", line_start=2,
                  line_end=2, quoted_snippet="y = 2", rationale="r")
    res = check_evidence(tmp_path, ev)
    assert res.ok is True
    assert res.reason == "ok"


def test_missing_file_flagged(tmp_path):
    ev = Evidence(claim="c", file_path="nope.py", line_start=1,
                  line_end=1, quoted_snippet=None, rationale="r")
    res = check_evidence(tmp_path, ev)
    assert res.ok is False
    assert "not found" in res.reason


def test_line_out_of_range_flagged(tmp_path):
    (tmp_path / "a.py").write_text("only one line\n")
    ev = Evidence(claim="c", file_path="a.py", line_start=5,
                  line_end=9, quoted_snippet=None, rationale="r")
    res = check_evidence(tmp_path, ev)
    assert res.ok is False
    assert "out of range" in res.reason


def test_snippet_mismatch_flagged(tmp_path):
    (tmp_path / "a.py").write_text("real = 1\n")
    ev = Evidence(claim="c", file_path="a.py", line_start=1,
                  line_end=1, quoted_snippet="fake = 999", rationale="r")
    res = check_evidence(tmp_path, ev)
    assert res.ok is False
    assert "snippet" in res.reason


def test_check_list_partitions(tmp_path):
    (tmp_path / "a.py").write_text("good = 1\n")
    evs = [
        Evidence(claim="ok", file_path="a.py", line_start=1, line_end=1,
                 quoted_snippet="good = 1", rationale="r"),
        Evidence(claim="bad", file_path="x.py", line_start=1, line_end=1,
                 quoted_snippet=None, rationale="r"),
    ]
    results = check_evidence_list(tmp_path, evs)
    assert results[0].ok is True
    assert results[1].ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_citation_check.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`src/project_evaluator/tools/citation_check.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class Evidence(BaseModel):
    claim: str
    file_path: str
    line_start: int
    line_end: int
    quoted_snippet: str | None = None
    rationale: str


@dataclass
class CitationResult:
    claim: str
    ok: bool
    reason: str


def check_evidence(repo_path: Path, ev: Evidence) -> CitationResult:
    target = Path(repo_path) / ev.file_path
    if not target.is_file():
        return CitationResult(ev.claim, False, f"file not found: {ev.file_path}")
    try:
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as e:
        return CitationResult(ev.claim, False, f"unreadable file: {e}")
    n = len(lines)
    if not (1 <= ev.line_start <= ev.line_end <= n):
        return CitationResult(
            ev.claim, False,
            f"line range {ev.line_start}-{ev.line_end} out of range (file has {n} lines)",
        )
    if ev.quoted_snippet is not None:
        actual = "\n".join(lines[ev.line_start - 1:ev.line_end])
        if ev.quoted_snippet.strip() not in actual.strip() and actual.strip() not in ev.quoted_snippet.strip():
            return CitationResult(ev.claim, False, "snippet does not match cited lines")
    return CitationResult(ev.claim, True, "ok")


def check_evidence_list(repo_path: Path, evidences: list[Evidence]) -> list[CitationResult]:
    return [check_evidence(repo_path, e) for e in evidences]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_citation_check.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/tools/citation_check.py tests/test_citation_check.py
git commit -m "feat: layer-0 deterministic citation check (0 tokens)"
```

---

## Task 7: Usage aggregation + table

**Files:**
- Create: `src/project_evaluator/tools/usage.py`
- Create: `tests/test_usage.py`

- [ ] **Step 1: Write the failing test**

`tests/test_usage.py`:
```python
from project_evaluator.tools.usage import SubagentUsage, aggregate_usage, render_usage_table


def test_aggregate_sums_per_subagent():
    records = [
        SubagentUsage("recon", 100, 20),
        SubagentUsage("tech-fit", 500, 80),
        SubagentUsage("recon", 50, 10),
    ]
    rows = aggregate_usage(records)
    by = {r["subagent"]: r for r in rows}
    assert by["recon"]["input_tokens"] == 150
    assert by["recon"]["output_tokens"] == 30
    assert by["tech-fit"]["input_tokens"] == 500
    total = [r for r in rows if r["subagent"] == "TOTAL"][0]
    assert total["input_tokens"] == 650
    assert total["output_tokens"] == 110


def test_render_returns_rich_table():
    from rich.table import Table
    t = render_usage_table(aggregate_usage([SubagentUsage("recon", 1, 1)]))
    assert isinstance(t, Table)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_usage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`src/project_evaluator/tools/usage.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from rich.table import Table


@dataclass
class SubagentUsage:
    subagent: str
    input_tokens: int
    output_tokens: int


def aggregate_usage(records: list[SubagentUsage]) -> list[dict]:
    agg: dict[str, dict] = {}
    for r in records:
        a = agg.setdefault(r.subagent, {"subagent": r.subagent,
                                        "input_tokens": 0, "output_tokens": 0})
        a["input_tokens"] += r.input_tokens
        a["output_tokens"] += r.output_tokens
    rows = sorted(agg.values(), key=lambda x: x["subagent"])
    rows.append({
        "subagent": "TOTAL",
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    })
    return rows


def render_usage_table(rows: list[dict]) -> Table:
    table = Table(title="Token usage by subagent")
    table.add_column("Subagent")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    for r in rows:
        style = "bold" if r["subagent"] == "TOTAL" else None
        table.add_row(r["subagent"], str(r["input_tokens"]),
                      str(r["output_tokens"]), style=style)
    return table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_usage.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/tools/usage.py tests/test_usage.py
git commit -m "feat: per-subagent token aggregation and rich table"
```

---

## Task 8: Agent-facing persistence tools

**Files:**
- Create: `src/project_evaluator/tools/persistence.py`
- Create: `tests/test_persistence.py`

These are the deepagents tool callables (plain functions, docstrings, JSON-string returns for the LLM). A factory binds them to a `Store` + workspaces dir.

- [ ] **Step 1: Write the failing test**

`tests/test_persistence.py`:
```python
import json

from project_evaluator.config import Config
from project_evaluator.db.store import Store
from project_evaluator.tools.persistence import make_persistence_tools


def test_record_then_history_roundtrip(tmp_path):
    store = Store(tmp_path / "e.db")
    store.init_db()
    cfg = Config.from_env_for_test(tmp_path / "data")
    tools = make_persistence_tools(store, cfg.workspaces_dir)

    history = json.loads(tools["get_repo_history"]("https://x/y.git"))
    assert history["status"] == "no_history"

    out = json.loads(tools["record_evaluation"](
        url="https://x/y.git", commit_sha="abc", summary="ok",
        file_evals=[{"file_path": "a.py", "content_hash": "h",
                     "verdict": "good", "notes": "n", "tech_tags": "py"}],
        tech_findings=[{"technology": "FastAPI", "purpose_fit": "yes",
                        "correctness": "idiomatic", "overengineering": "no",
                        "rationale": "r", "evidence": "a.py:1"}],
    ))
    assert out["status"] == "recorded"

    history = json.loads(tools["get_repo_history"]("https://x/y.git"))
    assert history["status"] == "ok"
    assert history["commit_sha"] == "abc"
    assert history["files"][0]["file_path"] == "a.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_persistence.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`src/project_evaluator/tools/persistence.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

from ..db.store import Store
from .repo import RepoAccessError, clone_or_update_repo, diff_since, workspace_path_for


def make_persistence_tools(store: Store, workspaces_dir: Path) -> dict:
    def get_repo_history(url: str) -> str:
        """Return the most recent stored evaluation for a git URL as JSON.
        Use this FIRST on any repo to decide full vs incremental evaluation."""
        last = store.get_last_evaluation(url)
        if last is None:
            return json.dumps({"status": "no_history", "url": url})
        return json.dumps({"status": "ok", **last})

    def diff_since_last(url: str) -> str:
        """Return files added/modified/deleted since the last recorded
        evaluation's commit, as JSON. Empty if no prior evaluation."""
        last = store.get_last_evaluation(url)
        if last is None or not last.get("commit_sha"):
            return json.dumps({"status": "no_history", "added": [],
                               "modified": [], "deleted": []})
        path = workspace_path_for(url, workspaces_dir)
        if not path.exists():
            return json.dumps({"status": "no_workspace",
                               "hint": "call recon to clone first"})
        d = diff_since(path, last["commit_sha"])
        return json.dumps({"status": "ok", "base_sha": last["commit_sha"], **d})

    def record_evaluation(url: str, commit_sha: str, summary: str,
                          file_evals: list[dict], tech_findings: list[dict]) -> str:
        """Persist a completed evaluation. file_evals items:
        {file_path, content_hash, verdict, notes, tech_tags}.
        tech_findings items:
        {technology, purpose_fit, correctness, overengineering, rationale, evidence}."""
        name = url.rstrip("/").split("/")[-1].removesuffix(".git")
        repo_id = store.upsert_repo(url, name, str(workspace_path_for(url, workspaces_dir)))
        eval_id = store.create_evaluation(repo_id, commit_sha, summary, "")
        for f in file_evals:
            store.add_file_evaluation(
                eval_id, repo_id, f["file_path"], f.get("content_hash", ""),
                f.get("verdict", ""), f.get("notes", ""), f.get("tech_tags", ""))
        for t in tech_findings:
            store.add_tech_finding(
                eval_id, t.get("technology", ""), t.get("purpose_fit", ""),
                t.get("correctness", ""), t.get("overengineering", ""),
                t.get("rationale", ""), t.get("evidence", ""))
        return json.dumps({"status": "recorded", "evaluation_id": eval_id})

    return {
        "get_repo_history": get_repo_history,
        "diff_since_last": diff_since_last,
        "record_evaluation": record_evaluation,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_persistence.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/tools/persistence.py tests/test_persistence.py
git commit -m "feat: agent-facing persistence tools (history/diff/record)"
```

---

## Task 9: Web search tool (graceful disable)

**Files:**
- Create: `src/project_evaluator/tools/websearch.py`
- Create: `tests/test_websearch.py`

- [ ] **Step 1: Write the failing test**

`tests/test_websearch.py`:
```python
import json

from project_evaluator.tools.websearch import make_web_search


def test_disabled_without_key():
    tool = make_web_search(api_key=None)
    out = json.loads(tool("react best practices"))
    assert out["status"] == "disabled"


def test_enabled_calls_backend(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["q"] = json["query"]

        class R:
            def raise_for_status(self): pass
            def json(self): return {"results": [{"title": "T", "url": "U", "content": "C"}]}
        return R()

    import project_evaluator.tools.websearch as W
    monkeypatch.setattr(W.httpx, "post", fake_post)
    tool = make_web_search(api_key="tvly-x")
    out = json.loads(tool("idiomatic fastapi"))
    assert out["status"] == "ok"
    assert captured["q"] == "idiomatic fastapi"
    assert out["results"][0]["title"] == "T"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_websearch.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`src/project_evaluator/tools/websearch.py`:
```python
from __future__ import annotations

import json

import httpx


def make_web_search(api_key: str | None):
    def web_search(query: str) -> str:
        """Search the web for framework/library best practices and idioms.
        Returns JSON {status, results:[{title,url,content}]}. Disabled if
        no TAVILY_API_KEY is configured."""
        if not api_key:
            return json.dumps({"status": "disabled",
                               "reason": "no TAVILY_API_KEY configured"})
        try:
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": 5, "api_key": api_key},
                headers={"Content-Type": "application/json"},
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            return json.dumps({"status": "error", "reason": str(e)})
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "content": r.get("content", "")[:1000]}
            for r in data.get("results", [])
        ]
        return json.dumps({"status": "ok", "results": results})

    return web_search
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_websearch.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/tools/websearch.py tests/test_websearch.py
git commit -m "feat: Tavily web-search tool with graceful disable"
```

---

## Task 10: Subagent definitions (recon, dependency-stack, tech-fit, evaluation-critic)

**Files:**
- Create: `src/project_evaluator/subagents/__init__.py` (empty)
- Create: `src/project_evaluator/subagents/recon.py`
- Create: `src/project_evaluator/subagents/dependency_stack.py`
- Create: `src/project_evaluator/subagents/tech_fit.py`
- Create: `src/project_evaluator/subagents/evaluation_critic.py`
- Create: `tests/test_subagents.py`

Each module exposes a builder returning a deepagents subagent dict (`name`, `description`, `system_prompt`, `tools`). Builders take already-bound tool callables so they are pure and testable without an LLM.

- [ ] **Step 1: Write the failing test**

`tests/test_subagents.py`:
```python
from project_evaluator.subagents.recon import build_recon
from project_evaluator.subagents.dependency_stack import build_dependency_stack
from project_evaluator.subagents.tech_fit import build_tech_fit
from project_evaluator.subagents.evaluation_critic import build_evaluation_critic


def _noop(*a, **k):  # stand-in tool callable
    return "{}"


REQUIRED = {"name", "description", "system_prompt", "tools"}


def test_recon_shape():
    sa = build_recon(clone_or_update_tool=_noop)
    assert REQUIRED <= set(sa)
    assert sa["name"] == "recon"
    assert callable(sa["tools"][0])
    assert "skip" in sa["system_prompt"].lower()


def test_dependency_stack_shape():
    sa = build_dependency_stack(parse_manifests_tool=_noop)
    assert sa["name"] == "dependency-stack"
    assert REQUIRED <= set(sa)


def test_tech_fit_enforces_evidence_contract():
    sa = build_tech_fit(read_tools=[_noop], web_search_tool=_noop)
    assert sa["name"] == "tech-fit"
    p = sa["system_prompt"]
    for key in ("claim", "file_path", "line_start", "line_end",
                "quoted_snippet", "rationale"):
        assert key in p
    assert "overengineering" in p.lower()


def test_critic_two_layers():
    sa = build_evaluation_critic(citation_check_tool=_noop, read_tools=[_noop])
    assert sa["name"] == "evaluation-critic"
    p = sa["system_prompt"].lower()
    assert "layer 0" in p or "layer-0" in p
    assert "independently" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_subagents.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementations**

`src/project_evaluator/subagents/__init__.py`: empty file.

`src/project_evaluator/subagents/recon.py`:
```python
def build_recon(clone_or_update_tool) -> dict:
    return {
        "name": "recon",
        "description": "Clones/updates the target repo into a cached workspace "
                       "and produces a project map: languages, structure, build "
                       "systems, entrypoints, size.",
        "system_prompt": (
            "You are the recon subagent. Use the clone/update tool to get the "
            "repo into its cached workspace, then read the filesystem to build a "
            "concise project map: detected languages, directory layout, build "
            "systems, likely entrypoints, and size. Skip vendored/build dirs "
            "(node_modules, .git, dist, build, target, vendor) and very large "
            "files (sample them). Write the map to a file named PROJECT_MAP.md "
            "so other subagents can read it. Return the map summary plus the "
            "current commit SHA."
        ),
        "tools": [clone_or_update_tool],
    }
```

`src/project_evaluator/subagents/dependency_stack.py`:
```python
def build_dependency_stack(parse_manifests_tool) -> dict:
    return {
        "name": "dependency-stack",
        "description": "Parses manifest files into a framework/library/version "
                       "inventory. Read-only parsing; no network or CVE lookups.",
        "system_prompt": (
            "You are the dependency-stack subagent. Parse the repo's manifests "
            "(package.json, pyproject.toml, requirements.txt, go.mod, Cargo.toml, "
            "pom.xml) into a clear inventory of frameworks, libraries and their "
            "versions. Note notable or unusually old pins. Do NOT run anything "
            "or query the network. Return a structured inventory."
        ),
        "tools": [parse_manifests_tool],
    }
```

`src/project_evaluator/subagents/tech_fit.py`:
```python
EVIDENCE_CONTRACT = (
    "EVIDENCE CONTRACT (MANDATORY): every claim MUST carry structured evidence "
    "as an object: {claim, file_path, line_start, line_end, quoted_snippet, "
    "rationale}. Never make a claim without resolvable file:line evidence and a "
    "verbatim quoted_snippet from those lines. Claims without evidence are "
    "rejected downstream."
)


def build_tech_fit(read_tools: list, web_search_tool) -> dict:
    return {
        "name": "tech-fit",
        "description": "Judges, per significant technology/pattern, whether it "
                       "is used for its intended purpose, idiomatically, and "
                       "whether the design is over- or under-engineered.",
        "system_prompt": (
            "You are the tech-fit subagent — the core evaluator. Read "
            "PROJECT_MAP.md and the dependency inventory, then assess the major "
            "technologies and architectural patterns on three axes: (a) "
            "purpose-fit (is the tech used for what it is for), (b) idiomatic "
            "correctness, (c) over-engineering vs under-engineering (unwarranted "
            "abstraction, microservices, message queues, etc.). Use web search "
            "to ground idiom/best-practice claims. " + EVIDENCE_CONTRACT +
            " Return a list of findings, each with its evidence object(s)."
        ),
        "tools": [*read_tools, web_search_tool],
    }
```

`src/project_evaluator/subagents/evaluation_critic.py`:
```python
def build_evaluation_critic(citation_check_tool, read_tools: list) -> dict:
    return {
        "name": "evaluation-critic",
        "description": "Always-run verifier. Layer 0: deterministic citation "
                       "check (0 tokens). Layer 1: independent LLM review of "
                       "whether each claim is justified by its evidence.",
        "system_prompt": (
            "You are the evaluation-critic subagent and you run on EVERY "
            "evaluation as the final step. Layer 0: call the citation-check "
            "tool on every evidence object from tech-fit; any failing citation "
            "is a hallucinated/stale reference — reject that claim. Layer 1: "
            "independently re-read the cited file subsets (do NOT trust "
            "tech-fit's reasoning chain) and decide for each surviving claim "
            "whether the evidence actually justifies it, whether over/under-"
            "engineering verdicts have explicit justification, and flag weak, "
            "unsupported, or overconfident claims. Return three lists: "
            "verified / needs-revision / rejected, plus a corrected summary."
        ),
        "tools": [citation_check_tool, *read_tools],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_subagents.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/subagents tests/test_subagents.py
git commit -m "feat: 4 subagent definitions (recon/deps/tech-fit/critic)"
```

---

## Task 11: Agent wiring

**Files:**
- Create: `src/project_evaluator/agent.py`
- Create: `tests/test_agent_wiring.py`

The exact `deepagents` signature must be confirmed against the installed version before wiring (Step 0). The documented API is `create_deep_agent(model, tools, system_prompt, subagents)`; checkpointer support follows LangGraph conventions but is verified, not assumed.

- [ ] **Step 0: Confirm installed deepagents API**

Run:
```bash
python -c "import inspect, deepagents; print(inspect.signature(deepagents.create_deep_agent))"
```
Expected: prints a signature including `model`, `tools`, `system_prompt`, `subagents`. Note whether a `checkpointer` parameter exists. If `create_deep_agent` is not exported, run `python -c "import deepagents; print(dir(deepagents))"` and use the documented constructor. Adapt Step 3's `_create` call to the real signature (do not invent parameters).

- [ ] **Step 1: Write the failing test**

`tests/test_agent_wiring.py`:
```python
from project_evaluator.config import Config
from project_evaluator import agent as A


def test_build_components_registers_four_subagents(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALUATOR_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    cfg = Config.from_env()
    cfg.ensure_dirs()
    comp = A.build_components(cfg)
    names = {sa["name"] for sa in comp["subagents"]}
    assert names == {"recon", "dependency-stack", "tech-fit", "evaluation-critic"}
    tool_names = {getattr(t, "__name__", "") for t in comp["main_tools"]}
    assert {"get_repo_history", "diff_since_last", "record_evaluation"} <= tool_names
    assert "evaluation-critic" in comp["system_prompt"]


def test_system_prompt_mandates_always_critic(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALUATOR_DATA_DIR", str(tmp_path))
    cfg = Config.from_env()
    cfg.ensure_dirs()
    comp = A.build_components(cfg)
    p = comp["system_prompt"].lower()
    assert "always" in p and "critic" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_wiring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'project_evaluator.agent'`

- [ ] **Step 3: Write implementation**

`src/project_evaluator/agent.py`:
```python
from __future__ import annotations

from .config import Config
from .db.store import Store
from .subagents.dependency_stack import build_dependency_stack
from .subagents.evaluation_critic import build_evaluation_critic
from .subagents.recon import build_recon
from .subagents.tech_fit import build_tech_fit
from .tools.citation_check import Evidence, check_evidence_list
from .tools.manifests import parse_manifests
from .tools.persistence import make_persistence_tools
from .tools.repo import build_project_map, clone_or_update_repo
from .tools.websearch import make_web_search

MAIN_SYSTEM_PROMPT = (
    "You are project-evaluator. Given a git URL, evaluate the QUALITY OF ITS "
    "TECHNOLOGY DECISIONS: purpose-fit, idiomatic use, and over/under-"
    "engineering. You never execute the repo's code.\n\n"
    "Flow: 1) call get_repo_history; if history exists, call diff_since_last "
    "and only re-evaluate changed/added files (reuse stored verdicts for "
    "unchanged ones, drop deleted). 2) Delegate to recon, then "
    "dependency-stack, then tech-fit via the task tool. 3) ALWAYS delegate to "
    "the evaluation-critic subagent as the final verification step — this is "
    "mandatory on every evaluation, full or incremental. 4) Incorporate the "
    "critic's verified/needs-revision/rejected lists, then call "
    "record_evaluation. 5) Present a clear report (delta report on re-runs).\n"
    "If a repo cannot be cloned, explain plainly and do not record anything."
)


def build_components(cfg: Config) -> dict:
    cfg.ensure_dirs()
    store = Store(cfg.db_path)
    store.init_db()

    def clone_tool(url: str, ref: str | None = None) -> str:
        """Clone or update the repo into its cached workspace and build a
        project map. Returns JSON with path, commit_sha, and the map."""
        import json
        r = clone_or_update_repo(url, cfg.workspaces_dir, ref)
        m = build_project_map(r["path"], skip_dirs=set(cfg.skip_dirs),
                              max_files=cfg.max_files, max_file_bytes=cfg.max_file_bytes)
        return json.dumps({"commit_sha": r["commit_sha"],
                           "fresh_clone": r["fresh_clone"],
                           "workspace": str(r["path"]), "project_map": m})

    def parse_manifests_tool(workspace_path: str) -> str:
        """Parse manifests under the given cached workspace path. Returns JSON
        inventory."""
        import json
        return json.dumps(parse_manifests(workspace_path))

    def citation_check_tool(workspace_path: str, evidences: list[dict]) -> str:
        """Layer-0 check: verify each evidence object resolves to real
        file:line and snippet. Returns JSON list of {claim, ok, reason}."""
        import json
        evs = [Evidence(**e) for e in evidences]
        res = check_evidence_list(workspace_path, evs)
        return json.dumps([r.__dict__ for r in res])

    web_search_tool = make_web_search(cfg.tavily_api_key)
    persistence = make_persistence_tools(store, cfg.workspaces_dir)
    main_tools = [persistence["get_repo_history"],
                  persistence["diff_since_last"],
                  persistence["record_evaluation"]]

    subagents = [
        build_recon(clone_tool),
        build_dependency_stack(parse_manifests_tool),
        build_tech_fit(read_tools=[], web_search_tool=web_search_tool),
        build_evaluation_critic(citation_check_tool, read_tools=[]),
    ]
    return {"store": store, "main_tools": main_tools, "subagents": subagents,
            "system_prompt": MAIN_SYSTEM_PROMPT}


def build_deep_agent(cfg: Config):
    """Construct the deepagents main agent. Call cfg.apply_observability_env()
    before this so LangSmith env is set."""
    from deepagents import create_deep_agent  # adapt name per Step 0 if needed

    comp = build_components(cfg)
    # Pass only parameters confirmed to exist in Step 0. checkpointer is added
    # here ONLY if Step 0 showed deepagents accepts it; otherwise omit it.
    agent = create_deep_agent(
        model=cfg.model,
        tools=comp["main_tools"],
        system_prompt=comp["system_prompt"],
        subagents=comp["subagents"],
    )
    return agent, comp
```

> Note for the implementing engineer: subagents' read tools use deepagents' built-in virtual-filesystem `read_file`/`ls` (auto-provided to subagents). `read_tools=[]` means "inherit built-ins"; do not fabricate custom read tools. recon writes `PROJECT_MAP.md` via the built-in `write_file`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_wiring.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/agent.py tests/test_agent_wiring.py
git commit -m "feat: deepagents wiring (tools + 4 subagents + main prompt)"
```

---

## Task 12: CLI / rich REPL

**Files:**
- Create: `src/project_evaluator/cli.py`
- Create: `tests/test_cli.py`

The LLM loop is isolated behind `run_repl(agent, console)`; pure helpers (`is_git_url`, `parse_command`, `render_report`) are unit-tested without an LLM.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from project_evaluator.cli import is_git_url, parse_command


def test_is_git_url():
    assert is_git_url("https://github.com/a/b.git")
    assert is_git_url("git@github.com:a/b.git")
    assert is_git_url("https://gitlab.com/a/b")
    assert not is_git_url("hello there")


def test_parse_command():
    assert parse_command("/exit") == ("exit", "")
    assert parse_command("/help") == ("help", "")
    assert parse_command("https://x/y.git") == ("message", "https://x/y.git")
    assert parse_command("how is the error handling?") == (
        "message", "how is the error handling?")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`src/project_evaluator/cli.py`:
```python
from __future__ import annotations

import re
import sys

from rich.console import Console
from rich.markdown import Markdown

from .config import Config

_GIT_RE = re.compile(r"^(https?://|git@)[\w.@:/\-~]+$")


def is_git_url(text: str) -> bool:
    t = text.strip()
    return bool(_GIT_RE.match(t)) and " " not in t


def parse_command(line: str) -> tuple[str, str]:
    line = line.strip()
    if line.startswith("/"):
        name = line[1:].split(" ", 1)[0].lower()
        rest = line[1 + len(name):].strip()
        return name, rest
    return "message", line


def render_report(console: Console, markdown_text: str) -> None:
    console.print(Markdown(markdown_text))


def run_repl(agent, comp, console: Console) -> None:
    thread = {"configurable": {"thread_id": "main"}}
    console.print("[bold]project-evaluator[/bold] — paste a git URL or ask. "
                  "/help, /exit")
    while True:
        try:
            line = console.input("\n[bold cyan]>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return
        cmd, arg = parse_command(line)
        if cmd == "exit":
            console.print("bye")
            return
        if cmd == "help":
            console.print("Paste a git URL to evaluate it. Re-paste later for "
                          "an incremental delta. Ask follow-up questions in "
                          "plain language. /exit to quit.")
            continue
        if not arg:
            continue
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": arg}]},
                                   config=thread)
            msg = result["messages"][-1]
            content = getattr(msg, "content", None) or msg.get("content", "")
            render_report(console, content if isinstance(content, str) else str(content))
        except Exception as e:  # surface model/agent errors plainly (spec §7)
            console.print(f"[red]Error:[/red] {e}")


def main() -> int:
    console = Console()
    cfg = Config.from_env()
    cfg.apply_observability_env()
    from .agent import build_deep_agent
    try:
        agent, comp = build_deep_agent(cfg)
    except Exception as e:
        console.print(f"[red]Failed to start:[/red] {e}")
        return 1
    run_repl(agent, comp, console)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/project_evaluator/cli.py tests/test_cli.py
git commit -m "feat: rich REPL CLI (pure helpers tested; main wires agent)"
```

---

## Task 13: Full-suite gate + live smoke test (flagged)

**Files:**
- Create: `tests/test_smoke_live.py`
- Create: `tests/fixtures/overengineered/` (toy repo)

- [ ] **Step 1: Write the (flagged) live test and fixture**

`tests/fixtures/overengineered/app.py`:
```python
class AbstractGreeterFactoryProvider:
    def create(self): return GreeterFactory()
class GreeterFactory:
    def make(self): return Greeter()
class Greeter:
    def greet(self, n): return f"hi {n}"

# A single print would do; this is deliberately over-engineered.
print(AbstractGreeterFactoryProvider().create().make().greet("world"))
```
`tests/fixtures/overengineered/requirements.txt`:
```
fastapi==0.110.0
```

`tests/test_smoke_live.py`:
```python
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                    reason="needs OPENAI_API_KEY")
def test_end_to_end_overengineered(tmp_path):
    """Initializes a local git repo from the over-engineered fixture and
    asks the agent to evaluate it; the report should mention
    over-engineering. Run with: pytest -m live"""
    from git import Repo
    from project_evaluator.config import Config
    from project_evaluator.agent import build_deep_agent

    src = tmp_path / "oe"
    subprocess.run([sys.executable, "-c",
                    "import shutil,sys;shutil.copytree(sys.argv[1],sys.argv[2])",
                    "tests/fixtures/overengineered", str(src)], check=True)
    repo = Repo.init(src)
    repo.config_writer().set_value("user", "email", "t@t.t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    repo.index.add(["app.py", "requirements.txt"])
    repo.index.commit("init")

    os.environ["EVALUATOR_DATA_DIR"] = str(tmp_path / "data")
    cfg = Config.from_env()
    cfg.apply_observability_env()
    agent, _ = build_deep_agent(cfg)
    out = agent.invoke(
        {"messages": [{"role": "user", "content": f"Evaluate {src}"}]},
        config={"configurable": {"thread_id": "t"}},
    )
    text = str(out["messages"][-1])
    assert "engineer" in text.lower()
```

- [ ] **Step 2: Verify the default suite stays green and excludes live**

Run: `python -m pytest -v`
Expected: All non-live tests PASS; `test_smoke_live.py` shows as deselected (`-m 'not live'` from pyproject).

- [ ] **Step 3: (Optional, manual) run the live smoke**

Run: `python -m pytest -m live -v` (requires `OPENAI_API_KEY`)
Expected: PASS — report text contains "engineer". If it fails on deepagents API mismatch, revisit Task 11 Step 0.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_live.py tests/fixtures/overengineered
git commit -m "test: flagged end-to-end smoke on over-engineered fixture"
```

---

## Task 14: README usage + finalize

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write usage docs**

Replace `README.md` content with: project purpose, install (`pip install -e .`), required env (`OPENAI_API_KEY`; optional `LANGSMITH_API_KEY` since tracing is on by default; optional `TAVILY_API_KEY`), run (`pe`), how incremental re-evaluation works, and the privacy note that LangSmith tracing (default on) sends source snippets to LangSmith cloud — set `LANGSMITH_TRACING=false` to disable.

- [ ] **Step 2: Run full suite**

Run: `python -m pytest -v`
Expected: all non-live tests PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README usage, env, privacy note"
```

---

## Self-Review (completed by plan author)

**1. Spec coverage:**
- §2 model/EVALUATOR_MODEL → Task 2. CLI TUI rich → Task 12. SQLite single file → Task 3 (conversation `SqliteSaver` checkpointer: Task 11 Step 0 verifies API; if unsupported, conversation memory is a documented follow-up — flagged below). File-unit incremental → Tasks 4/8/11 prompt. Always-on critic → Tasks 10/11. Cost table → Task 7 (wired for display in Task 12 follow-up — see gap). LangSmith default on → Tasks 2/12.
- §4 four subagents → Task 10. Evidence contract → Task 10 tech-fit. Layer-0 + Layer-1 critic → Tasks 6/10/11.
- §5 schema + 3 agent tools → Tasks 3/8. §6 first/re-run flow → Task 11 main prompt + Task 8 tools. §7 error handling → Task 4 (`RepoAccessError`), Task 2 (LangSmith), Task 12 (surfacing). §8 tests → every task is TDD; live flagged → Task 13. §9 layout → Tasks 1–12 create exactly the spec's files.

**2. Placeholder scan:** No "TBD/TODO". Task 11 Step 0 is a concrete executable verification step (not a placeholder) because the external `deepagents` signature must be confirmed, not assumed.

**3. Type consistency:** `Config.from_env`/`from_env_for_test`, `Store` method names, `Evidence` fields, `make_persistence_tools` keys, `build_*` subagent builders, and `build_components`/`build_deep_agent` are referenced consistently across tasks.

**Known gaps (intentional, flagged for execution):**
- **Conversation checkpointer** (spec §2/§3): wired only if Task 11 Step 0 confirms `deepagents` accepts a `checkpointer`. If not, add a thin LangGraph `SqliteSaver` integration task after Task 11 using the verified API — do not invent the parameter.
- **Usage table display** (spec §2 cost row): `usage.py` is built/tested (Task 7); surfacing per-subagent tokens in the REPL depends on the run-result/usage-metadata shape exposed by the installed `deepagents`/LangGraph. Add a small Task 12.5 once Task 11 Step 0 reveals where usage metadata lives on the result; render `render_usage_table` after each evaluation. Tracking this as an explicit execution-time follow-up rather than guessing the metadata path.
