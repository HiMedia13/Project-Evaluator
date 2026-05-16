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
