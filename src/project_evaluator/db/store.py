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
