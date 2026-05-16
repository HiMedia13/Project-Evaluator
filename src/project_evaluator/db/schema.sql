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
