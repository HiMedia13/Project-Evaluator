import logging
import os
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
    assert cfg.max_file_bytes == 500000
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
    monkeypatch.setenv("LANGSMITH_PROJECT", "project-evaluator")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    cfg = Config.from_env()
    with caplog.at_level(logging.WARNING):
        cfg.apply_observability_env()
    assert "LANGSMITH_API_KEY" in caplog.text
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_apply_observability_sets_env_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALUATOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "project-evaluator")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    cfg = Config.from_env()
    cfg.apply_observability_env()
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "project-evaluator"
