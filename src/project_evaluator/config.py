from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", ".venv", "venv",
    "__pycache__", ".mypy_cache", ".pytest_cache", "target",
    ".idea", ".gradle", "vendor", ".next", "out",
}

DEFAULT_MODEL = "openai:gpt-5.4"


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
            model=os.environ.get("EVALUATOR_MODEL", DEFAULT_MODEL),
            data_dir=data_dir,
            max_files=int(os.environ.get("EVALUATOR_MAX_FILES", "4000")),
            max_file_bytes=int(os.environ.get("EVALUATOR_MAX_FILE_BYTES", "500000")),
            skip_dirs=frozenset(DEFAULT_SKIP_DIRS),
            langsmith_tracing=_bool(os.environ.get("LANGSMITH_TRACING"), True),
            langsmith_project=os.environ.get("LANGSMITH_PROJECT", "project-evaluator"),
            tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        )

    @classmethod
    def from_env_for_test(cls, data_dir) -> "Config":
        """Construct a Config rooted at an explicit data dir (tests only)."""
        import os
        os.environ["EVALUATOR_DATA_DIR"] = str(data_dir)
        return cls.from_env()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

    def apply_observability_env(self) -> None:
        """Inject LangSmith env. Default on; degrade gracefully without key."""
        # LangSmith tracing is ON by default by design (user decision). If the
        # user never set an API key, degrade gracefully (warn once, disable)
        # rather than crash. LANGSMITH_PROJECT is intentionally only set when
        # tracing is enabled; when disabled we leave it untouched.
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
