"""Configuration loading.

Everything topicforge needs to know -- where the papers are, which endpoint to
talk to, how to slice a paper into sections, how large a taxonomy to induce --
lives in ``config.yaml`` at the root of a *project directory*, the one created
by ``topicforge init``.

Strings of the form ``${VAR}`` or ``${VAR:default}`` are expanded from the
environment when the file is read, so an API key never has to sit in a file that
might be committed.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

CONFIG_NAME = "config.yaml"


class ConfigError(RuntimeError):
    pass


def _expand(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` / ``${VAR:default}`` in strings."""
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) or "")

        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from *start* until a directory containing config.yaml is found."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / CONFIG_NAME).is_file():
            return candidate
    raise ConfigError(
        f"No {CONFIG_NAME} found in {here} or any parent directory.\n"
        "Run `topicforge init <dir>` to create a project, then run commands from "
        "inside it (or pass --config <path>)."
    )


class Paths:
    """Resolved absolute paths for every directory topicforge touches."""

    def __init__(self, root: Path, spec: dict[str, Any]):
        self.root = root

        def under(key: str, default: str) -> Path:
            value = spec.get(key) or default
            path = Path(os.path.expanduser(str(value)))
            return path if path.is_absolute() else (root / path)

        # Inputs.
        self.pdfs = under("pdfs", "pdfs")
        self.markdown = under("markdown", "data/markdown")
        self.prompts = under("prompts", "prompts")
        self.metadata = under("metadata", "metadata.csv")
        self.seed_taxonomy = under("seed_taxonomy", "seed_taxonomy.yaml")

        # TOPICFORGE_DATA_DIR redirects every derived artefact -- cache, interim
        # tables, exports -- without touching the source documents. Useful for
        # keeping two runs side by side, and for exercising the pipeline against a
        # throwaway backend without poisoning the real response cache.
        self.data = Path(os.environ.get("TOPICFORGE_DATA_DIR") or under("data", "data"))
        self.cache = self.data / "cache"
        self.llm_cache = self.cache / "llm_cache"
        self.interim = self.data / "interim"
        self.processed = self.data / "processed"
        self.figures = Path(os.environ.get("TOPICFORGE_FIGURES_DIR") or under("figures", "figures"))

    # Named artefacts, so nothing downstream has to spell a filename twice.
    @property
    def raw_topics(self) -> Path:
        return self.interim / "topics_raw.jsonl"

    @property
    def taxonomy(self) -> Path:
        return self.interim / "taxonomy.yaml"

    @property
    def stability(self) -> Path:
        return self.interim / "taxonomy_stability.csv"

    @property
    def assigned(self) -> Path:
        return self.interim / "topics_assigned.csv"

    @property
    def review(self) -> Path:
        return self.interim / "topics_review.csv"

    def ensure(self) -> None:
        """Create every writable directory. Safe to call repeatedly."""
        for d in (self.cache, self.llm_cache, self.interim, self.processed,
                  self.figures, self.markdown):
            d.mkdir(parents=True, exist_ok=True)


class Config:
    def __init__(self, raw: dict[str, Any], root: Path):
        self.raw = raw
        self.root = root
        self.project: str = raw.get("project") or root.name
        self.paths = Paths(root, raw.get("paths") or {})
        self.llm: dict[str, Any] = raw.get("llm") or {}
        self.excerpt: dict[str, Any] = raw.get("excerpt") or {}
        self.sections: dict[str, Any] = raw.get("sections") or {}
        self.topics: dict[str, Any] = raw.get("topics") or {}
        self.metadata: dict[str, Any] = raw.get("metadata") or {}
        self.viz: dict[str, Any] = raw.get("viz") or {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Config project={self.project!r} root={self.root} model={self.llm.get('model')!r}>"


@lru_cache(maxsize=None)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the project configuration.

    With no argument the project root is discovered by walking up from the
    current directory, which is what every CLI command does.
    """
    if path is None:
        root = find_project_root()
        path = root / CONFIG_NAME
    else:
        path = Path(path).resolve()
        if path.is_dir():
            path = path / CONFIG_NAME
        if not path.is_file():
            raise ConfigError(f"{path} does not exist.")
        root = path.parent
    raw = _expand(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    cfg = Config(raw, root)
    cfg.paths.ensure()
    return cfg
