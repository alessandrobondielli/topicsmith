"""Prompt templates.

Every prompt this tool sends is a Jinja template on disk, not a string in the
source. That is deliberate: the prompts *are* the domain configuration. Adapting
topicforge from NLP papers to clinical trials or to medieval history is a matter
of editing eight files, not of forking the package.

``topicforge init`` copies the shipped defaults into the project's ``prompts/``
directory. Resolution is project-first with the shipped template as fallback, so
deleting a file you did not want to customise restores the default rather than
breaking the run.

Each pass has two templates, ``<pass>.system.j2`` and ``<pass>.user.j2``, and
receives the same context, so anything available in one is available in both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import ChoiceLoader, Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from .config import Config, load_config

PASSES = ("extract", "taxonomy", "merge", "assign")
SHIPPED = Path(__file__).parent / "templates"
SHIPPED_PROMPTS = SHIPPED / "prompts"


class PromptError(RuntimeError):
    pass


class Prompts:
    """Loads and renders the templates for one project."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.env = Environment(
            loader=ChoiceLoader([
                FileSystemLoader(str(self.cfg.paths.prompts)),
                FileSystemLoader(str(SHIPPED_PROMPTS)),
            ]),
            undefined=StrictUndefined,   # a typo in a template is an error, not a blank
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

    def render(self, name: str, /, **context: Any) -> str:
        try:
            return self.env.get_template(f"{name}.j2").render(**context).strip()
        except TemplateNotFound as exc:
            raise PromptError(
                f"No template {name}.j2 in {self.cfg.paths.prompts} or in the shipped "
                "defaults. Run `topicforge init --prompts-only` to restore them."
            ) from exc

    def pair(self, pass_name: str, /, **context: Any) -> tuple[str, str]:
        """``(system, user)`` for one pass, rendered against the same context."""
        return (
            self.render(f"{pass_name}.system", **context),
            self.render(f"{pass_name}.user", **context),
        )

    def source(self, name: str) -> Path:
        """Where a template actually resolved from -- for `topicforge prompts`."""
        local = self.cfg.paths.prompts / f"{name}.j2"
        return local if local.is_file() else SHIPPED_PROMPTS / f"{name}.j2"


def shipped_files() -> list[Path]:
    """Every shipped prompt template, for `init` to copy."""
    return sorted(SHIPPED_PROMPTS.glob("*.j2"))
