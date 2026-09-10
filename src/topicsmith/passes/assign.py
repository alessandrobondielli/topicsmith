"""Pass 3: re-assign every paper to the frozen taxonomy.

Pass 1's labels are unconstrained, so they cannot be counted -- twenty phrasings
of the same idea are twenty rows. Here each paper is classified against the
canonical areas with the choice restricted to the names that exist in the
taxonomy, as a ``Literal`` in the response model. A compliant backend cannot
emit anything else, and a non-compliant one fails validation and is retried, so
the counts are trustworthy either way.

The paper's own pass-1 labels are supplied as evidence, which means this pass
reads a short summary rather than the paper again -- cheap enough to re-run
every time the taxonomy is edited, which is exactly what it is for.

Assignments below ``topics.assign_review_threshold`` go to a review CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config, load_config
from ..llm import LLMClient
from ..prompts import Prompts
from ..schemas import Taxonomy, assignment_model, extra_fields


def _display(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v is not None and str(v).strip()) or "n/a"
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return "n/a"
    return str(value)


def run(
    raw: pd.DataFrame,
    taxonomy: Taxonomy,
    cfg: Config | None = None,
    client: LLMClient | None = None,
) -> pd.DataFrame:
    """Assign every paper to a canonical area."""
    cfg = cfg or load_config()
    client = client or LLMClient(cfg)
    prompts = Prompts(cfg)
    schema = assignment_model(taxonomy)

    base = {
        "project": cfg.project,
        "taxonomy": taxonomy.render(),
        "areas": taxonomy.names(),
        "extra_fields": extra_fields(cfg),
    }
    items = raw.to_dict("records")
    for item in items:
        shown = {k: _display(v) for k, v in item.items()}
        # A field declared in config after pass 1 ran is absent from the raw
        # records. Degrade it to "n/a" rather than failing the whole prompt.
        for field in base["extra_fields"]:
            shown.setdefault(field["name"], "n/a")
        item["_display"] = shown

    system = prompts.render("assign.system", paper=None, **base)
    results = client.map(
        items,
        lambda item: prompts.render("assign.user", paper=item, **base),
        schema,
        system=system,
        label="assign",
        describe=lambda item: str(item["paper_id"]),
    )

    threshold = float(cfg.topics.get("assign_review_threshold", 0.7))
    carry = [c for c in raw.columns if c not in ("subtopics", "ok")]
    rows = []
    for item, assignment in zip(items, results):
        record = {c: item.get(c) for c in carry}
        record["raw_subtopics"] = item.get("subtopics")
        if assignment is None:
            record |= {
                "main_area": None, "secondary_area": None, "subtopics": [],
                "confidence": 0.0, "why": "LLM call failed",
            }
        else:
            record |= assignment.model_dump()
            if record["secondary_area"] == "none":
                record["secondary_area"] = None
        rows.append(record)

    result = pd.DataFrame(rows)
    result["needs_review"] = result["main_area"].isna() | (result["confidence"] < threshold)
    return result


def write_outputs(table: pd.DataFrame, cfg: Config) -> tuple[Path, Path]:
    table.to_csv(cfg.paths.assigned, index=False)
    table[table["needs_review"]].to_csv(cfg.paths.review, index=False)
    return cfg.paths.assigned, cfg.paths.review
