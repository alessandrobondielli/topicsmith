"""Pass 1: free-form labels, one call per paper.

Nothing here is constrained to a fixed vocabulary. The point is to let the model
describe each paper in its own words, so that the taxonomy induced in pass 2
comes from what the corpus actually contains rather than from a list written
before anyone read it.

Output is ``data/interim/topics_raw.jsonl``, and it is the expensive artefact of
the whole pipeline: passes 2 and 3 read it, so a taxonomy can be rebuilt, edited
by hand, or thrown away entirely without touching the papers again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..documents import excerpts
from ..llm import LLMClient
from ..prompts import Prompts
from ..schemas import extra_fields, paper_topics_model


def context(cfg: Config) -> dict:
    """Template context shared by both extract templates."""
    return {
        "project": cfg.project,
        "min_subtopics": cfg.topics.get("min_subtopics", 2),
        "max_subtopics": cfg.topics.get("max_subtopics", 5),
        "extra_fields": extra_fields(cfg),
    }


def run(
    corpus: pd.DataFrame,
    cfg: Config | None = None,
    client: LLMClient | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Extract free-form topics for every paper in *corpus*."""
    cfg = cfg or load_config()
    client = client or LLMClient(cfg)
    prompts = Prompts(cfg)
    schema = paper_topics_model(cfg)
    base = context(cfg)

    items = excerpts(corpus, cfg)
    if limit:
        items = items[:limit]

    system = prompts.render("extract.system", paper=None, **base)
    results = client.map(
        items,
        lambda item: prompts.render("extract.user", paper=item, **base),
        schema,
        system=system,
        label="extract",
        describe=lambda item: f"{item['paper_id']} {item['title'][:50]}",
    )

    blank = {name: None for name in schema.model_fields}
    rows = []
    for item, topics in zip(items, results):
        record = {
            "paper_id": item["paper_id"],
            "title": item["title"],
            "excerpt_strategy": item["strategy"],
            "excerpt_words": item["words"],
        }
        if topics is None:
            record |= blank | {"subtopics": [], "ok": False}
        else:
            record |= topics.model_dump()
            record["ok"] = True
        rows.append(record)
    return pd.DataFrame(rows)


def write_raw(table: pd.DataFrame, cfg: Config) -> Path:
    """Persist pass-1 output as JSONL -- the artefact passes 2 and 3 depend on."""
    path = cfg.paths.raw_topics
    with path.open("w", encoding="utf-8") as fh:
        for record in table.to_dict("records"):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_raw(cfg: Config) -> pd.DataFrame:
    path = cfg.paths.raw_topics
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `topicsmith extract` before this step."
        )
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty.")
    return pd.DataFrame([json.loads(ln) for ln in lines])
