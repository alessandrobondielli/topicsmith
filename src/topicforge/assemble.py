"""Joining the three passes into the tables everything else reads.

The interim artefacts are each one pass's output. This module turns them into
the three exports the notebook, the figures and any downstream script actually
use, plus a ``stats.json`` of the headline numbers -- so a hand-written summary
of the corpus cannot drift away from the charts.

The metadata sidecar is joined here rather than upstream, which means the
markdown does not have to still exist to re-export, and adding a column to
``metadata.csv`` costs one `topicforge export` and no LLM calls.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, load_config
from .documents import load_metadata
from .passes import taxonomy as taxonomy_pass
from .schemas import Taxonomy

LIST_COLUMNS = ("subtopics", "raw_subtopics")


def _parse_list(value: Any) -> list:
    """Read back a list column that went through CSV as its repr."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text in {"[]", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [t.strip() for t in text.split(",") if t.strip()]
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]


def read_assigned(cfg: Config) -> pd.DataFrame:
    path = cfg.paths.assigned
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `topicforge assign` before exporting."
        )
    frame = pd.read_csv(path)
    frame["paper_id"] = frame["paper_id"].astype(str)
    for column in LIST_COLUMNS:
        if column in frame:
            frame[column] = frame[column].apply(_parse_list)
    return frame


def build(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    """``{"papers", "subtopics", "areas"}`` -- everything downstream reads these."""
    cfg = cfg or load_config()
    papers = read_assigned(cfg)
    taxonomy = taxonomy_pass.load(cfg)

    meta = load_metadata(cfg)
    if meta is not None:
        title_col = cfg.metadata.get("title_column", "title")
        extra = [c for c in meta.columns if c not in ("paper_id", title_col)]
        if extra:
            papers = papers.merge(meta[["paper_id", *extra]], on="paper_id", how="left")

    return {
        "papers": papers,
        "subtopics": explode_subtopics(papers),
        "areas": area_table(papers, taxonomy),
    }


def explode_subtopics(papers: pd.DataFrame, column: str = "subtopics") -> pd.DataFrame:
    """One row per (paper, subtopic)."""
    work = papers[["paper_id", "main_area", column]].copy()
    work[column] = work[column].apply(lambda v: v if isinstance(v, list) else [])
    return (
        work.explode(column)
        .dropna(subset=[column])
        .rename(columns={column: "subtopic"})
        .reset_index(drop=True)
    )


def area_table(papers: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """One row per area: its definition, how many papers landed in it, and provenance.

    Areas with no papers are kept, at zero. An empty area is a finding -- either
    the taxonomy has a category the corpus does not support, or the assignment
    pass is avoiding it -- and dropping it would hide that.
    """
    main = papers["main_area"].value_counts()
    secondary = papers["secondary_area"].value_counts()
    total = max(len(papers), 1)
    rows = []
    for area in taxonomy.areas:
        n = int(main.get(area.name, 0))
        rows.append({
            "area": area.name,
            "definition": area.definition,
            "n_papers": n,
            "share_pct": round(100 * n / total, 1),
            "n_secondary": int(secondary.get(area.name, 0)),
            "seeded": bool(area.seeded),
            "subtopics": "; ".join(area.subtopics),
            "n_absorbed_labels": len(area.absorbs),
        })
    return pd.DataFrame(rows).sort_values("n_papers", ascending=False).reset_index(drop=True)


def stats(tables: dict[str, pd.DataFrame], cfg: Config | None = None) -> dict[str, Any]:
    """The headline numbers, in one place, so nothing hand-written can drift."""
    cfg = cfg or load_config()
    papers, areas, subtopics = tables["papers"], tables["areas"], tables["subtopics"]
    assigned = papers[papers["main_area"].notna()]
    confidence = pd.to_numeric(papers.get("confidence"), errors="coerce")

    out: dict[str, Any] = {
        "project": cfg.project,
        "model": cfg.llm.get("model"),
        "papers": int(len(papers)),
        "assigned": int(len(assigned)),
        "failed": int(len(papers) - len(assigned)),
        "areas": int(len(areas)),
        "areas_used": int((areas["n_papers"] > 0).sum()),
        "seeded_areas": int(areas["seeded"].sum()),
        "largest_area": (
            {"name": areas.iloc[0]["area"], "papers": int(areas.iloc[0]["n_papers"])}
            if len(areas) else None
        ),
        "distinct_subtopics": int(subtopics["subtopic"].nunique()),
        "papers_with_secondary_area": int(papers["secondary_area"].notna().sum()),
        "median_confidence": round(float(confidence.median()), 3) if confidence.notna().any() else None,
        "needs_review": int(papers["needs_review"].sum()) if "needs_review" in papers else 0,
    }
    if "excerpt_strategy" in papers:
        out["excerpt_strategy"] = papers["excerpt_strategy"].value_counts().to_dict()
    if "contribution_type" in papers:
        out["contribution_types"] = papers["contribution_type"].value_counts().to_dict()
    for facet in cfg.metadata.get("facets") or []:
        if facet in papers:
            out.setdefault("facets", {})[facet] = papers[facet].value_counts().to_dict()
    return out


def write(tables: dict[str, pd.DataFrame], cfg: Config | None = None) -> list[Path]:
    cfg = cfg or load_config()
    written = []
    for name, frame in tables.items():
        path = cfg.paths.processed / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)
    path = cfg.paths.processed / "stats.json"
    path.write_text(
        json.dumps(stats(tables, cfg), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    written.append(path)
    return written


def load(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    """Read back what :func:`write` wrote. What the notebook calls."""
    cfg = cfg or load_config()
    tables = {}
    for name in ("papers", "subtopics", "areas"):
        path = cfg.paths.processed / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run `topicforge export` first."
            )
        frame = pd.read_csv(path)
        if "paper_id" in frame:
            frame["paper_id"] = frame["paper_id"].astype(str)
        for column in LIST_COLUMNS:
            if column in frame:
                frame[column] = frame[column].apply(_parse_list)
        tables[name] = frame
    return tables
