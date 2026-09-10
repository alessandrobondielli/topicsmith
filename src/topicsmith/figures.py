"""The standard figure set.

Both `topicsmith figures` and the notebook call these, so the two cannot drift:
what you explore in Jupyter is what a Makefile renders on a server with no
display attached.

Each function takes the exported tables and returns a matplotlib figure, or
``None`` when the corpus has nothing to show for it -- an extra field nobody
declared, a metadata facet nobody supplied. :func:`render_all` saves everything
that came back.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from . import analysis, network, viz
from .config import Config, load_config
from .schemas import extra_fields


def areas(tables: dict[str, pd.DataFrame], cfg: Config):
    """Papers per research area -- the headline chart."""
    counts = analysis.area_counts(tables["papers"])
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 0.58 * len(counts) + 0.6))
    viz.hbar(ax, counts.index, counts.values)
    viz.figure_title(
        fig, f"{cfg.project} by research area",
        f"{len(tables['papers'])} papers · each counted once, in the area of its "
        "main contribution",
    )
    return fig


def subtopics(tables: dict[str, pd.DataFrame], cfg: Config, top: int = 20):
    """The commonest subtopic labels."""
    counts = analysis.subtopic_counts(tables["subtopics"])
    if counts.empty:
        return None
    shown = counts.head(top)
    fig, ax = plt.subplots(figsize=(12, 0.52 * len(shown) + 0.6))
    viz.hbar(ax, shown.index, shown.values)
    viz.figure_title(fig, "Most common subtopics",
                     f"the {len(shown)} largest of {len(counts)} subtopic labels")
    return fig


def subtopic_cloud(tables: dict[str, pd.DataFrame], cfg: Config):
    """The same distribution as a cloud -- shape at a glance rather than rank."""
    counts = analysis.subtopic_counts(tables["subtopics"])
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.imshow(
        analysis.wordcloud_image(collections.Counter(counts.to_dict()),
                                 width=1600, height=640),
        interpolation="bilinear",
    )
    ax.set_axis_off()
    viz.figure_title(fig, "Subtopics", "sized by the number of papers carrying each label")
    return fig


def subtopic_map(tables: dict[str, pd.DataFrame], cfg: Config, min_papers: int | None = None):
    """Subtopics that share papers, as a map."""
    min_papers = int(min_papers or cfg.viz.get("cooccurrence_min_papers", 3))
    graph = analysis.subtopic_cooccurrence(tables["subtopics"], min_papers=min_papers)
    if not graph.number_of_edges():
        return None
    return network.draw_topic_map(
        graph, figsize=(15, 10),
        title="Subtopics that appear on the same paper",
        subtitle=f"node size and shade = papers · edge width = papers sharing both "
                 f"subtopics · subtopics on at least {min_papers} papers",
    )


def contributions(tables: dict[str, pd.DataFrame], cfg: Config):
    """What kind of contribution the corpus is made of."""
    counts = analysis.value_counts(tables["papers"], "contribution_type")
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 0.52 * len(counts) + 0.8))
    viz.hbar(ax, counts.index, counts.values)
    viz.figure_title(fig, "What kind of contribution",
                     "as classified in the first reading pass")
    return fig


def facet(tables: dict[str, pd.DataFrame], cfg: Config, column: str):
    """Research area against one metadata column, as a heatmap.

    Rows are shaded against their own maximum: one dominant column (a venue that
    contributed half the corpus) otherwise paints every other cell the same
    near-white and throws away the comparison the chart exists to make.
    """
    table = analysis.crosstab(tables["papers"], column)
    if table.empty or table.shape[1] < 2:
        return None
    fig, ax = plt.subplots(
        figsize=(2.0 + 1.5 * table.shape[1], 0.62 * table.shape[0] + 1.8)
    )
    viz.heatmap(ax, table.values, table.index, table.columns, normalise_rows=True)
    viz.figure_title(fig, f"Research area by {column}",
                     "papers · each row shaded against its own maximum")
    return fig


def extra_field(tables: dict[str, pd.DataFrame], cfg: Config, name: str, top: int = 14):
    """A bar chart for one field declared in ``topics.extra_fields``."""
    counts = analysis.value_counts(tables["papers"], name, top=top)
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(counts) + 0.8))
    viz.hbar(ax, counts.index, counts.values)
    viz.figure_title(fig, name.replace("_", " ").capitalize(),
                     "a paper may count for more than one")
    return fig


def render_all(tables: dict[str, pd.DataFrame], cfg: Config | None = None,
               *, close: bool = True) -> list[Path]:
    """Render and save every figure the corpus supports."""
    cfg = cfg or load_config()
    viz.apply_style(theme=cfg.viz.get("theme"), scale=cfg.viz.get("scale"))

    jobs: list[tuple[str, Callable]] = [
        ("areas", areas),
        ("subtopics", subtopics),
        ("subtopic_cloud", subtopic_cloud),
        ("subtopic_map", subtopic_map),
        ("contributions", contributions),
    ]
    for column in cfg.metadata.get("facets") or []:
        jobs.append((f"area_by_{column}", lambda t, c, col=column: facet(t, c, col)))
    for field in extra_fields(cfg):
        jobs.append((field["name"], lambda t, c, n=field["name"]: extra_field(t, c, n)))

    written: list[Path] = []
    for name, build in jobs:
        fig = build(tables, cfg)
        if fig is None:
            continue
        written.append(viz.save(fig, name, cfg))
        if close:
            plt.close(fig)
    return written
