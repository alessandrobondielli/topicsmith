"""Derived tables the figures and the notebook plot.

Anything with a judgement call in it -- how a cross-tab is normalised, what
threshold a co-occurrence has to clear -- lives here rather than in a notebook
cell, so it can be read, reused and corrected in one place.
"""

from __future__ import annotations

import collections

import pandas as pd


def area_counts(papers: pd.DataFrame) -> pd.Series:
    """Papers per main area, largest first."""
    return papers["main_area"].value_counts()


def subtopic_counts(subtopics: pd.DataFrame) -> pd.Series:
    return subtopics["subtopic"].value_counts()


def area_pairs(papers: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """The commonest (main, secondary) area pairs -- where the corpus straddles."""
    pairs = papers.dropna(subset=["main_area", "secondary_area"])
    if pairs.empty:
        return pd.DataFrame(columns=["main_area", "secondary_area", "papers"])
    return (
        pairs.groupby(["main_area", "secondary_area"]).size()
        .sort_values(ascending=False).head(top)
        .rename("papers").reset_index()
    )


def crosstab(
    papers: pd.DataFrame,
    column: str,
    *,
    row: str = "main_area",
    min_col_total: int = 1,
    normalise: str | None = None,
) -> pd.DataFrame:
    """Papers counted across the areas and one metadata column.

    Columns below *min_col_total* fold into "Other" rather than becoming a
    fringe of near-empty stripes -- a heatmap of forty one-paper venues is not a
    picture of anything.
    """
    pairs = papers[[row, column]].dropna().copy()
    if pairs.empty:
        return pd.DataFrame()
    pairs[column] = pairs[column].astype(str)

    totals = pairs[column].value_counts()
    small = set(totals[totals < min_col_total].index)
    if small:
        pairs[column] = pairs[column].where(~pairs[column].isin(small), "Other")

    table = pd.crosstab(pairs[row], pairs[column])
    order = table.sum(axis=1).sort_values(ascending=False).index
    cols = table.sum(axis=0).sort_values(ascending=False).index.tolist()
    if "Other" in cols:  # keep the catch-all last, whatever its size
        cols = [c for c in cols if c != "Other"] + ["Other"]
    table = table.loc[order, cols]

    if normalise == "row":
        table = table.div(table.sum(axis=1).replace(0, 1), axis=0) * 100
    elif normalise == "column":
        table = table.div(table.sum(axis=0).replace(0, 1), axis=1) * 100
    return table


def subtopic_cooccurrence(subtopics: pd.DataFrame, min_papers: int = 3):
    """Graph of subtopics that appear together on the same paper.

    *min_papers* is the knob that decides whether the map is readable. Two is
    almost always too low: it admits every one-off pairing and the labels end up
    in a lattice of crossing leader lines.
    """
    import networkx as nx

    frequent = subtopics["subtopic"].value_counts()
    keep = set(frequent[frequent >= min_papers].index)

    graph = nx.Graph()
    for subtopic, count in frequent.items():
        if subtopic in keep:
            graph.add_node(subtopic, papers=int(count))
    for _, group in subtopics[subtopics["subtopic"].isin(keep)].groupby("paper_id"):
        subs = sorted(set(group["subtopic"]))
        for i, a in enumerate(subs):
            for b in subs[i + 1:]:
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)
    return graph


def value_counts(papers: pd.DataFrame, column: str, top: int | None = None) -> pd.Series:
    """Counts for a column that may hold either scalars or lists.

    ``contribution_type`` is one per paper; an extra field declared as
    ``list[str]`` is many. Both are worth charting, and neither should need its
    own function.
    """
    if column not in papers:
        return pd.Series(dtype=int)
    counter: collections.Counter = collections.Counter()
    for entry in papers[column]:
        items = entry if isinstance(entry, list) else [entry]
        for item in items:
            if isinstance(item, str):
                if item.strip():
                    counter[item.strip()] += 1
            elif item is not None and not isinstance(item, list) and not pd.isna(item):
                counter[str(item)] += 1
    series = pd.Series(counter).sort_values(ascending=False)
    return series.head(top) if top else series


def wordcloud_image(
    counts: collections.Counter, *, width: int = 1600, height: int = 900, max_words: int = 70,
):
    """A wordcloud in the house palette, sized by frequency.

    Two changes from the obvious defaults, both about being read from a slide:
    **fewer words** (120 words on one plate meant the tail rendered at four
    points and said nothing), and a colour ramp that starts well down the
    sequential scale, so even a once-used term clears the surface instead of
    fading into it.
    """
    from wordcloud import WordCloud

    from . import viz

    ramp = viz.SEQUENTIAL[6:]
    font = _wordcloud_font()
    ceiling = max(counts.values()) if counts else 1
    lookup = {str(k).lower(): v for k, v in counts.items()}

    def color(word, font_size, position, orientation, random_state=None, **kwargs):
        # Darker for the more frequent terms, so colour repeats the size cue
        # rather than adding an unrelated dimension.
        rank = lookup.get(word.lower(), 1) / ceiling
        return ramp[min(len(ramp) - 1, int(rank * len(ramp)))]

    return WordCloud(
        width=width, height=height, background_color=viz.SURFACE, font_path=font,
        color_func=color, prefer_horizontal=0.95, collocations=False,
        margin=8, max_words=max_words, min_font_size=13, relative_scaling=0.55,
    ).generate_from_frequencies({str(k): float(v) for k, v in counts.items()})


def _wordcloud_font() -> str | None:
    """A path to the house sans, so the clouds are not the odd plate out.

    ``wordcloud`` renders with Pillow and takes a file path, not a family name,
    so it ignores the matplotlib font stack and falls back to its bundled
    monospace. Resolving the same stack by hand keeps the typeface consistent.
    """
    from matplotlib import font_manager

    from . import viz

    for family in viz.FONT_STACK:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
        except Exception:  # noqa: BLE001 - family simply not installed
            continue
        # Pillow reads .ttf/.otf; a macOS .ttc collection works too but only via
        # its first face, which for these families is the regular weight.
        if path:
            return path
    return None
