"""Pass 2: induce a canonical taxonomy from the pooled free-form labels.

Pass 1 produces one main topic and a handful of subtopics per paper, phrased
however each paper's content suggested. Counting those directly gives a long
tail of near-synonyms and no usable categories. Here the label pool goes to the
model and comes back as a small tree of areas, each with a definition and the
raw labels it absorbs.

The result is written as YAML and is **meant to be edited by hand** -- renaming
an area, merging two, rewriting a definition. Pass 3 reads whatever is in the
file, and the file is part of no cache key, so disagreeing with the model costs
one re-run of pass 3 and nothing else. That editing step is the whole reason
this tool exists rather than a clustering script.

A **seed taxonomy** goes in the other direction: areas you already know you want
are handed to the induction, which must keep them and induce the rest around
them.

Large corpora take the sharded path described in :mod:`topicforge.reduce`; the
merge call folds the partial taxonomies into one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .. import reduce
from ..config import Config, load_config
from ..llm import LLMClient
from ..prompts import Prompts
from ..schemas import Area, InducedTaxonomy, Taxonomy


def load_seed(cfg: Config, path: str | Path | None = None) -> Taxonomy | None:
    """The optional seed taxonomy, or None.

    Accepts the same YAML shape the pipeline writes, so an edited
    ``taxonomy.yaml`` from a previous edition can be handed straight back in as
    the seed for the next one.
    """
    path = Path(path) if path else cfg.paths.seed_taxonomy
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    areas = raw.get("areas") if isinstance(raw, dict) else raw
    if not areas:
        return None
    # A seed may legitimately give only a name and a definition.
    return Taxonomy(areas=[
        Area(
            name=a["name"],
            definition=a.get("definition", ""),
            subtopics=a.get("subtopics") or [a["name"]],
            absorbs=a.get("absorbs") or [],
            seeded=True,
        )
        for a in areas
    ])


def context(cfg: Config, raw: pd.DataFrame, seed: Taxonomy | None) -> dict:
    """Template context shared by the taxonomy and merge templates."""
    ok = raw[raw["ok"]] if "ok" in raw else raw
    return {
        "project": cfg.project,
        "n_papers": len(ok),
        "min_areas": cfg.topics.get("min_areas", 8),
        "max_areas": cfg.topics.get("max_areas", 14),
        "min_area_papers": cfg.topics.get("min_area_papers", 3),
        "max_area_papers": cfg.topics.get("max_area_papers", max(4, len(ok) // 5)),
        "seed": seed,
    }


def induce(
    raw: pd.DataFrame,
    cfg: Config | None = None,
    client: LLMClient | None = None,
    *,
    seed: Taxonomy | None = None,
    temperature: float | None = None,
    cache_salt: str = "",
    plan: reduce.Plan | None = None,
) -> tuple[Taxonomy, reduce.Plan]:
    """Induce one taxonomy from the pooled labels, sharding if it does not fit."""
    cfg = cfg or load_config()
    client = client or LLMClient(cfg)
    prompts = Prompts(cfg)
    plan = plan or reduce.plan(raw, cfg)
    base = context(cfg, raw, seed)

    def one(pool: reduce.Pool, salt: str) -> InducedTaxonomy:
        system, user = prompts.pair(
            "taxonomy", pool=pool, sharded=len(plan.pools) > 1, **pool.blocks(), **base
        )
        return client.one(user, InducedTaxonomy, system=system,
                          temperature=temperature, cache_salt=salt)

    if len(plan.pools) == 1:
        induced = one(plan.pools[0], cache_salt)
    else:
        partials = [
            one(pool, f"{cache_salt}shard-{i}/{len(plan.pools)}")
            for i, pool in enumerate(plan.pools)
        ]
        system, user = prompts.pair("merge", partials=partials, **base)
        induced = client.one(user, InducedTaxonomy, system=system,
                             temperature=temperature, cache_salt=f"{cache_salt}merge")

    seed_names = {a.name for a in seed.areas} if seed else set()
    return Taxonomy.from_induced(induced, seed_names), plan


def stability(
    raw: pd.DataFrame,
    cfg: Config | None = None,
    client: LLMClient | None = None,
    *,
    seed: Taxonomy | None = None,
) -> pd.DataFrame:
    """Induce the taxonomy several times and report how much the areas move.

    Compares runs by how well each one's area names and absorbed labels overlap
    with the first. High overlap means the categories are a property of the
    corpus rather than of one sampling run -- which is the answer to the first
    question anyone asks about LLM-derived categories.
    """
    cfg = cfg or load_config()
    client = client or LLMClient(cfg)
    n_runs = int(cfg.topics.get("stability_runs", 3))
    temperature = float(cfg.topics.get("stability_temperature", 0.7))
    threshold = float(cfg.topics.get("stability_match_threshold", 0.4))

    plan = reduce.plan(raw, cfg)
    runs = [
        induce(raw, cfg, client, seed=seed, temperature=temperature,
               cache_salt=f"stability-{i}-", plan=plan)[0]
        for i in range(n_runs)
    ]
    reference = runs[0]
    ref_labels = [
        {s.casefold() for s in a.absorbs + a.subtopics} for a in reference.areas
    ]

    rows = []
    for i, taxonomy in enumerate(runs):
        matched = 0
        for area in taxonomy.areas:
            labels = {s.casefold() for s in area.absorbs + area.subtopics}
            best = max(
                (len(labels & ref) / max(len(labels | ref), 1) for ref in ref_labels),
                default=0.0,
            )
            matched += best >= threshold
        rows.append({
            "run": i,
            "n_areas": len(taxonomy.areas),
            "areas": ", ".join(taxonomy.names()),
            "areas_matching_run0": matched,
            "match_rate": round(matched / max(len(taxonomy.areas), 1), 3),
        })
    return pd.DataFrame(rows)


HEADER = """\
# Canonical topic taxonomy for {project}.
#
# Induced by `topicforge taxonomy`. EDIT THIS FILE FREELY -- rename areas, merge
# two into one, rewrite a definition, delete an area you do not believe in.
# `topicforge assign` reads whatever is here, and this file is part of no cache
# key, so an edit costs one re-run of the assignment pass and nothing else.
#
# name       the label that appears on every chart and in every export
# definition what belongs in the area and what does not; the model reads this
# subtopics  the finer labels offered to the assignment pass
# absorbs    the raw pass-1 labels this area was induced from (provenance only)
# seeded     true if the area came from your seed file rather than the model
"""


def save(taxonomy: Taxonomy, cfg: Config) -> Path:
    """Write the taxonomy as hand-editable YAML."""
    path = cfg.paths.taxonomy
    body = yaml.safe_dump(taxonomy.model_dump(), sort_keys=False,
                          allow_unicode=True, width=100)
    path.write_text(HEADER.format(project=cfg.project) + "\n" + body, encoding="utf-8")
    return path


def load(cfg: Config) -> Taxonomy:
    path = cfg.paths.taxonomy
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `topicforge taxonomy` before assigning."
        )
    return Taxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
