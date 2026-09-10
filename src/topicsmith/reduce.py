"""Getting the pass-1 label pool into a single prompt.

Pass 2 works by showing the model *every* label pass 1 produced, with counts, and
asking for a taxonomy that covers them. That is the right shape for the problem
-- the categories come from what the corpus actually contains -- but the pool
grows linearly with the corpus, and at some size the prompt no longer fits.

Three regimes, chosen automatically:

``single``
    The pool fits. One call, which is the behaviour this pipeline was built and
    validated with, and the only regime a few hundred papers ever reach.
``cutoff``
    The pool is over budget. Keep the most frequent labels that together account
    for ``topics.coverage`` of the total label mass and drop the singleton tail.
    A label that occurred once cannot define an area anyway -- the taxonomy
    prompt explicitly rejects categories that name a single paper.
``shards``
    Still over budget after the cutoff. Split into frequency-stratified batches,
    induce a partial taxonomy per batch, and merge the partials in one final
    call. Every shard sees the *head* of the distribution as shared context, so
    the partial taxonomies are drawn in comparable terms rather than each
    inventing its own frame.

Nothing here is lossy in the ``single`` regime, which is the point: a corpus that
used to work keeps working, byte for byte.
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

_WHITESPACE = re.compile(r"\s+")
_TRIM = re.compile(r"^[\s\"'`(\[]+|[\s\"'`)\].,;:]+$")


def normalise(label: str) -> str:
    """Canonical surface form of one label.

    Conservative on purpose: whitespace, wrapping punctuation and case. No
    stemming, no lemmatisation -- "language models" and "language modelling" are
    different labels here, and merging them is pass 2's job, where a model can
    see that they belong together and say so in a definition.
    """
    return _WHITESPACE.sub(" ", _TRIM.sub("", str(label))).strip()


def count(values: Iterable[Any]) -> collections.Counter:
    """Count labels case-insensitively, reporting each under its commonest spelling."""
    variants: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for value in values:
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if not isinstance(item, str):
                continue
            label = normalise(item)
            if label:
                variants[label.casefold()][label] += 1
    return collections.Counter({
        spellings.most_common(1)[0][0]: sum(spellings.values())
        for spellings in variants.values()
    })


def render(counter: collections.Counter, limit: int | None = None) -> str:
    """The ``- label (n)`` block a prompt shows."""
    items = counter.most_common(limit) if limit else counter.most_common()
    return "\n".join(f"- {label} ({n})" for label, n in items)


def estimate_tokens(text: str) -> int:
    """Rough token count. Deliberately generous -- being wrong low means a failed call."""
    return int(len(text) / 3.2)


def cover(counter: collections.Counter, coverage: float) -> collections.Counter:
    """The most frequent labels accounting for *coverage* of the total mass."""
    total = sum(counter.values())
    if total == 0:
        return counter
    target, running = total * coverage, 0
    kept: list[tuple[str, int]] = []
    for label, n in counter.most_common():
        kept.append((label, n))
        running += n
        if running >= target:
            break
    return collections.Counter(dict(kept))


@dataclass
class Pool:
    """One pass-2 input: three label distributions, possibly a slice of a larger set."""

    main_topics: collections.Counter
    subtopics: collections.Counter
    methods: collections.Counter
    shared: bool = False           # True for the head labels every shard sees

    def blocks(self) -> dict[str, str]:
        return {
            "main_topics": render(self.main_topics),
            "subtopics": render(self.subtopics),
            "methods": render(self.methods),
        }

    def tokens(self) -> int:
        return sum(estimate_tokens(b) for b in self.blocks().values())

    def n_labels(self) -> int:
        return len(self.main_topics) + len(self.subtopics) + len(self.methods)


@dataclass
class Plan:
    """What `topicsmith taxonomy` is about to do, and why."""

    mode: str                       # "single" | "cutoff" | "shards"
    pools: list[Pool]               # one entry for single/cutoff, N for shards
    n_labels_before: int
    n_labels_after: int
    tokens: int
    budget: int
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        dropped = self.n_labels_before - self.n_labels_after
        head = {
            "single": f"{self.n_labels_after} labels, ~{self.tokens:,} tokens — one call",
            "cutoff": (f"{self.n_labels_after} labels after dropping {dropped} rare ones, "
                       f"~{self.tokens:,} tokens — one call"),
            "shards": (f"{self.n_labels_after} labels over {len(self.pools)} shards "
                       f"(~{self.tokens:,} tokens) — {len(self.pools)} inductions + 1 merge"),
        }[self.mode]
        return "\n".join([head, *(f"  {n}" for n in self.notes)])


def _split(counter: collections.Counter, head: int, n_shards: int
           ) -> tuple[collections.Counter, list[collections.Counter]]:
    """Head labels shared by every shard; the tail dealt round-robin across shards."""
    ordered = counter.most_common()
    shared = collections.Counter(dict(ordered[:head]))
    shards = [collections.Counter() for _ in range(n_shards)]
    for i, (label, n) in enumerate(ordered[head:]):
        shards[i % n_shards][label] = n
    return shared, shards


def plan(raw: pd.DataFrame, cfg) -> Plan:
    """Decide how the pass-1 labels reach pass 2."""
    ok = raw[raw["ok"]] if "ok" in raw else raw
    counters = {
        "main_topics": count(ok["main_topic"]),
        "subtopics": count(ok["subtopics"]),
        "methods": count(ok["method_family"]),
    }
    before = sum(len(c) for c in counters.values())

    budget = int(cfg.topics.get("pool_budget_tokens", 12000))
    coverage = float(cfg.topics.get("coverage", 0.95))
    head_share = float(cfg.topics.get("shard_head_share", 0.25))

    whole = Pool(counters["main_topics"], counters["subtopics"], counters["methods"])
    if whole.tokens() <= budget:
        return Plan("single", [whole], before, whole.n_labels(), whole.tokens(), budget)

    counters = {k: cover(c, coverage) for k, c in counters.items()}
    trimmed = Pool(counters["main_topics"], counters["subtopics"], counters["methods"])
    notes = [
        f"pool exceeded {budget:,} tokens; kept the labels covering "
        f"{coverage:.0%} of occurrences"
    ]
    if trimmed.tokens() <= budget:
        return Plan("cutoff", [trimmed], before, trimmed.n_labels(),
                    trimmed.tokens(), budget, notes)

    n_shards = max(2, math.ceil(trimmed.tokens() / (budget * (1 - head_share))))
    pools: list[Pool] = []
    shared_parts, shard_parts = {}, {}
    for key, counter in counters.items():
        head = max(1, int(len(counter) * head_share))
        shared_parts[key], shard_parts[key] = _split(counter, head, n_shards)
    for i in range(n_shards):
        pools.append(Pool(
            main_topics=shared_parts["main_topics"] + shard_parts["main_topics"][i],
            subtopics=shared_parts["subtopics"] + shard_parts["subtopics"][i],
            methods=shared_parts["methods"] + shard_parts["methods"][i],
            shared=True,
        ))
    notes.append(
        f"still over budget; split into {n_shards} shards, each carrying the "
        f"{head_share:.0%} most frequent labels as shared context"
    )
    return Plan("shards", pools, before, trimmed.n_labels(), trimmed.tokens(), budget, notes)
