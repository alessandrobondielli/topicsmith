"""Section extraction from the converted paper markdown.

Topic labelling reads Title + Abstract + Keywords + Introduction + Conclusions
rather than the whole paper: it is roughly a quarter of the tokens, and -- more
importantly -- it leaves out Related Work, which otherwise drags the labels
towards what a paper *cites* instead of what it *does*.

Header conventions vary (numbered or not, levels 1-6, and not always in
English), so matching is by regex against a canonical set of section kinds. The
patterns are **configurable**: the ``sections:`` block of ``config.yaml``
replaces or extends :data:`DEFAULT_SECTIONS`, which is what makes the tool
usable on a corpus that does not look like an English CS paper.

Papers that yield no parseable Introduction or Conclusions -- typically extended
abstracts -- fall back to full text automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HEADER = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
# Leading section numbering: "3.", "A.1.", "IV)" ...
_NUMBERING = re.compile(r"^(?:[0-9]+|[A-Z]|[IVXL]+)(?:\.[0-9A-Za-z]+)*[.)]?\s+")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Order matters: the first pattern that matches a header wins, so the more
# specific kinds are listed before the catch-alls. Kept as an ordered mapping so
# a config block can override one kind without restating the rest.
DEFAULT_SECTIONS: dict[str, list[str]] = {
    "abstract": [r"^abstract", r"^sommario", r"^riassunto", r"^resumen", r"^zusammenfassung"],
    "keywords": [r"^keywords?", r"^index terms", r"^parole chiave"],
    "introduction": [r"^introduction", r"^introduzione", r"^background and motivation"],
    "related_work": [
        r"related work", r"previous work", r"state of the art", r"literature review",
        r"lavori correlati",
    ],
    "conclusion": [r"conclusion", r"conclusioni", r"final remarks", r"closing remarks"],
    "limitations": [r"limitation", r"limitazioni"],
    "references": [r"^references?", r"^bibliograph", r"^bibliografia", r"^works cited"],
    "acknowledgements": [r"acknowledge?ment", r"ringraziament"],
    "declaration": [
        r"^declaration on generative ai", r"^ethic", r"^data availability",
        r"^conflict of interest", r"^funding",
    ],
    "appendix": [r"^appendix", r"^appendice", r"^supplementary"],
    # Only reached when nothing above matched, so a real "Conclusions" section is
    # always preferred over a "Discussion" one.
    "discussion": [r"^discussion", r"^discussione", r"discussion$"],
}

# Lines matching these are dropped wholesale: running headers, copyright
# notices, the venue banner a proceedings template stamps on every page.
DEFAULT_BOILERPLATE: list[str] = [
    r"^©\s*20\d\d",
    r"^copyright\s*©",
    r"^published in proceedings",
    r"^proceedings of the",
]

# Sections that must never be folded into an excerpt: back matter, plus the one
# that this whole extraction exists to avoid.
EXCLUDED_FROM_EXCERPT = frozenset(
    {"related_work", "references", "acknowledgements", "appendix", "declaration"}
)
BACK_MATTER = frozenset({"references", "acknowledgements", "declaration", "appendix"})


class SectionRules:
    """Compiled header patterns and boilerplate filters for one project."""

    def __init__(self, sections: dict[str, Iterable[str]] | None = None,
                 boilerplate: Iterable[str] | None = None):
        merged = dict(DEFAULT_SECTIONS)
        for kind, patterns in (sections or {}).items():
            if patterns is None:  # `abstract: null` disables a kind entirely
                merged.pop(kind, None)
            else:
                merged[kind] = list(patterns)
        self.patterns: list[tuple[str, re.Pattern[str]]] = [
            (kind, re.compile("|".join(f"(?:{p})" for p in patterns), re.I))
            for kind, patterns in merged.items() if patterns
        ]
        rules = list(boilerplate) if boilerplate is not None else DEFAULT_BOILERPLATE
        self.boilerplate = [re.compile(p, re.I) for p in rules]

    @classmethod
    def from_config(cls, cfg) -> "SectionRules":
        spec = dict(cfg.sections or {})
        return cls(
            {k: v for k, v in spec.items() if k != "boilerplate"},
            spec.get("boilerplate"),
        )

    def classify(self, title: str) -> str:
        """Map a header string to a canonical section kind."""
        stripped = _NUMBERING.sub("", title).strip()
        for kind, pattern in self.patterns:
            if pattern.search(stripped):
                return kind
        return "other"

    def is_boilerplate(self, line: str) -> bool:
        return any(p.match(line) for p in self.boilerplate)


DEFAULT_RULES = SectionRules()


@dataclass
class Section:
    level: int
    title: str
    kind: str
    body: str

    @property
    def words(self) -> int:
        return len(self.body.split())


@dataclass
class Paper:
    """A parsed markdown paper."""

    paper_id: str
    doc_title: str
    preamble: str
    sections: list[Section]

    def first(self, kind: str) -> Section | None:
        """The first section of *kind*, or None."""
        return next((s for s in self.sections if s.kind == kind), None)

    def gather(self, kind: str) -> tuple[str, str] | None:
        """Full text of the first *kind* section, including its subsections.

        Papers often put an empty ``## 1. Introduction`` header above the real
        content in ``### 1.1 Background`` and ``### 1.2 ...``. Taking the
        header's own body alone would return nothing, so everything down to the
        next header at the same or shallower level is folded in -- minus any
        child that is itself related work or back matter, which is exactly the
        material this extraction exists to avoid.

        Returns ``(title, text)`` or None.
        """
        for i, section in enumerate(self.sections):
            if section.kind != kind:
                continue
            parts = [section.body] if section.body.strip() else []
            for child in self.sections[i + 1:]:
                if child.level <= section.level:
                    break
                if child.kind in EXCLUDED_FROM_EXCERPT:
                    continue
                header = child.title or ""
                parts.append(f"{header}\n{child.body}".strip())
            return section.title, "\n\n".join(p for p in parts if p.strip()).strip()
        return None

    def kinds(self) -> set[str]:
        return {s.kind for s in self.sections}

    def body_without_backmatter(self) -> str:
        """Everything except references, acknowledgements, declarations, appendices."""
        parts = [self.preamble] if self.preamble.strip() else []
        for s in self.sections:
            if s.kind in BACK_MATTER:
                continue
            parts.append(f"{'#' * s.level} {s.title}\n{s.body}")
        return "\n\n".join(parts).strip()


def clean_text(text: str, rules: SectionRules = DEFAULT_RULES) -> str:
    """Drop image embeds and recurring page boilerplate."""
    text = _IMAGE.sub("", text)
    lines = [ln for ln in text.splitlines() if not rules.is_boilerplate(ln.strip())]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse(path: str | Path, paper_id: str | None = None,
          rules: SectionRules = DEFAULT_RULES) -> Paper:
    """Split a markdown file into canonical sections."""
    path = Path(path)
    raw = clean_text(path.read_text(encoding="utf-8", errors="ignore"), rules)

    doc_title = ""
    preamble: list[str] = []
    sections: list[Section] = []
    current: Section | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            current.body = "\n".join(buffer).strip()
            sections.append(current)

    in_code = False
    for line in raw.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADER.match(line)
        if not match:
            (buffer if current is not None else preamble).append(line)
            continue
        level, title = len(match.group(1)), match.group(2)
        # A level-1 header before any section is the paper title.
        if level == 1 and current is None and not doc_title:
            doc_title = title
            continue
        flush()
        current, buffer = Section(level, title, rules.classify(title), ""), []
    flush()

    return Paper(
        paper_id=paper_id if paper_id is not None else path.stem,
        doc_title=doc_title.strip(),
        preamble="\n".join(preamble).strip(),
        sections=sections,
    )


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " […]"


def build_excerpt(
    paper: Paper,
    *,
    title: str | None = None,
    abstract: str | None = None,
    max_words: int = 3500,
    min_words: int = 150,
) -> tuple[str, str]:
    """Assemble the text sent to the LLM for one paper.

    *title* and *abstract* override anything parsed out of the conversion when a
    metadata sidecar supplied them, since a sidecar is the more reliable source.
    Returns ``(text, strategy)`` where strategy is ``"sections"`` or
    ``"full_text"`` -- recorded per paper, so a broken conversion shows up as a
    short excerpt rather than as a mysterious topic label.
    """
    blocks: list[str] = []
    heading = title or paper.doc_title
    if heading:
        blocks.append(f"# TITLE\n{heading}")

    abstract_section = paper.first("abstract")
    abstract_text = abstract or (abstract_section.body if abstract_section else "")
    if abstract_text:
        blocks.append(f"# ABSTRACT\n{abstract_text.strip()}")

    keywords = paper.first("keywords")
    if keywords and keywords.body.strip():
        blocks.append(f"# AUTHOR KEYWORDS\n{keywords.body.strip()}")

    intro = paper.gather("introduction")
    conclusion = paper.gather("conclusion") or paper.gather("discussion")

    if intro is None and conclusion is None:
        # Nothing structural to work with: send the paper, minus back matter.
        body = paper.body_without_backmatter()
        if body:
            blocks.append(f"# FULL TEXT\n{_truncate(body, max_words)}")
        return "\n\n".join(blocks).strip(), "full_text"

    if intro and intro[1]:
        blocks.append(f"# INTRODUCTION\n{_truncate(intro[1], int(max_words * 0.6))}")
    if conclusion and conclusion[1]:
        blocks.append(f"# CONCLUSIONS\n{_truncate(conclusion[1], int(max_words * 0.3))}")
    limitations = paper.gather("limitations")
    if limitations and limitations[1]:
        blocks.append(f"# LIMITATIONS\n{_truncate(limitations[1], int(max_words * 0.1))}")

    text = "\n\n".join(blocks).strip()
    # A parse can technically succeed while yielding almost nothing useful.
    if len(text.split()) < min_words:
        body = paper.body_without_backmatter()
        if len(body.split()) > len(text.split()):
            return (
                "\n\n".join(blocks[:3] + [f"# FULL TEXT\n{_truncate(body, max_words)}"]).strip(),
                "full_text",
            )
    return text, "sections"
