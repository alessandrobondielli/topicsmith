"""Finding the papers, converting them, and attaching optional metadata.

The corpus is a directory of PDFs. A paper's **id is its file stem**, and that
id is the join key for everything downstream: the markdown file, the metadata
sidecar, every interim table and every export. Nothing else about a paper is
assumed -- no submission system, no bibliographic database.

Conversion goes through `opendataloader-pdf`, in a single batched call because
each call spawns a JVM. It is an optional dependency: a project that already has
markdown or plain text points ``paths.markdown`` at it and never runs `ingest`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import markdown as md
from .config import Config, load_config

TEXT_SUFFIXES = (".md", ".markdown", ".txt")


class IngestError(RuntimeError):
    pass


def find_pdfs(cfg: Config) -> list[Path]:
    """Every PDF under ``paths.pdfs``, recursively, sorted by id."""
    root = cfg.paths.pdfs
    if not root.is_dir():
        raise IngestError(
            f"{root} does not exist. Set paths.pdfs in config.yaml to the directory "
            "holding your papers."
        )
    return sorted(root.rglob("*.pdf"), key=lambda p: p.stem)


def find_texts(cfg: Config) -> dict[str, Path]:
    """Converted text keyed by paper id, from ``paths.markdown``."""
    root = cfg.paths.markdown
    if not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            # First one wins, so a .md beats a .txt of the same name.
            found.setdefault(path.stem, path)
    return found


def ingest(cfg: Config | None = None, *, force: bool = False,
           limit: int | None = None) -> tuple[list[str], list[str]]:
    """Convert every PDF to markdown. Returns ``(converted_ids, skipped_ids)``.

    Papers whose markdown already exists are skipped unless *force*, so adding
    ten papers to a corpus of a thousand converts ten.
    """
    cfg = cfg or load_config()
    try:
        import opendataloader_pdf  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise IngestError(
            "opendataloader-pdf is not installed. Either\n"
            "  pip install 'topicforge[pdf]'          (it needs a Java runtime)\n"
            "or convert the PDFs yourself and point paths.markdown at the result --\n"
            "topicforge reads any directory of .md/.txt files named after the papers."
        ) from exc

    pdfs = find_pdfs(cfg)
    if limit:
        pdfs = pdfs[:limit]
    existing = set() if force else set(find_texts(cfg))
    todo = [p for p in pdfs if p.stem not in existing]
    skipped = [p.stem for p in pdfs if p.stem in existing]

    if todo:
        cfg.paths.markdown.mkdir(parents=True, exist_ok=True)
        # One call for all of them: each convert() starts a JVM, so per-file
        # calls spend more time booting Java than reading PDFs.
        opendataloader_pdf.convert(
            input_path=[str(p) for p in todo],
            output_dir=str(cfg.paths.markdown),
            format="markdown",
        )

    produced = find_texts(cfg)
    missing = [p.stem for p in todo if p.stem not in produced]
    if missing:
        raise IngestError(
            f"{len(missing)} PDF(s) produced no markdown: {', '.join(missing[:8])}"
            + (" ..." if len(missing) > 8 else "")
        )
    return [p.stem for p in todo], skipped


def load_metadata(cfg: Config) -> pd.DataFrame | None:
    """The optional metadata sidecar, indexed by paper id, or None.

    Any columns are allowed. ``metadata.id_column`` names the one holding the
    paper id (default ``paper_id``); ids are read as strings, because a corpus
    numbered 1..n and one named by DOI must behave the same.
    """
    path = cfg.paths.metadata
    if not path.is_file():
        return None
    id_col = cfg.metadata.get("id_column", "paper_id")
    frame = pd.read_csv(path, dtype=str).fillna("")
    if id_col not in frame.columns:
        raise IngestError(
            f"{path} has no '{id_col}' column (found: {', '.join(frame.columns)}). "
            "Set metadata.id_column in config.yaml."
        )
    frame = frame.rename(columns={id_col: "paper_id"})
    frame["paper_id"] = frame["paper_id"].astype(str).str.strip()
    duplicated = frame["paper_id"][frame["paper_id"].duplicated()].tolist()
    if duplicated:
        raise IngestError(f"{path} repeats paper ids: {', '.join(duplicated[:8])}")
    return frame.replace("", pd.NA)


def build_corpus(cfg: Config | None = None) -> pd.DataFrame:
    """The table every pass starts from: one row per paper with usable text.

    Columns: ``paper_id``, ``md_path``, ``title``, ``abstract``, plus every
    column of the metadata sidecar. Title resolution is sidecar → markdown H1 →
    file stem, so a paper is never labelled by its filename unless nothing else
    was available.
    """
    cfg = cfg or load_config()
    rules = md.SectionRules.from_config(cfg)
    texts = find_texts(cfg)
    if not texts:
        raise IngestError(
            f"No .md/.txt files in {cfg.paths.markdown}. Run `topicforge ingest` to "
            "convert the PDFs, or point paths.markdown at text you already have."
        )

    meta = load_metadata(cfg)
    title_col = cfg.metadata.get("title_column", "title")
    meta_titles: dict[str, str] = {}
    if meta is not None and title_col in meta.columns:
        meta_titles = {
            row.paper_id: str(getattr(row, title_col))
            for row in meta.itertuples()
            if pd.notna(getattr(row, title_col))
        }

    rows = []
    for paper_id, path in sorted(texts.items()):
        paper = md.parse(path, paper_id, rules)
        abstract = paper.first("abstract")
        rows.append({
            "paper_id": paper_id,
            "md_path": str(path),
            "title": meta_titles.get(paper_id) or paper.doc_title or paper_id,
            "title_source": (
                "metadata" if meta_titles.get(paper_id)
                else "markdown" if paper.doc_title else "filename"
            ),
            "abstract": abstract.body.strip() if abstract and abstract.body.strip() else None,
            "n_sections": len(paper.sections),
        })

    corpus = pd.DataFrame(rows)
    if meta is not None:
        overlap = set(corpus["paper_id"]) & set(meta["paper_id"])
        if not overlap:
            raise IngestError(
                f"{cfg.paths.metadata} shares no paper id with the corpus. The ids must "
                f"match the file stems (corpus has e.g. {corpus['paper_id'].iloc[0]!r})."
            )
        extra = [c for c in meta.columns if c not in ("paper_id", title_col)]
        corpus = corpus.merge(meta[["paper_id", *extra]], on="paper_id", how="left")
    return corpus


def excerpts(corpus: pd.DataFrame, cfg: Config) -> list[dict]:
    """Prepare the text sent to pass 1, recording which strategy each paper took."""
    rules = md.SectionRules.from_config(cfg)
    max_words = int(cfg.excerpt.get("max_words", 3500))
    min_words = int(cfg.excerpt.get("fallback_min_words", 150))

    items = []
    for row in corpus.itertuples():
        paper = md.parse(row.md_path, row.paper_id, rules)
        text, strategy = md.build_excerpt(
            paper,
            title=row.title,
            abstract=None if pd.isna(row.abstract) else row.abstract,
            max_words=max_words,
            min_words=min_words,
        )
        items.append({
            "paper_id": row.paper_id,
            "title": row.title,
            "text": text,
            "strategy": strategy,
            "words": len(text.split()),
        })
    return items
