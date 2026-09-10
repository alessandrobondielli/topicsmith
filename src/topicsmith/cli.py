"""Command line entry point.

    topicsmith init [dir]     scaffold a project: config, prompts, notebook
    topicsmith ingest         PDFs -> markdown (opendataloader)
    topicsmith ping           check the endpoint and its structured-output mode
    topicsmith extract        pass 1: free-form labels, one call per paper
    topicsmith taxonomy       pass 2: induce the canonical areas
    topicsmith stability      how much the taxonomy moves between runs
    topicsmith assign         pass 3: classify every paper against the taxonomy
    topicsmith export         write data/processed + stats.json
    topicsmith figures        render the standard figure set
    topicsmith status         what has run, what is next
    topicsmith prompts        where each template is resolving from
    topicsmith cache          size of the on-disk response cache

The one manual step is between `taxonomy` and `assign`: read
``data/interim/taxonomy.yaml`` and edit it. That is the point of the tool, which
is why `run` -- which does all three passes back to back -- says so.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import assemble, documents
from .config import Config, ConfigError, load_config
from .llm import LLMClient, cache_stats
from .prompts import PASSES, SHIPPED, Prompts, shipped_files

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="LLM-driven semi-supervised topic detection for research papers.")
console = Console()

_config_path: Path | None = None


@app.callback()
def main(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml (default: found by walking up)."
    ),
) -> None:
    global _config_path
    _config_path = config


def _cfg() -> Config:
    try:
        return load_config(_config_path)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _client(cfg: Config) -> LLMClient:
    try:
        return LLMClient(cfg)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _report(client: LLMClient) -> None:
    console.print(f"[dim]{client.usage.summary()}[/dim]")
    for name, error in client.usage.failures[:10]:
        console.print(f"  [red]failed[/red] {name}: {error}")
    if len(client.usage.failures) > 10:
        console.print(f"  [red]... and {len(client.usage.failures) - 10} more[/red]")


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


# ------------------------------------------------------------------ scaffolding

@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Where to create the project."),
    seed: bool = typer.Option(False, "--seed", help="Also write a seed_taxonomy.yaml to fill in."),
    prompts_only: bool = typer.Option(False, "--prompts-only",
                                      help="Only restore the prompt templates."),
    force: bool = typer.Option(False, "--force", help="Overwrite files that already exist."),
) -> None:
    """Create a project directory: config.yaml, prompts/, notebook."""
    directory = directory.resolve()
    (directory / "prompts").mkdir(parents=True, exist_ok=True)

    def copy(src: Path, dst: Path) -> None:
        if dst.exists() and not force:
            console.print(f"  [dim]kept[/dim]    {dst.relative_to(directory)}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        console.print(f"  [green]wrote[/green]   {dst.relative_to(directory)}")

    for template in shipped_files():
        copy(template, directory / "prompts" / template.name)
    if prompts_only:
        return

    copy(SHIPPED / "config.yaml", directory / "config.yaml")
    copy(SHIPPED / "notebook.ipynb", directory / "notebooks" / "topics.ipynb")
    if seed:
        copy(SHIPPED / "seed_taxonomy.example.yaml", directory / "seed_taxonomy.yaml")
    (directory / "pdfs").mkdir(exist_ok=True)

    console.print(f"\n[bold]{directory}[/bold] is ready. Next:")
    console.print("  1. put your PDFs in ./pdfs (or point paths.pdfs elsewhere)")
    console.print("  2. set OPENAI_BASE_URL / OPENAI_API_KEY / TOPICSMITH_MODEL, "
                  "or edit the llm block in config.yaml")
    console.print("  3. edit prompts/extract.system.j2 to name your field — it is "
                  "the single highest-leverage edit")
    console.print("  4. topicsmith ingest && topicsmith ping && topicsmith extract")


@app.command()
def prompts() -> None:
    """Show which template file each pass is actually using."""
    cfg = _cfg()
    loader = Prompts(cfg)
    table = Table("template", "resolves to")
    for pass_name in PASSES:
        for part in ("system", "user"):
            name = f"{pass_name}.{part}"
            source = loader.source(name)
            local = source.is_relative_to(cfg.paths.prompts)
            table.add_row(f"{name}.j2",
                          f"[green]{source}[/green]" if local else f"[dim]{source} (shipped)[/dim]")
    console.print(table)


# ---------------------------------------------------------------------- corpus

@app.command()
def ingest(
    force: bool = typer.Option(False, help="Re-convert papers that already have markdown."),
    limit: Optional[int] = typer.Option(None, help="Only the first N PDFs."),
) -> None:
    """Convert the PDFs to markdown."""
    cfg = _cfg()
    try:
        converted, skipped = documents.ingest(cfg, force=force, limit=limit)
    except documents.IngestError as exc:
        _fail(str(exc))
    console.print(f"[green]ingest[/green] {len(converted)} converted, "
                  f"{len(skipped)} already present -> {cfg.paths.markdown}")


@app.command()
def status() -> None:
    """What has run so far, and what to run next."""
    cfg = _cfg()
    pdfs = list(cfg.paths.pdfs.rglob("*.pdf")) if cfg.paths.pdfs.is_dir() else []
    texts = documents.find_texts(cfg)
    steps = [
        ("pdfs", f"{len(pdfs)} file(s) in {cfg.paths.pdfs}", bool(pdfs)),
        ("markdown", f"{len(texts)} file(s) in {cfg.paths.markdown}", bool(texts)),
        ("metadata", str(cfg.paths.metadata), cfg.paths.metadata.is_file()),
        ("seed taxonomy", str(cfg.paths.seed_taxonomy), cfg.paths.seed_taxonomy.is_file()),
        ("pass 1 · extract", str(cfg.paths.raw_topics), cfg.paths.raw_topics.is_file()),
        ("pass 2 · taxonomy", str(cfg.paths.taxonomy), cfg.paths.taxonomy.is_file()),
        ("pass 3 · assign", str(cfg.paths.assigned), cfg.paths.assigned.is_file()),
        ("export", str(cfg.paths.processed / "papers.csv"),
         (cfg.paths.processed / "papers.csv").is_file()),
    ]
    table = Table("step", "", "where")
    for name, where, done in steps:
        table.add_row(name, "[green]✓[/green]" if done else "[dim]·[/dim]",
                      f"[dim]{where}[/dim]")
    console.print(table)
    console.print(f"[dim]cache: {cache_stats(cfg)['entries']} entries[/dim]")


@app.command()
def ping() -> None:
    """Check the endpoint and report which structured-output mode it supports."""
    cfg = _cfg()
    client = _client(cfg)
    console.print(f"model      {client.model}")
    console.print(f"endpoint   {cfg.llm.get('base_url')}")
    try:
        mode = client.run_mode()
    except Exception as exc:  # noqa: BLE001
        _fail(f"no answer: {exc}")
    console.print(f"structured [green]{mode}[/green]")
    if mode == "prompt":
        console.print("[yellow]The endpoint constrains nothing; the schema is only "
                      "described in the prompt. Answers are still validated, but "
                      "expect more retries.[/yellow]")


# ------------------------------------------------------------------ the passes

@app.command()
def extract(
    limit: Optional[int] = typer.Option(None, help="Only the first N papers (smoke test)."),
) -> None:
    """Pass 1: free-form topic labels, one call per paper."""
    from .passes import extract as pass1

    cfg = _cfg()
    try:
        corpus = documents.build_corpus(cfg)
    except documents.IngestError as exc:
        _fail(str(exc))
    client = _client(cfg)
    table = pass1.run(corpus, cfg, client, limit=limit)
    path = pass1.write_raw(table, cfg)

    sources = corpus["title_source"].value_counts().to_dict()
    console.print(f"[green]extract[/green] {int(table['ok'].sum())}/{len(table)} papers -> {path}")
    console.print(f"[dim]titles from {sources} · excerpts "
                  f"{table['excerpt_strategy'].value_counts().to_dict()}[/dim]")
    _report(client)


@app.command()
def taxonomy(
    seed: Optional[Path] = typer.Option(None, help="Seed taxonomy YAML (default: paths.seed_taxonomy)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the label plan and stop."),
) -> None:
    """Pass 2: induce the canonical taxonomy from the pooled labels."""
    from . import reduce
    from .passes import extract as pass1
    from .passes import taxonomy as pass2

    cfg = _cfg()
    try:
        raw = pass1.read_raw(cfg)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    plan = reduce.plan(raw, cfg)
    console.print(f"[dim]{plan.describe()}[/dim]")
    if dry_run:
        return

    seed_taxonomy = pass2.load_seed(cfg, seed)
    if seed_taxonomy:
        console.print(f"[dim]seeded with {len(seed_taxonomy.areas)} area(s) from "
                      f"{seed or cfg.paths.seed_taxonomy}[/dim]")
    client = _client(cfg)
    result, _ = pass2.induce(raw, cfg, client, seed=seed_taxonomy, plan=plan)
    path = pass2.save(result, cfg)

    table = Table("area", "subtopics")
    for area in result.areas:
        table.add_row(f"{area.name}{' [dim](seeded)[/dim]' if area.seeded else ''}",
                      ", ".join(area.subtopics))
    console.print(table)
    console.print(f"\nwritten to [bold]{path}[/bold]")
    console.print("[yellow]Read it and edit it before assigning.[/yellow] Renaming an "
                  "area, merging two, or rewriting a definition costs one re-run of "
                  "`assign` and no re-reading of the papers.")
    _report(client)


@app.command()
def stability() -> None:
    """Re-induce the taxonomy several times and report how much it moves."""
    from .passes import extract as pass1
    from .passes import taxonomy as pass2

    cfg = _cfg()
    try:
        raw = pass1.read_raw(cfg)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    client = _client(cfg)
    report = pass2.stability(raw, cfg, client, seed=pass2.load_seed(cfg))
    report.to_csv(cfg.paths.stability, index=False)
    console.print(report.drop(columns=["areas"]).to_string(index=False))
    console.print(f"written to {cfg.paths.stability}")
    _report(client)


@app.command()
def assign() -> None:
    """Pass 3: assign every paper to the frozen taxonomy."""
    from .passes import assign as pass3
    from .passes import extract as pass1
    from .passes import taxonomy as pass2

    cfg = _cfg()
    try:
        raw = pass1.read_raw(cfg)
        frozen = pass2.load(cfg)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    client = _client(cfg)
    table = pass3.run(raw, frozen, cfg, client)
    _, review = pass3.write_outputs(table, cfg)

    counts = table["main_area"].value_counts()
    out = Table("area", "papers")
    for name in frozen.names():
        out.add_row(str(name), str(int(counts.get(name, 0))))
    console.print(out)
    empty = [n for n in frozen.names() if not counts.get(n)]
    if empty:
        console.print(f"[yellow]{len(empty)} area(s) took no papers: "
                      f"{', '.join(empty)}. Consider removing them from the "
                      f"taxonomy and re-running.[/yellow]")
    console.print(f"{int(table['needs_review'].sum())} low-confidence -> {review}")
    _report(client)


@app.command(name="run")
def run_all(
    limit: Optional[int] = typer.Option(None, help="Only the first N papers (smoke test)."),
) -> None:
    """Run all three passes back to back."""
    console.print("[yellow]`run` goes straight past the taxonomy-editing step. For a "
                  "real analysis, run the three passes separately and read "
                  "taxonomy.yaml in between.[/yellow]\n")
    extract(limit=limit)
    taxonomy(seed=None, dry_run=False)
    assign()


# ------------------------------------------------------------------- outputs

@app.command()
def export() -> None:
    """Assemble every artefact and write data/processed."""
    cfg = _cfg()
    try:
        tables = assemble.build(cfg)
    except FileNotFoundError as exc:
        _fail(str(exc))
    for path in assemble.write(tables, cfg):
        console.print(f"  [green]wrote[/green] {path}")


@app.command()
def figures() -> None:
    """Render the standard figure set."""
    from . import figures as figure_set

    cfg = _cfg()
    try:
        tables = assemble.load(cfg)
    except FileNotFoundError as exc:
        _fail(str(exc))
    for path in figure_set.render_all(tables, cfg):
        console.print(f"  [green]wrote[/green] {path}")


@app.command()
def stats() -> None:
    """Print the headline numbers."""
    cfg = _cfg()
    try:
        tables = assemble.build(cfg)
    except FileNotFoundError as exc:
        _fail(str(exc))
    console.print_json(json.dumps(assemble.stats(tables, cfg), ensure_ascii=False, default=str))


@app.command()
def cache() -> None:
    """Show the size of the on-disk LLM response cache."""
    console.print_json(json.dumps(cache_stats(_cfg())))


if __name__ == "__main__":  # pragma: no cover
    app()
