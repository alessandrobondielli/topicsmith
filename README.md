# topicsmith

**Semi-supervised topic detection for research papers, using an LLM you control.**

Point it at a directory of PDFs. Get back a taxonomy of research areas induced
from what those papers actually contain, every paper classified against it, tidy
CSV exports and a set of figures — from the command line, through any
OpenAI-compatible endpoint.

The word that matters is *semi-supervised*. The taxonomy is written to a YAML
file you are expected to open and edit: rename an area, merge two, throw one
out, rewrite a definition. Re-running the classification pass then costs a few
cheap calls, because the expensive reading pass is cached and never repeats. You
can also hand the tool a **seed taxonomy** of areas you already know you want,
and it will induce the rest around them.

```bash
# Not on PyPI yet (the name is taken by an unrelated project). Install from source:
git clone https://github.com/alessandrobondielli/topicsmith && cd topicsmith
pip install -e .          # add '.[pdf]' for PDF ingestion, '.[notebooks]' for the notebook

topicsmith init my-analysis && cd my-analysis
# put your PDFs in ./pdfs, then edit config.yaml and prompts/extract.system.j2

topicsmith ingest       # PDFs -> markdown
topicsmith extract      # pass 1: free-form labels, one call per paper
topicsmith taxonomy     # pass 2: induce the canonical areas
$EDITOR data/interim/taxonomy.yaml        # <- the point of the tool
topicsmith assign       # pass 3: classify every paper against your taxonomy
topicsmith export && topicsmith figures
```

---

## Why three passes

Asking a model to sort papers into categories requires knowing the categories.
Asking it to invent categories per paper gives you as many categories as papers.
So:

**Pass 1 — read.** One call per paper, no fixed vocabulary. The model describes
each paper in its own words: a main topic, a handful of subtopics, the problem,
the method family, the kind of contribution, a one-line summary. Nothing here is
countable yet, and that is deliberate.

It reads Title + Abstract + Keywords + Introduction + Conclusions rather than
the whole paper. That is about a quarter of the tokens, and — more usefully — it
leaves out Related Work, which otherwise drags labels toward what a paper
*cites* rather than what it *does*. Papers that do not parse into sections fall
back to full text automatically, and which strategy each paper took is recorded.

**Pass 2 — organise.** Every label from pass 1 goes into one call, with counts,
and comes back as a small tree of areas: a name, a one-sentence definition, its
subtopics, and the raw labels it absorbs. Because the model sees the whole
distribution at once, it can merge synonyms and see that a category applying to
one paper is noise.

**Pass 3 — count.** Every paper is re-classified against the frozen taxonomy,
with the area choice constrained to a `Literal` over the taxonomy's own names.
A compliant backend cannot emit anything else; a non-compliant one fails
validation and is retried. Either way the counts are trustworthy — nothing was
silently filed under a category that does not exist. This pass reads the pass-1
labels, not the paper, so it is cheap to re-run.

Between passes 2 and 3 is where you come in.

---

## What you get

```
data/interim/
  topics_raw.jsonl          pass 1, one JSON object per paper — the expensive artefact
  taxonomy.yaml             pass 2, hand-editable, the file you own
  taxonomy_stability.csv    how much the taxonomy moves between runs
  topics_assigned.csv       pass 3, one row per paper
  topics_review.csv         the assignments the model was least sure of
data/processed/
  papers.csv                one row per paper, joined with your metadata
  subtopics.csv             one row per (paper, subtopic)
  areas.csv                 one row per area: definition, counts, provenance
  stats.json                the headline numbers, so nothing hand-written drifts
figures/                    PNG + SVG, transparent background
notebooks/topics.ipynb      the same figures, explorable
```

Everything except `data/cache/` is regenerable. **Do not delete `data/cache/`** —
it holds every LLM response, keyed on model + prompt + schema + temperature, and
deleting it means paying for the whole corpus again.

---

## Setup

### The endpoint

Any OpenAI-compatible server. topicsmith probes once for how it constrains
structured output and reuses whichever works:

1. `response_format: {type: json_schema}` — preferred;
2. `guided_json` in the request body — older vLLM;
3. the schema described in the prompt — last resort. It still validates the
   answer, so the constraint holds; expect more retries.

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=...        # "dummy" for a local server
export TOPICSMITH_MODEL=Qwen/Qwen3-32B
topicsmith ping                  # says which mode it negotiated
```

**If your model reasons**, start vLLM with a matching `--reasoning-parser`
(`deepseek_r1`, `qwen3`, …). Without one, the whole `<think>` block arrives
inside `message.content`, and a model reasoning *about a JSON schema* writes
braces while it thinks — so JSON extraction fails on some papers and not others.
topicsmith strips think blocks anyway, but the parser is the right fix.

Two other settings worth getting right before the first run:

- `llm.max_tokens` must be generous. The taxonomy pass alone needs several
  thousand completion tokens for its JSON, and a reasoning model charges its
  thinking against the same budget. The default is 32768.
- `llm.temperature` — reasoning models loop under greedy decoding; ~0.6 is what
  DeepSeek-R1 and Qwen3-thinking recommend. Set it to 0.0 if your model does not
  think. **Change it before the first run:** temperature is part of the cache
  key, so changing it later re-pays every call. So does changing the model.

### The papers

`topicsmith ingest` converts PDFs to markdown with
[opendataloader-pdf](https://pypi.org/project/opendataloader-pdf/), batching all
of them into one call because each call spawns a JVM. Install it with
`pip install -e '.[pdf]'`; it needs a Java runtime.

Already have text? Point `paths.markdown` at any directory of `.md` or `.txt`
files and skip `ingest` entirely. **A paper's id is its file stem**, and that is
the join key for everything — the markdown, the metadata, every export.

### Optional: metadata

Drop a `metadata.csv` next to `config.yaml` with a `paper_id` column matching the
file stems and any other columns you like — year, venue, track, format. They are
joined into the exports, and each column named in `metadata.facets` gets a
research-area heatmap. Everything works without it.

### Optional: a seed taxonomy

Write the areas you already know you want into `seed_taxonomy.yaml` (name +
definition). Pass 2 keeps them by name, refines their definitions against the
evidence if warranted, and induces the rest of the taxonomy around them. Areas
that came from the seed are marked `seeded: true` in the output.

An edited `taxonomy.yaml` from a previous run works as a seed file unchanged —
which is how you carry categories across editions of a conference.

---

## Making it yours

**The prompts are the configuration.** All eight templates live in your project's
`prompts/` directory and are ordinary Jinja files. `topicsmith init` puts them
there; `topicsmith prompts` shows which file each pass is resolving from; delete
one and the shipped default takes over again.

The single highest-leverage edit is `prompts/extract.system.j2`. The shipped
version is field-neutral, which makes it weaker than it needs to be. Name your
discipline and give one example of the right level of specificity:

> Be specific: 'Italian dependency parsing' rather than 'syntax'; 'instruction
> tuning' rather than 'training'.

After that, `config.yaml`:

- `topics.extra_fields` adds fields to the pass-1 schema *and* to the prompt —
  the languages a paper concerns, the population a trial recruited, whatever
  your field counts. Costs one config entry and no code. (It re-pays pass 1: the
  schema is part of the cache key.)
- `sections` retargets the header patterns that split a paper into
  Introduction / Conclusions / Related Work. The defaults handle English plus a
  little Italian, Spanish and German.
- `viz.theme` picks one of eight validated palettes — `ink_ember`,
  `mediterranean`, `ultraviolet`, `forest`, `crimson`, `marigold`, `slate`,
  `classic_blue`. Every ramp has monotone lightness, adjacent-step gaps of at
  least 0.06 L, a light end clearing 2:1 on its own surface, and all-pairs CVD
  separation on the categorical slots. `viz.scale` moves every font size at once
  — raise it for a projector.

---

## Larger corpora

Pass 2 shows the model every label at once, which is the right shape for the
problem and does not scale forever. topicsmith sizes the pool against
`topics.pool_budget_tokens` and picks one of three regimes, telling you which:

| | |
|---|---|
| **single** | The pool fits. One call. What a few hundred papers will always do. |
| **cutoff** | Over budget: keep the most frequent labels covering `topics.coverage` (default 95%) of occurrences and drop the singleton tail. A label seen once cannot define an area anyway. |
| **shards** | Still over budget: split into frequency-stratified batches, induce a partial taxonomy per batch, and merge them in one final call. Every batch carries the head of the distribution as shared context, so the partials come back in comparable terms. |

`topicsmith taxonomy --dry-run` reports the plan without spending anything.

---

## How much should you believe it?

Three things are built in, and all three are worth using.

**`topicsmith stability`** re-induces the taxonomy several times at a non-zero
temperature and reports how much the areas move. High overlap means the
categories are a property of your corpus rather than of one sampling run. It is
the answer to the first question anyone asks about LLM-derived categories.

**`topics_review.csv`** holds every assignment below
`topics.assign_review_threshold`. Reading it is the fastest fifteen minutes you
can spend on the analysis.

**Empty areas are reported.** If an area takes no papers, either the taxonomy
has a category the corpus does not support or the assignment pass is avoiding
it. Both are findings.

And the structural guarantee: `main_area` is a `Literal` over the taxonomy's own
names, so a paper is never silently filed under an invented category. If the
model answers off-taxonomy, validation fails and the call is retried.

---

## Commands

```
topicsmith init [dir]     scaffold a project: config, prompts, notebook
topicsmith ingest         PDFs -> markdown
topicsmith ping           check the endpoint and its structured-output mode
topicsmith status         what has run, what is next
topicsmith extract        pass 1
topicsmith taxonomy       pass 2   (--seed FILE, --dry-run)
topicsmith stability      how much the taxonomy moves between runs
topicsmith assign         pass 3
topicsmith run            all three, back to back (skips your editing step)
topicsmith export         write data/processed + stats.json
topicsmith figures        render the standard figure set
topicsmith stats          print the headline numbers
topicsmith prompts        where each template is resolving from
topicsmith cache          size of the on-disk response cache
```

All of them take `--config PATH`; without it, the project root is found by
walking up from the current directory.

---

## Status

Extracted from the CLiC-it 2026 pipeline and repackaged as a standalone tool.
The individual passes are lifted from a working analysis, but the packaged CLI
has not yet been run end to end against a fresh corpus. Treat `0.1.0` as a
first cut and expect rough edges.

---

## Origin

topicsmith is the topic-detection half of the analytics pipeline built for
**CLiC-it 2026** (the Italian Conference on Computational Linguistics), where it
replaced a BERTopic-based approach over 119 accepted papers. The
conference-specific half — authors, affiliations, geography, co-authorship — is
not part of this repository.

MIT licensed.
