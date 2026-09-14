# topicsmith

Semi-supervised topic detection for a corpus of research papers, using an
OpenAI-compatible LLM endpoint.

Point it at a directory of PDFs (or plain text). It induces a taxonomy of
research areas from the content of those papers, lets you edit that taxonomy by
hand, classifies every paper against it, and writes CSV tables and a set of
figures. Everything runs from the command line.

The taxonomy is written to a YAML file that you are expected to open and edit:
rename an area, merge two, delete one, rewrite a definition. Re-running the
classification pass after an edit is cheap, because the expensive reading pass
is cached and does not repeat. You can also supply a seed taxonomy of areas you
already know you want, and the rest of the taxonomy is induced around them.

## How it works

Sorting papers into categories requires knowing the categories in advance.
Asking a model to invent categories one paper at a time produces roughly as many
categories as there are papers. topicsmith splits the problem into three passes.

**Pass 1 — extract.** One call per paper, with no fixed vocabulary. The model
describes each paper in its own words: a main topic, a few subtopics, the
problem, the method family, the type of contribution, and a one-line summary.

By default it reads Title, Abstract, Keywords, Introduction, and Conclusions
rather than the whole paper. That is about a quarter of the tokens, and it
leaves out Related Work, which otherwise pulls labels toward what a paper cites
rather than what it does. Papers that do not parse into sections fall back to
full text, and the strategy used for each paper is recorded.

**Pass 2 — taxonomy.** Every label from pass 1 goes into a single call, with
occurrence counts, and comes back as a small tree of areas: a name, a
one-sentence definition, a list of subtopics, and the raw labels each area
absorbs. Because the model sees the whole distribution at once, it can merge
synonyms and treat a label that applies to one paper as noise.

The output is `data/interim/taxonomy.yaml`. Edit it before running pass 3.

**Pass 3 — assign.** Every paper is re-classified against the frozen taxonomy.
The area choice is constrained to a `Literal` over the taxonomy's own names, so a
compliant backend cannot return anything else; a non-compliant one fails
validation and is retried. This pass reads the pass-1 labels, not the paper, so
it is cheap to re-run after a taxonomy edit.

## Install

Not on PyPI (the name is taken by an unrelated project). Install from source:

```bash
git clone https://github.com/alessandrobondielli/topicsmith
cd topicsmith
pip install -e .
```

Optional extras:

- `pip install -e '.[pdf]'` — PDF-to-markdown conversion (needs a Java runtime)
- `pip install -e '.[notebooks]'` — Jupyter, to run the generated notebook

Requires Python 3.10 or newer.

## Quick start

```bash
topicsmith init my-analysis
cd my-analysis
# put your PDFs in ./pdfs, set the endpoint (see below),
# then edit prompts/extract.system.j2 to name your field

topicsmith ingest      # PDFs -> markdown
topicsmith ping        # check the endpoint
topicsmith extract     # pass 1
topicsmith taxonomy    # pass 2

$EDITOR data/interim/taxonomy.yaml   # review and edit

topicsmith assign      # pass 3
topicsmith export      # write data/processed/ and stats.json
topicsmith figures     # render the figure set
```

`topicsmith status` shows what has run and what is next.

## Configuring the endpoint

topicsmith talks to any OpenAI-compatible server. On the first call it probes
how the server constrains structured output and reuses whichever method works:

1. `response_format: {type: json_schema}` — preferred
2. `guided_json` in the request body — older vLLM
3. schema described in the prompt — fallback; answers are still validated, but
   expect more retries

Configure it with environment variables or the `llm` block in `config.yaml`, for example:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy          # any non-empty value for a local server
export TOPICSMITH_MODEL=Qwen/Qwen3-32B
topicsmith ping                      # reports the negotiated mode
```

If your model produces reasoning traces, start the server with a matching
`--reasoning-parser` (`deepseek_r1`, `qwen3`, and so on). Without one, the
`<think>` block arrives inside `message.content`; a model reasoning about a JSON
schema writes braces while it thinks, and JSON extraction then fails on some
papers but not others. topicsmith strips think blocks regardless, but the parser
is the correct fix.

Two settings to get right before the first run:

- `llm.max_tokens` (default `32768`). The taxonomy pass needs several thousand
  completion tokens for its JSON, and a reasoning model spends part of the same
  budget on its trace.
- `llm.temperature` (default `0.6`). Reasoning models can loop under greedy
  decoding; `0.6` is what DeepSeek-R1 and Qwen3-thinking recommend. Set it to
  `0.0` for a non-reasoning model. Change it before the first run: temperature
  is part of the cache key, so changing it later re-pays every call. Changing
  the model does the same.

## Preparing the papers

`topicsmith ingest` converts PDFs to markdown with
[opendataloader-pdf](https://pypi.org/project/opendataloader-pdf/), converting
all of them in one call because each call starts a JVM.

If you already have text, point `paths.markdown` in `config.yaml` at a directory
of `.md` or `.txt` files and skip `ingest`. A paper's ID is its filename stem,
and that stem is the join key for the markdown, the metadata, and every export.

## Optional: metadata

Place a `metadata.csv` next to `config.yaml` with a `paper_id` column matching
the filename stems, plus any other columns (year, venue, track, format). These
columns are joined into the exports. Each column listed in `metadata.facets`
gets a research-area heatmap. The topic analysis works without metadata.

## Optional: a seed taxonomy

Write the areas you already know you want into `seed_taxonomy.yaml` (a name and a
definition each). Pass 2 keeps them by name, refines their definitions against
the evidence if warranted, and induces the rest of the taxonomy around them.
Seeded areas are marked `seeded: true` in the output.

An edited `taxonomy.yaml` from a previous run works as a seed file without
changes, which is how you carry a set of categories from one run to the next.

## Customizing

The prompt templates are the main configuration surface. All eight live in your
project's `prompts/` directory as Jinja files. `topicsmith init` writes them;
`topicsmith prompts` shows which file each pass resolves; delete a file and the
built-in default is used again.

The highest-impact edit is `prompts/extract.system.j2`. The shipped version is
field-neutral. Name your discipline and give one example of the level of
specificity you want (for example, "dependency parsing" rather than "syntax").

In `config.yaml`:

- `topics.extra_fields` adds fields to the pass-1 schema and prompt (the
  languages a paper covers, the population a trial recruited, and so on). This
  re-pays pass 1, because the schema is part of the cache key.
- `sections` retargets the header patterns that split a paper into Introduction,
  Conclusions, and Related Work. The defaults cover English plus some Italian,
  Spanish, and German.
- `viz.theme` selects one of eight palettes: `ink_ember`, `mediterranean`,
  `ultraviolet`, `forest`, `crimson`, `marigold`, `slate`, `classic_blue`. Each
  ramp has monotone lightness, adjacent-step gaps of at least 0.06 L, a light end
  above 2:1 contrast on its own surface, and CVD separation across all pairs of
  categorical slots. `viz.scale` multiplies every font size at once.

## Outputs

```
data/interim/
  topics_raw.jsonl          pass 1, one JSON object per paper (the expensive artifact)
  taxonomy.yaml             pass 2, hand-editable
  taxonomy_stability.csv    stability report, if you run it
  topics_assigned.csv       pass 3, one row per paper
  topics_review.csv         the lowest-confidence assignments
data/processed/
  papers.csv                one row per paper, joined with metadata
  subtopics.csv             one row per (paper, subtopic)
  areas.csv                 one row per area: definition, counts, provenance
  stats.json                the headline numbers
figures/                    PNG and SVG, transparent background
notebooks/topics.ipynb      the same figures, as a notebook
```

Everything except `data/cache/` is regenerable. Do not delete `data/cache/`: it
holds every LLM response, keyed on model, prompt, schema, and temperature.
Deleting it means paying for the whole corpus again.

## Larger corpora

Pass 2 shows the model every label at once. This does not scale indefinitely.
topicsmith sizes the label pool against `topics.pool_budget_tokens` and picks
one of three strategies, reporting which:

| Strategy | When | What it does |
|---|---|---|
| `single` | pool fits the budget | one call; the case for a few hundred papers |
| `cutoff` | over budget | keep the most frequent labels covering `topics.coverage` (default 95%) of occurrences, drop the singleton tail |
| `shards` | still over budget | split into frequency-stratified batches, induce a partial taxonomy per batch, and merge in one final call; each batch carries the frequent head of the distribution as shared context |

`topicsmith taxonomy --dry-run` reports the plan without making any calls.

## Checking the results

- `topicsmith stability` re-induces the taxonomy several times at a non-zero
  temperature and reports how much the areas move between runs. High overlap
  means the categories are a property of the corpus rather than of one sampling
  run.
- `topics_review.csv` contains every assignment below
  `topics.assign_review_threshold`. Read it.
- Empty areas are reported by `assign` and kept in `areas.csv` at zero. An area
  with no papers means either the taxonomy has a category the corpus does not
  support, or the assignment pass is avoiding it.
- `main_area` is a `Literal` over the taxonomy's names, so a paper is never
  filed under a category that does not exist. An off-taxonomy answer fails
  validation and is retried.

## Command reference

```
topicsmith init [dir]     scaffold a project: config.yaml, prompts/, notebook
topicsmith ingest         convert PDFs to markdown
topicsmith ping           check the endpoint and its structured-output mode
topicsmith status         what has run, what is next
topicsmith extract        pass 1
topicsmith taxonomy       pass 2   (--seed FILE, --dry-run)
topicsmith stability      re-induce the taxonomy and measure how much it moves
topicsmith assign         pass 3
topicsmith run            all three passes back to back (skips the editing step)
topicsmith export         write data/processed/ and stats.json
topicsmith figures        render the figure set
topicsmith stats          print the headline numbers
topicsmith prompts        show where each template resolves from
topicsmith cache          size of the on-disk response cache
```

Every command accepts `--config PATH`. Without it, the project root is found by
walking up from the current directory.

## Project status

Early release. The individual passes come from a working analysis, but the
packaged CLI has not yet been run end to end against a fresh corpus. Treat
version `0.1.0` as a first cut.

## License

MIT. See [LICENSE](LICENSE).
