# premsql (fork)

A fork of [premAI-io/premsql](https://github.com/premAI-io/premsql) at **v0.2.10**,
extended to evaluate text-to-SQL models on benchmarks upstream doesn't cover and
to fix bugs that made some of its results wrong.

Built for [prem_model_evaluation](https://github.com/knikolaevskii/prem_model_evaluation),
which uses it as its `premsql` dependency.

```
version = 0.2.10+kn.1     # upstream 0.2.10 plus local revision 1
```

The version carries a PEP 440 local identifier so `pip list` distinguishes this
from the PyPI release, which is otherwise also `0.2.10`.

## Install

```bash
pip install "git+https://github.com/knikolaevskii/premsql-text2sql-fork.git@text2sql-extensions"
```

---

# What changed

## Added

**WikiSQL and Defog dataset support, via a pluggable schema source.**
Upstream builds the schema section of a prompt by introspecting `sqlite_master`
in each row's database — which assumes every dataset ships one queryable SQLite
file per `db_id`. Neither of these does. `Text2SQLBaseInstance` now takes a
`schema_source`:

| value | schema comes from | used by |
|-------|-------------------|---------|
| `sqlite` | introspecting the row's database *(upstream behaviour, unchanged)* | Bird, Spider |
| `metadata_file` | one shared JSON keyed by `db_id` | WikiSQL |
| `schema_path` | a per-`db_id` JSON beside the database | Defog |

A dataset opts in with two class attributes (`SCHEMA_SOURCE`, `VALID_SPLITS`),
so `premsql/datasets/real/{wikisql,defog}.py` stay as thin as upstream's
`bird.py`.

**Subset-match partial credit.** `BaseExecutor.match_sqls` now reports whether
the gold result set is *contained* in the prediction's, alongside exact match.
A model that returns the right answer plus extra columns is scored differently
from one that is simply wrong, instead of both being zero.

**Bracket-expansion gold queries.** Defog expresses interchangeable column
choices as `SELECT {author.name,author.aid} FROM …`, and alternative acceptable
answers as several queries separated by `;`. The evaluator expands both into
the full set of acceptable queries.

**`PostgresExecutor`.** Defog is distributed as Postgres dumps and evaluated
against a live server. Credentials come from `POSTGRES_*` environment
variables, and the target database is resolved **per row** from `db_id` — a
single fixed database could only ever cover 25 of Defog's 210 questions, since
they span 11 databases.

**Generators for OpenAI-compatible endpoints that aren't OpenAI.**
`Text2SQLGeneratorAPI` (self-hosted servers: vLLM, TGI, LM Studio),
`Text2SQLGeneratorOpenRouter` (hosted models), and
`WikiSQLText2SQLGeneratorAPI`, which lowercases quoted literals to match how
WikiSQL's gold queries are written.

**Error-correction tracking.** Execution-guided decoding now reports how often
a first attempt failed and how often re-prompting with the error recovered it.

## Fixed

These are upstream bugs. Several silently produced wrong numbers rather than
failing, which is why they're worth listing individually.

**VES always reported 0%.** `compute_metric_with_errors` guarded its ratio sum
on `result["ves"] == 1`. VES is a runtime *ratio* — real values are floats like
`1.03` — so the guard never fired and every VES run scored zero regardless of
model quality.

**BIRD's `evidence` was dropped.** The prompt builder populates
`{additional_knowledge}` from a field named `knowledge`; BIRD names it
`evidence`. The hint was discarded for **1386 of 1534** validation rows, and
those hints are frequently the whole task — the first question asks for "the
highest eligible free rate", and its evidence defines the term:
`Free Meal Count (K-12) / Enrollment (K-12)`. Measured effect on 50 rows:
**18% → 34% accuracy**.

**A literal `%` was parsed as a parameter placeholder.** psycopg2 uses
`%`-style placeholders, and raw SQL was handed straight to pandas, so every
`LIKE '%foo%'` failed with `immutabledict is not a sequence`. On the full Defog
set this hit **39 of 210 rows** — all scored as model failures even though the
model's SQL executed fine in every case; it was the gold queries that could not
run. Measured effect: **63.3% → 82.9%**, execution errors **54 → 6**.

**Bird downloaded the entire repository.** `snapshot_download` was called
without `allow_patterns`, so requesting `split="validation"` also pulled the
training databases — over 100GB before a single row could be scored. Now
filtered to the requested split: **~1.4GB**.

**Result comparison ignored the values.** `OptimizedSQLiteExecutor` returned
rows as dicts and compared them with `set(map(tuple, …))`; `tuple(dict)` yields
the *column names*, so any two result sets sharing a schema compared equal
regardless of their contents.

**premsql could not be installed alongside current transformers.** The
`huggingface-hub = "^0.24.5"` pin resolves to `<0.25`, while transformers
≥4.49 requires `≥0.30` — an unsatisfiable pair that fails with
`ResolutionImpossible`. Widened to `<1.0.0`; `snapshot_download`'s signature is
unchanged across that range.

**No Postgres driver was declared.** `PostgresExecutor` builds a
`postgresql://` engine, but nothing provided a driver for SQLAlchemy, so every
query failed with a bare `No module named 'psycopg2'`. Added `psycopg2-binary`.

## Relationship to upstream

Bug fixes here are candidates for contribution back to premsql. Upstream
publishes no license file, so this fork is kept private rather than
redistributing modified copies of its source.
