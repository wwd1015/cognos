# Future work

What the review of 2026-11-11/12 found and did not change, what is proposed next, and where
work in hand stands. An item leaves this file when it is done (the CHANGELOG records it) or
when it is decided against (say so here, with the reason, so it is not found again).

Three kinds of item, kept apart:

- **Needs your decision** changes what COGNOS does. Nothing here is built before you confirm.
  (None open today.)
- **Suspected** is something that looks wrong and has not been proven or fixed.
- **Proposed** is structure, speed or look: no change in what a run records. These can be
  done without asking; they are listed so the order is visible.

## In progress (resume here)

Work under way when this file was last saved. A scheduled wake-up reads this section.
The user picked layout A and said yes to D1 to D11 on 2026-11-13. Simple, well-specified
work goes to cheaper helper agents (Opus); design and anything that touches a recorded
number stays with the main session. Nothing is committed.

1. ~~D1 to D11.~~ Done, each with a test; see the CHANGELOG (Changed, Removed).
2. ~~Layout A: the decision first, the evidence behind tabs, the activity behind a button
   (`panels.decision_first`, `DECISION_FIRST`).~~ Done, test added.
3. ~~The four small changes (waiting steps fold, names not ids, factor chips, "nothing
   moved" in a sentence).~~ Done, tests added.
4. ~~Ideas borrowed from GoogleChrome/modern-web-guidance.~~ Thirteen applied to
   `ui/assets/cognos.css` and `ui/theme.py`; see "the workbench" below.
5. ~~Looked at in a browser, light and dark.~~ ~~Whole suite and lint.~~ 421 passed,
   2 skipped; ruff clean.
6. ~~The user looked at the workbench and said to commit.~~ Committed on the branch
   `review-2026-11`.
7. **Now:** the architecture list below, in its order, one commit per item on the same
   branch. Mechanical moves go to helper agents; the whole suite runs before each commit.

Still waiting for a yes or no (none blocks the list): listing re-opens and retries in the
run record; px to rem and darker input borders; the deck's `--ink-3`.

## Decided and done on 2026-11-13

D1 to D11 of the review were confirmed and built. What is left open from them:

- **The run record does not list re-opens and retries.** Both now record who asked (the
  step's stamp, and `events.jsonl`, which travels in the export), but `process.journal`
  reads only decisions, lineage and the package. Listing them needs a durable stamp on
  `RunState`; that is a change to what a run records, so it waits for a yes.
- **A replaced main run names its successor by run id**, not by name: the runs of one model
  usually share a name.
- **`examples/live_commercial/run`** (a recorded run) cites `search.py#ratchet_search`,
  which is removed. Its live test was already skipped pending a re-record (ADR-0016).

## Suspected (not proven, not changed)

- **Stale question text.** The design stage's leakage question tells the sponsor to edit
  `data.drop_columns` and re-run. Exclusions are a data-gate decision now.
- **Stale config docstrings.** `StructuralConfig` and `MigrationConfig` say the benchmark is
  scored "on the sealed holdout"; it is scored on the test sample. `ColumnTransform` cites
  `modeling/transforms.py` for what lives in `column_transforms.py`. (Docstrings only.)
- **ADR-0025's file name** says "six-steps"; its title says seven. Renaming breaks links.
- **`CHANGELOG.md` → Unreleased** repeats "Added" and "Changed" headings.
- **`intake.py`**: `ctx.overrides.design.get(f.id) or ctx.overrides.compliance[f.id]` raises
  `KeyError` if a design override were ever stored empty. No path stores one today.
- **Provenance filter.** `datasources` drops options whose name contains "password" or "key"
  from the provenance. An option named `token` would be recorded. Options are documented as
  never a secret; the filter is narrower than the rule.
- **Environment.** The `.venv`'s `.pth` files carry the macOS hidden flag, so Python 3.12
  skips them and `.venv/bin/python -m pytest` cannot import `cognos` without
  `PYTHONPATH=src`. `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` should clear
  it (not tried). A stray `__editable__.cognos-1.0.0 2.pth` sits beside them.

### Looked at and dropped

- *"The dataset is re-read several times per stage"* and *"`Engine.results()` re-parses every
  result"*: measured at about 0.1 s and 0.05 s of a 7 to 13 s run. Not worth a change.

## Proposed: architecture

In the order I would do them. None changes what a run records.

1. **Split the three largest modules along lines they already have.**
   `ui/panels.py` (2,100 lines) into one module per stage page; `ui/app.py` (1,900) into
   layout, routing and one callback module per page; `stages/model.py` (1,150) into the stage
   and a `pool_builder` (leaders, the engine's pool, finalizing a member). Today a change to
   one page means reading all of them.
2. **Typed stage payloads.** Stages hand each other plain dictionaries. Two of the bugs fixed
   in this review (PBO, the fair-lending sentence) were a reader asking for a key the writer
   had stopped writing, and nothing noticed for months. A small pydantic model per stage
   payload (permissive: extra keys allowed, so old runs still load) turns that into a failing
   test. Start with `backtest` and `model`, the two most read.
3. **Paths in results are relative to the run.** `scorer_path` is recorded absolute (readers
   now resolve it, see `pool.scorer_path`). Record it relative and keep the resolver for old
   runs, so nothing in a result names a machine.
4. **One reader for "the results of a run".** `service.results`, `service._results_in`,
   `Engine.results` and `RunContext.get` each parse `stages/*/result.json` their own way
   (one raises on a corrupt file, one skips it). One function, one behaviour.
5. **A check that the docs name code that exists.** `CLAUDE.md` pointed at a function removed
   two commits earlier. The script used in this review (every `` `module.symbol` `` and
   `` `path::symbol` `` in the top-level docs resolves) is forty lines; as a unit test it keeps
   the "Extending COGNOS" section honest.
6. **Stop fitting through the full sklearn pipeline for one-column fits.** After the memo
   below, what is left of the single-factor analysis is about 4 ms of pipeline overhead per
   fold for a regression on one column. A direct fit gives the same numbers to the last bit
   in principle; it needs the same proof the memo got before it is trusted.

## Proposed: speed

Done in this pass: **the validation memo** (`modeling/cv_cache.py`). The single-factor
analysis was computed in full at the design stage, again on every redraw at the design gate,
and again by the model stage: about 40% of a run, 67% on the time-series demo.

| Demo | Two looks at the screen | Re-run after one rule changed |
|---|---|---|
| C&I scorecard | 0.9 s → 0.3 s | 1.9 s → 0.9 s |
| Loss forecast | 4.0 s → 0.5 s | 5.2 s → 1.1 s |
| Rating migration | 1.4 s → 0.7 s | 5.8 s → 3.4 s |
| Linked sources | 1.0 s → 0.3 s | 2.0 s → 0.9 s |

Every recorded number of five demo runs (results, ledgers, pool, backtest, validation) is
identical with and without it.

Next, by expected gain:

1. **Fit the pool members in parallel** at the end of the model stage (each is independent:
   a fit, a diagnostics battery, a saved scorer). Largest on the migration demo, where
   finalizing is most of what is left.
2. **One-column fits without the pipeline** (architecture item 6).
3. **The workbench polls once a second and re-reads `state.json` each time.** Fine for one
   person; on a shared folder over a network, read the file's modification time first and
   parse only when it moved.

## Proposed: the workbench

Kept inside the Ledger system (warm paper, ink, one oxblood accent, serif titles, mono
figures, hairline rules). "Modern" here means less to read and fewer forms, not a new skin.
Options are shown as pictures to pick from before anything is built (step 5 above).

Built on 2026-11-13: layout A (the decision first, evidence behind tabs, activity behind a
button) and the four small changes.

Borrowed from GoogleChrome/modern-web-guidance, applied: grey small text raised to 4.6:1
(`--ink-3`, both themes); the active tab no longer changes weight, so the row does not
shift; balanced titles and better line endings in prose; a 75-character reading width in
the white paper and the brief; scrollbars in key with the paper; one 2px focus ring; motion
only for people who have not asked for less; firmer rules under "increase contrast"; a
visible selection in forced colours; real underlines; hover actions that also show on
keyboard focus and touch; folds that open smoothly; 24px hit areas on three small buttons.

Left out, with the reason: a rule under the sticky table head when scrolled and a fade that
ends at the last row (Chrome-only today); px to rem for about sixty font sizes and darker
input borders (a wider change of look: ask first); parallax, scroll reveals, page
transitions (not this look). `docs/deck/index.html` still carries the old `--ink-3`; a
change there needs an edition-log row.
