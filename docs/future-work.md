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
7. ~~The architecture list, one commit per item.~~ All six done on `review-2026-11`
   (2026-11-13); 445 passed, 2 skipped; ruff clean.
8. **Waiting for the user:** merge `review-2026-11` into `main`, and the three answers
   below.

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

1. ~~Split the three largest modules.~~ Done: `ui/panels/` (one module per stage page),
   `ui/app/` (layout, routing, one callback module per page) and `stages/pool_builder.py`.
   Each was proved a pure move (same definitions, same callbacks and layout, same recorded
   numbers on nine demo runs). `stages/model.py` is still 840 lines because `run` alone is
   460; cutting `run` into steps is a further change, not a move.
2. ~~Typed stage payloads, first step.~~ Done for `model` and `backtest`:
   `payloads.py` declares the shapes and `tests/unit/test_payload_shapes.py` fails when a
   writer records an undeclared key, when a reader asks for one, or when a key that is
   read is no longer written by any of three real runs. Nothing uses the shapes at run
   time. Not covered yet: reads inside a pool member or a backtest candidate row; the
   gate handler in `engine/gates.py` (one function for every gate, so its reads are not
   attributed); the other stages' payloads. Seen on the way: `transforms` means
   `{column: form}` on the model payload and a list of transformations in the champion's
   view; a reader of one given the other would not be caught.
3. ~~Paths in results are relative to the run.~~ Done: a member's `scorer_path` is recorded
   as `models/candidates/<id>.joblib`; `pool.scorer_path` still resolves the absolute path an
   older run carries.
4. ~~One reader for "the results of a run".~~ Done: `artifacts.read_result` /
   `read_results` / `read_payload`. Behaviour kept: a damaged result raises for the engine
   and the pages, and is left out of a comparison.
5. ~~A check that the docs name code that exists.~~ Done:
   `tests/unit/test_docs_name_code.py` resolves every file, `file.py::symbol` and
   `module.symbol` in the five top-level docs.
6. ~~Stop fitting through the full sklearn pipeline for small linear fits.~~ Done
   (`fit.make_fit_predict`): a linear specification on numeric columns is scaled and fitted
   directly, with the same routines. Bit-identical in a test over every linear family, and
   every recorded file of nine demo runs (413 files) is the same before and after.

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

Also done: **the direct fit** (architecture item 6). Measured under the profiler on
2026-11-13: loss forecast 9.6 s to 7.2 s, rating migration 10.2 s to 8.9 s, C&I scorecard
6.9 s to 6.3 s for a whole run.

Looked at and dropped: *fitting the pool members in parallel*. Finalizing the whole pool
takes 0.1 to 0.4 s of a 7 to 10 s run (measured on three demos), so there is nothing to
gain; an earlier note here said otherwise and was wrong.

Next:

1. **What is left is validation itself.** After the memo and the direct fit, most of a
   run is still `cv_score` (about 2 ms a fold, thousands of folds in the single-factor
   analysis). The next step would be scoring a factor's folds in one vectorised pass for
   OLS; that changes the arithmetic, so it needs agreement to a tolerance, not to the bit,
   and therefore a decision.
2. **The workbench polls once a second and re-reads `state.json` each time.** Fine for one
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
