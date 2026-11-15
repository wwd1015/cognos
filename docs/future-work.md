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
8. ~~Merge `review-2026-11` into `main`.~~ Merged on 2026-11-15 (`daa3338`), not pushed.
9. ~~Two follow-ups: `ModelStage.run` cut into steps; the payload check extended to rows
   and the gate handler.~~ Done 2026-11-15.
10. ~~The open decisions, one at a time.~~ Run record: yes, built. px to rem: yes, done.
    Input borders: kept as they are. Deck grey: done. Provenance filter: yes, built.
    One-pass scoring: no.

**Waiting for the user:** a sealed package (`packages/vN.json`) binds the data given
(`material.data`), source options included, and the export ships it unfiltered. Proposed:
bind the data with literal secret-named options replaced, so the package never holds one
and its digest still verifies. A run's own files (`config.yaml`, `state.json`) keep every source
option as written, because the engine reads them back to fetch the data; an export replaces
literal options named like a secret and keeps environment references (2026-11-15).

## Decided and done on 2026-11-13

D1 to D11 of the review were confirmed and built. What is left open from them:

- ~~The run record does not list re-opens and retries.~~ Confirmed and built on
  2026-11-15: `state.actions` keeps each re-open, retry and take-over (who, when, which
  step) and `process.journal` lists them. Runs recorded before show none.
- **A replaced main run names its successor by run id**, not by name: the runs of one model
  usually share a name.
- **`examples/live_commercial/run`** (a recorded run) cites `search.py#ratchet_search`,
  which is removed. Its live test was already skipped pending a re-record (ADR-0016).

## Suspected (not proven, not changed)

- **ADR-0025's file name** says "six-steps"; its title says seven. Renaming breaks links.
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
   numbers on nine demo runs). `ModelStage.run` was then cut into named steps (445
   lines to 103), with the same nine-demo proof.
2. ~~Typed stage payloads, first step.~~ Done for `model` and `backtest`:
   `payloads.py` declares the shapes and `tests/unit/test_payload_shapes.py` fails when a
   writer records an undeclared key, when a reader asks for one, or when a key that is
   read is no longer written by any of three real runs. Nothing uses the shapes at run
   time. Reads inside a pool member or a backtest
   candidate row and the gate handler's reads are covered too (233 payload reads and 114
   row reads recognised). Not covered: the other stages' payloads; `_who` in
   `ui/panels/_shared.py`, which takes both kinds of row.
   Seen on the way: `transforms` means
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

Decided against on 2026-11-15: *scoring a factor's folds in one vectorised pass*. It
would be faster but agree with today's numbers to a tolerance, not to the bit. The user:
an analysis that takes seconds or even minutes is fine; do not over-engineer it. No
further speed work is planned.

Also looked at and dropped: *reading `state.json` only when it moved*. A state load is
0.15 ms and listing a run 0.45 ms (measured 2026-11-15); the page asks once a second and
the runs list every three. On a network share a check of the file's time costs a round
trip too, so there is little to save.

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
ends at the last row (Chrome-only today); darker input borders (decided against on 2026-11-15: the user compared four strengths
side by side and kept today's; px to rem was confirmed and done the same day); parallax, scroll reveals, page
transitions (not this look). The deck's `--ink-3` was matched on 2026-11-15 (edition 12). On the raised,
surface and sunken backgrounds the grey sits at 4.2 to 4.4:1 in the deck and the workbench
alike: it is measured against the base paper only.
