# Future work

What the review of November 2026 left behind that is recorded nowhere else: what was decided
against (so it is not proposed again), what is known to be incomplete, and what looks wrong
and has not been proven. What was done is in the CHANGELOG. An item leaves this file when it
is done or when its reason no longer holds.

## Decided against

- **Darker borders on form fields** (2026-11-15). Four strengths were compared side by side;
  today's was kept (2.0:1 on light paper, 2.9:1 on dark).
- **Scoring a factor's validation folds in one pass** (2026-11-15). Faster, but the numbers
  would agree with today's to a tolerance, not to the last bit. An analysis that takes
  seconds or even minutes is fine; no speed work that adds a code path or loosens exactness.
- **Fitting the pool members in parallel.** Finalizing the whole pool is 0.1 to 0.4 s of a
  7 to 10 s run.
- **Reading `state.json` only when it moved.** A state load is 0.15 ms and listing a run
  0.45 ms; the page asks once a second.
- **Caching the dataset or the results within a stage.** The repeated reads are about 0.1 s
  and 0.05 s of a run.
- **Filtering secret-named options in a run's own `config.yaml` and `state.json`.** The
  engine reads them back to fetch the data, and `password_env` names a variable, not a
  secret. The export and the sealed package replace literal ones instead.
- **Naming a replaced main run's successor by name.** It is named by run id: the runs of one
  model usually share a name.
- **From GoogleChrome/modern-web-guidance:** a rule under the sticky table head when
  scrolled and a fade that ends at the last row (Chrome-only today); parallax, scroll
  reveals and page transitions (not the Ledger look).

## Known gaps

- **`examples/live_commercial/run`** (a recorded run) cites `search.py#ratchet_search`, which
  is removed. Its live test is skipped until the run is recorded again (ADR-0016).
- **Payload shapes cover two stages.** `payloads.py` declares `model` and `backtest` only.
  Within those, two readers are not attributed: `_who` in `ui/panels/_shared.py` (it takes
  both a pool member and a candidate row) and `members` in `stages/backtest.py` after it is
  rebuilt.
- **`transforms` means two things**: `{column: form}` on the model payload, a list of
  transformations in the champion's view. A reader of one given the other is not caught.
- **`champion_single_score` and `headroom_vs_champion`** in the ensemble benchmark are named
  for a champion the model stage no longer picks: the value is the best single model among
  the top eight. Naming only.
- **Small grey text on raised, surface and sunken panels** sits at 4.2 to 4.4:1 in the
  workbench and the deck. `--ink-3` is measured against the base paper only.
- **Runs recorded before 2026-11-15** show no re-opens or retries in the run record; their
  activity log has them.

## Suspected (not proven, not changed)

- **ADR-0025's file name** says "six-steps"; its title says seven. Renaming breaks links.
- **Environment.** The `.venv`'s `.pth` files carry the macOS hidden flag, so Python 3.12
  skips them and `.venv/bin/python -m pytest` cannot import `cognos` without
  `PYTHONPATH=src`. `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` should clear
  it (not tried). A stray `__editable__.cognos-1.0.0 2.pth` sits beside them.
