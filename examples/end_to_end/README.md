# COGNOS end-to-end example

A single script that exercises every stage of the COGNOS pipeline on synthetic data and
demonstrates both operating modes.

```bash
pip install -e .          # from the repo root
python examples/end_to_end/run_demo.py
```

It runs three scenarios:

1. **Autonomous mode — clean regression.** The full 8-stage pipeline runs unattended:
   `explore → ideate → model → backtest → validate → comply → document → review`. It searches a
   space of statistical models with a ratchet (accept-if-better) loop, runs the statistical
   diagnostic battery, scores the sealed holdout through the IMPACT feature-table engine (with a
   built-in fallback if IMPACT is not installed), and emits a white paper as an **OKF bundle**
   (Google Open Knowledge Format) with a Google Model Card and an EU AI Act Annex IV pack, then
   verifies the docs match the deployment code.

2. **Compliance gate firing.** A synthetic credit model carries an injected disparity between two
   groups. COGNOS's fair-lending check computes the four-fifths disparate-impact ratio (~0.68 < 0.80)
   and **BLOCKs** the pipeline before the model can be documented or shipped.

3. **Stage-by-stage mode.** Each agent is invoked individually; every stage reads the previous
   stages' on-disk artifacts, so a human can inspect or approve between any two steps.

The equivalent one-liners via the CLI:

```bash
cognos demo --task regression       # scenario 1
cognos demo --task credit           # scenario 2 (BLOCKs at comply)
cognos demo --task classification   # a clean classification run
```

All artifacts (fitted models, diagnostics, experiment ledgers, OKF white papers) are written under
the printed working directory's `runs/<run_id>/`.
