# COGNOS end-to-end example

A single script that exercises the COGNOS pipeline on synthetic data and demonstrates both operating
modes.

```bash
pip install -e .          # from the repo root
python examples/end_to_end/run_demo.py
```

It runs three scenarios:

1. **Autonomous mode — commercial credit model.** The full 8-stage pipeline runs unattended:
   `explore → ideate → model → backtest → validate → comply → document → review`. It searches a space
   of statistical models with a ratchet (accept-if-better) loop, fits the single interpretable
   champion with full-rank statistical inference, scores the sealed **out-of-time** holdout through the
   IMPACT feature-table engine (built-in fallback if IMPACT is not installed), and runs **SR 11-7
   outcomes analysis** — discrimination (Gini/KS), calibration, and population stability (PSI). It then
   emits a non-gating model-risk **readiness report** and a white paper as an **OKF bundle** (Google
   Open Knowledge Format) with a Model Card, then verifies the docs match the deployment code.

2. **Validation gate firing.** A model that uses a leaking feature is **BLOCKed** by the independent
   validation gate before it can be documented or shipped (confirmed target leakage is the hard BLOCK;
   compliance is a non-gating report).

3. **Stage-by-stage mode.** Each agent is invoked individually; every stage reads the previous stages'
   on-disk artifacts, so a human can inspect or approve between any two steps.

The equivalent one-liners via the CLI:

```bash
cognos demo --task commercial       # scenario 1 (out-of-time outcomes analysis)
cognos demo --task regression
```

To let the **reasoning layer** drive the work (LLM-driven feature engineering + LLM-guided search),
configure an LLM brain (`brain.kind: llm` + `ANTHROPIC_API_KEY`) and set `search.guided: true`. The
LLM only *proposes*; the deterministic engine verifies every proposal before it is kept, and records
the prompt/response trajectory to `runs/<id>/reasoning/transcript.jsonl` for replay and audit.

All artifacts (fitted models, diagnostics, experiment ledgers, OKF white papers) are written under the
printed working directory's `runs/<run_id>/`.
