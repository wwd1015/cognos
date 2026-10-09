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

To let **LLM agents** make the recommendations, pass a provider (`cognos demo --provider
claude_cli`, or `agents.provider: anthropic` + `ANTHROPIC_API_KEY` in a profile; `cognos providers`
lists what is available) and optionally set `search.guided: true` so the modeler proposes extra
experiments. Agents only *recommend*; the engine checks every answer before it counts, and records
every call under `runs/<id>/agents/` for audit and replay. Add `--interactive` (or use `cognos ui`)
to decide at the review gates yourself.

All artifacts (fitted models, diagnostics, experiment ledgers, OKF white papers) are written under the
printed working directory's `runs/<run_id>/`.
