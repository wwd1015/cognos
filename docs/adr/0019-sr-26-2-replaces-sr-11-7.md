# SR 26-2 replaces SR 11-7 (v1.8)

On April 17, 2026 the Federal Reserve, the OCC and the FDIC issued revised interagency guidance on
model risk management: Fed SR 26-2, OCC Bulletin 2026-13 and FDIC FIL-15-2026. SR 26-2 supersedes
SR 11-7 (2011) and SR 21-8; the OCC withdrew Bulletin 2011-12 and the Comptroller's Handbook
booklet on model risk management; the FDIC rescinded FIL-22-2017 and FIL-27-2021. COGNOS named
SR 11-7 as its governing guidance in code, prompts, templates and documents.

What carries over: the guidance still names conceptual soundness, outcomes analysis (including
back-testing) and ongoing monitoring as the components of model validation, still asks for
effective challenge by people with the expertise, independence and standing to change the model,
and still expects a model inventory and documentation. What is new: it is principles-based and
says plainly that it sets no enforceable standards; it scales with materiality (exposure and
purpose); it is aimed at banking organizations above $30 billion in assets; and generative and
agentic AI models are outside its scope.

## Decision

1. **SR 26-2 is the guidance COGNOS names.** The default regime is `SR26-2`; the readiness
   report's evidence block is `sr26_2` (facts `comply.sr26_2.*`, findings `sr26-2-*`); stages,
   prompts, templates, the deck and the docs cite SR 26-2. Agents are told never to cite SR 11-7.
2. **The old name is read as the new one.** `config.SUPERSEDED_REGIMES` maps `SR11-7` to
   `SR26-2` when a profile or a run's config is loaded, so older profiles and runs report against
   the guidance in force. Results recorded before the change keep `sr11_7` on disk; the comply
   page and the facts read either key.
3. **What COGNOS checks does not change.** The three components it evidences are the same in
   both letters. Wording that quoted SR 11-7 ("alternatives considered", validation "must be"
   independent) is restated in SR 26-2's terms: the comparison of alternative methodologies
   under model testing, and effective challenge with enough independence to stay objective.

## Consequences

- Recorded runs (`examples/live_commercial/run/`) and earlier changelog entries keep SR 11-7:
  they are a record of what was produced then.
- The Model-Risk Analyst's reading is a readiness report, not a compliance verdict (ADR-0006).
  SR 26-2 being non-binding makes that stance more accurate, not less.
- COGNOS's agents are generative AI used as development tools; the models COGNOS builds are
  traditional statistical models, which are in SR 26-2's scope.
