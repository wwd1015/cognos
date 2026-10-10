# Worked example — a new model development, then an update of it

Both development modes in one offline script: the business intent document, the Intake Analyst's
interview, and a model update that starts from the first run.

```bash
pip install -e .                                # from the repo root
python examples/model_update/run_demo.py        # deterministic agents: no key, no network
```

## What happens

1. **The sponsor fills in the template** (`cognos intent-template -o business_intent.md`) and
   leaves two sections empty: the outcome definition and the horizon.
2. **New model development.** The run starts from that document. Intake reads it, finds the two
   required fields unstated, and the run pauses at the intent gate with two blocking questions:

   ```
   [new] intent is 'needs clarification'; the Intake Analyst asks:
      ? [design-default_definition] What event should the model predict (e.g. 90+ DPD, …)?
      ? [design-horizon] Over what outcome window is the event observed (e.g. 12 months)?
   [new] after the answers: intent is 'clear', 0 blocking question(s)
   ```

   The answers re-run intake; the brief now shows them as *answered by the sponsor* beside the
   positions *stated in the intent document* ([`sample_output/new_brief.md`](sample_output/new_brief.md)).
   Confirming the intent writes the stated positions into the run's design brief, and the other
   eight stages run.
3. **Model update.** A year later the sponsor files an update request on the same template
   (`cognos intent-template --kind update`). The second run is given the request, the existing
   model's scoring code, and the first run as `prior_run`:

   ```
   [update] scope: re estimate; existing family: ridge_logit
      - Refresh the development data through the latest vintage  (data refresh, from explore)
      - Re-estimate the coefficients on the refreshed sample  (re estimation, from ideate)
      - Close validation finding V-12 on calibration  (remediation, from model)
   [update] run …: approved; first on the slate: ridge_logit / all
   ```

   The engine read the existing model's family and scores from the first run's record; the design
   kept that family first on the slate; and the white paper gained a model change record with the
   existing model beside the update ([`sample_output/engagement.md`](sample_output/engagement.md)).
   Here both runs use the same synthetic data, so the two columns agree; on a refreshed sample
   they would not.

## The same thing from the command line

```bash
cognos intent-template -o intent.md                       # fill it in
cognos run --config cognos.yaml --intent intent.md --interactive

cognos intent-template --kind update -o request.md        # fill it in
cognos run --config cognos.yaml --kind update --intent request.md \
           --prior score_v1.py --prior-run <first run id> --interactive
```

With a live provider (`--provider claude_cli`), the Intake Analyst also reads documents that do
not follow the template, and asks about a goal that is stated but vague. The deterministic agent
reads the template's sections as written.
