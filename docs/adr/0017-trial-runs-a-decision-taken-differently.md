# Trial runs: a decision taken differently, nothing lost and nothing hidden (v1.6)

A developer at a gate often wants to know what the other choice would have done: a different
plan, a different pool, a different champion. Until now there were two ways to find out, and
both lost something. Re-opening the gate re-runs everything after it and keeps one previous
result, so a third try overwrites the first. Starting a new run keeps both, and repeats every
earlier decision by hand.

The obvious feature is "duplicate this run". The strongest argument against it is that it
makes a best-of-many cheap and quiet. Each copy looks at the test sample again, and a run
counts only its own looks. A developer could copy five times, keep the copy that looks best,
delete the rest, and hand the validator a run that reports six looks at the test sample when
the model had thirty. Worse, each copy could go on to validation and open the sealed holdout.
That is the selection bias ADR-0016 was written to make visible.

The same exploration already happens outside the system, in a second run or a notebook, with
no record at all. So the choice is not between exploring and not exploring. It is between
exploring off the record and on it.

## Decision

1. **A model has one main run and any number of trial runs.** A trial run is a copy of a run
   made at a developer gate (`gate_intent`, `gate_data`, `gate_design`, `gate_candidates`,
   `gate_champion`). It waits at that gate; the developer decides it differently. Results,
   data and the search cache before the gate come across, so only what the decision changes
   runs again. The run it was made from is not touched. The developer says what the trial is
   for, and that note is kept.

2. **A trial run stops before validation.** The engine never schedules `validate` on a trial
   run, so a trial run cannot open the sealed holdout. Nothing from validation on is copied
   into one, and it carries no package.

3. **To go on, a trial run is made the main run.** The run it replaces becomes a trial run
   and keeps everything it recorded; a package it had sealed is no longer in force. There is
   always at most one main run of a model, so the holdout is opened by one line of work. A
   holdout already opened for an earlier main run is counted for the model: a second
   champion scored on it raises the existing reuse finding.

4. **Looks at the test sample are counted for the model.** A run's own looks are what it
   scored after it was copied (its comparisons, and a search it ran with different inputs).
   The model's total is the sum over all of its runs, deleted ones included. Validation
   reports the total, as a note for the record when a sealed holdout exists and nobody chose
   on it, and as a finding when the test result is all there is.

5. **Deleting a trial run keeps its line.** Its files go. What it tried, how far it went, the
   champion it reached and how often it used the test sample stay in the model's record
   (`<runs>/_models/<model>.json`). The main run of a model cannot be deleted while trial
   runs of it remain, unless they are deleted with it.

6. **One record, read everywhere.** `trials.record` lists every run of a model on one line
   each. The run record page, validation (`stages/validate/trials.json`), the white paper
   ("Alternatives tried"), the sealed package and the export (`MODEL_RECORD.json`) read the
   same list.

```
main run     … → gate_design → model → gate_candidates → backtest → gate_champion → validate → …
                      │
trial run             └→ gate_design* → model → gate_candidates → backtest → gate_champion ■
                         (* decided differently)                                  (stops here)
```

## What is stored, and what is computed

Stored: `Lineage` on each run's state (the model, the parent, the gate, the note, who and
when, the overrides and counts it inherited, and the stamps of a change of role), and the
model's record: its events (made, made the main run, deleted) and the lines of deleted runs.

Computed on read: every live run's line, from its own state and stage results, and the
model's totals. Nothing about a live run is copied into the record, so the record cannot
disagree with the run.

## Consequences

- A copied run is a few megabytes; the search cache is the largest part and is what makes a
  trial run cheap to carry on.
- Validation records the runs of the model as they stood when it ran. A trial run made
  afterwards does not make that validation stale; the run record page shows the live list.
- A trial run whose plan changes the split cuts its own samples. If it is later made the main
  run, the holdout it opens is not the same rows as its predecessor's; the count still treats
  it as the model's holdout, which errs on the side of saying so.
- The search's own looks at the test sample (a ranking on the test sample) are counted once
  per distinct search. A re-run of the same search inside one run was never counted twice
  and still is not.
- Not closed: a developer can still start a completely new run on the same data instead of a
  trial run. Recognising the same data across unrelated runs is a separate piece of work.
- Seats: making a trial run and choosing the main run belong to the developer and record who
  did it.
