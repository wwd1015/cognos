# Templates, and the data after the intent (v1.7)

The form that started a run asked first for "Data & design", one list with three kinds of
answer in it: your own data (a file, Snowflake, several sources), a synthetic demo, or a
project profile from `projects/`. It asked two questions at once: where the data comes from,
and how much is decided before the run starts. It also asked for the data before anyone had
confirmed what the model is for.

That order is backwards for the work. Which data to use follows from the intent: the use, the
horizon, the default definition and the segment decide which population, which outcome window
and which drivers are relevant. A developer who uploads a file before the intent is settled
has chosen the data without the question it answers. The demos did not belong in the form
either: they are runs, shown on the runs page. The profiles were the right idea under the
wrong name. Each one was an example with a data path in it, and none of the four ran as
shipped, because their data files did not exist.

## Decision

1. **The developer gives the data once the intent is confirmed.** A new developer gate,
   `gate_source` ("Provide the data"), sits between `gate_intent` and `explore`. It reviews
   no stage. It opens when the intent is confirmed, and the workbench shows it on the Explore
   page. The developer gives a file, a Snowflake table or query, or several sources, and may
   name the dependent variable and the date column. The decision is recorded with `by`, like
   every other gate. Autonomous mode waits here too, because nobody can give the data for the
   developer.

2. **The data given is an override.** `overrides.data` holds the `data` mapping as given.
   `RunContext` lays it over the profile's data section (`CognosConfig.with_data`), so every
   stage reads it as if the profile had it. The profile YAML is never edited. New data starts
   the data work over: the dependent variable, the join and the exclusions decided on the old
   data are cleared, the snapshot under `data/` is removed, and everything from `explore` on
   is marked stale with the reason.

3. **A run whose profile gives the data does not ask.** `gate_source` closes by itself,
   recording "the data came with the run" and no decision. `cognos demo`, `cognos run
   --config` with a data path, the evals and every run recorded before this change behave as
   before. The gate is not in `workflow.gates`: it is always part of a run.

4. **A template is a profile with no data.** It holds the standard setup for one kind of
   model: the task and metric, the design brief a sponsor usually gives, the development
   plan, the families, the scenarios and the compliance regime, plus `data.guide`, a note on
   what data this kind of model is usually built on. The guide is shown to the developer at
   `gate_source`; the engine never reads it. Every value is a starting point. The plan
   resolver already puts the developer above the profile, so anything a template sets can be
   changed at the gates.

5. **Two places, no third.** Built-in templates ship in the package (`cognos/templates/`):
   PD logistic scorecard, PD structural simulation, loss forecast time series, rating
   migration, LGD regression, EAD credit conversion factor. A team saves its own into the
   shared runs folder (`<runs>/_templates/`), so everyone on that folder sees them (ADR-0015).
   There is no personal store: a template is meant to be a shared standard.

6. **A finished run can be kept as a team template.** `service.save_template` takes a main
   run that has finished (`completed` or `approved`) and writes the template it started from
   with the run's decisions laid over it: design answers, intended use, the plan as edited,
   the model families of the slate. Nothing about the data goes in except one line in the
   guide naming the run it came from, what it was built on, and the dependent variable. A
   title already taken is refused, never overwritten. The template records who saved it,
   when, and from which run.

7. **Starting a run.** The form asks for the model's name and what to start from: a built-in
   template, a team template, or from scratch (defaults only). The data, the demos and the
   profile list are gone from it. `cognos run --template <ref>` and `cognos data` do the same
   from the command line; `cognos templates` lists them and `cognos save-template` saves one.

## Consequences

- The workflow has eight gates. The deck, the trial-run gates (a trial run can be made at
  `gate_source` to try different data) and the run comparison (a "Data" row) follow.
- `projects/` is removed. Its examples live on as the demo presets and the built-in templates.
- Column names in a template (expected signs, the migration or structural columns) are
  conventions the guide states. Data that names them differently is changed in the data or in
  a team template, not guessed by the engine.
