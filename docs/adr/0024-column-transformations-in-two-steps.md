# Column transformations, decided in two steps

A variable rarely enters a model as it is, and what it needs depends on the model. A scorecard
needs every factor to rank risk in order: risk falls with interest coverage until about 3x,
then stops falling, so the ratio is capped and its line allowed to bend (a "knee"), or smoothed
and held to one direction, often after reading a LOWESS curve of the default rate. A loss
regression needs a straight-line relationship that a few extreme values do not drive (a log, a
cap). A loss forecast needs drivers that do not drift on their own (a change, a lag).

Every library transformation was a function of one column, never of the target, which is why
the library is built before the split. A knee or a smooth curve is placed with the outcome in
view, so it cannot be one of those; and the Explore page reads every row, the future holdout
included, before the split exists.

## Decision

1. **The model's intent names the property each column needs** (`column_transforms.intent_of`):
   rank ordering (classification), a straight line (regression), drivers that do not drift
   (time series). Each intent offers its own kinds (`INTENTS`): `as_is`, `cap`, `knee`,
   `smooth`, `log` for a scorecard; `as_is`, `log`, `cap`, `knee` for a regression; the plan's
   lags and changes (`library`, the screen picks), `as_is`, `change`, `lag`, `log` for a
   forecast. A new kind goes in `KINDS`, `config.ColumnTransform`, `fit` and, when it is a
   function of the column alone, `factors.own_factor`.

2. **Step one, whether: explore and the data gate.** `column_transforms.review` reads every row
   explore has, checks each column for its intent's property (ranks in order through five
   bands beyond noise, straight, no long tail, even spread, no drift, acts at once) and
   recommends a kind with a reason, never a parameter. Trying a kind previews what it would do
   (`column_transforms.sketch`, `service.sketch_transform`: default parameters, nothing kept)
   over the curve as explored; a separate confirmation makes it the decision. Leakage suspects are not reviewed. The
   developer decides every reviewed column on the Explore page, one variable at a time; an
   edit that leaves one open is refused, and accepting takes the engine's kind for all
   (`overrides.transforms`). New data, a new join or a new target clears the decisions.

3. **Step two, how: ideate and the design gate.** For each transformed column, ideate fits the
   parameters on the development sample as the model stage will cut it under the plan in
   force (`ModelStage.development`: the library, then the split; the test sample and the
   holdout are dropped unread). The design page shows each one with its chart; a person may
   change its parameters (knees and knots, the smoothing span, the cap's percentiles, which
   change, how many periods, the expression) and refit, which redraws only the chart
   (`service.preview_transform`, nothing recorded). The decision carries the parameters; an
   edit must set every transformed column, and accepting keeps the engine's settings. Knots a
   person gives are kept as given. **The kind is decided once, at the data gate.** The design
   gate shows it locked and refuses another; using a different one means reopening the data
   gate, so one decision on record says which transformation a column gets.

4. **The model stage fits again on its own development sample and freezes the points**
   (`column_transforms.library`): the test sample and the sealed holdout are computed from the
   points, never from an outcome. This is the one place a factor depends on the target, and it
   is kept that narrow. A column with its own transformation enters only in that form; a column
   kept as it is enters as its level; `library` leaves it to the plan's `transforms`. A knee or
   a smooth factor's expected sign is `+` (turned to rise with the outcome); a cap, a log and a
   change keep the column's sign.

5. **A person's own transformation is a custom expression** (kind `custom`, offered for every
   intent): written in `x`, the column itself, under the safe expression rules of
   `modeling/transforms.py` (arithmetic and whitelisted numpy functions, no other column, no
   outcome; `column_transforms.check_expr`). It is a function of one column, so it needs no
   split, is previewed like any kind, and is recomputed for scoring from the expression kept on
   the factor (`Factor.expr`). A kind that must be fitted (a team's binning, say) is a plugin
   for later, not an expression.

6. **A transformation never hides leakage.** Validation matches the champion's factors to the
   leakage suspects by source column, so `leak__log` or `leak__knee1` blocks like `leak`.

## Consequences

The Explore page follows the work in order (the data, the join, the outcome and the variables,
how each relates to the outcome, the transformations, the summary and the decision), with the
data gate's fields in the step they belong to. The model page shows each transformed column as
fitted (`stages/model/transforms.json`).

Versions B (one plan-wide setting) and C (knees adjusted on the model page, after a model run)
were built and compared; this design keeps C's parameter editor but moves it before the model,
and keeps A's per-variable decision on the Explore page.
