# Shaped factors: caps and knees, fitted after the split

A financial ratio rarely enters a scorecard as it is. Risk falls with interest coverage until
about 3x, then stops falling; a leverage ratio has a long tail that one obligor can drag a
coefficient with. Modelers cap such a column and let its line bend once or twice (a "knee"),
keeping one direction so the factor is monotonic, often after reading a smoothed (LOWESS) curve
of the default rate against the ratio. A loss forecast needs other transformations (lags,
changes), which the factor library already has.

Every library transformation was a function of one column, never of the target, which is why
the library is built before the split. A knee is chosen with the outcome in view, so it cannot
be one of those.

## Decision

1. **Two shaped operations, `cap` and `knee`** (`factors.SHAPE_OPS`). A cap holds a column to
   its 1st and 99th percentiles. A knee factor is a capped line that may bend at up to two
   knees, each segment held to the expected sign (else the data's), so it is monotonic; the
   number of knees used is the one with the best BIC. Both are stored as points (x, value)
   and computed by interpolation, flat beyond the ends.

2. **They are fitted after the split, on the development sample only**
   (`factors.shape_library`, in the model stage), then frozen: the test sample and the sealed
   holdout are computed from the points, never from an outcome. This is the one place a
   factor depends on the target, and it is kept that narrow.

3. **Who sets the shape.** A person's shape for a column (`overrides.shapes`) wins over the
   plan's `factors.shapes`, which wins over a plan-wide `cap` / `knee` in
   `factors.transforms`. Knots a person gives are kept as given and recorded as such. A knee
   factor's expected sign is `+` (it is turned to rise with the outcome); a cap keeps its
   column's sign.

4. **Explore surveys the shapes as advice** (`factors.survey`, `profile.shape_survey`): the
   observed curve per column, whether it runs one way, how many knees a fit would use. It is
   read on every row explore has, the future holdout included, exactly like the weight of
   evidence look already was, so it may suggest a *form* (cap, how many knees) but never place
   a knee: knee positions come only from the development sample or from a person looking at
   the development sample's curve on the model page.

5. The model page shows each shaped column as fitted (`stages/model/shapes.json`); a shaped
   factor competes with its column as it is in the single-factor screen.

## Under comparison

Where a person sets the form is being compared in three versions of the workbench:
A (per column at the data gate, on the Explore page), B (one plan-wide setting at the design
gate), C (advice on Explore, the form per column in the plan, knees adjustable on the model
page). This ADR records the engine they share; the chosen version is recorded here when made.
