# Several forms per column, all decided at the data gate

ADR-0024 decided one kind per column at the data gate and its parameters at the design gate.
Two things did not hold. A modeller often wants the same variable in several forms at once
and lets the evidence choose: an amount as it is and as its log; a loss-forecast driver as it
is, lagged one and two periods, and its change at each lag. And the forms were decided in two
places: the data gate's kind, and the design page's "forms tried for every other column", which
applied only to columns left to "the plan's lags and changes" and so, on cross-sectional data,
to none of them.

## Decision

1. **A column's decision is a set of forms** (`overrides.transforms[col] = {"forms": [...]}`,
   `column_transforms.forms_of`). A form is a kind (as it is, log, change or growth, each at a
   lag where rows follow one another in time; capped, knees, smooth; a custom expression). Every
   form is a variable of its own in the single-factor screen (`factors.build_library`), and
   the per-column rule (`best_per_source`) keeps the strongest. The column enters as it is only
   when that is one of its forms. A decision made before (one kind, one spec, `library`) reads
   as its form, or as the driver's forms.

2. **Every form is decided at the data gate**, on a grid of base forms by lag
   (`column_transforms.GRIDS` by the model's intent), with the fitted shapes and an expression
   beside it. The engine recommends a set with its reason (`_recommend`): a column with a long
   right tail as it is and as its log; a driver as it is and as its change, now and up to two
   periods back (further when the outcome follows it later). Accepting takes the engine's sets.

3. **The design gate reviews only what needs fitting or testing**: each capped, knees, smooth
   or custom form, keyed `column::kind` (`column_transforms.item_key`, `decided`). Logs, lags
   and changes are used as decided. The design page shows the variables and their forms as one
   table with totals ("20 variables → 23 forms"); the plan-wide extra forms field shows only
   when no column was reviewed (a profile run without a data gate review).

## Consequences

- The model stage no longer drops a fitted column's raw level itself: the library does, for
  every column not decided to enter as it is.
- Model and ideate reports of fitted forms are keyed `column::kind`.
- More forms widen the screen; the per-column rule, the correlation rule and `top_n` bound
  what reaches selection.
