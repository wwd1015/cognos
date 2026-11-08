# The design stage in six steps, and the single-factor analysis before the search

The design page used to open on a conclusion ("reduced-form PD") and put every choice in one
long form: the slate, the split, the screen, the selection, the ranking. A modeller works in an
order: what kind of model, how the data are cut, which variables, how variables and models will
be judged, which variables are strong on their own. And the single-factor analysis, the most
consequential judgment in a scorecard, ran inside the model stage, where the developer met it
only as thresholds set beforehand and a ledger read afterwards.

## Decision

1. **Six steps, each confirmed before the next opens** (`ui/design_steps.py`): model form (the
   frameworks set against each other, the families to search), data treatment (the rows as one
   bar: development, test, holdout, with the validation inside development), variables (each
   transformation's parameters, ADR-0024; the forms tried for the rest; what every model must
   carry), assessment plan, single-factor analysis, summary and decision. A confirmed step folds
   to one line and can be opened again, which makes the steps after it wait. The confirmations
   are the page's own: nothing is recorded until the design gate is decided, and the gate reads
   every field wherever it sits. "Accept design" and "Apply my design" open once steps 1 to 5
   are confirmed.

2. **The assessment plan sets rules at two levels, then the ranking.** Variables are kept when
   they pass the screen (missing, p-value, sign, forms per column, correlation, shortlist).
   Models are kept when they pass the model rules (factors, every p-value, every sign,
   correlation between two factors, and now **variance inflation**, `selection.max_vif`,
   recommended at 5 when the slate has a family with coefficients). What is left is ranked
   **in order** (the first line decides, later lines break ties) or **blended by weight**
   (`evaluation.rank_mode: weights`): each line becomes the share of models a model beats, and
   the shares are averaged by the weights (`selection.blend`). A line may also be the number of
   factors or the spread of the score across validation folds (`plan.RANK_EXTRAS`, both better
   when lower); the first line must be a score, since single factors are ranked on it.
   Every number blended is an engine number; the score and each line's share are kept on the
   model (`info["blend"]`). A ranking written as a formula is left for later.

3. **The single-factor analysis is shown at the design gate, after the rules.** Ideate runs it
   under the plan in force (`ModelStage.screen_preview`): the library, the split and each
   column's own transformation on the development sample, exactly as the model stage cuts it;
   the test sample and the holdout are dropped unread. The page draws it only once the
   assessment plan is confirmed, as a funnel over a table: one bar per rule with its toll, a
   bar filters the table to who that rule removed and why. The model stage and the preview
   call one function (`ModelStage.screen_factors`), so they cannot disagree; the model stage
   runs it again under the decided rules and its result is the record.

4. **Every look is kept; a rule changed after a look is counted.** Drawing the analysis is a
   look (`service.preview_screen`, kept in `state.screen_looks` with `by`, the rules and how
   many factors they kept). A change of rules between two looks is a revision after the result
   was seen (`plan.screen_revisions`): the design summary shows the count, and validation raises
   it as a finding with what changed (`ValidateStage._screen_record`). Looks are not decisions:
   they never re-render the page or invalidate a step.

5. **An exception is per variable and carries a reason.** A factor may be kept although a rule
   drops it, or left out although the rules keep it (`overrides.screen_exceptions`, with `by`;
   a reason is required). The screen records the rule a kept factor overruled
   (`overruled`), and a kept factor never pushes a stronger one out under the correlation rule;
   only a factor kept by rule (an engineered factor) is seen first. Exceptions show as their own
   bar in the funnel, travel on the model payload (`exceptions`) and reach validation as a
   finding. New data, a new join or a new target clears them.

6. **The candidate pool page shows the model funnel** (`selection.model_funnel`,
   `stages/model/model_funnel.json`): the best `FUNNEL_MODELS` distinct specifications through
   the model rules in a fixed order, each leaving at the first rule it breaks, with the reason.

7. **The developer sets the model form.** Each framework the engine can fit here (applicable,
   with a family of its own) has a role: main form, also competes, benchmark only, not used
   (`overrides.framework_roles`, with `by`; a role that departs from the Design agent's needs a
   reason, and exactly one framework is the main form). A benchmark-only family is fitted and
   ranked but never in the engine's pool or the champion (`benchmark_families`; the check now
   holds for every benchmark, not only under required interpretability). The families follow
   the roles: one set to not used leaves the slate, one given a role brings its families. A form
   the engine cannot fit is proposed with a reason (`overrides.proposed_forms`): recorded in the
   design brief, listed as not fitted in the documentation, and raised at validation. The page
   shows one line per family (what it is, which rules can judge it, whether it can win): the
   slate's "top"/"all" feature strategies were never read by the model stage, whose variables
   come from the screen.

## Why the single-factor analysis moved, and what keeps it honest

The case against: shown before the search, the screen invites a person to adjust thresholds
until a favourite variable survives, and nothing would show it. The case for: choosing the
variables is the judgment a reviewer asks about first (conceptual soundness, SR 26-2), and it
can only be made by looking at each variable's own evidence. A per-variable exception with a
written reason is a better instrument than a loosened global threshold. The guards above make
the trade: rules before results (the page), every look counted (the engine), exceptions on
record (the gate), the model stage as the record.

## Consequences

- `gates.handle` takes `by`, so a handler can stamp who set what (exceptions do).
- A new screen rule appears in the funnel by having a log line in `factors.screen`; a new model
  rule needs an entry in `selection.MODEL_RULES` matching its violation text.
- The design page no longer has the "Development plan" table; who set each choice is on the
  field itself ("recommended by the engine", "set by the developer").
