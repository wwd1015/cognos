# Open design questions (ideate, before the MD answered)

- **[design-use_case]** What decision will the model support (origination underwriting, portfolio surveillance, CECL/IFRS 9, IRB, stress testing)? The use case fixes the target definition, horizon, and documentation depth.
- **[design-horizon]** What outcome window defines the target (e.g. 12-month default)? The window must match how the target column was labelled.
- **[design-default_definition]** What event definition labels 'default' (e.g. 90+ DPD, nonaccrual, bankruptcy)? Validation re-derives risk from this.
- **[design-segment]** What portfolio segment does this sample represent (C&I, CRE, small business…)? Pooling heterogeneous segments biases coefficients.
- **[data-leakage]** Explore flagged dpd_at_outcome as possible target leakage — confirm whether each is in the information set at prediction time; if not, add it to data.drop_columns and re-run.
- **[data-epv]** Only 93 events for 13 candidate features (≈7.2 events per variable, below the ~10 rule of thumb) — confirm appetite for a short feature list, a coarser segmentation, or a longer sampling window.
