---
type: dataset
title: Dataset Profile
description: Shape, features, and leakage notes from exploration.
resource: stages\explore\profile.json
tags:
- data
timestamp: '2026-09-30T02:35:19.198601+00:00'
---

# Dataset Profile

- **Rows:** 1200
- **Columns:** 8
- **Model features (6):** leverage, interest_coverage, current_ratio, log_assets, profit_margin, sector
- **Numeric features:** 5
- **Categorical features:** 1

## Leakage notes

No high-correlation target-leakage suspects were flagged during exploration.

Protected attributes are excluded from the model feature set (disparate-treatment avoidance). See [methodology](./methodology.md).
