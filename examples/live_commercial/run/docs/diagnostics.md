---
type: diagnostics
title: Statistical Diagnostics
description: The full statistical test battery.
tags:
- diagnostics
timestamp: '2026-09-30T02:35:19.201696+00:00'
---

# Statistical Diagnostics

2 of 2 tests passed; max failed severity: INFO.

| test | category | statistic | p-value | result | interpretation |
| --- | --- | --- | --- | --- | --- |
| max_vif | multicollinearity | 1.5673 | n/a | pass | Max VIF=1.6 on 'sector_industrials'. |
| condition_number | multicollinearity | 4.8950 | n/a | pass | Design condition number=4.9. |

The battery is implemented in {@code:src/cognos/stages/stat_tests.py#run_battery}. See [model](./model.md).
