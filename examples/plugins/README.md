# Example plugin

[`credit_tools.py`](credit_tools.py) adds one analysis tool (`information_value`: weight of
evidence and IV of a feature against a binary target) and one data source (`fixed_width`).

```bash
export PYTHONPATH=examples/plugins
COGNOS_PLUGINS=credit_tools cognos plugins        # the tool and the source are listed
```

or in a profile:

```yaml
plugins: [credit_tools]
```

or, for a package you install, an entry point:

```toml
[project.entry-points."cognos.plugins"]
credit_tools = "credit_tools"
```

Once loaded, the Data Analyst sees `information_value` in its tool list and can request it by
name; the engine runs it, keeps the result and chart under `stages/explore/analyses/`, and the
workbench labels it with the plugin it came from.

## Tools for other stages

[`validation_tools.py`](validation_tools.py) adds a custom model test for outcomes analysis and
validation (`score_band_monotonicity`, on the champion and the sealed holdout) and a sample-size
screen for design (`events_per_feature`).

```python
from cognos.plugins import Tool

registry.add_tool(Tool("score_band_monotonicity", "Does the outcome rise with the score?",
                       score_band_monotonicity, {"bands": "number of score bands"},
                       needs_target=True, stages=("backtest", "validate"),
                       needs=("model", "holdout"), version="0.1"))
```

- `stages`: whose agent may request it (`intake`, `explore`, `ideate`, `model`, `backtest`,
  `validate`, `comply`, `document`).
- `needs`: what it is handed. `data` arrives as `df`; `train`, `holdout`, `documents` and
  `results` arrive in `env`; `model` gives `env["score"](frame)` and `env["model_path"]`.
  `holdout` and `model` exist only from `backtest` on.
- `checks` in the result (`name`, `passed`, `severity`, `detail`) make it a test: a failed check
  is a finding of the stage, and a `high` one fails validation.
- `available`: return `(False, "why")` when a dependency is missing; the tool is then listed as
  not run instead of being offered.

## Writing a tool

```python
from cognos.plugins import AnalysisTool

def my_test(df, params, env):          # env: target, task, features, datetime_col
    ...
    return {"title": "...", "summary": {"statistic": 1.23},          # small numbers: citable facts
            "table": {"columns": [...], "rows": [[...]]},            # optional
            "chart": {"kind": "bar", "x": [...], "series": [{"name": "...", "y": [...]}]}}  # optional

def register(registry):
    registry.add_tool(AnalysisTool("my_test", "What it answers, in one sentence.", my_test,
                                   {"column": "the feature"}, needs_target=True))
```

A tool is your reviewed code: keep it a pure function of the data and its parameters. Chart kinds
are `bar`, `line`, `scatter`, `histogram` and `heatmap` (see `cognos/analysis/charts.py`).
