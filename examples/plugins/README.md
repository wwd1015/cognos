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
