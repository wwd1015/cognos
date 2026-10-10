# Data sources, plugins, and analysis the agent requests or writes (v1.2)

Through v1.1 the data stage read one CSV or Parquet file, required the profile to name the target,
and gave the Data Analyst a fixed profile to react to. The workbench showed that profile as tables.
Three things were missing from a real first look at data: the data usually lives in a warehouse;
which column is the outcome is a judgment made against the business intent; and an analyst
explores, with charts, using the firm's own tests and, when none fits, a few lines of code.

## Decision

1. **Data comes through a connector, and the run keeps a snapshot.** `data.source` names a
   connector: `file` (local or uploaded; CSV, Parquet, Excel, JSON lines), `sqlite`, `snowflake`,
   or one a plugin registers. A database source runs exactly one read-only statement (a table, or
   a single `SELECT`). The engine fetches once, stores `data/dataset.parquet`, and records
   provenance beside it (`data/source.json`: connector, table or query, row count, time, the
   snapshot's SHA-256). Every later stage reads the snapshot. Secrets are read from the
   environment at fetch time and never written to a profile, a run, or provenance. A missing
   driver or credential makes the source unavailable with a message.
2. **The dependent variable may be left open.** `data.target` and `task` are optional. When they
   are, the engine ranks the columns that could be an outcome (mechanically: the name, a binary
   coding, words shared with the intent), the Data Analyst chooses one against the confirmed
   business intent, the engine infers the task from that column, and the model developer confirms
   or changes it at `gate_data`. The decision is an override like any other (`overrides.target`);
   changing it re-runs explore. A target fixed in the profile cannot be changed at the gate.
3. **The agent asks; the engine runs.** The Data Analyst still has no tools of its own (it runs
   isolated, as every agent does). It returns *requests*; the engine executes them and shows the
   agent a summary in the next round (`analysis.rounds`). This is the guided-search pattern
   (ADR-0001 stage B) applied to exploration. A request is one of:
   - **a tool**: a named function in the plugin registry. A tool is ordinary reviewed code, so
     what it reports is the engine's number.
   - **Python the agent writes**, when no tool fits.
4. **Plugins extend COGNOS without editing it.** A plugin is a module with `register(registry)`
   that adds analysis tools and data sources. It is found through the `cognos.plugins`
   entry-point group, the profile's `plugins:` list, or `COGNOS_PLUGINS`. A plugin that fails to
   load is reported (`cognos plugins`) and skipped.
5. **Agent-written code runs restricted, and is an artifact.** Before it runs, the script's
   syntax tree is checked: numerical imports only, no files, network, OS, private attributes or
   string evaluation. It then runs in a separate isolated Python process with restricted
   builtins, an empty working directory, a time limit, and a copy of the dataset; it returns
   results only through `emit_value` / `emit_table` / `emit_chart`. A rejected script goes back
   to the agent with the reason, like any failed engine check. Every script is then:
   - **kept** as `stages/explore/analyses/<id>.py`, under a header naming the run, the author,
     the purpose and the code's hash, with its result beside it;
   - **reviewed**: shown in full at `gate_data`, whose decision records the id and hash of every
     script the developer accepted;
   - **delivered to validation**: the validate stage re-runs each script from the saved file and
     compares what it finds with the recorded result (a mismatch, or a file that no longer
     matches its hash, is a finding), and the Independent Validator receives the code;
   - **documented and exported**: printed in the white paper and included in the export.
6. **Charts are specs, not figures.** Tools and scripts emit a small JSON chart description.
   The core package gains no plotting dependency, a chart is a text artifact that travels, and
   the workbench draws every chart in its own theme.

## What this does to "no LLM math"

A tool's numbers are the engine's. A script's numbers are computed by the engine executing code
the agent wrote; they are citable facts (`explore.analysis.<id>.<name>`), and what makes them
trustworthy is not the agent but the chain above: the code is on the record, a person read it,
and validation reproduced it. The agent still never types a number.

## Limits, stated plainly

- The restriction is defence in depth against mistakes and casual misuse, not a security boundary
  against a hostile model. Where that matters, set `analysis.allow_code: false`: tools and
  plugins still work.
- Exploration, like the profile it extends, sees the whole dataset, including rows that later
  become the sealed holdout. That was already true of explore's correlations. The modeler's
  blindness to holdout *performance* is unchanged.
- A target must be numeric: binary coded 0/1, or continuous. Multi-class is not supported.
- Snowflake needs `snowflake-connector-python`; Excel needs `openpyxl`. Neither is a core
  dependency.
