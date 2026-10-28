# The data catalog: the tables a team keeps, chosen after the intent (v1.9)

ADR-0018 moved the data after the intent: a run waits at `gate_source` until the developer gives
it a file, a Snowflake table or query, or several sources. That asked each developer to find the
data from nothing every time, although a commercial-credit team builds its models on the same
tables: obligor financials, facility exposures, macro series, rating history, workout
recoveries. What each table is, what one row means, which column is an outcome and how the
tables join is known to the people who own them, and was written down nowhere the run could
read. A template could only say in prose what data "this kind of model is usually built on".

## Decision

1. **One catalog in force per runs folder.** `<runs>/_catalog/catalog.yaml` lists the tables
   a team keeps; while it does not exist, the default catalog shipped with COGNOS is in force. Each entry has an id, a title, a description (what the table is and what it is used
   for), its grain, its key columns, its date column, its outcome columns, its owner and refresh,
   a source (a file, a Snowflake table or a read-only query, as `SourceConfig`; secrets stay in
   the environment) and what each column means. `links` documents how two tables join, with a
   note. `defaults` is what a run from scratch starts with. Everyone on the folder sees the same
   catalog (ADR-0015).

2. **COGNOS ships a default catalog.** `cognos/default_catalog/catalog.yaml` is read from the
   package, never copied; its synthetic data (`synth.make_catalog_tables`) is written to
   `<runs>/_catalog/default_data/` on first use: obligor quarters with the 12-month default, facility
   exposures, the obligor master, market data for public obligors, quarterly macro with the
   net charge-off rate, an agency rating panel, resolved defaults and defaulted revolvers. Every
   built-in template runs on it. A team writes its own file, which replaces the default
   entirely (no merging, so what is in force is always one file); `cognos catalog --init`
   starts it as a copy of the default. The page and the CLI say which catalog is in force.

3. **A template names its starting tables.** `data.catalog` lists catalog ids; the first is
   the proposed base table. A run from scratch starts with the catalog's `defaults`; a revised
   decision starts with the tables chosen last. A team template saved from a run keeps the
   tables that run chose. Every selection is a starting point: any table can be unchecked and
   any other checked.

4. **The data step shows what it offers.** At `gate_source` the developer sees every catalog
   table with its description, owner, grain, size, date range and whether its keys identify a
   row (measured from the file and cached until it changes; a database table is measured when
   asked, `cognos catalog --measure`), the meaning of its columns, and the documented links. Data
   the catalog does not have is added beside it: files (CSV, Excel, Parquet), Snowflake tables or
   read-only queries.

5. **The catalog informs, it does not decide.** The chosen tables become `data.sources`, each
   carrying its description and column meanings, copied when the data is given so the run keeps
   them if the catalog changes. The documented links between them become `data.links`: always
   offered at explore and measured like any other link (`documented`, with the owner's note),
   preferred by the deterministic agent, never binding (the agent chooses, the developer
   confirms at the data gate; ADR-0014 still holds). A link whose columns the data does not have
   is reported, not raised. The Data Analyst reads the column meanings (`dictionary`) beside the
   numbers. The decision records the catalog ids chosen, with `by`.

## Consequences

- The catalog is configuration, not a data store: it never holds a secret, and a run reads the
  snapshot it took (`data/sources/`), never the catalog again.
- What the catalog declares feeds what explore measures, never a decision. The base table's
  outcome columns rank first among the target candidates; its other outcomes, and the columns
  that identify a row (declared keys, or text columns with many distinct values), are listed for
  the Data Analyst, who recommends excluding them. The developer confirms both at the data gate,
  as before. Without this a joined table's identifiers entered the model as categories (one
  column per obligor) and the largest number on the page was proposed as the target.
- A wrong catalog (a renamed column, a stale link) shows on the page that uses it: undescribed
  and missing columns in the summary, unusable links at explore.
