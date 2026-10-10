# Linked sources: several inputs, a join that is proposed and confirmed (v1.4)

ADR-0012 gave a run one source: a file, or one read-only statement. A person who already knows
the join writes it as that statement. Two cases stayed outside: inputs of different kinds (a
Snowflake table and a CSV cannot be joined in one query), and inputs whose connection the
developer wants help finding.

## Decision

1. **A profile may list several sources** (`data.sources`, each with a `name`), of any kind a
   connector exists for. The engine fetches each one once and keeps it in the run
   (`data/sources/<name>.parquet`, with provenance and a hash).
2. **The engine measures the links** (`linking.profile`): for every pair of tables, the columns
   that could be a key, the share of each table's rows that find a match, and whether a key
   identifies one row or many. Keys are compared as text, so `7`, `"7"` and `7.0` are the same
   key across a database and a file. A link needs more than overlapping values (the same name,
   or a side that identifies its rows); columns that only together identify a row are offered
   as a composite key. These are the only links there are.
3. **The Data Analyst proposes the plan** (`data_linker`): a base table, the links to join on,
   what to do when a key matches several rows (`aggregate` or `first`) and when a base row has
   no match (`keep` or `drop`). It chooses from the measured links and cannot name another key.
   The agent still executes nothing.
4. **The engine runs the plan** (`linking.apply`). The base table's rows are the modelling
   rows. A join never adds or repeats one: a table with several rows per key is reduced to one
   row first, and the row count is asserted. A base row is dropped only when the plan says so,
   and every dropped row is counted and raised as a finding.
5. **The developer confirms it at the data gate.** The page shows the sources, the join as run
   with its match rates, the links not used and the analyst's concerns. Accepting fixes the
   plan (`overrides.join`); changing it rebuilds the dataset and re-runs the exploration. The
   decision records the digest of the plan that was on the page (`reviewed_join`).
   **Keep or drop is recommended from risk, not by default.** For each link the engine compares
   the target between the rows that match and the rows that do not (`a_outcome` / `b_outcome`:
   means, counts, and the difference in standard errors). The recommendation is to drop only
   when the unmatched rows are at most a tenth of the table and their outcome is within chance
   of the rest (|z| < 2, tested from 20 rows up); otherwise to keep them, with the reason on
   the page (`linking.unmatched_advice`). The developer can overrule it per run at the gate.
6. **A stated join needs no proposal.** `data.base` and `data.join` in the profile are run as
   written; no agent is asked, and the gate cannot change them (the profile is the authority,
   as it is for a stated target).
7. **Downstream is unchanged.** The joined table is `data/dataset.parquet`; every stage reads
   it as before. The white paper prints the sources and the join.

## Consequences

- **Not solved: time.** The engine does not know which rows of a joined table are dated after
  the observation. Averaging a table of periods can leak the future. The prompt tells the
  analyst to raise it and the page repeats the concern; nothing checks it mechanically. An
  as-of join (latest row at or before a date) is the next step.
- **Not solved: missing values.** The model families do not accept them. When the
  recommendation is to keep unmatched rows, the added columns have gaps and the model stage
  fails unless the developer excludes those columns, fixes the source, or chooses to drop. The
  page says so; nothing imputes.
- The outcome comparison needs the target in the table that loses rows. With an open target,
  or a target in another table, the advice falls back to the share of rows alone and says so.
- Aggregation is one rule for a whole table (numbers averaged, text takes the first value, a
  row count added). A developer who needs sums or a latest value writes the query or prepares
  the table.
- Sources are held in memory as DataFrames. Large tables should be reduced in the source
  (`query`, `limit`).
