# Catalog sources, a short table picker, and links drawn on the join

The data catalog (ADR-0020) was one YAML file, and "Provide the data" listed every table in it
with everything open: about 2,000 px for eight tables. A bank's metadata system holds hundreds
of tables, with schemas, field definitions, keys and owners, and it is where a team would want
the catalog to come from, possibly through a Claude skill that also reads the model's intent.

The join canvas (ADR-0014) could already add or remove a measured link and change the base
table, but nothing on it looked clickable, two join settings applied to every link at once, and
a key the engine had not paired (names that differ, `obl_id` / `obligor_id` at 85%) could not
be used at all.

## Decision

1. **A catalog source is a plugin contract** (`catalog.CatalogSource`): `search(text, intent)`,
   `describe(id)`, `links(ids)`, `defaults()` and, optionally, `assess(ids, intent)`. The YAML
   catalog is the built-in source (`YamlCatalog`); a plugin adds another with
   `registry.add_catalog`, and `COGNOS_CATALOG` names the one in force. A named source that is
   missing or unavailable is an error, never a quiet switch to other tables. Everything the
   workbench and a run ask of the catalog goes through the source.

2. **A source informs, it never decides.** `assess` returns advice (`fits` / `partly` /
   `unclear`, each with reasons), shown with the source's name. Without it the engine assesses
   mechanically (`catalog.fit`): the first table should carry an outcome, every other table
   should have a documented link or a shared key to one chosen before it, and the dates should
   overlap. The gate_source decision records the source and the advice the developer saw. A
   source runs on the engine's side as reviewed plugin code: agents still have no tools.

3. **The picker shows what is chosen; the rest is found by searching.** The chosen tables are
   one line each (base marked, fit badge, remove), a line opens to the description, the fit
   reasons and the column meanings; search runs on the server so it works with any catalog
   size. Documented links are shown where they matter: on the join canvas.

4. **Each link has its own rules.** Clicking a link on the canvas opens its settings: in the
   join or not, several rows per key (average or first), base rows with no match (as
   recommended, keep, drop). The decision carries them per link (`join.rules`); the old
   one-answer-for-all fields still work.

5. **A person may draw a link, and the engine measures it first.** Dragging from a column of
   one table to a column of another asks the engine to measure that pair
   (`service.measure_link`, nothing recorded); the decision measures it again
   (`linking.measure_drawn`, before the run's lock) and refuses one that cannot be. Drawn links
   are kept in `overrides.drawn_links` with who drew them, in order, so explore measures them
   with the rest under stable ids (`D1`, `D2`, ...). New data at gate_source clears them. A
   join still never adds or repeats a base row: a drawn link joins through `linking.apply`
   like any other.

6. **Not on the canvas:** filters, unions, calculated columns, many-to-many joins. Prepared
   data belongs upstream (the warehouse, a catalog table). Where a table sits on the canvas is
   a view, never a decision.

## Consequences

- A team plugs its metadata system in without changing COGNOS (`examples/plugins/
  metadata_catalog.py` reads an export of one). One source is in force at a time.
- The fit check answers the open question from ADR-0020 ("warn when a chosen table has no
  documented link"): it says so as advice and never blocks.
- A link the developer names is a measured link like the Data agent's: "the agent chooses
  among measured links, it never names a key" still holds; a person may name one.
