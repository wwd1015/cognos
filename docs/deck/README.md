# The COGNOS deck

`index.html` is the presentation of COGNOS: how a run moves, and the thinking behind the
design. It is one self-contained file. Open it in a browser:

```bash
open docs/deck/index.html
```

| Key | Does |
|---|---|
| `→` `↓` `Space` `PageDown` | next slide |
| `←` `↑` `PageUp` | previous slide |
| `Home` / `End` | first / last slide |
| `D` | dark theme |
| `F` | full screen |

The workflow slide plays by itself; Pause, Back and Step (or a click on any stage or gate)
take it over. `#5` in the address opens slide 5. Printing from the browser gives one slide per page.

## Keeping it current

The deck is documentation, and `tests/unit/test_deck.py` holds it to the code:

- the **edition block** (`<script id="edition">`) must name `cognos.__version__`;
- the **`FLOW` list** (between `FLOW:BEGIN` and `FLOW:END`) must equal the engine's steps,
  labels, seats and agent names;
- every **`data-fact`** figure must equal what the code counts.

When the system changes:

1. Change the slide that describes it. A new principle or stage gets its own slide; keep one
   idea per slide. Every slide is a fixed 1920 × 1080 canvas that the page scales to the
   window, so lay out in canvas pixels and keep everything inside it.
2. If the workflow, a label, a count or the version changed, the test says which.
3. Raise `edition` and `date` in the edition block and add a row to the edition log on the
   last slide saying what changed.

The look is the workbench's Ledger design system; the tokens at the top of the file are the
same values as `src/cognos/ui/assets/cognos.css`.
