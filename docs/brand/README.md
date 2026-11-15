# The COGNOS logo

| File | Use |
|---|---|
| `cognos-logo.svg`, `cognos-logo-dark.svg` | The mark and the name, on light and dark grounds |
| `cognos-lockup.svg`, `cognos-lockup-dark.svg` | The mark, the name and the creed, for covers, title slides and the white paper |
| `cognos-mark.svg` | The mark alone: the tab icon (`src/cognos/ui/assets/favicon.svg`) and the deck's icon |
| `cognos-seal.svg` | The stamp a person's decision carries at a gate (`src/cognos/ui/assets/seal.svg`) |

The mark is the creed drawn as one C, read around the curve in the creed's order:

- **Solid ink: the engine disposes.** The base of the C, the structure everything stands on.
- **Dashed ink: agents recommend.** Proposals, not yet final.
- **The oxblood block: people decide.** It locks the C closed, as a person's decision closes a gate.

The same code holds wherever the logo appears: solid is the engine, dashed is an agent, oxblood
is a person. The creed line reads *Engine disposes · Agents recommend · People decide*, with
"People decide" in the accent.

Drawn in the Ledger palette (ink `#16140f`, oxblood `#7d2230`; on dark, `#f0ede4` and `#e0939e`).
The name is set in Source Serif 4 (600, wide tracking) and the creed in IBM Plex Mono, with
system fallbacks. The letters are text rather than outlines (`textLength` keeps their width
fixed under a fallback face), so they can still be refined.

The seal is one colour so it can serve as a CSS mask; there the dashed part reads by its gaps.
