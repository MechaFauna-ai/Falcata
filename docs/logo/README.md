# Falcata logo

## Files

| File | Use |
|---|---|
| `falcata-mark.svg` | Primary mark. README, docs header, slides, anywhere ≥64px. |
| `falcata-icon.svg` | Simplified mark for small sizes: favicons, avatars, ≤64px. |
| `falcata-mark-{180,256,512}.png` | Raster exports of the primary mark. |
| `falcata-icon-{16,32,64,180,512}.png` | Raster exports of the icon. |

Regenerate the PNG files after editing an SVG:

```python
import cairosvg
for src, sizes in (("falcata-mark", (512, 256, 180)),
                   ("falcata-icon", (512, 180, 64, 32, 16))):
    for sz in sizes:
        cairosvg.svg2png(url=f"docs/logo/{src}.svg",
                         write_to=f"docs/logo/{src}-{sz}.png",
                         output_width=sz, output_height=sz)
```

## The idea

Falcata is named for *Falcataria moluccana* — one of the fastest-growing trees
on earth. The mark is that tree, and its branching **is a decision tree**: every
fork carries a split node, every twig ends in a leaf. The crown is drawn flat
and very wide, which is the falcata's actual silhouette, and the foliage is its
bipinnate (feathery, twice-divided) leaf.

The simplified icon strips the foliage and keeps only the split structure —
hollow nodes for splits, filled nodes for leaves — so at favicon size it still
reads as both a tree and the algorithm.

## Relationship to the MechaFauna mark

Falcata is built under [MechaFauna](https://mechafauna.ai), and the mark is
drawn as a sibling of MechaFauna's:

- **Same palette:** terracotta `#A57057` on warm cream `#FBFAF6`.
- **Same treatment:** single-weight monoline, no fills (except leaf nodes),
  rounded caps and joins, one colour.
- **Same idea, inverted subject:** MechaFauna draws an organic animal with
  machinery inside; Falcata draws an organic tree with an algorithm inside.

## Palette

| Role | Hex |
|---|---|
| Stroke | `#A57057` |
| Background | `#FBFAF6` |

Both marks ship with the cream background baked in, matching MechaFauna. For a
transparent version, delete the `<rect>` in the SVG.
