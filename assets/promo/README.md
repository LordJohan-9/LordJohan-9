# Profile animation

The profile README is generated from the files in this directory. It contains
only the animated hero and the self-hosted GitHub and email badges.

## Regenerate

Use Python 3.12 and the pinned dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
python assets\promo\build.py
python -m unittest discover -s assets\promo -p "test_*.py"
```

The default export is 2x layout resolution at a 2x frame rate. Set
`PROMO_EXPORT` or `PROMO_Q` to positive integers to adjust those output
settings; leave them unset for the committed assets.

`build.py` runs `train_trace.py` and then `generate.py`. The training program
uses a fixed seed to train a 2-8-8-1 classifier on a deterministic two-moons
dataset. It writes the data, checkpoint metrics, and decision grids to
`trace.json`; generation only reads and replays this committed trace. Training
fails if the final training accuracy is below 95%.

## Content and design

- `content.json` is the source of truth for identity, roles, education, focus,
  skills, brain-region mapping, links, badge labels, and terminal UI copy.
- `tokens.json` defines the palette, canvas, timing, fonts, and accessibility
  thresholds. Palette colors are based on the sources documented in
  [`../../CREDITS.md`](../../CREDITS.md).
- `brain.json` defines the outline and 40 stable node positions; the first 19
  represent neurons from the trained network.
- `train_trace.py` records the model run; `generate.py` draws each frame and
  emits the GIF, WebP, identity-card poster, and SVG badges.
- Fonts and their licenses are stored in `assets/fonts/`.

The animation uses a 880 by 440 layout rendered at 2x. It contains six beats:
CRT power-on, boot log, identity card, training trace, cortex formation, and
sign-off. The GIF is the README animation for broad GitHub compatibility;
the WebP is an optional full-color copy. The README's reduced-motion poster is
included as a `<picture>` source, but GitHub web, mobile web, and app behavior
still needs validation on the live profile.

## Automated checks

The renderer checks the recorded model architecture and accuracy, contrast,
font coverage, loop duration, frame-level flash and saturated-color budgets,
image size limits, and badge size limits. The workflow rebuilds the assets and
fails if the committed output is stale; it never commits generated changes.

Some rendering and encoding libraries can vary between operating systems even
at identical package versions. If the stale-asset workflow reports a binary
difference after a local rebuild, reproduce using its Python 3.12 Ubuntu
environment before changing committed outputs.
