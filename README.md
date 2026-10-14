# Paper capture

Run `.venv/bin/python example.py`. Space or S captures the detected sheet;
Q quits. The RealSense color stream defaults to 1920 × 1080 at 8 FPS.

Each capture saves two matching A4 PNGs in `captures/`:

- `*_bw.png`: pure black ink on white paper, stored as a 1-bit image.
- `*_color.png`: the original perspective-corrected color crop for comparison.

The cleanup corrects gradual lighting differences before selecting dark marks.
It runs only when capturing. Both files retain the same dimensions, orientation,
and DPI, so cleanup does not move or resize the drawing.

For faint pencil, lower the ink contrast:

```bash
.venv/bin/python example.py --ink-contrast 8
```

Higher contrast values reject more paper texture but may erase light strokes.
`--background-mm 25` increases the lighting correction window for thick strokes
or filled shapes; the window should be larger than their width. Hard shadow
edges can still look like ink, so use diffuse lighting and check the preview.

Optional `--min-ink-area 8` removes isolated components smaller than eight output
pixels; this can remove intentional dots too. `--border-mm 1` clears a 1 mm edge
without changing coordinates, including any drawing in that strip. Both options
default to zero to preserve your marks.

Process an existing camera photo with
`.venv/bin/python example.py --image photo.jpg`. All four paper corners must be
visible against a contrasting background.

Use the black/white image for image completion. To keep the robot from retracing
your drawing, the later completion step should preserve the original strokes
and generate an additional-strokes mask in these same A4 coordinates. Sending
to a model and robot motion are not implemented by this capture script.

Check cleanup with `.venv/bin/python -m unittest test_paper_cleanup.py`.
