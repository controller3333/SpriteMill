#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_walk_preview.py  (Claude seat, roundtable relay 2026-07-02)

Compose a single animated WebP that shows all 8 walk directions cycling at once,
built from the per-direction 64px cells that build_T_sheet_from_mp4.py already
saved under <build-dir>/cells/<dir>/<dir>_walk{1..5}T.png.

Layout mirrors the T sheet's block structure:
    col 0 = cardinals   front / left / right / back      (rows 0..3)
    col 1 = diagonals   front_left / front_right /
                        back_left / back_right           (rows 0..3)
Canvas = 2*cell_w x 4*cell_h. Frame k (0..4) shows walk[k+1] for every tile.

Memory-safe: only the 8 chosen walk cells for the current frame are in RAM.
"""
import argparse
import os
from PIL import Image

# (direction -> (col, row)) in the 2x4 preview grid
GRID = {
    "front": (0, 0), "left": (0, 1), "right": (0, 2), "back": (0, 3),
    "front_left": (1, 0), "front_right": (1, 1),
    "back_left": (1, 2), "back_right": (1, 3),
}
WALKS = ["walk1", "walk2", "walk3", "walk4", "walk5"]


def cell_path(cells_dir, direction, name):
    return os.path.join(cells_dir, direction, f"{direction}_{name}T.png")


def detect_cell_size(cells_dir):
    """Cell size comes from the actual build output -- cells may be 64x128
    (biped default) or any --cell-size the round used (e.g. 128x96 for
    quadrupeds). Hardcoding 64x128 stretched non-default cells."""
    for direction in GRID:
        for name in ["idle"] + WALKS:
            p = cell_path(cells_dir, direction, name)
            if os.path.exists(p):
                with Image.open(p) as im:
                    return im.size
    raise SystemExit(f"no cells found under {cells_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", required=True,
                    help="<build out-dir>/cells produced by build_T_sheet_from_mp4.py")
    ap.add_argument("--out", required=True, help="output .webp path")
    ap.add_argument("--scale", type=int, default=2, help="integer upscale (NEAREST)")
    ap.add_argument("--duration", type=int, default=140)
    a = ap.parse_args()

    src_w, src_h = detect_cell_size(a.cells_dir)
    cw, ch = src_w * a.scale, src_h * a.scale
    canvas_w, canvas_h = cw * 2, ch * 4

    frames = []
    for wk in WALKS:
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        for direction, (col, row) in GRID.items():
            p = cell_path(a.cells_dir, direction, wk)
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGBA")
            if a.scale != 1:
                im = im.resize((cw, ch), Image.NEAREST)
            canvas.alpha_composite(im, (col * cw, row * ch))
        frames.append(canvas)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=a.duration, loop=0, disposal=2)
    print(f"wrote {a.out}  {frames[0].size}  frames={len(frames)}")


if __name__ == "__main__":
    main()
