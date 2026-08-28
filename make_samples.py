#!/usr/bin/env python
"""Emit sample glyph output — SVG per meaning, plus a PNG contact sheet.

This is what "MEANDER outputs language" concretely means at B1: a closed vector
curve per concept, with the Layer-2 parameter spec riding in <metadata> so the
v1 cheap-tier decode is exact. Note honestly what that implies — the cheap tier
reads the sidecar, not the drawing. Until B5 the mark is not yet the channel.

    python make_samples.py --n 12 --m 6
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from meander import efd, lock as lock_mod, phi, render                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    lk = lock_mod.load()
    cfg = render.RenderConfig.from_lock(lk)
    noise = render.NoiseConfig.from_lock(lk)
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    params = phi.sample_params(args.m, rng, args.n)

    tiles, n_simple = [], 0
    for i, z in enumerate(params):
        coeffs = efd.vector_to_coeffs(z, args.m)
        n_simple += efd.is_simple(efd.reconstruct(coeffs, 96))

        meta = {
            "schema": lk["schema"],
            "version": {"E_fp": None, "phi_fp": None, "R_fp": None},
            "orientation": {"beacon": "canonical", "frame": "A7"},
            "head": {"efd_free_params": [round(float(v), 5) for v in z], "m": args.m},
            "legs": [],
            "global": {"negation": False, "mood": "decl", "tense": "pres"},
            "role_parity": None,
            "checksum": None,
            "HONESTY": ("role_parity absent -> A8 downgraded: this glyph is "
                        "ROLE-SWAP-UNVERIFIED and may decode confident-wrong"),
        }
        svg = render.to_svg(coeffs, cfg, metadata=meta, seed=i)
        with open(os.path.join(args.out, f"glyph_{i:02d}.svg"), "w",
                  encoding="utf-8") as fh:
            fh.write(svg)

        clean = render.rasterize(coeffs, cfg, noise=render.NoiseConfig.none())
        inked = render.rasterize(coeffs, cfg, noise=noise, seed=i)
        tiles.append(np.hstack([clean, inked]))

    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    h, w = tiles[0].shape
    sheet = np.zeros((rows * h, cols * w), dtype=np.float32)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    png = os.path.join(args.out, "contact_sheet.png")
    Image.fromarray(((1.0 - sheet) * 255).astype(np.uint8)).save(png)

    print(f"wrote {args.n} SVGs + {png}")
    print(f"m={args.m}  free params/glyph={efd.n_free_params(args.m)}  "
          f"simple contours {n_simple}/{args.n}")
    print("each tile: clean render | same meaning through ink+scan noise")


if __name__ == "__main__":
    main()
