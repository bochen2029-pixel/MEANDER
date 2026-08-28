"""Layer 3 — the render (spec §3.3). params -> raster, and params -> SVG.

Two render modes, and the distinction matters:

  "stroke"  the drawn glyph: a variable-width closed stroke. This is what an eye
            sees, so it is what the perceptual metric consumes.
  "fill"    the silhouette. Used ONLY for coefficient re-extraction in the
            capacity measurement, because recovering a stroked curve's medial
            axis is a B5 (hard-tier decoder) problem and folding it into the B0
            capacity number would silently measure the wrong channel.

The noise model is the honest part of this file. `ink` noise perturbs the curve
BEFORE it is drawn (a hand and a nib: smooth, correlated wobble plus width
variation — not white noise, because a pen does not jitter independently at
every point). `scan` noise degrades the raster AFTER drawing (sensor blur plus
additive Gaussian). Both sigmas are `nominal_unmeasured` in meander.lock and
every law that consumes them is listed there as disabled.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from . import efd

__all__ = ["RenderConfig", "NoiseConfig", "rasterize", "to_svg"]

SUPERSAMPLE = 4


class RenderConfig:
    def __init__(self, raster_px=128, stroke_width_px=2.5, contour_points=256,
                 margin=0.12, scale=None):
        self.raster_px = int(raster_px)
        self.stroke_width_px = float(stroke_width_px)
        self.contour_points = int(contour_points)
        self.margin = float(margin)
        # Fixed scale, NOT fit-to-bbox. The A7 frame already pins a1=1, so the
        # size is known; fitting each glyph to its own bbox would erase real
        # shape differences that happen to change the bounding box.
        self.scale = float(scale) if scale is not None else 0.30

    @classmethod
    def from_lock(cls, lock):
        r = lock["render"]
        return cls(r["raster_px"], r["stroke_width_px"], r["contour_points"], r["margin"])


class NoiseConfig:
    def __init__(self, ink_sigma=0.0, ink_width_sigma=0.0, scan_sigma=0.0,
                 scan_blur_px=0.0):
        self.ink_sigma = float(ink_sigma)
        self.ink_width_sigma = float(ink_width_sigma)
        self.scan_sigma = float(scan_sigma)
        self.scan_blur_px = float(scan_blur_px)

    @classmethod
    def from_lock(cls, lock, mult=1.0):
        n = lock["noise"]
        return cls(n["ink_sigma"] * mult, n["ink_width_sigma"] * mult,
                   n["scan_sigma"] * mult, n["scan_blur_px"] * mult)

    @classmethod
    def none(cls):
        return cls()

    @property
    def is_clean(self):
        return (self.ink_sigma == 0 and self.ink_width_sigma == 0
                and self.scan_sigma == 0 and self.scan_blur_px == 0)


# --------------------------------------------------------------- ink

def _smooth_field(n: int, sigma: float, rng: np.random.Generator,
                  n_modes: int = 6) -> np.ndarray:
    """A periodic, band-limited random field over the contour parameter.

    Correlated by construction: a hand wobbles slowly. White noise here would
    make the ink model far kinder than reality at high harmonics — it would show
    up as a flat noise floor and inflate the measured capacity.
    """
    if sigma <= 0:
        return np.zeros(n)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    out = np.zeros(n)
    for k in range(1, n_modes + 1):
        amp = rng.normal(0.0, 1.0 / k)          # 1/f: slow wobble dominates
        phase = rng.uniform(0.0, 2.0 * np.pi)
        out += amp * np.cos(k * t + phase)
    s = out.std()
    return out / s * sigma if s > 1e-12 else out


def _apply_ink(contour: np.ndarray, noise: NoiseConfig, stroke_px: float,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Perturb the curve along its normals; return (contour, per-point width)."""
    n = len(contour)
    widths = np.full(n, stroke_px, dtype=float)
    if noise.ink_width_sigma > 0:
        widths = widths * (1.0 + _smooth_field(n, noise.ink_width_sigma, rng))
        widths = np.clip(widths, 0.4, None)
    if noise.ink_sigma > 0:
        tangent = np.roll(contour, -1, axis=0) - np.roll(contour, 1, axis=0)
        norm = np.hypot(tangent[:, 0], tangent[:, 1])
        norm = np.where(norm < 1e-12, 1e-12, norm)
        normal = np.stack([-tangent[:, 1] / norm, tangent[:, 0] / norm], axis=1)
        # sigma is in units of stroke width, converted to contour units by the
        # caller's scale; here contour is already in pixel space.
        offset = _smooth_field(n, noise.ink_sigma * stroke_px, rng)
        contour = contour + normal * offset[:, None]
    return contour, widths


# --------------------------------------------------------------- raster

def rasterize(coeffs: np.ndarray, cfg: RenderConfig, mode: str = "stroke",
              noise: NoiseConfig | None = None, seed: int = 0) -> np.ndarray:
    """Coefficients -> float32 raster in [0,1], 1.0 = ink.

    The cosmetic seed is independent of the parameters (A6 / M-CANON): changing
    `seed` must never change the decode, only the calligraphy.
    """
    noise = noise or NoiseConfig.none()
    rng = np.random.default_rng(seed)

    contour = efd.ensure_ccw(efd.reconstruct(coeffs, cfg.contour_points))

    px = cfg.raster_px * SUPERSAMPLE
    stroke = cfg.stroke_width_px * SUPERSAMPLE
    half = px / 2.0
    pts = contour * (px * cfg.scale) + half

    pts, widths = _apply_ink(pts, noise, stroke, rng)

    img = Image.new("L", (px, px), 0)
    draw = ImageDraw.Draw(img)

    if mode == "fill":
        draw.polygon([tuple(p) for p in pts], fill=255)
    elif mode == "stroke":
        # Variable width: neither SVG nor PIL can taper a stroke, so the curve is
        # drawn in chunks at locally-averaged width with rounded joins. Chunking
        # (rather than a disc per point) keeps this ~20x faster, which matters:
        # the harness renders the whole lexicon once per map per resolution.
        closed = np.vstack([pts, pts[:1]])
        n_chunks = 24
        bounds = np.linspace(0, len(pts), n_chunks + 1).astype(int)
        for ci in range(n_chunks):
            lo, hi = bounds[ci], bounds[ci + 1] + 1
            if hi - lo < 2:
                continue
            seg = closed[lo:hi]
            w = max(1, int(round(float(widths[lo:hi - 1].mean()))))
            draw.line([tuple(p) for p in seg], fill=255, width=w, joint="curve")
    else:
        raise ValueError(f"unknown render mode {mode!r}")

    img = img.resize((cfg.raster_px, cfg.raster_px), Image.BOX)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    if noise.scan_blur_px > 0:
        arr = gaussian_filter(arr, noise.scan_blur_px)
    if noise.scan_sigma > 0:
        arr = arr + rng.normal(0.0, noise.scan_sigma, arr.shape).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


# --------------------------------------------------------------- SVG

def to_svg(coeffs: np.ndarray, cfg: RenderConfig, metadata: dict | None = None,
           seed: int = 0) -> str:
    """Layer-3 emit: one closed variable-width path + the Layer-2 payload.

    The Layer-2 spec rides in <metadata> — that is what makes the v1 "cheap tier"
    decode exact (§5). Note honestly what that means: the cheap tier reads the
    sidecar, it does not read the drawing. Until B5 the mark is not yet the
    channel.
    """
    import json

    contour = efd.ensure_ccw(efd.reconstruct(coeffs, cfg.contour_points))
    px = cfg.raster_px
    pts = contour * (px * cfg.scale) + px / 2.0

    d = f"M {pts[0,0]:.3f},{pts[0,1]:.3f} " + " ".join(
        f"L {x:.3f},{y:.3f}" for x, y in pts[1:]) + " Z"

    meta = ""
    if metadata:
        payload = json.dumps(metadata, separators=(",", ":"))
        payload = payload.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        meta = f'  <metadata id="meander-layer2">{payload}</metadata>\n'

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{px}" height="{px}" '
        f'viewBox="0 0 {px} {px}">\n'
        f'{meta}'
        f'  <path d="{d}" fill="none" stroke="#111" '
        f'stroke-width="{cfg.stroke_width_px}" stroke-linejoin="round" '
        f'data-cosmetic-seed="{seed}"/>\n'
        f'</svg>\n'
    )
