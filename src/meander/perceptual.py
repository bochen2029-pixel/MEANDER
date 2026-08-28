"""The perceptual metric — and the honest admission about it.

v0.2 §4.1 asks for "perceptual contour distance"; v0.1 §4.1 named LPIPS or human
triplets. Both need either a weights download or human subjects. Neither exists
here yet, so what is pinned is an explicit PROXY:

    low-pass the raster at a foveal scale, then RMS difference.

That is a crude model of one real thing (visual acuity discards high spatial
frequency before similarity is judged) and it is not a human eye. meander.lock
tags it `nominal_unmeasured` and lists F-METRIC's human-legible mode among the
disabled laws. A PASS measured with this metric licenses the MACHINE-READABLE
claim only (v0.2 §8) — never the human-legibility claim. Saying otherwise would
be exactly the kind of quiet upgrade the v0.2 red-team struck out.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

__all__ = ["lowpass", "distance", "distance_matrix", "calibrate_jnd"]


def lowpass(raster: np.ndarray, blur_px: float) -> np.ndarray:
    if blur_px <= 0:
        return np.asarray(raster, dtype=np.float32)
    return gaussian_filter(np.asarray(raster, dtype=np.float32), blur_px)


def distance(a: np.ndarray, b: np.ndarray, blur_px: float) -> float:
    la, lb = lowpass(a, blur_px), lowpass(b, blur_px)
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def _features(rasters: np.ndarray, blur_px: float) -> np.ndarray:
    """(N,H,W) -> (N, H*W) low-passed, so pairwise distance is one cdist call."""
    out = np.empty((len(rasters), rasters[0].size), dtype=np.float32)
    for i, r in enumerate(rasters):
        out[i] = lowpass(r, blur_px).reshape(-1)
    return out


def distance_matrix(rasters: np.ndarray, blur_px: float) -> np.ndarray:
    """Full (N,N) perceptual distance matrix, in the same RMS units as distance()."""
    f = _features(rasters, blur_px)
    npix = rasters[0].size
    return (cdist(f, f, metric="euclidean") / np.sqrt(npix)).astype(np.float32)


def calibrate_jnd(rasters_a: np.ndarray, rasters_b: np.ndarray, blur_px: float,
                  percentile: float = 95.0) -> float:
    """The just-noticeable-difference, calibrated rather than guessed.

    Two independently-noised renders of the SAME meaning define the floor: any
    pair of DIFFERENT meanings closer than that is, at this resolution and this
    sigma, a collision. Taking the p95 rather than the max keeps one unlucky
    render from inflating the threshold and hiding real collisions.
    """
    d = np.array([distance(a, b, blur_px) for a, b in zip(rasters_a, rasters_b)])
    return float(np.percentile(d, percentile))
