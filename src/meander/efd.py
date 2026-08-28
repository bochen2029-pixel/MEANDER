"""Elliptic Fourier descriptors — the carrier for node signatures (spec §4.1).

A closed contour <-> a set of harmonic coefficients (a_n, b_n, c_n, d_n).
Low harmonics = gross shape (category); high harmonics = fine detail (nuance).
Truncating high harmonics IS the resolution dial (A5). Kuhl & Giardina 1982.

A7 (ORIENTATION) requires a canonical frame: node contours are parameterised in
the beacon frame with a canonical start point. Without it, coefficient
extraction is discontinuous in start-point and rotation — an A2 violation at the
normalisation seam. `normalize()` kills exactly those degrees of freedom.

CONSEQUENCE THE SPEC DID NOT NOTICE. After canonical normalisation harmonic 1 is
pinned to a1=1, b1=c1=0, leaving only d1 (eccentricity) free. So m harmonics
carry **1 + 4*(m-1)** free parameters, not 4*m. The v0.2 §6 prior
`efd_bits_by_harmonic: [4,3,2,1,1,<1]` implicitly credits harmonic 1 with ~4
bits of payload; under A7 harmonic 1 is a frame, not a payload, and carries one
parameter. This shifts the capacity arithmetic and is measured, not asserted,
in capacity.py.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "n_free_params",
    "extract",
    "reconstruct",
    "reconstruct_consistent",
    "normalize",
    "truncate",
    "vector_to_coeffs",
    "coeffs_to_vector",
    "signed_area",
    "ensure_ccw",
    "is_simple",
]


# --------------------------------------------------------------- parameter count

def n_free_params(m: int) -> int:
    """Free parameters carried by m harmonics in the A7 canonical frame."""
    if m < 1:
        raise ValueError("m must be >= 1")
    return 1 + 4 * (m - 1)


# --------------------------------------------------------------- forward / inverse

def extract(contour: np.ndarray, m: int) -> np.ndarray:
    """Contour (N,2), closed, -> coefficients (m,4) as [a,b,c,d] per harmonic.

    The DC term (centroid) is discarded: node identity is shape, not position.
    """
    xy = np.asarray(contour, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("contour must be (N,2)")
    if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if len(xy) < 4:
        raise ValueError("contour too short")

    d = np.roll(xy, -1, axis=0) - xy                 # closed differences
    dt = np.hypot(d[:, 0], d[:, 1])
    dt = np.where(dt < 1e-12, 1e-12, dt)
    t = np.concatenate([[0.0], np.cumsum(dt)])       # (N+1,)
    T = t[-1]

    coeffs = np.zeros((m, 4), dtype=float)
    for n in range(1, m + 1):
        w = 2.0 * np.pi * n / T
        c1, c0 = np.cos(w * t[1:]), np.cos(w * t[:-1])
        s1, s0 = np.sin(w * t[1:]), np.sin(w * t[:-1])
        k = T / (2.0 * n * n * np.pi * np.pi)
        dx_dt, dy_dt = d[:, 0] / dt, d[:, 1] / dt
        coeffs[n - 1, 0] = k * np.sum(dx_dt * (c1 - c0))
        coeffs[n - 1, 1] = k * np.sum(dx_dt * (s1 - s0))
        coeffs[n - 1, 2] = k * np.sum(dy_dt * (c1 - c0))
        coeffs[n - 1, 3] = k * np.sum(dy_dt * (s1 - s0))
    return coeffs


def reconstruct_consistent(coeffs: np.ndarray, n_points: int = 256,
                           iters: int = 6, tol: float = 2e-3) -> np.ndarray:
    """The contour whose MEASURED descriptors equal `coeffs`. Use this to draw.

    Why this exists. `extract(reconstruct(c)) != c`, and the gap does not shrink
    with denser sampling (measured: 0.136 at both 1024 and 4096 points). It is
    not sampling error — it is structural. The EFD series is defined in the
    ARC-LENGTH parameter, but a TRUNCATED series evaluated at uniform t is not
    arc-length parameterised, so re-analysis lands somewhere else.

    That matters because arc length is the only parameterisation a reader can
    observe: nothing in a drawn curve reveals t. So the canonical code must be
    defined by what analysis RETURNS, not by what synthesis was handed. This
    fixed-point iteration finds the pre-image, driving the residual from 0.136
    to ~1.7e-3. Skipping it would have shown up in the capacity measurement as
    channel noise that is really a synthesis artefact, understating capacity,
    and would have forced the B5 decoder to learn a distortion carrying no
    meaning at all.
    """
    target = np.asarray(coeffs, dtype=float)
    cur = target.copy()
    out = reconstruct(cur, n_points)
    for _ in range(iters):
        measured = normalize(extract(ensure_ccw(out), len(target)))
        err = target - measured
        if np.abs(err).max() < tol:
            break
        cur = cur + err
        out = reconstruct(cur, n_points)
    return out


def reconstruct(coeffs: np.ndarray, n_points: int = 256) -> np.ndarray:
    """Coefficients (m,4) -> closed contour (n_points, 2), centred on origin."""
    coeffs = np.asarray(coeffs, dtype=float)
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    for n in range(1, len(coeffs) + 1):
        a, b, c, d = coeffs[n - 1]
        cn, sn = np.cos(n * t), np.sin(n * t)
        x += a * cn + b * sn
        y += c * cn + d * sn
    return np.stack([x, y], axis=1)


# --------------------------------------------------------------- A7 canonical frame

def _rotate_start(coeffs: np.ndarray, theta: float) -> np.ndarray:
    """Shift the start point by theta (phase of the parameterisation)."""
    out = np.empty_like(coeffs)
    for n in range(1, len(coeffs) + 1):
        a, b, c, d = coeffs[n - 1]
        ct, st = np.cos(n * theta), np.sin(n * theta)
        # [a b; c d] @ [[ct, -st],[st, ct]]
        out[n - 1] = [a * ct + b * st, -a * st + b * ct,
                      c * ct + d * st, -c * st + d * ct]
    return out


def _rotate_frame(coeffs: np.ndarray, psi: float) -> np.ndarray:
    """Rotate the shape in the plane by -psi (fixes orientation to the beacon)."""
    cp, sp = np.cos(psi), np.sin(psi)
    R = np.array([[cp, sp], [-sp, cp]])
    out = np.empty_like(coeffs)
    for n in range(len(coeffs)):
        M = np.array([[coeffs[n, 0], coeffs[n, 1]], [coeffs[n, 2], coeffs[n, 3]]])
        M2 = R @ M
        out[n] = [M2[0, 0], M2[0, 1], M2[1, 0], M2[1, 1]]
    return out


def normalize(coeffs: np.ndarray, size_invariant: bool = True) -> np.ndarray:
    """Put coefficients in the A7 canonical frame: b1 = c1 = 0, a1 = 1, d1 free.

    Start-point phase and plane rotation are removed; with size_invariant the
    scale is removed too (node size is not a payload channel — it is fragile
    under scan and a legibility hazard inside a ring).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    a1, b1, c1, d1 = coeffs[0]

    num = 2.0 * (a1 * b1 + c1 * d1)
    den = a1 * a1 + c1 * c1 - b1 * b1 - d1 * d1
    theta0 = 0.5 * np.arctan2(num, den)

    # theta is only defined mod pi/2; pick the branch that puts the semi-major
    # axis on +x with a1 > 0, so the frame is a function of the shape alone.
    best, best_score = None, -np.inf
    for k in range(4):
        cand = _rotate_start(coeffs, theta0 + k * np.pi / 2.0)
        psi = np.arctan2(cand[0, 2], cand[0, 0])
        cand = _rotate_frame(cand, psi)
        score = cand[0, 0] - abs(cand[0, 3])   # prefer a1 large and positive
        if score > best_score:
            best, best_score = cand, score

    out = best
    if size_invariant:
        e = abs(out[0, 0])
        if e > 1e-12:
            out = out / e
    # numerical hygiene: the frame guarantees these are zero
    out[0, 1] = 0.0
    out[0, 2] = 0.0
    return out


def truncate(coeffs: np.ndarray, k: int) -> np.ndarray:
    """The A5 resolution dial: keep harmonics 1..k, zero the rest (same shape)."""
    out = np.array(coeffs, dtype=float, copy=True)
    if k < len(out):
        out[k:] = 0.0
    return out


# --------------------------------------------------------------- packing

def coeffs_to_vector(coeffs: np.ndarray) -> np.ndarray:
    """Canonical coefficients -> the free-parameter vector [d1, a2,b2,c2,d2, ...]."""
    coeffs = np.asarray(coeffs, dtype=float)
    return np.concatenate([[coeffs[0, 3]], coeffs[1:].reshape(-1)])


def vector_to_coeffs(vec: np.ndarray, m: int) -> np.ndarray:
    """Free-parameter vector -> canonical coefficients (m,4), a1=1, b1=c1=0."""
    vec = np.asarray(vec, dtype=float)
    if len(vec) != n_free_params(m):
        raise ValueError(f"expected {n_free_params(m)} params for m={m}, got {len(vec)}")
    coeffs = np.zeros((m, 4), dtype=float)
    coeffs[0] = [1.0, 0.0, 0.0, vec[0]]
    if m > 1:
        coeffs[1:] = vec[1:].reshape(m - 1, 4)
    return coeffs


# --------------------------------------------------------------- geometry guards

def signed_area(contour: np.ndarray) -> float:
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw(contour: np.ndarray) -> np.ndarray:
    """Canonical traversal direction (A7 handedness). Sign of d1 depends on it."""
    return contour if signed_area(contour) >= 0 else contour[::-1].copy()


def is_simple(contour: np.ndarray, samples: int = 96) -> bool:
    """True if the closed contour does not self-intersect.

    A legibility guard, not a formality: unconstrained high-harmonic coefficients
    produce spiky self-crossing blobs that a hand cannot draw and an eye cannot
    parse. phi must be constrained to the simple-contour region or its shapes
    are not glyphs. Coarse O(n^2) segment test on a resampled contour.
    """
    idx = np.linspace(0, len(contour), samples, endpoint=False).astype(int)
    p = contour[idx]
    q = np.roll(p, -1, axis=0)
    n = len(p)

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent through the wrap
            d1 = _cross(p[i], q[i], p[j])
            d2 = _cross(p[i], q[i], q[j])
            d3 = _cross(p[j], q[j], p[i])
            d4 = _cross(p[j], q[j], q[i])
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                return False
    return True
