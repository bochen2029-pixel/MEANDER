"""The bit budget (spec §6) — measured through the real pipeline, not asserted.

v0.2 §11 marks every budget number `prior`. This module replaces the physical
derivation with a measurement: push known coefficients through
render -> ink noise -> scan noise -> re-extract, and read off the per-parameter
SNR. Bits follow from the Gaussian channel capacity,

        bits_j = 0.5 * log2(1 + var_signal_j / var_noise_j)

Two things this measurement settles that the paper estimate could not:

  * Harmonic 1 is a FRAME, not a payload. Under the A7 canonical normalisation
    a1=1 and b1=c1=0, so harmonic 1 carries one free parameter (d1), not four.
    The §6 prior [4,3,2,1,1,<1] implicitly credits it with ~4 bits.
  * Re-extraction is done from the FILLED silhouette, not the stroked glyph.
    Recovering a stroked curve's medial axis is the B5 hard-tier problem;
    folding it in here would measure the decoder's weakness and call it the
    channel's capacity. This number is therefore an UPPER BOUND on what a real
    reader gets, and is labelled as one.

WHAT COUNTS AS THE TRANSMITTED SIGNAL. Not the synthesis parameters. Measured:
`extract(reconstruct(p)) != p` by ~0.14, and the gap does NOT shrink with denser
sampling, because a truncated EFD series evaluated at uniform t is not
arc-length parameterised while analysis can only measure arc length. That offset
is a property of the parameterisation, not of the ink. Charging it to the
channel would understate capacity. (The obvious repair — iterate to the
fixed point so the drawn curve measures back as p — was tried and REJECTED: the
correction inflates amplitudes and drives contours out of the legible region,
collapsing the simple-contour rate from ~96% to ~12%. Consistency and legibility
are in direct conflict, which is itself a finding.) So the reference here is the
CLEAN render's own measurement, and capacity is clean-extract vs noisy-extract.
"""

from __future__ import annotations

import numpy as np
from skimage import measure

from . import efd, phi, render

__all__ = ["extract_from_raster", "measure_bits", "summarise_by_harmonic"]


def extract_from_raster(raster: np.ndarray, cfg: render.RenderConfig,
                        m: int) -> np.ndarray | None:
    """Raster -> canonical free-parameter vector, or None if no contour survives."""
    contours = measure.find_contours(np.asarray(raster, dtype=float), 0.5)
    if not contours:
        return None
    c = max(contours, key=len)
    if len(c) < 16:
        return None
    xy = np.stack([c[:, 1], c[:, 0]], axis=1)          # (row,col) -> (x,y)
    xy = (xy - cfg.raster_px / 2.0) / (cfg.raster_px * cfg.scale)
    xy = efd.ensure_ccw(xy)
    try:
        return efd.coeffs_to_vector(efd.normalize(efd.extract(xy, m)))
    except Exception:                                   # noqa: BLE001
        return None


def _bits_from_regression(sent: np.ndarray, got: np.ndarray,
                          ridge: float = 1e-6, var_floor: float = 1e-9):
    """Gaussian channel capacity, estimated the way leakage demands.

    bits_j = 0.5 * log2( var(z_j) / var(z_j - E[z_j | z_hat]) )

    The conditional expectation is a least-squares fit on the WHOLE received
    vector, and that is the point. Two cheaper estimators were tried and both
    lied:

      variance ratio (var_signal / var_noise)
          reported harmonic 3 carrying 17 bits, more than harmonics 1 and 2
          combined. Re-extraction reparameterises by arc length and dumps
          energy into high harmonics, inflating their variance but not their
          noise.

      marginal correlation (-0.5 log2(1-rho^2))
          physically sane totals, but still gave harmonic 3 ~1.9 bits against
          harmonic 2's ~0.1. Arc-length reparameterisation of an ellipse has
          period pi, so it generates ODD-harmonic content: the h3 channel was
          re-measuring eccentricity (d1) and being credited for it twice.

    Regressing on the full received vector removes what other parameters already
    explain, so each parameter is charged only for information no other
    parameter carries. Requires n_samples > n_free_params.
    """
    n, p = sent.shape
    if n <= p + 2:
        return None, f"need n_samples > n_free_params+2 ({n} <= {p + 2})"
    x = np.hstack([got, np.ones((n, 1))])
    xtx = x.T @ x + ridge * np.eye(x.shape[1])
    beta = np.linalg.solve(xtx, x.T @ sent)
    resid = sent - x @ beta
    # unbiased residual variance: n - (p+1) degrees of freedom consumed by the fit
    dof = max(n - (p + 1), 1)
    var_res = np.maximum((resid ** 2).sum(axis=0) / dof, var_floor)
    var_sig = np.maximum(sent.var(axis=0), var_floor)
    return 0.5 * np.log2(np.maximum(var_sig / var_res, 1.0)), None


def measure_bits(m: int, cfg: render.RenderConfig, noise: render.NoiseConfig,
                 n_samples: int = 160, seed: int = 23) -> dict:
    """Per-free-parameter channel capacity through the actual render pipeline."""
    rng = np.random.default_rng(seed)
    p = efd.n_free_params(m)

    sent, got, clean_got = [], [], []
    n_lost, n_selfcross = 0, 0

    for i, z in enumerate(phi.sample_params(m, rng, n_samples)):
        coeffs = efd.vector_to_coeffs(z, m)
        if not efd.is_simple(efd.reconstruct(coeffs, 128)):
            n_selfcross += 1
            continue
        # The reference is the CLEAN render's measurement, not the synthesis
        # parameters. See the module docstring: the parameter -> observable-code
        # map is not the identity, and that offset is a property of the
        # parameterisation, not of the channel. Comparing z to z_hat would
        # charge the channel for a synthesis artefact and understate capacity.
        clean = render.rasterize(coeffs, cfg, mode="fill",
                                 noise=render.NoiseConfig.none(), seed=1000 + i)
        z_ref = extract_from_raster(clean, cfg, m)
        noisy = render.rasterize(coeffs, cfg, mode="fill", noise=noise, seed=1000 + i)
        z_hat = extract_from_raster(noisy, cfg, m)
        if z_ref is None or z_hat is None:
            n_lost += 1
            continue
        sent.append(z)
        clean_got.append(z_ref)
        got.append(z_hat)

    if len(sent) < 8:
        return {"ok": False, "reason": f"only {len(sent)} usable samples",
                "n_lost": n_lost, "n_selfcross": n_selfcross}

    sent = np.array(sent, dtype=np.float64)
    got = np.array(got, dtype=np.float64)
    clean_got = np.array(clean_got, dtype=np.float64)

    bits, err = _bits_from_regression(sent, got)
    if err:
        return {"ok": False, "reason": err, "n_samples_used": len(sent)}
    bits_clean, _ = _bits_from_regression(sent, clean_got)

    return {
        "ok": True,
        "m": m,
        "n_free_params": p,
        "bits_per_param": bits.tolist(),
        "total_bits": float(bits.sum()),
        "bits_per_param_noiseless": bits_clean.tolist(),
        "total_bits_noiseless": float(bits_clean.sum()),
        "lost_to_noise_bits": float(bits_clean.sum() - bits.sum()),
        "n_samples_used": len(sent),
        "n_lost": n_lost,
        "n_selfcross": n_selfcross,
        "selfcross_rate": n_selfcross / max(n_samples, 1),
        "estimator": ("0.5*log2(var(z) / var(z - E[z|z_hat])), the conditional "
                      "expectation being a least-squares fit on the whole "
                      "received vector, so cross-parameter leakage is removed"),
        "caveat": ("UPPER BOUND: re-extracted from the filled silhouette, so it "
                   "excludes the stroked-glyph medial-axis recovery problem (B5), "
                   "and the estimator is linear-Gaussian, so any nonlinear "
                   "recoverability is not counted."),
    }


def summarise_by_harmonic(result: dict) -> list[float]:
    """Free-parameter bits regrouped per harmonic: h1 has 1 param, h2+ have 4."""
    if not result.get("ok"):
        return []
    bits = result["bits_per_param"]
    out = [float(bits[0])]
    for k in range(1, result["m"]):
        out.append(float(sum(bits[1 + 4 * (k - 1): 1 + 4 * k])))
    return out
