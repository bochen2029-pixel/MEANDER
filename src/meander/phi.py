"""phi — the learned concept -> shape map, and its four conflicting losses (§4.1).

    1. round-trip    phi^-1(phi(e)) ~= e
    2. continuity    shape distance tracks cosine distance          (A2 upper)
    3. separation    min pairwise shape distance >= delta           (A2 LOWER)
    4. nested        truncate_k(phi(e)) decodes into radius r(k), r monotone (A5)

They genuinely conflict, and the conflict is the experiment (M-PARETO). The same
architecture with weights (1,0,0,0) IS the `roundtrip_only_phi` null — which is
what makes that null strong: it is not a crippled straw man, it is this exact
network denied only the continuity, separation and nesting objectives.

THE SHAPE-DISTANCE PROXY, and why it is principled rather than convenient.
The perceptual metric is a raster operation and is not differentiable through
the renderer. But by Parseval the L2 distance between two EFD contours as
functions of the parameter is exactly proportional to the Euclidean distance
between their coefficient vectors. Low-passing the raster attenuates high
harmonics, so the perceptual metric is well modelled by a HARMONIC-WEIGHTED
coefficient distance. Those weights are not guessed: `measure_harmonic_weights`
recovers them from the actual render + low-pass pipeline. Training therefore
optimises a measured surrogate of the metric the falsifiers will judge with, and
the residual gap between surrogate and truth is reported instead of assumed away.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import efd

__all__ = ["PhiNet", "param_scale", "param_offset", "sample_params", "train",
           "LossWeights", "measure_harmonic_weights"]


# --------------------------------------------------------------- amplitude budget

D1_LO, D1_HI = 0.25, 0.90        # eccentricity bounds; see param_offset
HARMONIC_BASE = 0.20             # amplitude budget at harmonic 2, decaying 1/n


def param_offset(m: int) -> np.ndarray:
    """Centre of each free parameter's range. Only d1 is off-centre, and that
    is the whole point.

    MEASURED, not chosen for taste. Sampling uniformly in coefficient space
    produces self-INTERSECTING contours 71% of the time at a generous amplitude
    budget, and tightening the budget alone never gets below ~30%. The cause is
    d1: with a1 pinned to 1 by the A7 frame, d1 is the ellipse's minor axis, so
    d1 -> 0 collapses the base curve to a segment traversed out and back, and
    then ANY harmonic perturbation crosses it. Bounding d1 into [0.25, 0.90]
    drops self-intersection to ~4% at base 0.22 and ~0.4% at base 0.16.

    The consequence belongs in the spec: the legible region of shape space is a
    small, awkward subset of the coefficient space, so usable capacity is
    strictly less than the free-parameter count suggests. §6 costed harmonics
    but never costed LEGIBILITY. This is a third axis on the Pareto frontier.
    """
    off = np.zeros(n_free_params_local(m), dtype=np.float32)
    off[0] = (D1_HI + D1_LO) / 2.0
    return off


def param_scale(m: int, base: float = HARMONIC_BASE) -> np.ndarray:
    """Half-range of each free parameter; harmonic amplitude decays as 1/n."""
    scales = [(D1_HI - D1_LO) / 2.0]
    for n in range(2, m + 1):
        scales.extend([base / n] * 4)
    return np.asarray(scales, dtype=np.float32)


def n_free_params_local(m: int) -> int:
    return 1 + 4 * (m - 1)


def sample_params(m: int, rng, n: int = 1, base: float = HARMONIC_BASE) -> np.ndarray:
    """Uniform draw inside the pinned, legibility-constrained parameter box."""
    off, sc = param_offset(m), param_scale(m, base)
    return (off + rng.uniform(-1.0, 1.0, size=(n, len(sc))) * sc).astype(np.float32)


class PhiNet(nn.Module):
    def __init__(self, d: int, m: int, hidden: int = 512):
        super().__init__()
        p = efd.n_free_params(m)
        self.m, self.p = m, p
        self.enc = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p), nn.Tanh(),
        )
        self.dec = nn.Sequential(
            nn.Linear(p, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )
        self.register_buffer("scale", torch.from_numpy(param_scale(m)))
        self.register_buffer("offset", torch.from_numpy(param_offset(m)))

    def encode(self, e):                       # -> free-parameter vector
        return self.enc(e) * self.scale + self.offset

    def decode(self, z):
        out = self.dec((z - self.offset) / self.scale)
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# --------------------------------------------------------------- perceptual weights

def measure_harmonic_weights(m: int, cfg, blur_px: float, n_probe: int = 24,
                             eps: float = 0.05, seed: int = 11) -> np.ndarray:
    """How much does the RENDERED, LOW-PASSED image move per unit of each param?

    Perturb one free parameter at a time, render, low-pass, measure the change.
    The result is the diagonal of the metric the eye actually applies — high
    harmonics move the raster far less than low ones, which is precisely why
    training against unweighted coefficient distance would over-invest in detail
    the reader cannot see.
    """
    from . import perceptual, render

    rng = np.random.default_rng(seed)
    p = efd.n_free_params(m)
    scales = param_scale(m)
    weights = np.zeros(p, dtype=np.float64)

    for base in sample_params(m, rng, n_probe):
        r0 = render.rasterize(efd.vector_to_coeffs(base, m), cfg,
                              noise=render.NoiseConfig.none())
        for j in range(p):
            probe = base.copy()
            probe[j] += eps * scales[j]
            r1 = render.rasterize(efd.vector_to_coeffs(probe, m), cfg,
                                  noise=render.NoiseConfig.none())
            weights[j] += perceptual.distance(r0, r1, blur_px) / (eps * scales[j])

    weights /= n_probe
    return (weights / max(weights.max(), 1e-12)).astype(np.float32)


# --------------------------------------------------------------- training

class LossWeights:
    def __init__(self, roundtrip=1.0, continuity=0.0, separation=0.0, nested=0.0):
        self.roundtrip = float(roundtrip)
        self.continuity = float(continuity)
        self.separation = float(separation)
        self.nested = float(nested)

    def as_dict(self):
        return {"roundtrip": self.roundtrip, "continuity": self.continuity,
                "separation": self.separation, "nested": self.nested}


def _weighted_pdist(z, w):
    zw = z * w
    return torch.cdist(zw, zw)


def train(vectors: np.ndarray, m: int, weights: LossWeights,
          harmonic_w: np.ndarray, delta: float = 0.30, epochs: int = 400,
          batch: int = 128, lr: float = 1e-3, seed: int = 0, device: str = "cpu",
          verbose: bool = False):
    """Train one phi to a single frozen operating point. Returns (net, history)."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    e_all = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(dev)
    n, d = e_all.shape

    net = PhiNet(d, m).to(dev)
    w = torch.from_numpy(np.asarray(harmonic_w, dtype=np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # truncation levels for the nested loss: free-param counts at each k < m
    trunc_p = [efd.n_free_params(k) for k in range(1, m)]
    history = []

    for ep in range(epochs):
        idx = torch.randperm(n, device=dev)[:batch]
        e = e_all[idx]
        z = net.encode(e)
        total = torch.zeros((), device=dev)
        parts = {}

        # 1. round-trip
        rec = net.decode(z)
        l_rt = (1.0 - (rec * e).sum(-1)).mean()
        parts["roundtrip"] = float(l_rt.detach())
        total = total + weights.roundtrip * l_rt

        # 2. continuity — shape distance tracks cosine distance
        if weights.continuity > 0:
            sem = 1.0 - e @ e.T
            vis = _weighted_pdist(z, w)
            sm, vm = sem.mean(), vis.mean()
            num = ((sem - sm) * (vis - vm)).mean()
            den = sem.std().clamp_min(1e-8) * vis.std().clamp_min(1e-8)
            l_cont = 1.0 - num / den
            parts["continuity"] = float(l_cont.detach())
            total = total + weights.continuity * l_cont

        # 3. separation — the expensive half of A2; fights continuity
        if weights.separation > 0:
            vis = _weighted_pdist(z, w)
            off = vis + torch.eye(len(e), device=dev) * 1e6
            l_sep = torch.relu(delta - off.min(dim=1).values).mean()
            parts["separation"] = float(l_sep.detach())
            total = total + weights.separation * l_sep

        # 4. nested — coarse code must already carry the category; fights round-trip
        if weights.nested > 0 and trunc_p:
            losses_k = []
            for p_k in trunc_p:
                zt = z.clone()
                zt[:, p_k:] = 0.0
                l_k = (1.0 - (net.decode(zt) * e).sum(-1)).mean()
                losses_k.append(l_k)
            l_nest = torch.stack(losses_k).mean()
            # monotonicity: error must not increase as harmonics are ADDED
            mono = torch.zeros((), device=dev)
            chain = losses_k + [l_rt]
            for i in range(len(chain) - 1):
                mono = mono + torch.relu(chain[i + 1] - chain[i])
            parts["nested"] = float(l_nest.detach())
            total = total + weights.nested * (l_nest + 2.0 * mono)

        opt.zero_grad()
        total.backward()
        opt.step()

        if ep % 50 == 0 or ep == epochs - 1:
            parts["total"] = float(total.detach())
            parts["epoch"] = ep
            history.append(dict(parts))
            if verbose:
                print("   ", {k: round(v, 4) for k, v in parts.items()})

    net.eval()
    return net, history
