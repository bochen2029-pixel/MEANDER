"""B0 invariants. Run: python -m pytest tests -q   (or: python tests/test_b0.py)

These guard the properties the spec's axioms actually rest on, not the plumbing.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from meander import capacity, efd, falsifiers, lock, perceptual, phi, render  # noqa: E402


# --------------------------------------------------------------- A7 canonical frame

def test_free_param_count_is_not_4m():
    # The A7 frame pins a1=1, b1=c1=0, so harmonic 1 keeps ONE parameter.
    assert efd.n_free_params(1) == 1
    assert efd.n_free_params(4) == 13
    assert efd.n_free_params(8) == 29


def test_normalize_reaches_canonical_frame_and_is_idempotent():
    rng = np.random.default_rng(0)
    for _ in range(8):
        c = rng.normal(size=(4, 4))
        n1 = efd.normalize(c)
        assert abs(n1[0, 0] - 1.0) < 1e-9      # a1 == 1
        assert abs(n1[0, 1]) < 1e-9            # b1 == 0
        assert abs(n1[0, 2]) < 1e-9            # c1 == 0
        assert np.abs(efd.normalize(n1) - n1).max() < 1e-9


def test_vector_coeff_roundtrip_is_exact():
    rng = np.random.default_rng(1)
    for m in (1, 3, 6):
        v = phi.sample_params(m, rng, 1)[0]
        assert np.allclose(efd.coeffs_to_vector(efd.vector_to_coeffs(v, m)), v)


def test_truncate_is_the_resolution_dial():
    c = efd.vector_to_coeffs(phi.sample_params(5, np.random.default_rng(2), 1)[0], 5)
    t = efd.truncate(c, 2)
    assert np.allclose(t[:2], c[:2]) and np.allclose(t[2:], 0.0)


# --------------------------------------------------------------- legibility

def test_d1_bound_keeps_contours_drawable():
    # The measured finding: d1 -> 0 collapses the base ellipse to a segment and
    # any perturbation self-crosses. The pinned bound must hold it under 10%.
    rng = np.random.default_rng(3)
    bad = sum(not efd.is_simple(efd.reconstruct(efd.vector_to_coeffs(z, 4), 96))
              for z in phi.sample_params(4, rng, 120))
    assert bad / 120 < 0.10, f"self-intersection {bad / 120:.1%} exceeds the pinned budget"


def test_unbounded_d1_is_what_breaks_it():
    # Guards the DIAGNOSIS, not just the fix: with d1 free to approach zero the
    # rate must be far worse, else the bound is cargo cult.
    rng = np.random.default_rng(4)
    z = phi.sample_params(4, rng, 120)
    z[:, 0] = rng.uniform(-0.05, 0.05, size=len(z))
    bad = sum(not efd.is_simple(efd.reconstruct(efd.vector_to_coeffs(r, 4), 96)) for r in z)
    assert bad / len(z) > 0.30


# --------------------------------------------------------------- A6 param/render split

def test_cosmetic_seed_never_touches_meaning():
    # M-CANON: with noise off, the cosmetic seed is inert. Two renders of one
    # meaning must be bit-identical.
    c = efd.vector_to_coeffs(phi.sample_params(4, np.random.default_rng(5), 1)[0], 4)
    cfg = render.RenderConfig()
    a = render.rasterize(c, cfg, noise=render.NoiseConfig.none(), seed=1)
    b = render.rasterize(c, cfg, noise=render.NoiseConfig.none(), seed=999)
    assert np.array_equal(a, b)


def test_cosmetic_variance_stays_under_the_jnd():
    # F-VARIANCE in miniature: noisy renders of ONE meaning must stay closer than
    # the JND, or cosmetic variation is leaking into meaning.
    cfg, nz = render.RenderConfig(), render.NoiseConfig(0.06, 0.10, 0.03, 0.8)
    rng = np.random.default_rng(6)
    zs = phi.sample_params(4, rng, 12)
    ra = [render.rasterize(efd.vector_to_coeffs(z, 4), cfg, noise=nz, seed=i) for i, z in enumerate(zs)]
    rb = [render.rasterize(efd.vector_to_coeffs(z, 4), cfg, noise=nz, seed=500 + i) for i, z in enumerate(zs)]
    jnd = perceptual.calibrate_jnd(np.array(ra), np.array(rb), 1.6)
    assert 0.0 < jnd < 0.2


# --------------------------------------------------------------- capacity

def test_capacity_is_finite_and_far_below_the_prior():
    cfg, nz = render.RenderConfig(), render.NoiseConfig(0.06, 0.10, 0.03, 0.8)
    cap = capacity.measure_bits(4, cfg, nz, n_samples=120)
    assert cap["ok"]
    assert 0.0 < cap["total_bits"] < 15.0, "if this reaches the §6 prior, re-derive"
    assert len(capacity.summarise_by_harmonic(cap)) == 4


# --------------------------------------------------------------- falsifiers

def test_f_metric_is_perfect_when_perception_equals_semantics():
    rng = np.random.default_rng(7)
    v = rng.normal(size=(60, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sem = falsifiers.semantic_distances(v)
    tri = falsifiers.make_triples(sem, n_per_anchor=2, dist_rank=(4, 12))
    assert falsifiers.f_metric(sem, sem, tri)["forced_choice_acc"] == 1.0


def test_f_degrade_rejects_a_category_flip():
    r = falsifiers.f_degrade({1: 0.5, 2: 0.3, 3: 0.1}, {1: 0.0, 2: 0.4, 3: 0.0})
    assert r["monotone"] and not r["pass"]      # monotone radii, but a flip kills it


# --------------------------------------------------------------- M-PIN

def test_lock_refuses_to_invent_a_threshold():
    lk = lock.load()
    assert lock.require(lk, "floors.f_metric_forcedchoice_min") == 0.75
    for missing in ("floors.nonexistent_floor", "version.phi_fp"):
        try:
            lock.require(lk, missing)
        except (KeyError, ValueError):
            continue
        raise AssertionError(f"{missing} should have refused to resolve")


def test_lock_blocks_a_real_peek_but_discloses_a_synthetic_one():
    lk = lock.load()
    disclosed = lock.assert_no_peek(lk)           # current lock: synthetic peeks only
    assert disclosed, "the synthetic-only revisions must be surfaced, not swallowed"

    lk["revisions"] = [{"key": "floors.x", "peeked": True}]
    try:
        lock.assert_no_peek(lk)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a revision peeked at real results must halt the run")

    lk["revisions"] = [{"key": "floors.x", "peeked": False}]
    assert lock.assert_no_peek(lk) == []


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
