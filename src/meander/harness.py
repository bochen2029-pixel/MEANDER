"""B0 — the falsifier harness and the strong-null measurement.

B0's acceptance (spec §9): the harness runs, the strong nulls are measured, and
the bit budget is pinned in meander.lock. It does NOT train the candidate phi to
a verdict; that is B1, and B1 is only meaningful once a real sense-disambiguated
E is on disk.

The reporting rule is M-PARETO: never a row of independent green checkboxes.
Every falsifier is reported at ONE frozen map and ONE resolution m, together
with what it cost the others.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import capacity as cap_mod
from . import efd, falsifiers, lexicon as lex_mod, lock as lock_mod, maps as maps_mod
from . import perceptual, phi as phi_mod, render

__all__ = ["render_all", "calibrate_jnd", "evaluate_map", "run"]


def render_all(coeffs: np.ndarray, cfg: render.RenderConfig,
               noise: render.NoiseConfig, seed0: int = 0) -> np.ndarray:
    return np.stack([render.rasterize(c, cfg, mode="stroke", noise=noise,
                                      seed=seed0 + i) for i, c in enumerate(coeffs)])


def calibrate_jnd(m: int, cfg: render.RenderConfig, noise: render.NoiseConfig,
                  blur_px: float, n: int = 48, seed: int = 31) -> float:
    """Two independent noisy renders of the SAME meaning define the noise floor."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for i, z in enumerate(phi_mod.sample_params(m, rng, n)):
        c = efd.vector_to_coeffs(z, m)
        a.append(render.rasterize(c, cfg, noise=noise, seed=7000 + i))
        b.append(render.rasterize(c, cfg, noise=noise, seed=9000 + i))
    return perceptual.calibrate_jnd(np.array(a), np.array(b), blur_px)


def _degrade(smap, vectors, m):
    """F-DEGRADE inputs: decode each truncated code, measure radius and flips."""
    if smap.decode(smap.encode(vectors[:1])) is None:
        return None
    z = smap.encode(vectors)
    full = smap.decode(z)
    full_nn = np.argmax(full @ vectors.T, axis=1)
    n = len(vectors)
    radii, flips, ranks = {}, {}, {}
    for k in range(1, m + 1):
        zt = np.array(z, copy=True)
        zt[:, efd.n_free_params(k):] = 0.0
        dec = smap.decode(zt)
        radii[k] = float(np.mean(1.0 - np.sum(dec * vectors, axis=1)))
        sim = dec @ vectors.T
        flips[k] = float(np.mean(np.argmax(sim, axis=1) != full_nn))
        # rank of the TRUE concept in the truncated decode: the vagueness-vs-error
        # discriminator. Staying near = vagueness (legal); falling out = error.
        true_sim = sim[np.arange(n), np.arange(n)][:, None]
        ranks[k] = float(np.mean((sim > true_sim).sum(axis=1) + 1))
    return falsifiers.f_degrade(radii, flips, ranks=ranks, n_lexicon=n)


def evaluate_map(smap, vectors, m, cfg, noise, blur_px, jnd, sem, triples, ctx,
                 coarse_triples=None):
    t0 = time.time()
    if not smap.available:
        return {"name": smap.name, "kind": smap.kind, "skipped": True,
                "note": smap.note}

    smap.fit(vectors, m, ctx)
    coeffs = smap.coeffs(vectors)

    simple = [efd.is_simple(efd.reconstruct(c, 96)) for c in coeffs[:64]]
    rasters = render_all(coeffs, cfg, noise)
    perc = perceptual.distance_matrix(rasters, blur_px)

    out = {
        "name": smap.name,
        "kind": smap.kind,
        "skipped": False,
        "note": smap.note,
        "n_codes": getattr(smap, "n_codes", None),
        "simple_contour_rate": float(np.mean(simple)),
        "f_metric": falsifiers.f_metric(perc, sem, triples,
                                        coarse_triples=coarse_triples),
        "f_collision": falsifiers.f_collision(perc, jnd),
        "f_roundtrip": falsifiers.f_roundtrip(smap.decode(smap.encode(vectors)), vectors),
        "f_degrade": _degrade(smap, vectors, m),
        "seconds": round(time.time() - t0, 1),
    }
    if getattr(smap, "history", None):
        out["train_history"] = smap.history[-1]
    return out


def run(m: int = 4, n_lexicon: int | None = None, epochs: int = 3000,
        include_candidate: bool = True, out_dir: str = "results",
        lock_path: str | None = None, data_dir: str = "data") -> dict:
    t0 = time.time()
    lock = lock_mod.load(lock_path)
    disclosed_peeks = lock_mod.assert_no_peek(lock)
    for rev in disclosed_peeks:
        print(f"[B0] DISCLOSED: '{rev.get('key')}' was revised after a "
              f"{rev.get('peeked')} peek (no floor moved: "
              f"{rev.get('no_floor_was_moved')})")

    if n_lexicon:
        lock["lexicon"]["size_target"] = n_lexicon
    lex = lex_mod.load(lock, data_dir=data_dir)

    cfg = render.RenderConfig.from_lock(lock)
    noise = render.NoiseConfig.from_lock(lock)
    blur_px = lock["perceptual"]["foveal_blur_px"]

    print(f"[B0] lexicon: {lex.source}  N={len(lex)}  d={lex.dim} "
          f"synthetic={lex.synthetic}")
    for note in lex.notes:
        print(f"      note: {note}")

    print(f"[B0] measuring perceptual weights per free parameter (m={m}) ...")
    harmonic_w = phi_mod.measure_harmonic_weights(m, cfg, blur_px)

    print("[B0] calibrating JND from same-meaning noisy renders ...")
    jnd = calibrate_jnd(m, cfg, noise, blur_px)

    print("[B0] measuring channel capacity through render+noise+re-extract ...")
    cap = cap_mod.measure_bits(m, cfg, noise)
    cap["bits_by_harmonic"] = cap_mod.summarise_by_harmonic(cap)
    measured_bits = cap["total_bits"] if cap.get("ok") else lock["budget"]["glyph_capacity_bits"]

    sem = falsifiers.semantic_distances(lex.vectors)
    triples = falsifiers.make_triples(sem)
    coarse_triples = falsifiers.make_triples(
        sem, seed=4243, dist_rank=falsifiers.COARSE_DIST_RANK)
    difficulty = falsifiers.triple_difficulty(sem, triples)
    print(f"[B0] triple difficulty: positive d={difficulty['mean_positive_distance']:.4f} "
          f"distractor d={difficulty['mean_distractor_distance']:.4f} "
          f"gap={difficulty['mean_gap']:.4f} "
          f"({difficulty['gap_over_positive_distance']:.1%} of positive distance)")
    ctx = {"harmonic_w": harmonic_w, "delta": 0.30, "jnd": jnd}

    results = []
    for smap in maps_mod.build_all(measured_bits, epochs=epochs,
                                   include_candidate=include_candidate):
        print(f"[B0]   {smap.name} ...", flush=True)
        r = evaluate_map(smap, lex.vectors, m, cfg, noise, blur_px, jnd, sem,
                         triples, ctx, coarse_triples=coarse_triples)
        if r.get("skipped"):
            print(f"         SKIPPED: {r['note']}")
        else:
            fm = r["f_metric"]
            rt = r["f_roundtrip"]["top1"]
            print(f"         2AFC={fm['forced_choice_acc']:.3f} "
                  f"[{fm['ci95'][0]:.3f},{fm['ci95'][1]:.3f}]  "
                  f"P@5={fm['precision_at_5']:.3f}  "
                  f"coarse={fm['coarse_acc_diagnostic_only']:.3f}  "
                  f"collide={r['f_collision']['rate']:.4f}  "
                  f"rt={'n/a' if rt is None else f'{rt:.3f}'}  ({r['seconds']}s)")
        results.append(r)

    scored = [r for r in results if not r.get("skipped")]
    nulls = [r for r in scored if r["kind"] == "null"]
    cand = next((r for r in scored if r["kind"] == "candidate"), None)
    best_null = max(nulls, key=lambda r: r["f_metric"]["forced_choice_acc"]) if nulls else None

    floors = lock["floors"]
    verdict = {"rung": "B0", "b1_claim_licensed": False}
    if cand and best_null:
        acc = cand["f_metric"]["forced_choice_acc"]
        margin = acc - best_null["f_metric"]["forced_choice_acc"]
        ci_disjoint = cand["f_metric"]["ci95"][0] > best_null["f_metric"]["ci95"][1]
        verdict.update({
            "candidate_2afc": acc,
            "best_null": best_null["name"],
            "best_null_2afc": best_null["f_metric"]["forced_choice_acc"],
            "margin": margin,
            "ci_disjoint": bool(ci_disjoint),
            "clears_absolute_floor": bool(acc >= floors["f_metric_forcedchoice_min"]),
            "clears_margin": bool(margin >= floors["f_metric_margin_over_best_null"]
                                  and ci_disjoint),
        })

    out = {
        "schema": lock["schema"],
        "rung": "B0",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code_fp": lock_mod.code_fingerprint(),
        "m": m,
        "n_free_params": efd.n_free_params(m),
        "lexicon": {"source": lex.source, "n": len(lex), "dim": lex.dim,
                    "synthetic": lex.synthetic,
                    "disambiguation": lex.disambiguation, "notes": lex.notes},
        "SYNTHETIC_NOT_A_RESULT": bool(lex.synthetic),
        "jnd": jnd,
        "triple_difficulty": difficulty,
        "capacity": cap,
        "floors": floors,
        "disclosed_peeks": disclosed_peeks,
        "disabled_laws": lock_mod.disabled_laws(lock),
        "maps": results,
        "verdict": verdict,
        "seconds": round(time.time() - t0, 1),
    }

    os.makedirs(out_dir, exist_ok=True)
    tag = "synthetic" if lex.synthetic else "real"
    path = os.path.join(out_dir, f"b0_{tag}_m{m}_n{len(lex)}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)   # YAML dates in disclosed_peeks
    out["path"] = path
    print(f"[B0] wrote {path}  ({out['seconds']}s)")
    return out
