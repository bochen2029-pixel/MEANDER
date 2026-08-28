#!/usr/bin/env python
"""Held-out F-METRIC — the check every earlier number in this project skipped.

Everything before this was IN-SAMPLE: phi trained on all 500 words, F-METRIC then
measured on those same 500. v0.1 §7 asked for held-out concepts and this
implementation never did it.

The bias is not symmetric. phi is a ~500k-parameter network; PCA and UMAP are
rank-2 projections. Given the same words to fit and to be judged on, phi has far
more room to memorise, so in-sample scoring flatters phi specifically. That is
why it does NOT weaken the FAIL verdicts -- phi lost while holding the advantage
-- but would make any pass unsafe. Pinned as a rider on B1-EXT-1 in
meander.lock, declared before that run's result existed.

Protocol, fixed by the rider: fit phi AND every null on a 400-word train split,
measure F-METRIC on the 100 held-out words, bars unchanged.

The nulls get their estimator's real transform() for the held-out words (see
maps._embed_new), not a nearest-neighbour stand-in. A null crippled at
generalisation would hand phi a free win, which is exactly the fake floor M-NULL
exists to prevent.

    python run_holdout.py --m 10 20 --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

CONFIGS = {
    "continuity_all_four": (0.5, 3.0, 0.5, 0.5),
    "continuity_dominant": (0.2, 3.0, 0.0, 0.0),
    "b1_baseline":         (1.0, 1.0, 0.5, 0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*", default=[10])
    ap.add_argument("--config", default="continuity_all_four")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=20260828)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)

    from meander import (capacity as cap_mod, falsifiers, harness,
                         lexicon as lex_mod, lock as lock_mod, maps as maps_mod,
                         perceptual, phi as phi_mod, render)

    code_fp_at_start = lock_mod.code_fingerprint()   # before any work
    lk = lock_mod.load()
    floors = lk["floors"]
    legibility_min = 0.95
    for pr in lk.get("preregistrations") or []:
        legibility_min = (pr.get("new_bar_pinned_now") or {}).get(
            "simple_contour_rate_min", legibility_min)

    lk["lexicon"]["size_target"] = args.n
    lex = lex_mod.load(lk, data_dir="data")
    cfg = render.RenderConfig.from_lock(lk)
    noise = render.NoiseConfig.from_lock(lk)
    blur = lk["perceptual"]["foveal_blur_px"]

    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(lex))
    test_idx = np.sort(perm[:args.holdout])
    train_idx = np.sort(perm[args.holdout:])
    V_train = lex.vectors[train_idx]
    V_test = lex.vectors[test_idx]
    print(f"lexicon {lex.source} N={len(lex)}  train={len(V_train)} "
          f"test={len(V_test)}  seed={args.split_seed}  threads={args.threads}")

    # Everything scored below is computed on the TEST words only.
    sem = falsifiers.semantic_distances(V_test)
    triples = falsifiers.make_triples(sem)
    diff = falsifiers.triple_difficulty(sem, triples)
    print(f"held-out triple difficulty: gap={diff['mean_gap']:.4f} "
          f"({diff['gap_over_positive_distance']:.1%} of positive distance), "
          f"{len(triples)} triples")

    def score(smap, m, ctx, jnd):
        smap.fit(V_train, m, ctx)                       # FIT ON TRAIN
        coeffs = smap.coeffs(V_test)                    # SCORE ON TEST
        simple = float(np.mean([efd_is_simple(c) for c in coeffs[:64]]))
        rasters = harness.render_all(coeffs, cfg, noise)
        perc = perceptual.distance_matrix(rasters, blur)
        fm = falsifiers.f_metric(perc, sem, triples)
        dec = smap.decode(smap.encode(V_test))
        rt = falsifiers.f_roundtrip(dec, V_test)
        return {"name": smap.name, "kind": smap.kind, "f_metric": fm,
                "f_collision": falsifiers.f_collision(perc, jnd),
                "f_roundtrip": rt, "simple_contour_rate": simple}

    from meander.efd import is_simple, reconstruct

    def efd_is_simple(c):
        return is_simple(reconstruct(c, 96))

    rows = []
    for m in args.m:
        print(f"\n{'=' * 22} m = {m} {'=' * 22}")
        t0 = time.time()
        harmonic_w = phi_mod.measure_harmonic_weights(m, cfg, blur)
        jnd = harness.calibrate_jnd(m, cfg, noise, blur)
        cap = cap_mod.measure_bits(m, cfg, noise)
        bits = cap["total_bits"] if cap.get("ok") else lk["budget"]["glyph_capacity_bits"]
        ctx = {"harmonic_w": harmonic_w, "delta": 0.30, "jnd": jnd}
        print(f"  setup {time.time() - t0:.0f}s  capacity {bits:.2f} bits")

        best_null = None
        for smap in maps_mod.build_all(bits, epochs=args.epochs,
                                       include_candidate=False):
            if not smap.available:
                print(f"    {smap.name}: SKIPPED - {smap.note}")
                continue
            r = score(smap, m, ctx, jnd)
            print(f"    null {r['name']:28s} 2AFC={r['f_metric']['forced_choice_acc']:.3f}")
            rows.append({"m": m, **_flat(r, m)})
            if best_null is None or (r["f_metric"]["forced_choice_acc"]
                                     > best_null["f_metric"]["forced_choice_acc"]):
                best_null = r

        w = CONFIGS[args.config]
        cand = maps_mod.LearnedPhi(f"phi_{args.config}", phi_mod.LossWeights(*w),
                                   kind="candidate", epochs=args.epochs)
        r = score(cand, m, ctx, jnd)
        fm = r["f_metric"]
        acc, ci = fm["forced_choice_acc"], fm["ci95"]
        bn = best_null["f_metric"]
        margin = acc - bn["forced_choice_acc"]
        disjoint = ci[0] > bn["ci95"][1]
        checks = {
            "f_metric_absolute": acc >= floors["f_metric_forcedchoice_min"],
            "f_metric_margin": margin >= floors["f_metric_margin_over_best_null"] and disjoint,
            "f_metric_precision": fm["precision_at_5"] >= floors["f_metric_precision_at_5_min"],
            "f_collision": r["f_collision"]["rate"] <= floors["f_collision_max_rate"],
            "f_roundtrip": (r["f_roundtrip"]["top1"] or 0) >= floors["f_roundtrip_min_recovery"],
            "legibility": r["simple_contour_rate"] >= legibility_min,
        }
        row = {"m": m, **_flat(r, m), "best_null": best_null["name"],
               "best_null_acc": bn["forced_choice_acc"], "margin": margin,
               "ci_disjoint": disjoint, "checks": checks,
               "n_cleared": sum(checks.values()), "PASS": all(checks.values())}
        rows.append(row)
        print(f"    CAND {r['name']:28s} 2AFC={acc:.3f} [{ci[0]:.3f},{ci[1]:.3f}] "
              f"margin={margin:+.3f} P@5={fm['precision_at_5']:.3f} "
              f"-> {row['n_cleared']}/6 {'PASS' if row['PASS'] else 'FAIL'}")

    out = {
        "rung": "B1-EXT-1-HOLDOUT",
        "code_fp": code_fp_at_start,
        "code_fp_at_end": lock_mod.code_fingerprint(),
        "SYNTHETIC_NOT_A_RESULT": lex.synthetic,
        "protocol": (f"fit on {len(V_train)} train words, F-METRIC scored on "
                     f"{len(V_test)} held-out words, bars unchanged"),
        "split_seed": args.split_seed,
        "config": args.config,
        "held_out_triple_difficulty": diff,
        "legibility_min": legibility_min,
        "any_pass": any(r.get("PASS") for r in rows),
        "rows": rows,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "b1_holdout.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nANY PASS (held-out): {out['any_pass']}\n  -> {path}")


def _flat(r, m):
    return {"name": r["name"], "kind": r["kind"],
            "acc": r["f_metric"]["forced_choice_acc"],
            "ci95": r["f_metric"]["ci95"],
            "precision_at_5": r["f_metric"]["precision_at_5"],
            "collision": r["f_collision"]["rate"],
            "roundtrip": r["f_roundtrip"]["top1"],
            "simple_contour_rate": r["simple_contour_rate"]}


if __name__ == "__main__":
    main()
