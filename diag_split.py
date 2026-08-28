#!/usr/bin/env python
"""DIAGNOSTIC — separate "harder test" from "memorisation". Not a pass test.

The held-out run produced an ambiguous result: phi fell 0.753 -> 0.706 while the
best null fell 0.715 -> 0.621. Two effects are tangled in phi's drop:

    generalisation   the 100 test words were never seen during fitting
    difficulty       triples drawn from 100 words sit closer together than
                     triples from 500 (measured band gap 16.9% vs 25.0%),
                     so the held-out task is simply harder for everyone

Absolute scores across the two runs are therefore not comparable, and no honest
verdict can rest on them.

This isolates the first effect by holding the second fixed. Both models are
scored on THE SAME 100 test words and THE SAME triples; only the fitting set
differs:

    seen-during-fit    fitted on all 500 (so the test words were in training)
    unseen-during-fit  fitted on the 400 train words only

Whatever difficulty the 100-word task has, both models face it identically, so
the gap between them is generalisation and nothing else.

THIS CANNOT PRODUCE A PASS. It runs the seen-during-fit arm deliberately, which
is exactly the contaminated condition the pinned protocol forbids quoting. Its
output is a gap, reported to interpret an existing result. The pinned bars are
untouched and the B1 verdict is unaffected whatever this returns.

    python diag_split.py --m 10 --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

WEIGHTS = (0.5, 3.0, 0.5, 0.5)          # continuity_all_four, the B1 best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=10)
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

    from meander import (capacity as cap_mod, efd, falsifiers, harness,
                         lexicon as lex_mod, lock as lock_mod, maps as maps_mod,
                         perceptual, phi as phi_mod, render)

    code_fp = lock_mod.code_fingerprint()
    lk = lock_mod.load()
    lk["lexicon"]["size_target"] = args.n
    lex = lex_mod.load(lk, data_dir="data")
    cfg = render.RenderConfig.from_lock(lk)
    noise = render.NoiseConfig.from_lock(lk)
    blur = lk["perceptual"]["foveal_blur_px"]
    m = args.m

    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(lex))
    test_idx = np.sort(perm[:args.holdout])
    train_idx = np.sort(perm[args.holdout:])
    V_all, V_train, V_test = lex.vectors, lex.vectors[train_idx], lex.vectors[test_idx]

    # ONE evaluation task, used by both arms.
    sem = falsifiers.semantic_distances(V_test)
    triples = falsifiers.make_triples(sem)
    diff = falsifiers.triple_difficulty(sem, triples)
    print(f"m={m}  evaluation = the SAME {len(V_test)} held-out words for both arms")
    print(f"  band gap {diff['mean_gap']:.4f} "
          f"({diff['gap_over_positive_distance']:.1%}), {len(triples)} triples")

    harmonic_w = phi_mod.measure_harmonic_weights(m, cfg, blur)
    jnd = harness.calibrate_jnd(m, cfg, noise, blur)
    cap = cap_mod.measure_bits(m, cfg, noise)
    bits = cap["total_bits"] if cap.get("ok") else lk["budget"]["glyph_capacity_bits"]
    ctx = {"harmonic_w": harmonic_w, "delta": 0.30, "jnd": jnd}

    def evaluate(smap, fit_vectors, label):
        smap.fit(fit_vectors, m, ctx)
        coeffs = smap.coeffs(V_test)
        rasters = harness.render_all(coeffs, cfg, noise)
        perc = perceptual.distance_matrix(rasters, blur)
        fm = falsifiers.f_metric(perc, sem, triples)
        print(f"    {label:20s} {smap.name:28s} 2AFC={fm['forced_choice_acc']:.3f} "
              f"[{fm['ci95'][0]:.3f},{fm['ci95'][1]:.3f}] P@5={fm['precision_at_5']:.3f}")
        return {"arm": label, "name": smap.name,
                "acc": fm["forced_choice_acc"], "ci95": fm["ci95"],
                "precision_at_5": fm["precision_at_5"]}

    rows = []
    for label, fitset in (("seen-during-fit", V_all), ("unseen-during-fit", V_train)):
        print(f"  arm: {label}  (fitted on {len(fitset)} words)")
        cand = maps_mod.LearnedPhi(f"phi_continuity_all_four",
                                   phi_mod.LossWeights(*WEIGHTS),
                                   kind="candidate", epochs=args.epochs)
        rows.append(evaluate(cand, fitset, label))
        for smap in maps_mod.build_all(bits, epochs=args.epochs,
                                       include_candidate=False):
            if not smap.available:
                continue
            rows.append(evaluate(smap, fitset, label))

    def get(arm, name):
        return next((r for r in rows if r["arm"] == arm and r["name"] == name), None)

    names = sorted({r["name"] for r in rows})
    gaps = {}
    print("\n  GENERALISATION GAP (seen - unseen), difficulty held identical:")
    for nm in names:
        a, b = get("seen-during-fit", nm), get("unseen-during-fit", nm)
        if a and b:
            gaps[nm] = a["acc"] - b["acc"]
            print(f"    {nm:28s} {a['acc']:.3f} -> {b['acc']:.3f}   gap {gaps[nm]:+.3f}")

    out = {"rung": "DIAGNOSTIC-not-a-pass-test", "code_fp": code_fp, "m": m,
           "split_seed": args.split_seed, "difficulty": diff,
           "rows": rows, "generalisation_gap": gaps,
           "cannot_produce_a_pass": True,
           "note": ("the seen-during-fit arm is the contaminated condition the "
                    "pinned protocol forbids quoting; it exists only to hold "
                    "task difficulty fixed so the gap is interpretable")}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"diag_split_m{m}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
