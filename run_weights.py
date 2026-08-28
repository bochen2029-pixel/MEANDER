#!/usr/bin/env python
"""The loss-weight frontier — closing the honest gap in B1.

B1 swept the resolution dial m at ONE fixed loss weighting (1, 1, 0.5, 0.5).
But M-PARETO is about the coupled continuity/separation/round-trip/nested
tradeoff, so a single weighting is one point on that surface, not the surface.
Declaring the language dead without sweeping it would repeat, at larger scale,
the mistake that produced this session's premature "A5 is heading for
retraction": mistaking one badly-chosen configuration for the shape of the thing.

A PREDICTION THIS FILE MADE AND THE DATA REFUTED, kept here because the refutation
is the useful part. The original argument was: `continuity_only` = (0, 1, 0, 0)
switches off every objective competing with F-METRIC, so whatever it scores is
THE CEILING — no weighting can beat a config optimising the target and nothing
else. Measured, that is false:

    m=3    continuity_only 0.705   continuity_all_four 0.721
    m=10   continuity_only 0.747   continuity_all_four 0.753

Switching the "competing" objectives back ON made F-METRIC BETTER. The argument
was structurally wrong, not just numerically off. F-METRIC is a forced-choice
DISCRIMINATION task: it requires the positive to be nearer than the distractor,
which needs SEPARATION as much as continuity. The metric law and the identity
law are not purely opposed here — this test consumes both, so a config that
optimises correlation alone is not an upper bound on it. (`continuity_only` also
scores round-trip 0.002, so it could never satisfy B1's conjunction anyway.)

Corollary worth keeping: there is no cheap ceiling argument available. The loss
surface has to be sampled, not reasoned about from which objectives "compete".

PRE-REGISTRATION, unchanged from meander.lock and restated so it cannot drift:
a rescue requires BOTH
    2AFC >= 0.75                              (f_metric_forcedchoice_min)
    margin >= 0.05 over the best null, CIs disjoint  (f_metric_margin_over_best_null)
No bar is moved for this run. Beating the null while missing the absolute floor
is not a rescue, and clearing the floor while tying a 2-D squash is not either.

    python run_weights.py --m 3 10 --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*", default=[3, 10])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads; kept well under core count on purpose")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="restrict to named configs (B1-EXT-1 freezes one)")
    ap.add_argument("--prereg", default=None,
                    help="pre-registration id to stamp into the output")
    ap.add_argument("--tag", default="b1_loss_weight_frontier")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # Politeness: this is CPU-only work (no CUDA anywhere in the package) but
    # torch would otherwise take every core. Set before torch is imported.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    import torch
    torch.set_num_threads(args.threads)

    from meander import (capacity as cap_mod, falsifiers, harness,
                         lexicon as lex_mod, lock as lock_mod, maps as maps_mod,
                         perceptual, phi as phi_mod, render)

    # (roundtrip, continuity, separation, nested)
    CONFIGS = [
        ("continuity_only",      (0.0, 1.0, 0.0, 0.0)),   # THE CEILING
        ("continuity_dominant",  (0.2, 3.0, 0.0, 0.0)),
        ("continuity_plus_sep",  (0.2, 3.0, 0.5, 0.0)),
        ("continuity_all_four",  (0.5, 3.0, 0.5, 0.5)),
        ("b1_baseline",          (1.0, 1.0, 0.5, 0.5)),   # what B1 actually ran
    ]

    if args.configs:
        keep = set(args.configs)
        unknown = keep - {n for n, _ in CONFIGS}
        if unknown:
            raise SystemExit(f"unknown config(s): {sorted(unknown)}")
        CONFIGS = [(n, w) for n, w in CONFIGS if n in keep]

    code_fp_at_start = lock_mod.code_fingerprint()   # before any work
    lk = lock_mod.load()
    floors = lk["floors"]
    # Legibility floor, pinned in meander.lock preregistrations[B1-EXT-1]. A
    # margin bought with contours a hand cannot draw is not a pass.
    legibility_min = 0.95
    for pr in lk.get("preregistrations") or []:
        legibility_min = (pr.get("new_bar_pinned_now") or {}).get(
            "simple_contour_rate_min", legibility_min)
    lk["lexicon"]["size_target"] = args.n
    lex = lex_mod.load(lk, data_dir="data")
    cfg = render.RenderConfig.from_lock(lk)
    noise = render.NoiseConfig.from_lock(lk)
    blur = lk["perceptual"]["foveal_blur_px"]

    print(f"lexicon {lex.source}  N={len(lex)}  synthetic={lex.synthetic}  "
          f"threads={args.threads}/{os.cpu_count()}")

    sem = falsifiers.semantic_distances(lex.vectors)
    triples = falsifiers.make_triples(sem)
    coarse = falsifiers.make_triples(sem, seed=4243,
                                     dist_rank=falsifiers.COARSE_DIST_RANK)

    report = []
    for m in args.m:
        print(f"\n{'=' * 22} m = {m} {'=' * 22}")
        t0 = time.time()
        harmonic_w = phi_mod.measure_harmonic_weights(m, cfg, blur)
        jnd = harness.calibrate_jnd(m, cfg, noise, blur)
        cap = cap_mod.measure_bits(m, cfg, noise)
        bits = cap["total_bits"] if cap.get("ok") else lk["budget"]["glyph_capacity_bits"]
        ctx = {"harmonic_w": harmonic_w, "delta": 0.30, "jnd": jnd}
        print(f"  setup {time.time() - t0:.0f}s  capacity {bits:.2f} bits  jnd {jnd:.5f}")

        # Nulls do not depend on phi's loss weights, so measure them ONCE per m.
        best_null, null_rows = None, []
        for smap in maps_mod.build_all(bits, epochs=args.epochs,
                                       include_candidate=False):
            r = harness.evaluate_map(smap, lex.vectors, m, cfg, noise, blur, jnd,
                                     sem, triples, ctx, coarse_triples=coarse)
            if r.get("skipped"):
                continue
            null_rows.append(r)
            if best_null is None or (r["f_metric"]["forced_choice_acc"]
                                     > best_null["f_metric"]["forced_choice_acc"]):
                best_null = r
        bn_acc = best_null["f_metric"]["forced_choice_acc"]
        print(f"  best null: {best_null['name']} {bn_acc:.3f} "
              f"{best_null['f_metric']['ci95']}")

        for name, w in CONFIGS:
            smap = maps_mod.LearnedPhi(f"phi_{name}", phi_mod.LossWeights(*w),
                                       kind="candidate", epochs=args.epochs)
            r = harness.evaluate_map(smap, lex.vectors, m, cfg, noise, blur, jnd,
                                     sem, triples, ctx, coarse_triples=coarse)
            fm = r["f_metric"]
            acc, ci = fm["forced_choice_acc"], fm["ci95"]
            margin = acc - bn_acc
            disjoint = ci[0] > best_null["f_metric"]["ci95"][1]
            legible = r["simple_contour_rate"] >= legibility_min
            rescue = (acc >= floors["f_metric_forcedchoice_min"]
                      and margin >= floors["f_metric_margin_over_best_null"]
                      and disjoint
                      and legible
                      and r["f_metric"]["precision_at_5"] >= floors["f_metric_precision_at_5_min"]
                      and (r["f_roundtrip"]["top1"] or 0) >= floors["f_roundtrip_min_recovery"]
                      and r["f_collision"]["rate"] <= floors["f_collision_max_rate"])
            row = {
                "m": m, "config": name, "weights": w, "capacity_bits": bits,
                "acc": acc, "ci95": ci, "best_null": best_null["name"],
                "best_null_acc": bn_acc, "margin": margin, "ci_disjoint": disjoint,
                "precision_at_5": fm["precision_at_5"],
                "collision": r["f_collision"]["rate"],
                "roundtrip": r["f_roundtrip"]["top1"],
                "simple_contour_rate": r["simple_contour_rate"],
                "clears_absolute": acc >= floors["f_metric_forcedchoice_min"],
                "clears_margin": margin >= floors["f_metric_margin_over_best_null"] and disjoint,
                "clears_legibility": legible,
                "legibility_min": legibility_min,
                "RESCUE": rescue,
            }
            report.append(row)
            print(f"    {name:22s} 2AFC={acc:.3f} [{ci[0]:.3f},{ci[1]:.3f}] "
                  f"margin={margin:+.3f} P@5={fm['precision_at_5']:.3f} "
                  f"rt={row['roundtrip']:.3f} simple={r['simple_contour_rate']:.3f}"
                  f"  {'RESCUE' if rescue else 'fail'}")

    ceiling = [r for r in report if r["config"] == "continuity_only"]
    out = {
        "rung": "B1-weights",
        "prereg_id": args.prereg,
        "legibility_min": legibility_min,
        "code_fp": code_fp_at_start,
        "code_fp_at_end": lock_mod.code_fingerprint(),
        "SYNTHETIC_NOT_A_RESULT": lex.synthetic,
        "floors_used": {k: floors[k] for k in
                        ("f_metric_forcedchoice_min", "f_metric_margin_over_best_null")},
        "any_rescue": any(r["RESCUE"] for r in report),
        "ceiling_configs": ceiling,
        "ceiling_max_acc": max((r["acc"] for r in ceiling), default=None),
        "rows": report,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    print(f"\n{'=' * 56}")
    print(f"ANY RESCUE: {out['any_rescue']}")
    print(f"CEILING (continuity-only, all competing objectives off): "
          f"{out['ceiling_max_acc']:.3f} vs floor "
          f"{floors['f_metric_forcedchoice_min']}")
    if not out["any_rescue"]:
        print("  No loss weighting clears the pinned floors. The B1 verdict of")
        print("  NO_OPERATING_POINT is not an artefact of one configuration.")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
