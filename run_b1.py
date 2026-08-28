#!/usr/bin/env python
"""B1 — the node gate. One frozen phi, all four objectives, the Pareto frontier.

Spec §9 B1: train phi on round-trip + continuity + separation + nested; FREEZE;
then require F-METRIC and F-ROUNDTRIP and F-COLLISION to pass at that single phi
and that single m, and ship the frontier.

Two rules this runner exists to enforce, because prose cannot:

  * ONE operating point. The verdict is read at a single m chosen by a rule
    fixed in advance (below), never by picking whichever m looked best per
    falsifier. Nine green ticks harvested from nine different resolutions is
    exactly the checkbox theatre M-PARETO forbids.
  * NO OPERATING POINT IS A LEGITIMATE ANSWER. If no m clears all pinned floors
    at once, that is reported as the result -- MEANDER-the-language dies cheaply
    and honestly -- not smoothed into "promising at m=6".

Selection rule, pinned here before the sweep runs: among resolutions where the
candidate clears every pinned floor, take the SMALLEST m (simplest glyph, best
legibility, fewest bits demanded of hand and eye). If none qualifies, take the m
clearing the most floors and report it as a FAILURE with the shortfall named.

    python run_b1.py                  # full sweep from the lock
    python run_b1.py --m 4 6 8        # a subset
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from meander import harness, lock as lock_mod                          # noqa: E402


def score_against_floors(run: dict, floors: dict) -> dict:
    """Which pinned floors does the candidate clear at this m?"""
    cand = next((m for m in run["maps"]
                 if not m.get("skipped") and m["kind"] == "candidate"), None)
    if cand is None:
        return {"ok": False, "reason": "no candidate map in this run"}

    nulls = [m for m in run["maps"] if not m.get("skipped") and m["kind"] == "null"]
    best_null = max(nulls, key=lambda r: r["f_metric"]["forced_choice_acc"])

    acc = cand["f_metric"]["forced_choice_acc"]
    margin = acc - best_null["f_metric"]["forced_choice_acc"]
    ci_disjoint = cand["f_metric"]["ci95"][0] > best_null["f_metric"]["ci95"][1]
    degrade = cand.get("f_degrade") or {}
    rt = cand["f_roundtrip"]["top1"]

    checks = {
        "f_metric_absolute": acc >= floors["f_metric_forcedchoice_min"],
        "f_metric_margin": (margin >= floors["f_metric_margin_over_best_null"]
                            and ci_disjoint),
        "f_metric_precision": (cand["f_metric"]["precision_at_5"]
                               >= floors["f_metric_precision_at_5_min"]),
        "f_collision": cand["f_collision"]["rate"] <= floors["f_collision_max_rate"],
        "f_roundtrip": rt is not None and rt >= floors["f_roundtrip_min_recovery"],
        "f_degrade": bool(degrade.get("pass")),
    }
    return {
        "ok": True,
        "m": run["m"],
        "checks": checks,
        "n_cleared": sum(checks.values()),
        "n_total": len(checks),
        "all_clear": all(checks.values()),
        "candidate_2afc": acc,
        "best_null": best_null["name"],
        "best_null_2afc": best_null["f_metric"]["forced_choice_acc"],
        "margin": margin,
        "precision_at_5": cand["f_metric"]["precision_at_5"],
        "collision_rate": cand["f_collision"]["rate"],
        "roundtrip_top1": rt,
        "capacity_bits": run["capacity"].get("total_bits"),
        "simple_contour_rate": cand["simple_contour_rate"],
        "synthetic": run["SYNTHETIC_NOT_A_RESULT"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    lk = lock_mod.load()
    floors = lk["floors"]
    m_list = args.m or lk["resolution"]["m_sweep"]

    frontier, synthetic = [], False
    for m in m_list:
        print(f"\n{'=' * 20} m = {m} {'=' * 20}")
        run = harness.run(m=m, n_lexicon=args.n, epochs=args.epochs, out_dir=args.out)
        synthetic = synthetic or run["SYNTHETIC_NOT_A_RESULT"]
        row = score_against_floors(run, floors)
        frontier.append(row)
        if row.get("ok"):
            marks = " ".join(f"{k}={'Y' if v else 'n'}"
                             for k, v in row["checks"].items())
            print(f"  -> cleared {row['n_cleared']}/{row['n_total']}   {marks}")

    usable = [r for r in frontier if r.get("ok")]
    clear = [r for r in usable if r["all_clear"]]
    if clear:
        chosen, verdict = min(clear, key=lambda r: r["m"]), "OPERATING_POINT_FOUND"
    elif usable:
        chosen = max(usable, key=lambda r: (r["n_cleared"], -r["m"]))
        verdict = "NO_OPERATING_POINT"
    else:
        chosen, verdict = None, "NO_USABLE_RUN"

    out = {
        "rung": "B1",
        "schema": lk["schema"],
        "code_fp": lock_mod.code_fingerprint(),
        "SYNTHETIC_NOT_A_RESULT": synthetic,
        "selection_rule": ("smallest m clearing ALL pinned floors; else the m "
                           "clearing the most, reported as a failure"),
        "verdict": verdict,
        "m_operating": chosen["m"] if chosen else None,
        "frontier": frontier,
        "chosen": chosen,
    }
    path = os.path.join(args.out, "b1_frontier.json")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'=' * 56}\nB1 VERDICT: {verdict}")
    if chosen:
        print(f"  m_operating = {chosen['m']}  "
              f"({chosen['n_cleared']}/{chosen['n_total']} floors cleared)")
        for k, v in chosen["checks"].items():
            if not v:
                print(f"    SHORTFALL: {k}")
    if synthetic:
        print("  *** SYNTHETIC LEXICON - NOT A MEANDER RESULT ***")
    if verdict == "NO_OPERATING_POINT":
        print("  Per M-PARETO this is a legitimate outcome, not a setback to tune "
              "away: no single frozen phi clears every floor at any resolution.")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
