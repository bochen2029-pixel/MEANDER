#!/usr/bin/env python
"""B0 runner — pin the numbers, measure the strong nulls, prove the harness runs.

    python run_b0.py                 # default: m=4, whatever E is on disk
    python run_b0.py --m 6 --n 300
    python run_b0.py --sweep         # the M-PARETO frontier over resolution
    python run_b0.py --no-candidate  # nulls only (B0's actual acceptance)

Paths use forward slashes so the command behaves identically in cmd, PowerShell
and Git Bash.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from meander import harness                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4, help="retained harmonics")
    ap.add_argument("--n", type=int, default=None, help="lexicon size override")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--sweep", action="store_true", help="run the m_sweep frontier")
    ap.add_argument("--no-candidate", action="store_true",
                    help="nulls only — B0 acceptance does not require phi")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.sweep:
        from meander import lock as lock_mod
        frontier = []
        for m in lock_mod.load()["resolution"]["m_sweep"]:
            print(f"\n=== m = {m} " + "=" * 40)
            r = harness.run(m=m, n_lexicon=args.n, epochs=args.epochs,
                            include_candidate=not args.no_candidate, out_dir=args.out)
            frontier.append({
                "m": m,
                "n_free_params": r["n_free_params"],
                "capacity_bits": r["capacity"].get("total_bits"),
                "jnd": r["jnd"],
                "maps": {mm["name"]: {
                    "2afc": mm["f_metric"]["forced_choice_acc"],
                    "p@5": mm["f_metric"]["precision_at_5"],
                    "collision": mm["f_collision"]["rate"],
                    "roundtrip": mm["f_roundtrip"]["top1"],
                    "simple_contours": mm["simple_contour_rate"],
                } for mm in r["maps"] if not mm.get("skipped")},
            })
        path = os.path.join(args.out, "b0_pareto_frontier.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(frontier, fh, indent=2)
        print(f"\n[B0] frontier -> {path}")
    else:
        harness.run(m=args.m, n_lexicon=args.n, epochs=args.epochs,
                    include_candidate=not args.no_candidate, out_dir=args.out)


if __name__ == "__main__":
    main()
