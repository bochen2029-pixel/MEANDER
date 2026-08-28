"""The falsifiers (spec §7), as a coupled frontier rather than a checklist.

M-PARETO: F-METRIC, F-COLLISION and F-DEGRADE are ONE multi-objective problem on
the resolution dial. This module computes each one, but the harness is forbidden
from printing them as independent green ticks — they are reported at a single
frozen phi and a single m, with the frontier alongside.

Triple construction is pre-registered HERE, in code, so it cannot be tuned after
a peek. The positive is a rank-1..3 semantic neighbour; the distractor is drawn
from ranks 4..12 — ALSO a near neighbour. Both from a fixed seed.

The distractor band is narrow on purpose. The first implementation drew
distractors from ranks 50..150 and every map scored 0.98-1.00, nulls included,
which made the pinned 0.75 floor unfalsifiable. That was a deviation from the
stake, not a bar that needed moving: v0.2 §7 asks for "confusable NEAR-NEIGHBOUR
triples", and rank 50 of 300 is not a near neighbour. Logged in meander.lock
`revisions` (peeked: synthetic only — no real-E result existed when it changed).

`coarse` is reported alongside as a diagnostic only. It uses the old far band
and is NOT the pinned floor; a map can ace it while failing the real test, which
is precisely what the first run demonstrated.
"""

from __future__ import annotations

import numpy as np

__all__ = ["semantic_distances", "make_triples", "f_metric", "f_roundtrip",
           "f_collision", "f_degrade", "bootstrap_ci"]

POS_RANK = (1, 3)
DIST_RANK = (4, 12)          # the pinned, hard band: distractor is ALSO near
COARSE_DIST_RANK = (50, 150)  # diagnostic only, never the floor


def semantic_distances(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    return (1.0 - v @ v.T).astype(np.float32)


def make_triples(sem: np.ndarray, n_per_anchor: int = 8, seed: int = 4242,
                 dist_rank=DIST_RANK):
    """(anchor, positive, distractor) triples. Fixed before any map is run."""
    rng = np.random.default_rng(seed)
    n = len(sem)
    order = np.argsort(sem, axis=1)                # column 0 is self
    hi = min(dist_rank[1], n - 1)
    lo = min(dist_rank[0], hi - 1)
    if lo < POS_RANK[1] + 1:
        raise ValueError("lexicon too small for the pinned triple construction")

    triples = []
    for i in range(n):
        for _ in range(n_per_anchor):
            pos = order[i, rng.integers(POS_RANK[0], POS_RANK[1] + 1)]
            dist = order[i, rng.integers(lo, hi + 1)]
            if pos != dist:
                triples.append((i, int(pos), int(dist)))
    return np.array(triples)


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 9,
                 alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(correct)
    if n == 0:
        return (float("nan"), float("nan"))
    draws = correct[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))


# --------------------------------------------------------------- F-METRIC

def f_metric(perc: np.ndarray, sem: np.ndarray, triples: np.ndarray, k: int = 5,
             coarse_triples: np.ndarray | None = None):
    """Forced-choice discriminability + retrieval precision, on RENDERED glyphs."""
    a, p, d = triples[:, 0], triples[:, 1], triples[:, 2]
    correct = (perc[a, p] < perc[a, d]).astype(np.float64)
    acc = float(correct.mean())
    lo, hi = bootstrap_ci(correct)

    coarse = None
    if coarse_triples is not None and len(coarse_triples):
        ca, cp, cd = coarse_triples[:, 0], coarse_triples[:, 1], coarse_triples[:, 2]
        coarse = float(np.mean(perc[ca, cp] < perc[ca, cd]))

    n = len(perc)
    big = np.array(perc, dtype=np.float32, copy=True)
    np.fill_diagonal(big, np.inf)
    sem_big = np.array(sem, dtype=np.float32, copy=True)
    np.fill_diagonal(sem_big, np.inf)
    perc_nn = np.argsort(big, axis=1)[:, :k]
    sem_nn = np.argsort(sem_big, axis=1)[:, :k]
    prec = float(np.mean([len(set(perc_nn[i]) & set(sem_nn[i])) / k for i in range(n)]))

    return {"forced_choice_acc": acc, "ci95": [lo, hi], f"precision_at_{k}": prec,
            "n_triples": int(len(triples)), "coarse_acc_diagnostic_only": coarse}


def triple_difficulty(sem: np.ndarray, triples: np.ndarray) -> dict:
    """How far apart ARE the two candidates, semantically?

    Without this the forced-choice number is uninterpretable in both directions.
    A band gap near zero means the two candidates are effectively equidistant in
    meaning, so even a perfect map scores near chance and a wall of failures says
    nothing about phi. A huge gap means the opposite -- the first implementation
    of this test drew distractors so far away that every map scored 0.98-1.00.
    Report the gap next to the score, always.
    """
    a, p, d = triples[:, 0], triples[:, 1], triples[:, 2]
    d_pos, d_dist = sem[a, p], sem[a, d]
    gap = d_dist - d_pos
    return {"mean_positive_distance": float(d_pos.mean()),
            "mean_distractor_distance": float(d_dist.mean()),
            "mean_gap": float(gap.mean()),
            "gap_over_positive_distance": float(gap.mean() / max(d_pos.mean(), 1e-9)),
            "frac_gap_negative": float(np.mean(gap <= 0))}


# --------------------------------------------------------------- F-ROUNDTRIP

def f_roundtrip(decoded: np.ndarray | None, vectors: np.ndarray):
    """Top-1 identity recovery through phi -> phi^-1, same frozen phi as F-METRIC."""
    if decoded is None:
        return {"top1": None, "note": "map has no inverse (structural, not a failure)"}
    sim = np.asarray(decoded, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T
    return {"top1": float(np.mean(np.argmax(sim, axis=1) == np.arange(len(vectors))))}


# --------------------------------------------------------------- F-COLLISION

def f_collision(perc: np.ndarray, jnd: float):
    """Fraction of DISTINCT pairs closer than the measured JND at read resolution."""
    n = len(perc)
    iu = np.triu_indices(n, k=1)
    d = perc[iu]
    return {"rate": float(np.mean(d < jnd)), "jnd": float(jnd),
            "n_pairs": int(len(d)), "min_distance": float(d.min())}


# --------------------------------------------------------------- F-DEGRADE

def f_degrade(radii: dict[int, float], flips: dict[int, float],
              ranks: dict[int, float] | None = None, n_lexicon: int | None = None,
              neighbourhood_frac: float = 0.05):
    """A5: meaning must degrade into VAGUENESS, never into ERROR.

    `radii[k]`  mean semantic error when decoding a k-truncated code
    `flips[k]`  fraction whose k-truncated decode has a different nearest concept
    `ranks[k]`  mean rank of the TRUE concept under the k-truncated decode

    Flip RATE is the wrong criterion and an earlier version of this function got
    it wrong. Requiring zero flips at every k is unachievable by arithmetic, not
    by any fault of the encoding: at k=1 the code carries a single free parameter
    against the ~8 bits needed to name one of a few hundred concepts, so nearly
    everything "flips" no matter how good phi is. Reading that as a failure of A5
    confuses a counting limit with a design defect — and it is stricter than the
    pinned floor, which asks only for monotonicity.

    What A5 actually claims is about flip DISTANCE. Landing on a near neighbour
    with a wide radius is vagueness and is legal; landing far away with a tight
    radius is a confident-wrong read and is fatal. So the test is whether the
    true concept STAYS INSIDE A NEIGHBOURHOOD as resolution drops. On the first
    real profile this cleanly separated the maps that flip rate could not: at k=1
    phi-with-nested-loss sat at radius 0.175 while the round-trip-only null sat
    at 0.897, i.e. nearly orthogonal — which is precisely the failure mode A5
    describes, and precisely what the nested loss is there to prevent.
    """
    ks = sorted(radii)
    seq = [radii[k] for k in ks]
    monotone = all(seq[i] >= seq[i + 1] - 1e-6 for i in range(len(seq) - 1))

    out = {"radii": {int(k): float(radii[k]) for k in ks},
           "flips": {int(k): float(flips[k]) for k in sorted(flips)},
           "monotone_radii": bool(monotone),
           "worst_flip_rate": float(max(flips.values()) if flips else 0.0)}

    if ranks and n_lexicon:
        rseq = [ranks[k] for k in sorted(ranks)]
        mono_rank = all(rseq[i] >= rseq[i + 1] - 1e-6 for i in range(len(rseq) - 1))
        bound = max(2.0, neighbourhood_frac * n_lexicon)
        stays = all(r <= bound for r in rseq)
        out.update({
            "mean_true_rank": {int(k): float(ranks[k]) for k in sorted(ranks)},
            "neighbourhood_bound": float(bound),
            "monotone_rank": bool(mono_rank),
            "stays_in_neighbourhood": bool(stays),
            "pass": bool(monotone and mono_rank and stays),
            "criterion": ("radii and true-concept rank both monotone in k, and the "
                          "true concept never leaves the top "
                          f"{neighbourhood_frac:.0%} of the lexicon = vagueness, "
                          "not error"),
        })
    else:
        out["pass"] = bool(monotone)
        out["criterion"] = "monotone radii only (rank profile unavailable)"
    return out
