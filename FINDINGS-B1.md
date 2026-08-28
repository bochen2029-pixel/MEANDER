# B1 findings — the node gate, on real E

**2026-08-27/28 · rung B1 · `glove.6B.300d`, 500 monosemous common nouns · against [MEANDER-SPEC-v0.2.md](MEANDER-SPEC-v0.2.md)**

B1 is the make-or-break rung ([§9](MEANDER-SPEC-v0.2.md:181)): train φ on all four
objectives, **freeze it**, and require F-METRIC ∧ F-ROUNDTRIP ∧ F-COLLISION to
pass at that single φ and single m — or kill the language claim cheaply.

**Verdict: `NO_OPERATING_POINT`. The pre-registered B1 test FAILS.**

It fails narrowly, on one bar, and the bar it fails is the one specifically
pinned to prevent this outcome from being mistaken for success.

---

## 0 · What E actually is

`glove.6B.300d`, rank window [200, 20000), filtered to **monosemous common
nouns**: exactly one WordNet synset, part of speech noun, and no
`instance_hypernyms`. 1877 of 18350 survive; 500 sampled.

The last two conditions are load-bearing. Filtering on synset count alone — the
strategy the lock originally pinned — returned a lexicon roughly 40% proper nouns
(*italy, michael, malaysia, nigeria*) and function words (*within, sometimes,
mainly, non*). Each has exactly one synset and none is a concept: proper-noun
geometry encodes entity co-occurrence, function-word geometry encodes syntactic
distribution. F-METRIC run on that lexicon would have measured a structure §2
never asked about, and any pass would have been meaningless.

**Test difficulty**, reported so the score is interpretable: positive distance
0.581, distractor distance 0.726, gap 0.145 — **25% of the positive distance**.
Hard, but not degenerate. (The first implementation of this test drew distractors
from ranks 50–150 and every map scored 0.98–1.00; see `revisions` in the lock.)

---

## 1 · The resolution frontier — and why it misled me

At the B1 baseline weighting (1, 1, 0.5, 0.5):

| m | free params | capacity | φ 2AFC | best null (umap-2D) | margin | P@5 | legible |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.21 | 0.580 | 0.690 | −0.109 | 0.125 | 1.00 |
| 2 | 5 | 3.80 | 0.701 | 0.720 | −0.020 | 0.337 | 1.00 |
| **3** | 9 | 4.70 | **0.735** | 0.706 | +0.029 | 0.398 | 1.00 |
| 4 | 13 | 4.36 | 0.729 | 0.718 | +0.011 | 0.358 | 1.00 |
| 5 | 17 | 5.76 | 0.712 | 0.716 | −0.005 | 0.342 | 1.00 |
| 6 | 21 | 6.02 | 0.733 | 0.720 | +0.013 | 0.311 | 0.91 |
| 8 | 29 | 6.83 | 0.722 | 0.718 | +0.003 | 0.314 | 0.83 |
| 10 | 37 | 7.28 | 0.717 | 0.715 | +0.002 | 0.342 | 0.80 |

Read alone, this says resolution buys nothing: 4× the parameters from m=3 to
m=10, 55% more capacity, and 2AFC goes *down*. I reported exactly that, and
**it was an artifact of the single weighting** (§2).

What does survive from this table: **UMAP-2D scores ~0.71 at every resolution**,
and φ at 37 free parameters barely touches it. Also **legibility decays with m**
(1.00 → 0.80) — the drawable region shrinks as resolution rises, which is the
third Pareto axis B0 found.

---

## 2 · The loss-weight frontier — the honest gap in B1, closed

B1 swept m at **one** loss weighting. M-PARETO is about the coupled
continuity/separation/round-trip/nested tradeoff, so that is one point on a
surface, not the surface. Sweeping it changed the picture materially:

| config | weights (rt, cont, sep, nest) | m=3 | m=10 | Δ |
|---|---|---|---|---|
| continuity_only | 0, 1, 0, 0 | 0.705 | 0.747 | **+0.041** |
| continuity_dominant | 0.2, 3, 0, 0 | 0.709 | 0.748 | **+0.038** |
| continuity_plus_sep | 0.2, 3, 0.5, 0 | 0.709 | 0.744 | **+0.035** |
| **continuity_all_four** | **0.5, 3, 0.5, 0.5** | 0.721 | **0.753** | **+0.032** |
| b1_baseline | 1, 1, 0.5, 0.5 | 0.734 | 0.716 | **−0.018** |

**Every configuration improves with resolution except the one B1 ran.** The
"flat frontier" was a property of a hyperparameter, not of the carrier. There is
also a real interaction: at m=3 the baseline is *best* (0.734) and
continuity_all_four is worst-but-one; at m=10 they swap completely.

### The best frozen φ

`continuity_all_four` at m=10, one frozen map, every falsifier read at that
single operating point:

| floor | value | bar | |
|---|---|---|---|
| F-METRIC absolute | 0.753 [0.740, 0.766] | ≥ 0.75 | ✅ |
| F-METRIC precision@5 | 0.413 | ≥ 0.40 | ✅ |
| CI disjoint from best null | 0.740 > 0.729 | required | ✅ |
| F-ROUNDTRIP top-1 | 1.000 | ≥ 0.90 | ✅ |
| F-COLLISION | 0.000024 | ≤ 0.01 | ✅ |
| legibility (simple contours) | 0.984 | ≥ 0.95 | ✅ |
| **F-METRIC margin over best null** | **0.038** | **≥ 0.05** | ❌ |

**Five of six. Failing by 0.012 on one bar.**

That bar was pinned on 2026-08-27, before φ existed, with this rationale written
at the time: *"Clearing 0.75 while merely tying PCA-2D is a FAIL: it would mean
the learned map bought nothing."* This is that situation, and the bar is doing
its job. It is not moved.

---

## 3 · What I got wrong, collected

Three published claims of mine were refuted by later measurement in this repo.
They are listed together because the pattern matters more than any one of them:
**every error was over-concluding from a single configuration.**

1. **"F-DEGRADE fails; A5 is heading for retraction."** Withdrawn. The criterion
   (`worst_flip == 0` at every truncation) is unachievable by arithmetic at k=1,
   where the code carries one parameter against the ~8 bits needed to name one of
   500 concepts — and it was stricter than the lock's own pinned floor. The
   per-k profile shows the nested loss *working*: φ lands at semantic radius
   0.175 at the coarsest truncation where the round-trip-only null lands at
   0.897, nearly orthogonal. A5 is about flip **distance** — vagueness versus
   error — and now measures it.
2. **"The frontier is flat; resolution buys nothing."** An artifact of the one
   weighting swept. Four of five configurations gain +0.032 to +0.041 from m=3
   to m=10.
3. **"continuity_only is the ceiling for F-METRIC."** Structurally wrong, and
   the most interesting of the three. Switching the *competing* objectives back
   on made F-METRIC **better** (0.747 → 0.753). F-METRIC is a forced-choice
   **discrimination** task: it needs the positive nearer than the distractor,
   which requires **separation** as much as continuity. The metric law and the
   identity law are not purely opposed here — this test consumes both. There is
   no cheap ceiling argument; the loss surface has to be sampled.

A fourth, methodological: I edited `maps.py` while a run was in flight, and
`code_fp` was stamped at write time rather than run start, so the result would
have carried a hash of code that never executed. Fixed at the root
(`code_fp_at_end` is now recorded alongside, making mid-run edits visible).

---

## 4 · The two things that are solid

**UMAP-2D is a formidable null, and installing it changed a verdict.** At m=4 φ
scores 0.729 and umap 0.718. With umap absent — as it was for all of B0 — the
best null is `roundtrip_only_phi` at 0.649 and φ shows a **0.08 margin instead
of 0.011**. A false positive produced purely by a missing baseline. M-NULL's
"a crippled null is a fake floor" stopped being a maxim and became a measurement.

**Round-trip and collision are not the problem.** At every resolution above m=2,
F-ROUNDTRIP is ~1.000 and F-COLLISION is ~0. The entire question is F-METRIC:
whether the rendered shapes preserve semantic neighbourhood structure well enough
to beat a 2-D squash by a margin worth the machinery.

---

## 5 · Open at the time of writing

- **B1-EXT-1** — pre-registered extension to m ∈ {12, 14, 16, 20}, config frozen
  to `continuity_all_four`, bars unchanged, plus a **legibility floor of 0.95**
  pinned in advance because the specific way this could win dishonestly is by
  buying margin with glyphs a hand cannot draw. Declared in `meander.lock`
  `preregistrations` and committed *alone, containing no data*, before the run.
- **The held-out check** — and this is the significant caveat on everything
  above. **Every number in this document is in-sample:** φ is trained on all 500
  words and F-METRIC is then measured on those same 500. v0.1 §7 asked for
  held-out concepts; this implementation did not do it. The bias is asymmetric —
  φ is a ~500k-parameter network, PCA and UMAP are rank-2 projections, so
  in-sample scoring flatters φ specifically. That is why it does **not** weaken
  the FAIL (φ lost while holding the advantage) but would make any pass unsafe.
  Declared as a rider on B1-EXT-1 while that run was still executing.

## 6 · Standing limits

- The perceptual metric is a **low-pass-L2 proxy**, not a human eye. Per the
  lock's `disabled_laws`, results license the machine-readable claim only.
- `noise.ink_sigma` / `scan_sigma` remain `nominal_unmeasured`.
- One architecture (512-hidden MLP), one lexicon draw, one split seed.
- N = 500 of 1877 available monosemous common nouns. A larger lexicon makes the
  test harder and would need its own pre-registration.
