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

## 5b · B1-EXT-1 — RAN. `NO_RESCUE`, and the trend hypothesis is refuted

Pre-registered extension to m ∈ {12, 14, 16, 20}, config frozen to
`continuity_all_four`, bars unchanged, plus a **legibility floor of 0.95** pinned
in advance. Declared in `meander.lock` and committed *alone, containing no data*,
before the run — the commit ordering is the receipt.

| m | 2AFC | best null | margin | P@5 | simple contours | abs | marg | leg |
|---|---|---|---|---|---|---|---|---|
| 12 | 0.743 | 0.728 | +0.015 | 0.382 | 0.922 | · | · | · |
| **14** | **0.767** | 0.719 | **+0.048** | 0.409 | 0.938 | ✅ | · | · |
| 16 | 0.758 | 0.725 | +0.033 | 0.403 | 0.812 | ✅ | · | · |
| 20 | 0.750 | 0.728 | +0.022 | 0.414 | 0.984 | · | · | ✅ |

**The hypothesis that motivated the extension is refuted.** The margin was
supposed to be climbing toward 0.05. Across every resolution now tested — m = 3,
10, 12, 14, 16, 20 — it reads **+0.015, +0.038, +0.015, +0.048, +0.033, +0.022**.
It does not trend. It plateaus in a 0.015–0.048 band and never reaches the bar.

**The legibility bar was binding, and that is the point of having pinned it.**
Simple-contour rates at m=12/14/16 are 0.922 / 0.938 / 0.812 — all under 0.95.
**The resolutions where φ scores best are the ones where the glyphs are least
drawable.** That is precisely the "buy margin with undrawable glyphs" failure the
bar was written to catch; without it, m=14 would have been reported as a
one-bar miss rather than a two-bar miss.

Two caveats recorded rather than buried:

- **The high-m capacity numbers are probably inflated and should not be quoted.**
  15.89 bits at m=16 and 15.00 at m=20, against 7.28 at m=10. The estimator fits
  1+4(m−1) parameters — 61 at m=16, 77 at m=20 — against ~400 samples, so
  residual variance is under-estimated. The non-monotonicity (m=16 > m=20) is
  estimator noise, not physics.
- **This run's artifact cannot attest to its own provenance.** Its `code_fp` was
  stamped at *write* time by the pre-patch runner, after `maps.py` had been
  edited mid-run — the exact failure that prompted the fingerprint-at-start fix.
  The numbers are sound (the edit was in-sample-neutral), but the record is
  weaker than it should be. Reproduced under `B1-EXT-1-REPRO` with correct
  fingerprinting; that is a reproduction of the same seeded config, not a second
  trial.

## 5c · The generalisation diagnostic — settles the confound

`diag_split.py` scores a 500-fitted and a 400-fitted φ on the **same** 100 test
words and the **same** triples, so difficulty cancels and the residual is
generalisation alone. It cannot produce a pass — it deliberately runs the
contaminated arm — and no bar moved.

| map | fitted on 500 | fitted on 400 | gap |
|---|---|---|---|
| **φ continuity_all_four** | 0.744 | **0.706** | **+0.037** |
| umap_2d_render | 0.679 | 0.621 | +0.057 |
| roundtrip_only_phi | 0.656 | 0.598 | +0.059 |
| pca_2d_render | 0.600 | 0.614 | −0.014 |

Three readings:

1. **The difficulty confound was small.** φ scores 0.753 on the 500-word task and
   0.744 on the 100-word task with identical training — only ~0.009 of the earlier
   drop was "harder test". The real generalisation loss is **0.037**.
2. **φ generalises better than its strongest null** (0.037 against umap's 0.057),
   confirming the held-out result under controlled difficulty.
3. **Even the contaminated arm reaches only 0.744.** φ does not clear 0.75 on this
   task under *any* condition tested — trained on the test words included.
## 5a · The held-out check — RUN, and it inverts the failure mode

Every number in §1–§2 is **in-sample**: φ trained on all 500 words and F-METRIC
measured on those same 500. v0.1 §7 asked for held-out concepts and this
implementation did not do it. Declared as a rider on B1-EXT-1 *while that run was
still executing*, then run: fit φ and every null on 400 words, score F-METRIC on
the 100 unseen, bars unchanged.

| | in-sample (500) | held-out (100 unseen) |
|---|---|---|
| φ 2AFC | 0.753 | **0.706** [0.674, 0.739] |
| best null (umap-2D) | 0.715 | **0.621** |
| **margin** | +0.038 ❌ | **+0.085 ✅** |
| precision@5 | 0.413 ✅ | 0.356 ❌ |
| absolute floor 0.75 | ✅ | ❌ |
| band gap (task difficulty) | 25.0% | 16.9% |

**The margin bar — the one B1 failed — clears on unseen words**, +0.085 against a
required 0.05. Verdict still **FAIL** (3 of 6), but on different bars.

**The predicted bias ran backwards.** The rider argued in-sample scoring would
flatter φ, a ~500k-parameter network, relative to rank-2 projections. Measured,
the nulls fell *further*: umap −0.094, pca −0.104, against φ's −0.047. φ
generalises **better** than the 2-D squashes. UMAP is transductive — its
`transform()` on unseen points is a genuinely weaker object than its fitted
embedding — and it lost ground even given the fair version rather than a
nearest-neighbour stand-in.

**One confound blocks a clean reading, and it is not small.** Triples drawn from
100 words sit closer together than triples from 500: band gap 16.9% against
25.0%. The held-out task is simply harder for everyone, so φ's absolute drop
mixes memorisation with a harder test and **these numbers cannot separate them.**
`diag_split.py` isolates it by scoring a 500-fitted and a 400-fitted φ on the
*same* 100 test words and the *same* triples, holding difficulty fixed by
construction. It cannot produce a pass — it deliberately runs the contaminated
arm the protocol forbids quoting — and it leaves every bar untouched.

## 6 · What B1 actually establishes

Six independent tests now agree, and they agree on something more specific than
"it failed":

**φ works. It just doesn't work *enough*.**

The learned map is not a null result. It beats every strong baseline the spec
names — round-trip-only, PCA-2D, UMAP-2D, and both discrete codebooks — at every
resolution above m=2. The advantage **survives held-out testing** (+0.085 on
unseen words, larger than in-sample) and **survives difficulty control** (0.037
generalisation gap against umap's 0.057). It round-trips at 1.000 and collides at
~0. Something real is being learned, and it generalises better than a 2-D squash.

But its **absolute discriminability sits at 0.70–0.77 in every condition tested**
— in-sample, held out, difficulty-controlled, across ten resolutions and five
loss weightings — and the pinned floor is 0.75. MEANDER straddles its own bar. It
is not refuted as *"φ does nothing"*; it is refuted as *"φ does enough to read"*.

Three structural facts explain why, and all three were measured before any of
this was known:

1. **The carrier is small.** ~4–7 bits per glyph against §6's ~15 prior, and the
   channel is not noise-limited — at nominal σ, ink and scanner cost 0.15 bits
   out of 6.02. The bits are lost inside the representation.
2. **Legibility fights resolution.** The drawable region shrinks as m rises, and
   the resolutions where φ scores best are the ones where the glyphs are least
   drawable. This is a third Pareto axis §6 never costed.
3. **A 2-D projection is nearly as good.** UMAP-2D scores ~0.62–0.73 everywhere.
   φ with 37–77 free parameters beats it, but only by 0.02–0.09. Most of what a
   perceptual metric can read off a closed contour, two dimensions already carry.

The spec anticipated this outcome and wrote the disposal in advance
([§7](MEANDER-SPEC-v0.2.md:161)): *"No operating point ⇒ the language claim dies
honestly; a machine-readable analog lexicon may still survive at a stated
capacity."* That is exactly where B1 lands. The survival path is real — φ
round-trips, separates, and generalises — but it is thin, because
[§8](MEANDER-SPEC-v0.2.md:170)'s razor cuts backward on it: a machine that
already holds the embedding gains nothing from a lossier re-encode of it.

## 7 · What v0.3 should record

- **A2 (bi-Lipschitz) — half validated.** The *separation* lower bound is
  comfortably achievable (collision ~0 at every m ≥ 3). The *continuity* upper
  bound is where it fails. v0.2 called separation "the expensive half"; measured,
  it was the cheap half.
- **A5 (resolution) — survives.** The nested loss demonstrably beats the
  round-trip-only null on coarse-truncation error (0.175 vs 0.897). The earlier
  retraction was withdrawn.
- **§6 bit budget — revise down**, with the caveat that high-m estimates are
  estimator-strained and need re-measuring at n ≫ p.
- **New constraint, unnamed in v0.2: LEGIBILITY.** The drawable region is a small
  subset of coefficient space that *shrinks with resolution*. It belongs in the
  axioms, not in a footnote — it bound the outcome here.
- **F-METRIC — the bar was right and MEANDER failed it.** No case for moving it.
  The margin requirement in particular did the work it was pinned for: without
  it, the in-sample run would have read as a pass.
- **M-NULL — vindicated hardest of all.** Installing one absent baseline
  (umap-learn, thirty seconds) took φ's apparent margin from 0.08 to 0.011. Every
  null the spec names must be present before any margin is believed.

## 8 · Standing limits

- The perceptual metric is a **low-pass-L2 proxy**, not a human eye. Per the
  lock's `disabled_laws`, results license the machine-readable claim only.
- `noise.ink_sigma` / `scan_sigma` remain `nominal_unmeasured`.
- One architecture (512-hidden MLP), one lexicon draw, one split seed.
- N = 500 of 1877 available monosemous common nouns. A larger lexicon makes the
  test harder and would need its own pre-registration.
