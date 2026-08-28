# B0 findings — what the harness measured, and what it is not allowed to claim

**2026-08-27 · rung B0 · schema `v0.2-b0` · against [MEANDER-SPEC-v0.2.md](MEANDER-SPEC-v0.2.md)**

B0's acceptance in [§9](MEANDER-SPEC-v0.2.md:180) is: *the harness runs, the strong
nulls are measured, the bit budget is pinned in `meander.lock`.* All three are done.
B1 is **not** done and is **blocked** — see §5.

---

## 0 · The line between the two kinds of number in here

There is no embedding file on this machine. The harness therefore ran against a
structured **synthetic** stand-in space, and every result derived from it is
stamped `SYNTHETIC_NOT_A_RESULT` in the JSON. But the two halves of B0 are not
equally affected, and the distinction is the most important thing on this page:

| | touches the lexicon? | status |
|---|---|---|
| **capacity, legibility, carrier behaviour** (§1–§3) | **no** — the probe samples shape parameters directly and renders them | **real results.** They stand whatever E turns out to be. |
| **F-METRIC, F-COLLISION, F-ROUNDTRIP, F-DEGRADE comparisons** (§4) | yes | **not results.** Indicative of harness behaviour only. |

Everything in §1–§3 is a genuine measurement of the mark, made before a single
real word was encoded. That was the point of doing B0 first.

---

## 1 · The bit budget is worse than §6's prior — by roughly a factor of three

[§6](MEANDER-SPEC-v0.2.md:137) puts a human-legible glyph at **~15 bits** on two
converging priors (Miller ~14–20; ink-SNR ~12–18) and calls the resulting ~5×
starvation "the design's central constraint." Measured through the actual
render → ink noise → scan noise → re-extract loop, at 128 px and nominal σ:

| m | free params | measured bits | bits by harmonic |
|---|---|---|---|
| 2 | 5 | **3.87** | 3.83, 0.04 |
| 4 | 13 | **4.36** | 2.15, 0.03, 2.09, 0.10 |
| 6 | 21 | **5.87** | 2.38, 0.07, 1.79, 0.04, 1.48, 0.12 |
| 8 | 29 | **6.49** | 2.43, 0.05, 1.84, 0.02, 1.33, 0.02, 0.78, 0.03 |

**Twenty-nine free parameters carry six and a half bits.** Against the ~13 bits
needed to index a 10k lexicon, starvation is not ~5× but closer to **12–15×**.
§6 was the honest core of v0.2 and it was still optimistic.

Estimator caveats, both directions: the extraction is from the filled silhouette,
which *excludes* the stroked-glyph medial-axis problem (B5) and so reads **high**;
the estimator is linear-Gaussian and cannot see nonlinear recoverability, so it
reads **low**. A learned decoder is the obvious way to test the second.

## 2 · The channel is not noise-limited. It is carrier-limited.

Sweeping σ at m = 6 against a fixed 6.02-bit noiseless ceiling:

| σ (× nominal) | bits | lost to noise |
|---|---|---|
| 0.25 | 5.95 | 0.07 |
| 1.00 | **5.87** | **0.15** |
| 2.00 | 5.42 | 0.60 |
| 4.00 | 3.41 | 2.61 |

At nominal σ the ink and the scanner cost **0.15 bits out of 6.02**. You must
quadruple the noise before they cost anything real.

This relocates the engineering problem. The bits are not being lost to the pen or
the paper — they are lost inside the representation, before any noise is added.
Buying a finer nib, a better scanner or more raster resolution will not move this
number. Changing the shape basis, or building a decoder that inverts the
reparameterisation, might.

## 3 · Three structural facts about the EFD carrier

**3.1 Harmonic 1 is a frame, not a payload.** Under the A7 canonical
normalisation a1 = 1 and b1 = c1 = 0, so harmonic 1 retains exactly one free
parameter (d1, eccentricity). *m* harmonics carry **1 + 4(m−1)** parameters, not
4m. §6's prior credits harmonic 1 with ~4 bits; it measures 2.2–2.4, from one
parameter.

**3.2 Even harmonics are nearly dead — consistently, at every m.** Odd harmonics
carry 1.3–2.4 bits; even harmonics carry 0.02–0.12. Arc-length reparameterisation
of a near-elliptical contour has period π, so it generates odd-harmonic content
and scrambles even-harmonic content. **Roughly half of the EFD parameters are not
a channel.** Any capacity estimate that counts parameters rather than measuring
them will be about 2× too generous.

**3.3 Analysis and synthesis do not agree, and cannot be made to.**
`extract(reconstruct(c)) ≠ c` by ~0.14, and the gap does **not** shrink with
denser sampling (identical at 1024 and 4096 points). A truncated EFD series
evaluated at uniform *t* is not arc-length parameterised, and arc length is the
only parameterisation a reader can observe — nothing in a drawn curve reveals *t*.

The obvious repair — iterate to the fixed point so the drawn curve measures back
as *c* — **was implemented and rejected.** It converges (residual 0.136 → 0.0017)
but the correction inflates amplitudes and pushes contours out of the drawable
region: the simple-contour rate collapses from ~96% to ~12%. **Descriptor
consistency and legibility are in direct conflict.** B0 resolves it by defining
the transmitted signal as what analysis returns, not what synthesis was handed.

## 3.4 · Legibility is a Pareto axis §6 never costed

Sampling uniformly in coefficient space produces **self-intersecting contours 71%
of the time**, and tightening the amplitude budget alone never gets below ~30%.
The cause is specific: with a1 pinned to 1, d1 is the ellipse's minor axis, so
d1 → 0 collapses the base curve to a segment traversed out and back, and *any*
harmonic perturbation then crosses it. Bounding d1 into [0.25, 0.90] fixes it:

| m | self-intersection rate |
|---|---|
| 2 | 0% |
| 4 | 4% |
| 6 | 7% |
| 8 | 16% |

The drawable region is a small, awkward subset of coefficient space **and it
shrinks as resolution rises.** So the resolution dial trades against *three*
things, not two: continuity, separation, and now legibility.

---

## 4 · Harness behaviour on the synthetic stand-in — NOT results

Reported because B0's acceptance is "the nulls are measured" and because the
harness's own failures are the point of this rung. **No number below may be
quoted as a MEANDER result.** m = 4, N = 300, 3000 epochs.

| map | 2AFC (95% CI) | P@5 | collision | round-trip |
|---|---|---|---|---|
| `phi_learned` (candidate) | **0.921** [0.910, 0.932] | 0.729 | 0.0002 | 1.000 |
| `roundtrip_only_phi` | 0.873 [0.859, 0.886] | 0.667 | 0.0000 | 1.000 |
| `pca_2d_render` | 0.744 [0.726, 0.761] | 0.385 | 0.0257 | n/a |
| `discrete_code_matched_bits` | 0.620 [0.600, 0.639] | 0.322 | 0.0512 | 0.067 |
| `discrete_semantic_ordered` | 0.619 [0.599, 0.637] | 0.291 | 0.0562 | 0.067 |
| `umap_2d_render` | **skipped** — umap-learn absent, recorded not dropped | | | |

**φ clears the absolute floor (0.75) and misses the margin floor by 0.002.**
Margin over the best null = 0.0479 against a pinned 0.05, with disjoint CIs. The
verdict object says `clears_margin: false`. Pre-registration doing exactly its
job: on synthetic data, where it costs nothing, the bar held.

**F-DEGRADE: the nested loss works, and my first criterion was wrong.**

A first pass reported "F-DEGRADE fails, 95–100% category flips" and concluded
[A5](MEANDER-SPEC-v0.2.md:46) was heading for retraction. **That was a defect in
the test, not in the axiom.** The per-k profile:

| map | radii k=1→4 | flips k=1→4 |
|---|---|---|
| `phi_learned` (nested loss on) | 0.175 → 0.042 → 0.023 → **0.009** | 0.953 → 0.603 → 0.133 → 0.0 |
| `roundtrip_only_phi` | 0.897 → 0.719 → 0.442 → 0.0 | 0.997 → 0.940 → 0.523 → 0.0 |

At k=1 the code carries **one** free parameter against the ~8.2 bits needed to
name one of 300 concepts. A high flip rate there is arithmetic, not a design
defect, so `worst_flip == 0` was an unachievable bar — and one *stricter than the
pinned floor*, which asks only for monotonicity. Reading its failure as the
axiom's was the error.

What the profile actually shows is **the nested loss doing precisely its job.**
At k=1, φ lands at radius 0.175 while the round-trip-only null lands at 0.897 —
nearly orthogonal to the truth. That null is the exact failure A5 predicts
(round-trip hides discriminative bits in high harmonics, so gist dies first), and
the Matryoshka objective suppresses it: **5× better coarse-truncation error, 4×
fewer flips at k=3.** Radii and flips are monotone for every map.

A5's real claim is about flip **distance**, not flip rate — "degrades into
vagueness, never into error." Landing on a near neighbour with a wide radius is
vagueness and is legal; landing far away is a confident-wrong read and is fatal.
`f_degrade` now measures the mean **rank of the true concept** under each
truncation and requires it to stay inside the top 5% of the lexicon, monotonically.
That is the criterion the spec's words describe, and it separates φ from the
round-trip-only null where flip rate could not.

**The discrete nulls were capacity-bound before they were metric-bound.** At the
measured 4.36 bits they get **20 codes for 300 words**, so ~15 words share a mark
and the near-neighbour discrimination collapses to coin-flipping between
collisions. `discrete_semantic_ordered` — the sorted-syllabary adversary — scored
identically to the arbitrary codebook (0.619 vs 0.620), meaning **its ordering
was never actually tested.** To test it you would have to hand it more bits than
the glyph can carry, which breaks the matched-bits discipline. Unresolved, and
flagged rather than buried.

That has a sharp corollary. **A 4.4-bit glyph is too small a channel for anybody
— MEANDER or Chinese.** Functional Han literacy is ~3,500 characters ≈ 11.8 bits,
and that script demonstrably works. If the measured capacity of a MEANDER node is
4–6 bits, the mark is roughly *half a Han character* and no amount of clever
mapping fixes that. Raising the carrier's capacity is the precondition for
everything else.

---

## 5 · B1 is no longer blocked — it ran. See `FINDINGS-B1.md`

At the time of writing this page there was no embedding file and no WordNet on
the machine, so B1 could not run. **Both were installed the same day and B1 ran
on real E**; its results, the resolution frontier, the loss-weight frontier and
the verdict live in [`FINDINGS-B1.md`](FINDINGS-B1.md). Everything in §1–§4 above
is unaffected: the capacity, legibility and carrier measurements never touch a
lexicon.

Also still unmeasured: `noise.ink_sigma` and `noise.scan_sigma` remain
`nominal_unmeasured` (measuring them needs a real pen and a real scanner), and
the perceptual metric is a low-pass-L2 **proxy**, not a human eye. Per the lock's
`disabled_laws`, a pass measured with that proxy licenses the
**machine-readable** claim only ([§8](MEANDER-SPEC-v0.2.md:171)) — never the
human-legibility claim.

## 6 · What the harness caught in itself

Both logged in `meander.lock` `revisions` with `peeked: synthetic_only`:

1. **The forced-choice test was vacuous.** Distractors drawn from ranks 50–150
   gave every map 0.98–1.00, nulls included. §7 asks for *confusable
   near-neighbour* triples; rank 50 of 300 is not one. Corrected to ranks 4–12.
   The old band is retained as `coarse_acc_diagnostic_only` — it still reads
   1.000 for nearly every map, so the vacuity stays visible in every report
   instead of being deleted from the record.
2. **A false kill was nearly recorded.** At 400 epochs F-ROUNDTRIP read 0.25–0.34
   and looked like structural capacity starvation. It was undertraining — the
   same network reaches 0.983 at 1500 epochs and 1.000 at 4000.

Two estimator errors were also caught and are documented in `capacity.py`:
variance-ratio SNR reported harmonic 3 carrying **17 bits**, more than harmonics
1 and 2 combined; marginal correlation still credited h3 with ~1.9 bits against
h2's ~0.1, because it was re-measuring eccentricity and being paid twice. Only
regression on the full received vector removes the leakage.

---

## 7 · Next

1. ~~Unblock E~~ — **done.** GloVe 6B 300d + WordNet installed; 500 monosemous
   common nouns. See [`FINDINGS-B1.md`](FINDINGS-B1.md).
2. ~~Re-run on real E, then B1 with the frontier~~ — **done.** Verdict there.
3. ~~Confirm or retract A5~~ — **A5 survived**; the retraction claim was withdrawn
   (§4 above). The nested loss demonstrably beats the round-trip-only null.
4. **Attack the carrier, not the ink** — still open, and now the main event. The
   capacity ceiling is structural: dead even harmonics, a legibility region that
   shrinks with m, and an analysis/synthesis mismatch that cannot be repaired
   without breaking legibility. **This is where the remaining headroom is.** A
   different shape basis should be costed against the same lexicon-free capacity
   probe before any further φ tuning — that probe needs no embeddings and runs
   in seconds.
5. **Measure σ for real** — a pen, a scanner, and one afternoon would move
   `noise` off `nominal_unmeasured` and re-enable F-COLLISION as a real bar.
   Note B0's finding first, though: at nominal σ the channel is not noise-limited,
   so this is about honesty rather than about recovering capacity.

*Reproduce:*

```bash
python run_b0.py --m 4 --n 300
```
