# MEANDER

**A continuous analog lexicon over an explicit discrete role-grammar — a research probe into metric-preserving glyphs.**

Can "similar meaning" be made to "look similar," and *how much of a script can stay continuous before identity forces it digital?* A MEANDER glyph renders a concept as a shape learned so that near meanings render to near shapes; relations and grammar are a finite, discrete inventory. The winding line (a *meander*) is the analog lexicon; the grammar it rides is digital — and the spec says so, out loud.

> A relation is a direction you travel; a thought closes into a ring. But continuity buys "similar looks similar" only by fighting "distinct stays distinct" — the same tradeoff every writing system already resolved by going digital. MEANDER is the experiment that measures how much analog a script can afford.

- **The spec:** [`MEANDER-SPEC-v0.2.md`](MEANDER-SPEC-v0.2.md) — axioms, the object model, the learned map φ and its four conflicting losses, the bit budget, the coupled falsifier frontier, honest limits, the build ladder. ([`v0.1`](MEANDER-SPEC-v0.1.md) kept for the arc.)
- **Status: B0 built and run** ([`FINDINGS-B0.md`](FINDINGS-B0.md), [`meander.lock.yaml`](meander.lock.yaml)). The falsifier harness, the strong nulls and the capacity probe exist and execute; every kill bar was pinned before anything was trained. v0.2 remains the stake — red-teamed by a 5-lane adversarial swarm (verdict: MAJOR-FIX), retracting v0.1's two overclaims.
- **The one make-or-break** is B1's F-METRIC: can a learned continuous φ beat *strong* baselines (PCA/UMAP→2D + a discrete code at matched bits) at an absolute forced-choice discriminability floor? **B1 is blocked** — no embedding file and no WordNet on disk, and §2 is explicit that sense-blended vectors corrupt the test itself.
- **The central constraint just got worse.** §6 estimated ~15 bits per legible glyph on two converging priors. Measured through the real render → ink → scan → re-extract loop: **3.9 bits at m=2, 6.5 at m=8** — twenty-nine free parameters carry six and a half bits. Starvation is nearer 12–15× than 5×. Three structural reasons, all measured: harmonic 1 is a *frame* not a payload under A7; **even harmonics carry ~nothing** (arc-length reparameterisation has period π); and the drawable region shrinks as resolution rises.
- **The channel is not noise-limited — it is carrier-limited.** At nominal σ, ink and scanner cost **0.15 bits out of 6.02**. A better pen will not help; a different shape basis might.
- **A5 survives its first test, and the nested loss earns its keep.** At the coarsest truncation φ lands at semantic radius 0.175 where the round-trip-only null lands at 0.897 — nearly orthogonal, the exact failure A5 predicts. 5× better coarse error, 4× fewer category flips. (A first pass called F-DEGRADE a failure on a raw flip-rate criterion; that criterion was unachievable by arithmetic at low k and stricter than the pinned floor. A5 is about *flip distance* — vagueness versus error — and is now measured as such.)
- **Not:** a self-teaching or universal script, a lossless code, a machine-to-machine format (for machines the embedding already exists — the razor cuts backward), or honest-by-construction until an anti-role-swap parity ships.

Lineage: the free-tail razor (the embedding is already computed; MEANDER renders it), BRAIN's *"estimate analog, commit digital,"* XENOSTELE's orientation + self-validation, word2vec's displacement geometry (with the caveat it holds for paradigmatic relations, not thematic roles), and *Arrival*'s Heptapod ring — made honest.

MIT © 2026 Bo Chen.
