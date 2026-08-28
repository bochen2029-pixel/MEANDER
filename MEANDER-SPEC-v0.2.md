# MEANDER — specification

**A continuous analog lexicon over an explicit discrete role-grammar — a research probe into metric-preserving glyphs: can "similar meaning" be made to "look similar," and how much of a script can stay continuous before identity forces it digital?**

Spec **v0.2** · 2026-08-10 · Bo Chen · MIT · status: **PRE-BUILD STAKE, RED-TEAMED.** Supersedes v0.1 (kept on disk forever; the arc is append-only, never implemented against). v0.2 folds in a 5-lane adversarial QC swarm (see §12–§13); it **retracts two v0.1 claims outright** and puts numbers where v0.1 had slogans.

> *meander, n. — a winding path; also the ancient continuous-line ornament (Gk. Maíandros, a river). A MEANDER glyph is a continuous stroke that winds through meaning and closes on a thought — but the winding is only half the story: the grammar it rides is discrete, and v0.2 says so.*

---

## 0 · What MEANDER is (and is not)

MEANDER renders meaning as a mark in two registers at once, and the honest description keeps them separate:

- **The lexicon is continuous.** A concept is a point in a pinned embedding space, rendered by a learned map φ as a shape whose *perceptual* similarity is meant to track *semantic* similarity — brother and sister should look nearly alike, offset by the same small move that separates any near-neighbours. This is the novel, unproven bet.
- **The grammar is discrete.** Relations are indices into a finite inventory; roles bind by a fixed canonical order; global features (negation, mood, tense) are enum flags. This is a combinatorial symbol system, exactly like every other writing system's grammar.

So MEANDER is **a continuous analog lexicon over an explicit discrete role-grammar.** v0.1 called itself "the first writing system designed to be continuous rather than combinatorial" with "no symbol inventory." *That claim is retracted* (§12): the object model indexes a finite inventory and orders legs by a fixed rule, which is a discrete grammar by definition. What survives — and is genuinely worth building — is smaller and truer.

**The real question MEANDER poses.** Continuity buys "similar looks similar"; identity demands separation ("distinct looks distinct"). Those fight on one axis. Every human script resolved that fight by going **digital exactly where identity lives** (phonemes, graphemes, morphology) — the *duality of patterning*. MEANDER is the experiment that asks **how much of a script can stay analog before that tradeoff forces it discrete** — and its answer so far, on paper, is: *only the lexicon, and even there only within a measured bit budget.* That is a real scientific probe and an aesthetic object, not a universal communication technology.

**It is:** (a) a research instrument for whether a learned continuous shape-map preserves local semantic metric better than strong baselines; (b) an analog lexicon over a discrete role-grammar for a controlled, sense-disambiguated proposition space; (c) an aesthetic/monument object.

**It is not:** a self-teaching or universal script (it is keyed); a lossless code (analog by design); a mind-reader of arbitrary English (controlled, growable lexicon with a hard expressivity ceiling, §8); a machine-to-machine format (for machines the embedding already exists — the razor cuts backward: MEANDER would be a lossy, costlier re-encode of its own input, so its value is human-facing and scientific, not machine comms); and — until an anti-role-swap parity exists — it is **not** honest-by-construction (§ A8 retraction).

**Lineage.** The free-tail razor (the embedding is already computed; MEANDER renders it) — and its honest converse (for a machine reader that already holds the vector, rendering it is pure loss; MEANDER earns its keep only where a human eye or an aesthetic is the consumer). BRAIN's *"estimate analog, commit digital; degrade resolution, never closure."* XENOSTELE's orientation beacon + self-validation. word2vec displacement geometry (`king − man + woman ≈ queen`) — with the sharp caveat (§2) that it holds for *paradigmatic* relations and fails for *thematic* roles. *Arrival*'s Heptapod ring as aesthetic ancestor.

---

## 1 · The one rule, and the axioms

**The one rule.** Two invariants; a design choice is legal only if it serves both:
1. **The metric law** — similar meanings must look similar (continuity).
2. **The closure law** — a complete thought must be *structurally* checkable (closure detects incompleteness; it does **not**, by itself, detect substitution — see A8).

But note the tension the swarm surfaced and v0.2 elevates to first-class: **the metric law and identity are in opposition.** "Similar looks similar" and "distinct stays distinguishable" trade against each other on one dial. v0.2 therefore treats them as a **coupled frontier**, never as independent guarantees (M-PARETO, §7).

### Axioms (constitution; stable IDs)

| # | Axiom | Statement |
|---|-------|-----------|
| **A1** | **MANIFOLD** | Meaning lives in a fixed, fingerprinted embedding space `E` (dim `d`). A glyph renders a point/path in `E`; the representation is the truth, the mark a view. |
| **A2** | **BI-LIPSCHITZ** *(revised)* | φ must be Lipschitz **both ways**: an upper bound (small semantic change → small visual change, "similar looks similar") **and a lower bound** (distinct meanings stay perceptually separated ≥ δ at read resolution). v0.1 stated only the upper bound, which is near-vacuous — any smooth map has it, and its degenerate optimum is φ≈const (everything a blob). The lower bound (L_lo) is what makes it a language, and it is the expensive half. State (L_lo, L_hi); the ratio bounds achievable packing (feeds F-COLLISION). |
| **A3** | **DISPLACEMENT — TWO FAMILIES** *(revised)* | Relations split into **paradigmatic** (plural, tense, gender, comparative) — for which displacement geometry genuinely holds and legs may be analog — and **thematic/argument-structure** roles (agent, patient, theme) — for which it does **not** (fillers scatter across the whole noun subspace; mean displacement ≈ 0, variance enormous). Thematic roles are modeled as **discrete classes**, not displacements. v0.1 conflated the two families. |
| **A4** | **CLOSURE (funded)** *(revised)* | Geometric closure is always drawn (carries ~0 bits). **Semantic closure = a complete predication**, which requires a per-predicate **valency lexicon** (give=3 roles, sleep=1, rain=0), pinned in `meander.lock`. Closure detects *structural incompleteness* (missing/extra required role) — nothing more. It is not free and not geometric; it is a valency check. |
| **A5** | **RESOLUTION — NESTED OR RETRACTED** *(revised)* | The slide-rule claim ("truncate high harmonics → lose nuance, keep gist") is **false under the v0.1 losses** and is only recovered by an added **Matryoshka/nested loss** that binds coarse semantics to low harmonics *against* the round-trip loss's incentive (round-trip hides discriminative bits in high, low-SNR harmonics, which die first under noise → gist degrades before nuance). A5 holds **only if** the nested loss trains and F-DEGRADE passes; otherwise A5 is retracted. Not asserted as free. |
| **A6** | **PARAM/RENDER SPLIT** | Meaning lives in parameters (invariant); calligraphic variation lives in rendering (cosmetic, independently seeded). Two drawings of one meaning decode identically. |
| **A7** | **ORIENTATION** | Every glyph carries a canonical beacon (pole + handedness). Node contours are parameterized **in the beacon frame with a canonical start point**, killing the EFD phase/start-point degrees of freedom that otherwise make coefficient extraction discontinuous (a v0.1 A2 violation at normalization seams). |
| **A8** | **HONESTY — DOWNGRADED** *(revised)* | v0.1's "never a fabricated certainty" is **retracted.** A topology/leg-order/node↔leg slip at trace time is a **silent role-swap** ("dog bites man" → "man bites dog") that closes and decodes confident-wrong. Until an explicit **anti-role-swap parity** over concept-identity + leg-order exists (costing bits the budget can barely spare, §6), MEANDER **can** emit confident-wrong reads. A8 is restored only when that parity ships and F-CLOSURE catches substitution, not just omission. |

### Mechanism laws (M-*)

- **M-LOCAL** *(revised)* — the metric law is guaranteed **locally** (small change → small change). Global 2-D isometry is impossible (Johnson–Lindenstrauss). But local continuity **without** A2's lower bound is worthless (φ≈const passes it), so M-LOCAL is always cited **with** the separation floor.
- **M-PARETO** *(new)* — F-METRIC (continuity), F-COLLISION (separation), and F-DEGRADE (nested ordering) are **one coupled multi-objective problem on the resolution dial**, not independent tests. The spec ships a **Pareto frontier at a single frozen φ**, never a row of independent green checkboxes. There may be no simultaneously-passing operating point; finding that out cheaply is the point.
- **M-CLOSE** *(revised)* — closure is a valency check funded by the pinned valency lexicon; it detects structural incompleteness only. Substitution/role-swap detection requires explicit parity (A8).
- **M-PIN** — `E`, φ, `R`, the valency lexicon, the noise model, and every threshold are fingerprinted in `meander.lock`. Version = the tuple of fingerprints. A stake with no numbers is not a stake (§11 carries the numbers).
- **M-DEGRADE** *(revised)* — holds only if A5's nested loss trains. As written in v0.1 it was false (round-trip hides bits in the lowest-SNR subband). Degradation must be *shown* monotone, not asserted.
- **M-NULL** *(revised)* — every learned artifact ships with a null it must beat, and the null must be **strong, not crippled**: φ vs (i) a round-trip-only φ, (ii) PCA/UMAP→2D→render, (iii) a discrete code at **matched bit-capacity** (capacity defined operationally as bits recoverable per glyph at pinned read-noise σ — computable now, §6). Beating random-φ proves nothing and is banned as the null.

---

## 2 · The space `E` and the inventories (Layer 0)

A version pins: **`E`** (a fingerprinted embedding model); **φ** (learned concept→shape map); **`R`** (relation inventory, now two-family, §A3); the **valency lexicon** (funds M-CLOSE); and the **noise model** σ (ink + scan; required to define capacity and train φ⁻¹ on the right distribution).

Reference choices:
- **B1 (nodes):** word embeddings with clean paradigmatic geometry — GloVe or fastText, `d = 300` — **but sense-disambiguated**. Static embeddings sense-blend (bank/crane), which corrupts the F-METRIC test itself; v1 requires disambiguated lexicon entries (a sense-tagged subset, or a contextual model reduced to fixed sense vectors).
- **later (propositions):** a sentence-embedding space for whole-proposition points and the closure check.

**Relations, honestly (A3).** `R` = `R_para` (paradigmatic: analog displacement legs, small |R_para|) ∪ `R_theme` (thematic roles: discrete classes, drawn by distinct signatures, carrying ~0 bits of *displacement* geometry — their information is which discrete role, not a direction in E). §4.2's decode already classifies a discrete signature, so thematic `u_r` never did decode work; v0.2 stops pretending it did.

---

## 3 · The object model (three layers)

Meaning lives at Layer 2; Layers 1 and 3 derive from it.

### 3.1 Layer 1 — graph → path
A proposition is a semantic graph (head predicate + typed role edges + modifiers + global features). Canonicalize to a closed path. **Leg order is a fixed role-priority template** (agent < patient < theme < …), *not* content-dependent — because content-dependent ordering makes one concept swap permute all legs, violating F-COMPOSE. This fixed template **is the discrete grammar** (and it breaks on repeated roles / coordination "X and Y" with no tiebreak but content — a stated v1 limit, §8).

### 3.2 Layer 2 — the parameter spec (the payload)
```
Glyph {
  version:      { E_fp, phi_fp, R_fp, valency_fp, noise_fp, schema }
  orientation:  beacon                            # A7: pole + handedness + canonical start
  head:         concept_descriptor                # φ(e_head)
  legs: [                                          # ordered by FIXED role-priority (§3.1)
    Leg {
      role_id            # discrete class (thematic) OR paradigmatic-relation id
      magnitude          # analog, PARADIGMATIC legs only; ~2–3 bits meaningful (§6)
      node: concept_descriptor
      curvature, width, turn_mark
      crossing: [ {at, over|under} ]
    }, ...
  ]
  global:       { negation, mood, tense, aspect, modality, emphasis }   # enum flags, read once
  role_parity:  ECC over (node-identity + leg-order)   # REQUIRED for A8; costs bits (§6)
  checksum:     valency_closure                   # structural-incompleteness only (M-CLOSE)
}
```

### 3.3 Layer 3 — the render
Deterministic `params → SVG` + an independent cosmetic seed. Ring as closed variable-width Bézier; node contours stamped in the beacon frame; leg signatures along edges; crossings over/under; splatter cosmetic and/or carrying the role-parity ECC (honest-loss as texture). Layer-2 spec embedded in SVG metadata for cheap decode.

---

## 4 · The mark

### 4.1 Node signatures — φ, trained against all objectives at once
A node is a closed micro-contour whose **elliptic-Fourier descriptors** (in the beacon frame, canonical start point per A7) are φ(e). φ is a small MLP `E(d)→EFD(m)` with inverse φ⁻¹, trained jointly on **four** losses that must reach a single operating point:
1. **round-trip** — φ⁻¹(φ(e)) ≈ e.
2. **continuity** — perceptual contour distance correlates with cosine distance (A2 upper bound).
3. **separation** — min pairwise perceptual distance ≥ δ (A2 **lower** bound; the expensive half; *fights* continuity — this is the Pareto coupling, M-PARETO).
4. **nested/Matryoshka** — for every truncation level k, φ⁻¹(truncate_k(φ(e))) lands in a radius-r(k) neighbourhood with r monotone decreasing (funds A5; *fights* round-trip, which hides bits in high harmonics).

These objectives conflict (round-trip is Euclidean-reconstruction; continuity is cosine-alignment; separation opposes continuity; nested opposes round-trip). **They may have no common optimum.** v0.2's entire empirical claim is: find the frozen-φ operating point, or prove none exists — cheaply (§9).

### 4.2 Leg signatures — `R`, two families (A3)
Paradigmatic legs: an analog signature scaled by magnitude (real displacement geometry). Thematic legs: a **discrete** signature classifying the role (agent/patient/…), magnitude near-vestigial. Both recovered by signature classification, not plane-angle.

### 4.3 Path, closure, crossings, and the role-swap hazard
One continuous closed stroke. Geometric closure always drawn (~0 bits). **The load-bearing hazard:** tracing one stroke through self-crossings and threading over/under is a topology problem; a slip in **node↔leg assignment** silently swaps roles and still closes. This is why A8 is downgraded and why `role_parity` (an error-*detecting* code over node-identity + leg-order) is mandatory before any honesty guarantee is restored.

### 4.4 Global modulations
Whole-proposition enum flags (negation/mood/tense/aspect/modality/emphasis), read once. **Limit (stated):** "read once, globally" cannot represent *scope* — "not every" vs "every not," "must not" (□¬) vs "need not" (¬□), quantifier interaction. See §8 ceiling.

---

## 5 · Encode / decode
- **Encode:** proposition → semantic graph → canonical path (fixed role-priority) → Layer-2 params (+ valency closure + role-parity ECC) → SVG.
- **Decode, cheap tier (v1):** read params from SVG metadata. Exact.
- **Decode, hard tier:** *not* the classical trace→segment→classify pipeline (fragile at crossings). Instead a **learned regressor trained on unlimited synthetic (params→SVG→raster) pairs** — free perfect labels — with domain randomization for the hand-drawn gap and the role-parity ECC as verifier. Real risk is sim-to-real, not the pipeline. Aspirational for v1.
- **Honesty:** emit per-element confidence and a neighbourhood; on parity failure, report damage. **Absent role-parity, flag every decode as role-swap-unverified** (A8).

---

## 6 · The bit budget — the number v0.1 refused to compute

`d ≫ 2`, so meaning is carried by *identity + magnitude*, not plane-position; "similar looks similar" holds locally/compositionally, not as a global isometry. The cost is **capacity**, and it is derivable now, not deferred:

- **Human-legible glyph capacity ≈ 15 bits** — two independent derivations converge: perceptual (Miller ~7 discriminable levels × ~5 effective shape dims ⇒ ~14–20 bits) and physical (EFD coefficient SNR under ink+scan σ: ~3–4 bits on harmonic 1 decaying to <1 bit by ~h5–6 ⇒ SNR-weighted ~12–18 bits/node).
- **Payload demand:** one concept indexing a 10k-word lexicon ≈ 13 bits; a 3-leg proposition ≈ 75 bits; useful magnitude ≈ 2–3 bits/role max (and even that is discrete morphology — truly continuous magnitude is paralinguistic, near-vestigial for propositional meaning).
- **Verdict:** in human-legible mode a glyph carries **≈ 1 concept before collision — ~5× starved** for a 3-leg proposition. Role-parity (A8) needs bits this budget can barely spare, forcing a real tradeoff between honesty and payload.

This is not a footnote; it is the design's central constraint. It is pinned in `meander.lock` (§11) and it converts F-COLLISION from an un-failable "measure and report" into a hard numeric bar.

---

## 7 · Falsifiers — a coupled frontier with numbers, not a checklist

Per **M-PARETO**, the make-or-break falsifiers are **one experiment at a single frozen φ**, reported as a frontier. Every bar below is pinned in `meander.lock`; "below bar" with no number is banned.

- **F-METRIC** ★ *(rewritten)* — **claim:** a learned continuous φ preserves local semantic metric well enough to be read. **Test:** absolute, pre-registered **forced-choice discriminability floor** on confusable near-neighbour triples at fixed perceptual resolution — k-NN retrieval precision on rendered glyphs — **benchmarked against strong nulls** (round-trip-only φ; PCA/UMAP→2D→render; discrete code at matched bits). **Kill:** φ fails to beat the strong nulls at the pinned floor. (v0.1's "correlation beats random-φ" is retired — it returned a false "alive.")
- **F-COLLISION** *(now falsifiable)* — **claim:** distinct meanings stay distinguishable at working resolution. **Test:** measured collision rate vs the §6 bit-budget ceiling at pinned (m, N, σ). **Kill:** rate exceeds the pinned ceiling / a 3-leg proposition cannot clear a usefulness bar. (No longer "not pass/fail.")
- **F-DEGRADE** *(mechanized)* — **claim:** truncation loses nuance before category, monotone. **Test:** nested-loss validation — decode at each k lands in radius r(k), r monotone. **Kill:** any low-k decode flips category (confident-wrong), i.e. the nested loss failed to overcome the round-trip incentive.
- **F-ROUNDTRIP** — recovers the proposition on the controlled, sense-disambiguated set at the pinned rate, **at the same frozen φ as F-METRIC** (not deferred to a later rung).
- **F-COMPOSE** — one relation changed → one leg changed; verified under the **fixed** role-priority order (content-dependent order would auto-fail this).
- **F-CLOSURE** *(conditioned)* — catch-rate reported **by error type**: structural incompleteness (closure catches) vs substitution/role-swap (closure does **not** — only role-parity does). Kill if role-parity fails to catch swaps at the pinned rate.
- **F-REL-LEXICAL / F-REL-THEMATIC** *(split)* — paradigmatic relations classify/compose as displacements; thematic roles classify as discrete classes. Kill the *displacement* modeling of any role whose exemplar displacements have mean≈0 / variance≈noun-space.

**The Pareto report** replaces v0.1's nine independent rows: a single curve over resolution m showing (continuity ↑) vs (separation ↑) vs (round-trip ↑) vs (nested ↑), with the operating point (if any) that clears all pinned floors at once. **No operating point ⇒ the language claim dies honestly; a machine-readable analog lexicon may still survive at a stated capacity.**

---

## 8 · Honest limits (expanded)

- **Capacity starvation** (~1 concept/glyph legible; ~5× short of a 3-leg proposition) — the central constraint, §6, pinned.
- **Expressivity ceiling** *(new, hard):* v1 = semantic-role-labeled **simple clauses with atomic arguments.** NO recursion/embedding (nodes are points, not sub-rings — "John thinks [that Mary left]" is unreachable), NO quantifier/negation/modal **scope** (global flags read-once cannot compose), NO coordination with repeated roles. These are **not** reachable by growing the lexicon; they need a compositional mechanism MEANDER lacks. It is a phrasebook of simple predications, honestly bounded.
- **Keyed, not self-teaching.** Decodes only against pinned `{E, φ, R, valency, noise}`.
- **The razor cuts backward.** For a machine that already holds the embedding, MEANDER is a lossy costlier re-encode of its own input; it has value only where the consumer is a human eye or an aesthetic — not machine communication.
- **Perceptual ≠ machine-readable.** Without the perceptual/human-legibility validation, MEANDER is a machine-readable analog lexicon, a narrower (still legitimate) object; §0 is explicit about which mode a given claim lives in.
- **Honesty is conditional.** Until role-parity ships and F-CLOSURE catches substitution, MEANDER can decode confident-wrong (A8).

---

## 9 · Build ladder (rewritten around the frozen-φ frontier)

| Rung | Scope | Acceptance |
|------|-------|------------|
| **B0** | pin `E`, φ-arch, `R` (two families), valency lexicon, **noise model σ**, and the §11 bit-budget; build the falsifier harness + the **strong** nulls (round-trip-only φ, PCA/UMAP, discrete@matched-bits) | harness runs; strong nulls measured; bit-budget pinned in `meander.lock` |
| **B1** ★ *(rewritten)* | **nodes, one frozen φ, all objectives at once.** Train φ on round-trip + continuity + separation + nested; freeze; then require **F-METRIC ∧ F-ROUNDTRIP ∧ F-COLLISION to pass at that single φ / single m**, and ship the **Pareto frontier**. | the frontier has an operating point clearing all pinned floors vs strong nulls — **or it doesn't, and MEANDER-the-language is killed cheaply and honestly.** B1 is a **node-gate**, NOT "decides aliveness" (composition is B2/B3, where JL bites hardest). |
| **B2** | paradigmatic relations + thematic role classes; two-leg analogies | F-REL-LEXICAL, F-REL-THEMATIC, F-COMPOSE pass under fixed role-order |
| **B3** | closed proposition-glyphs + valency closure + role-parity | F-CLOSURE (by error type incl. role-swap) passes; A8 restorable |
| **B4** | organic SVG renderer; crossing/knot handling; F-VARIANCE | cosmetic variance decodes identically; crossings legible |
| **B5** | learned hard-tier decoder (synthetic-pairs regressor); sim-to-real | image→params at pinned fidelity; degradation graceful (F-DEGRADE) |
| **B6** | discourse (linked rings, honest as parataxis not embedding) + 3-D monument renderer | linked-ring read; 3-D round-trips with canonical pole |
| **Gate** | **F-WHOLE** | held-out decoder recovers propositions at pinned fidelity; every falsifier green or its number published; no claim exceeds §8. |

**The cheapest honest experiment is still B1 — but re-scoped:** it tests *nodes*, against *strong* nulls, at *one frozen φ*, with the bit-budget known in advance. It can kill or save the *lexicon* bet in days. It cannot, by itself, validate composition — that was v0.1's over-sell.

---

## 10 · Glossary
**`E`** pinned embedding space. **concept/node** a point in `E`, rendered by φ as an EFD contour. **paradigmatic relation** a real displacement (plural/tense) — analog leg. **thematic role** an argument slot (agent/patient) — discrete class, not a displacement. **path/ring** the closed glyph = a proposition. **head/beacon** predicate anchor + orientation + canonical start. **φ** learned concept→shape map (four conflicting losses). **valency lexicon** per-predicate required-role counts; funds closure. **closure** structural-incompleteness check (not substitution). **role-parity** ECC over identity+leg-order; prerequisite for A8. **bit budget** capacity in bits vs payload entropy (§6). **Pareto frontier** the coupled continuity/separation/round-trip/nested tradeoff at one frozen φ. **resolution m** retained harmonics.

---

## 11 · `meander.lock` — the pinned numbers (pre-registration)

> **Superseded by the real artifact.** The block below is v0.2's paper stake, kept
> verbatim for the arc. The authoritative lock is [`meander.lock.yaml`](meander.lock.yaml):
> it carries the pinned floors, an append-only `revisions:` log, and the numbers B0
> actually **measured** — which contradict the `budget:` priors here by ~3×. See
> [`FINDINGS-B0.md`](FINDINGS-B0.md). Where the two disagree, the measurement wins.

```yaml
version:   { E_model: TODO(GloVe|fastText d=300, sense-disambiguated), E_fp: TODO,
             phi_fp: TODO, R_fp: TODO, valency_fp: TODO, noise_fp: TODO, schema: v0.2 }
noise:     { ink_sigma: TODO_measure, scan_sigma: TODO_measure }   # SAFETY-class until set
budget:                                                            # §6, priors to be measured
  glyph_capacity_bits:      15        # prior; two-method convergence (Miller ~14-20; ink-SNR ~12-18)
  bits_per_node_lexicon:    13        # log2(10k)
  bits_per_proposition_3leg: 75       # prior estimate
  bits_per_magnitude_role:  3         # max meaningful; discrete morphology
  starvation_factor:        5         # payload / capacity, human-legible mode
  efd_bits_by_harmonic:     [4,3,2,1,1,"<1"]   # h1..h6 under nominal sigma (prior)
floors:                                                            # F-* kill bars (pin real numbers)
  f_metric_forcedchoice_min:   TODO   # absolute; must beat STRONG nulls, not random-phi
  f_collision_max_rate:        TODO   # at (m, N, sigma)
  f_roundtrip_min_recovery:    TODO   # SAME frozen phi as F-METRIC
  f_closure_structural_catch:  TODO
  f_closure_roleswap_catch:    TODO   # requires role-parity; A8 gate
  f_degrade_monotone:          required
nulls:     [ roundtrip_only_phi, pca_2d_render, umap_2d_render, discrete_code_matched_bits ]
retracted: [ "v0.1 §0 'first continuous writing system / no symbol inventory'",
             "v0.1 A8 'never a fabricated certainty'",
             "v0.1 A5 'resolution=precision' as free (now nested-loss-conditional)",
             "v0.1 F-METRIC 'correlation beats random-phi'" ]
```

---

## 12 · Changelog v0.1 → v0.2 (the red-team fold)

v0.2 folds a 5-lane adversarial QC swarm (§13). Unanimous verdict: **MAJOR-FIX** (core φ bet survives; overclaims struck). Changes:
1. **§0 identity retracted & reframed** — not a "continuous writing system"; it is an analog lexicon over a discrete role-grammar, and the honest research question (duality of patterning: how much can stay analog) is named.
2. **A8 downgraded** — "never confident-wrong" retracted; silent role-swap demonstrated; role-parity made a prerequisite.
3. **A2 → bi-Lipschitz** — added the separation lower bound (the expensive half; φ≈const was the degenerate hole).
4. **A3 split** — paradigmatic (displacement) vs thematic (discrete) relations; the "mean displacement" error fixed.
5. **A4/M-CLOSE funded** — valency lexicon pinned; closure = structural-incompleteness only.
6. **A5/F-DEGRADE mechanized** — nested/Matryoshka loss required (round-trip hides bits in low-SNR high harmonics → gist dies first); EFD canonical start-point (A7) kills the discontinuity.
7. **§6 bit budget added** — ~15 bits/glyph, ~1 concept/glyph, ~5× starvation, pinned. F-COLLISION now falsifiable.
8. **§7 restructured** — M-PARETO: one coupled frontier at a frozen φ, not nine checkboxes; F-METRIC rewritten (forced-choice floor vs strong nulls); F-RELATION split; strong-null discipline (M-NULL).
9. **§8 ceiling + §9 B1 rewrite** — hard expressivity ceiling (no recursion/scope); B1 = node-gate at one frozen φ with all objectives, bit-budget precomputed; demoted from "decides aliveness."

---

## 13 · Provenance

Designed 2026-08-10 with Claude (Fable 5) at the operator's direction. **v0.2 QC'd by a 5-lane Opus adversarial swarm** coordinating live over the Intercom message bus (room `proj-meander-6e84a002`, run `meander-qc-1`): lanes *skeptic, infotheory, linguist, mlbuild, devil*, three barrier rounds (open → cross-engage → verdict) with a synthesizer refereeing contradictions. Two independent methods (perceptual Miller-bound; physical ink-SNR) converged on the ~15-bit capacity number. Convergence noted honestly as partly shared-lineage (all Opus) — priced, not hidden. Sibling doctrine: STELE/XENOSTELE, BRAIN ("estimate analog, commit digital"), the free-tail razor. Staked before the build, dated, structured to lose honestly — now with the numbers the discipline demands.

*End — MEANDER spec v0.2.*
