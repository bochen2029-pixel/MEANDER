# MEANDER — specification

**An analog writing system where meaning is a path through an embedding manifold, a complete thought is a closed loop, and resolution is precision.**

Spec **v0.1** · 2026-08-10 · Bo Chen · MIT · status: **SKETCH → FULL DRAFT** (nothing built; this is the pre-build stake, per the estate discipline — claims dated, structured to lose honestly).

> *meander, n. — a winding path; also the ancient continuous-line ornament (Gk. Maíandros, a river). A MEANDER glyph is a single continuous stroke that winds through meaning and closes on a thought.*

---

## 0 · What MEANDER is (and is not)

MEANDER is a **constructed writing system** in which a glyph is not a symbol drawn from a finite alphabet but a **continuous, analog rendering of a trajectory through a semantic embedding space**. Concepts are points; relations are displacements; a proposition is a closed path that winds from concept to concept and returns to its head. Two meanings that are close in the embedding space render to glyphs that *look* close; a low-resolution rendering recovers the gist and loses the nuance, exactly as a slide rule loses significant figures — analog, lossy, graceful.

It is the first writing system designed to be **continuous** rather than combinatorial. Every human script — alphabetic, syllabic, logographic — is digital: a finite inventory of discrete symbols combined by rules. MEANDER has no symbol inventory. Meaning is position-and-path on a manifold, and the glyph is a chart of it. Continuity — not any particular shape — is the departure.

**It is:** a keyed, analog, compositional orthography over a pinned embedding space, with a learned meaning→mark map, a closed-loop proposition unit, self-validation by closure, and graceful degradation into vagueness.

**It is not:** (v0.1) a self-teaching or universal script decodable with no key (that is a later fork, §12); a lossless code (it is analog by design); a from-arbitrary-English encoder (v1 covers a controlled, growable lexicon); or a claim of exactness (it trades precision for continuity and honest failure).

**Lineage.** The razor (the embedding is *already computed* by the model — MEANDER renders a vector the forward pass produced, it does not invent meaning). BRAIN's economics law — *"estimate analog, commit digital; degrade resolution, never closure."* XENOSTELE's orientation beacon and self-validating content. The word2vec analogy geometry (`king − man + woman ≈ queen`) as the source of "a relation is a displacement." *Arrival*'s Heptapod B as the aesthetic ancestor (the closed ring = a complete thought, apprehended at once) — but MEANDER is generative and analog where Heptapod B was a finite hand-drawn dictionary.

---

## 1 · The one rule, and the axioms

**The one rule.** Everything derives from two invariants, and a design choice is legal only if it serves both:
1. **The metric law** — similar meanings must look similar.
2. **The closure law** — a complete thought must close.

### Axioms (constitution; stable IDs)

| # | Axiom | Statement |
|---|-------|-----------|
| **A1** | **MANIFOLD** | Meaning lives in a fixed, fingerprinted embedding space `E` (dim `d`). A glyph is a *rendering* of a point or path in `E`, never the truth. The representation is the truth; the mark is a view. |
| **A2** | **CONTINUITY** | The meaning→mark map φ is Lipschitz **locally and compositionally**: a small semantic change yields a small visual change. (Refined by M-LOCAL — this is *not* a claim of global 2-D isometry, which is impossible.) |
| **A3** | **DISPLACEMENT** | Legs encode *relations* — displacement vectors drawn from a learned inventory `R` — never absolute points. Composition is a path; a relation is a direction you travel, not a place you name. |
| **A4** | **CLOSURE** | A complete proposition is a *closed* path. Geometric closure is always drawn; **semantic closure** = a complete predication (a head with its required roles filled). A decode that does not close is incomplete or misread, and is reported as such. |
| **A5** | **RESOLUTION** | The glyph is a lossy analog projection. Drawing resolution = recovered dimensions = precision. Meaning **degrades into vagueness, never into error.** (the slide rule) |
| **A6** | **PARAM/RENDER SPLIT** | Meaning lives in *parameters* (invariant); calligraphic variation lives in *rendering* (cosmetic). Two drawings of one meaning decode identically. |
| **A7** | **ORIENTATION** | Every glyph carries a canonical pole/beacon so it is viewpoint-unambiguous. There is no "which way is up" question. |
| **A8** | **HONESTY** | An unreadable or damaged glyph decodes to a *neighborhood + confidence*, never a fabricated certainty. Vagueness is a first-class output; a wrong-but-confident decode is a defect. |

### Mechanism laws (M-*; cited by falsifiers forever)

- **M-LOCAL** — the metric law (A2) is guaranteed *locally* (small change → small change) and *compositionally* (one relation different → one leg different), **not** as a global isometry. Johnson–Lindenstrauss forbids embedding `E` (d≫2) into the plane without gross distortion; MEANDER never requires plane-distance to equal semantic-distance globally, and says so.
- **M-CANON** — one meaning → exactly one parameter spec. Canonicalization (leg ordering, head choice) is deterministic. The cosmetic RNG (render variation) is seeded independently and **never touches the parameters**.
- **M-CLOSE** — geometric closure is drawn always; semantic closure is checked always; a mismatch between them is the built-in checksum and is surfaced, never smoothed.
- **M-PIN** — `E`, φ, and `R` are fingerprinted artifacts. A glyph is only decodable against the same fingerprints. Version = (E-fp, φ-fp, R-fp); revisions and vintages, never silent drift. (STELE discipline.)
- **M-DEGRADE** — under damage or low resolution, the decoder loses *resolution* (meaning gets vaguer, lands in a wider neighborhood) and never loses *closure honesty* (it never emits a confident wrong meaning). Degradation in resolution is legal; degradation in closure kills the read.
- **M-NULL** — every learned artifact ships with the null it must beat: φ vs a random shape-map; the analog language vs a discrete-symbol baseline at matched channel capacity. No organ outlives its null.

---

## 2 · The space `E` (Layer 0)

MEANDER is defined *over* an embedding space, not tied to one. A version pins:

- **`E`** — a specific, fingerprinted embedding model. Reference choices:
  - **B1 (concepts/words):** a classical word-embedding space with clean analogy geometry — GloVe or fastText, `d = 300`. Fast, offline, analogy-friendly; ideal for testing the make-or-break metric bet cheaply.
  - **later (propositions):** a sentence-embedding space (a sentence-transformer, `d = 384–1024`) for whole-proposition points and for the closure check.
- **φ** — the **concept map**: a learned function `E → shape-params` (node signatures). Frozen per version.
- **`R`** — the **relation inventory**: a learned/curated set of relations, each = `{id, canonical displacement direction u_r ∈ E, drawn signature}`. Frozen per version.

All three carry a hash. `syncytium`-style: `meander.lock` records `{E-model-id, E-fp, phi-fp, R-fp, harmonics, schema-version, thresholds}` — the single source of every number; every unset threshold names the law it disables.

**Honest note on relations-as-displacement.** That relations are consistent displacement vectors is the word2vec analogy property — real but *approximate* (polysemy and context bend it). v1 curates a role inventory (AMR-style roles: `agent, patient, theme, time, location, manner, degree, negation, plural, tense, modality, …`) and assigns each a canonical direction estimated as the mean displacement over exemplar pairs, accepting the imperfection and measuring it (F-RELATION).

---

## 3 · The object model

A MEANDER glyph is defined at three layers. Meaning lives at Layer 2; Layers 1 and 3 are derivations of it.

### 3.1 Layer 1 — graph → path
A proposition is a **semantic graph**: a head (predicate) with typed edges (roles) to argument concepts, plus modifiers and global features (negation, mood, tense, modality, emphasis). Canonicalize the graph to an ordered, closure-symmetric **path**: start at the head anchor, walk out to each argument along its role-leg, and close the ring back to the head. Vertices are concept-points; legs are relation-displacements; global features are ring-level modulations applied to the whole loop at once.

### 3.2 Layer 2 — the parameter spec (the payload, the truth)
The invariant, machine-readable description of one glyph:

```
Glyph {
  version:      { E_fp, phi_fp, R_fp, schema }
  orientation:  beacon_angle                     # A7 — fixes rotation & handedness
  head:         concept_descriptor               # φ(e_head): the predicate anchor node
  legs: [                                         # ordered by M-CANON
    Leg {
      relation_id                                # index into R
      magnitude        # analog λ along u_r      # nuance
      node: concept_descriptor  # φ(e_target): the argument this leg lands on
      curvature, width          # analog nuance / emphasis
      turn_mark                 # vertex punctuation, segmentation aid
      crossing: [ {at, over|under} ]             # knot resolution (§4.3)
    }, ...
  ]
  global:       { negation, mood, tense, aspect, modality, emphasis }   # ring-level flags
  checksum:     closure_signature                # M-CLOSE
}
```

`concept_descriptor` = the truncated shape-parameters of a node (see §4.1). This layer is the "QR payload": rigid, exact, versioned.

### 3.3 Layer 3 — the render (the calligraphy)
Deterministic function `params → SVG`, plus an *independent* cosmetic RNG (M-CANON). The ring is a closed variable-width Bézier through the vertices; node signatures are stamped at vertices; leg signatures are rendered along edges; the orientation beacon marks the head; crossings drawn over/under; optional ink-splatter texture that may carry parity (honest-loss redundancy disguised as aesthetic). The Layer-2 spec is embedded in the SVG metadata for cheap decode. Two renders of one `params` differ cosmetically and decode identically (F-VARIANCE).

---

## 4 · The mark (how meaning becomes geometry)

### 4.1 Node signatures — the concept map φ
A node is a small **closed micro-contour** whose **elliptic Fourier descriptors (EFD)** are φ(e). Properties that make EFD the right carrier:
- A closed curve ↔ a set of harmonic coefficients; **low harmonics = gross shape (category), high harmonics = fine detail (nuance)**. Truncating high harmonics = lowering resolution = the slide rule (A5), directly.
- Continuity: nearby `e` → nearby coefficients → nearby contour (A2), *if* φ is trained for it.
- Recoverable: read the contour → its EFD → `φ⁻¹` (a learned inverse) → nearest concepts in `E`.

**φ is learned** (the make-or-break, §7 F-METRIC). Architecture: a small MLP `φ: E(d) → EFD(m)` and an inverse `φ⁻¹: EFD(m) → E(d)`, trained jointly with three losses:
1. **round-trip:** `φ⁻¹(φ(e)) ≈ e`.
2. **continuity/metric:** perceptual distance between contours correlates with cosine distance in `E` (a correlation or triplet loss).
3. **perceptual (optional, for human-legibility):** rasterize + LPIPS or human triplets, so the similarity a *person* feels tracks cosine. Without this, MEANDER is machine-readable only (stated fork).

`m` (number of retained coefficients × channels) is the capacity knob; `m < d`, so φ is lossy — the quantization. Resolution at read time = how many harmonics survive.

### 4.2 Leg signatures — the relation inventory `R`
Each relation `r` has a **drawn signature**: a characteristic open stroke (a curvature profile + a turn-mark) distinct enough to classify by shape, not by plane-angle (dodging the JL floor — high-D direction is recovered by *signature classification*, not by 2-D orientation). A leg in a glyph = the signature of `r`, scaled/shaded by analog `magnitude, curvature, width` (the nuance). Reading a leg = classify signature → `r`; measure geometry → magnitude.

### 4.3 The path, closure, and crossings
The glyph is one continuous closed stroke: `head → (leg₁) → node₁ → (leg₂) → node₂ → … → (legₙ) → head`. **Geometric closure** (the ring literally closes) is always drawn. **Semantic closure** = the decoded role-set is a complete predication (M-CLOSE); a dangling leg or missing required role = incomplete, reported (A8). Because a high-D path projected to 2-D self-intersects, **crossings are resolved knot-style**: an over/under mark at each crossing, or — the single sanctioned use of the third dimension — a local lift of one strand out of the plane at the crossing only (§ ties to the 2-D-language / 3-D-monument decision: 3-D appears *only* to disambiguate crossings and to render hero glyphs as held objects, never as the base script).

### 4.4 Global modulations
Whole-proposition features, applied to the ring at once (faithful to "the thought is apprehended simultaneously"): negation = ring stroke doubling; question = a hook/spur at the head (the one feature borrowed straight from Heptapod B); emphasis/urgency = ring weight; conditional = a break-and-rejoin; tense/aspect/modality = beacon-adjacent diacritics. Read once, globally.

---

## 5 · Encode / decode

### 5.1 Encode
1. Input: a controlled-English proposition, or a semantic graph directly.
2. **Parse** → semantic graph (v1: controlled grammar / AMR-subset).
3. **Map** concepts → `E` points (lexicon/embed); roles → `R` ids.
4. **Canonicalize** (M-CANON): choose head, order legs deterministically, compute closure.
5. **Emit** Layer-2 params + checksum.
6. **Render** → SVG (independent cosmetic seed).

### 5.2 Decode
- **Cheap tier (v1):** read Layer-2 params from SVG metadata / path geometry (you hold the source). Trivial and exact.
- **Hard tier (aspirational):** from a raster image — locate ring + beacon → fix frame; trace the closed stroke; segment at turn-marks/crossings into legs and nodes; classify leg signatures → relations; classify node contours → `φ⁻¹` → nearest concepts (with neighborhood + confidence); read global modulations; check closure.
- **Reconstruct** the graph → controlled-English. Per A8: emit per-element confidence; on closure failure or low confidence, **report a neighborhood / vagueness, never fabricate** a crisp wrong reading.

---

## 6 · The projection problem, stated honestly

`d ≫ 2`. You cannot make plane-distance equal semantic-distance globally (JL). MEANDER's resolution:
- **High-D content is carried by *identity + magnitude*, not by plane-position.** A leg's relation (a high-D direction) is recovered by **classifying its drawn signature**, and its magnitude by geometry. A node's concept is recovered by its **EFD signature**, not its `(x, y)`.
- Therefore "similar looks similar" holds **locally and compositionally** (A2, M-LOCAL): change one concept slightly → its contour shifts slightly; change one relation → one leg changes. It does **not** hold as a global map (two unrelated thoughts can look arbitrarily different) — which is the correct, honest version of the property.
- The cost is **capacity**: at resolution `m`, only so many concepts/relations stay distinguishable; beyond that, collisions. This is the analog approximation, measured and published (F-COLLISION), not hidden.

---

## 7 · Falsifiers (the contract; each can kill a piece; published either way)

| ID | Claim under test | Kill condition | Consequence |
|----|------------------|----------------|-------------|
| **F-METRIC** ★ | perceptual glyph-similarity correlates with cosine similarity on held-out concepts | correlation below bar (no better than the random-φ null) | A2 fails — it is a cipher, not a language. **Run first, on words, cheaply.** |
| **F-RELATION** | curated relations are consistent-enough displacements to classify and compose | relation directions do not separate; legs unclassifiable | drop learned displacements; fall back to a discrete role-tag per leg (less analog) |
| **F-ROUNDTRIP** | meaning → glyph → meaning recovers the proposition on the controlled set | recovery below bar | encode/decode or canonicalization is broken |
| **F-COMPOSE** | one relation changed → one leg changed (compositional continuity) | small semantic edit causes global glyph scramble | M-LOCAL violated; the path model is wrong |
| **F-CLOSURE** | well-formed propositions close; single-leg misreads caught by non-closure | catch-rate below bar | the free checksum does not work; add explicit parity |
| **F-DEGRADE** | as resolution drops, error grows into the right neighborhood, monotone, never flips to a wrong meaning | a low-res glyph decodes confidently wrong | A5/M-DEGRADE dead; the slide-rule claim fails |
| **F-VARIANCE** | N cosmetic renders of one meaning decode identically | any cosmetic render changes the decode | M-CANON violated; render is leaking into meaning |
| **F-CANON** | one meaning → one parameter spec, deterministically | same meaning yields two specs | canonicalization non-deterministic |
| **F-COLLISION** | the collision rate vs resolution is measured and reported | (not pass/fail) — an unstated ceiling is the failure | the honest capacity ceiling; must ship as a number |

Standing guards: **null-arm first** (a crippled null is a fake floor); **no bar set after a peek**; every threshold pinned in `meander.lock`.

---

## 8 · Honest limits (stated, per estate discipline)

- **Bounded capacity** per glyph → real collisions at low resolution. That *is* the analog approximation; the number ships (F-COLLISION).
- **Perceptual ≠ cosine** unless φ is trained against perception. Skip the perceptual loss and MEANDER is machine-readable only — a real and legitimate mode, but named.
- **Free English → graph is imperfect** (AMR parsing). v1 covers a controlled lexicon and proposition space, growable; it is not a mind-reader of arbitrary prose.
- **Keyed, not self-teaching.** v1 decodes only against pinned `{E, φ, R}`. A XENOSTELE-style self-bootstrapping mode (carry the primer, teach the decode) is a later fork (§12), not a v1 claim.
- **Analog by design.** Exactness is not the goal; continuity and graceful failure are. Nobody should expect a barcode.
- **The hard-tier decoder (image → meaning) is aspirational.** v1 ships the cheap tier (SVG metadata). Vision-from-a-photograph is a later rung.

---

## 9 · Build ladder (each rung earns the next; nothing ships at a stage)

| Rung | Scope | Acceptance |
|------|-------|------------|
| **B0** | pin `E`; define schema + `meander.lock`; build the falsifier harness, a perceptual-sim metric, and the random-φ / discrete-symbol nulls | harness runs; nulls measured |
| **B1** ★ | **nodes only.** Learn φ for single concepts (words, GloVe/fastText). | **F-METRIC** passes on words (brother/sister look alike) — the make-or-break, days of work |
| **B2** | **relations.** Learn/curate `R` leg-signatures; two-leg analogies render and decode | F-RELATION, F-COMPOSE pass |
| **B3** | **the path.** Closed proposition-glyphs over the controlled set | F-CLOSURE, F-ROUNDTRIP, F-CANON pass |
| **B4** | **the renderer.** Organic SVG calligraphy; knot/crossing handling | F-VARIANCE passes; crossings legible |
| **B5** | **degradation + honesty.** Confidence reporting; collision measurement | F-DEGRADE passes; F-COLLISION number published |
| **B6** | **discourse + monument.** Linked rings for arguments; 3-D held-object renderer for hero glyphs | linked-ring read; 3-D round-trips with a canonical pole |
| **Gate** | **F-WHOLE** | a held-out decoder recovers propositions at stated fidelity; similar-looks-similar confirmed; degradation graceful; every falsifier green or its number published |

**The cheapest real experiment is B1's F-METRIC**, and it needs no glyphs: take an off-the-shelf embedding space, a few hundred words, and test whether *any* learned continuous shape-map makes similar words look perceptually similar above the random-φ null. That single number decides whether MEANDER is alive.

---

## 10 · Glossary

**`E`** the pinned embedding space (meaning lives here). **concept** a point in `E`. **relation / leg** a displacement in `E`, drawn from inventory `R`. **path / ring** the closed glyph; a complete proposition. **head** the predicate anchor + orientation beacon. **φ** the learned concept→shape map; **`R`** the learned relation inventory. **node signature** a concept's EFD contour; **leg signature** a relation's drawn stroke. **magnitude** analog scalar along a relation. **closure** geometric (drawn) + semantic (complete predication) = the checksum. **global modulation** a ring-level feature (negation/mood/tense/emphasis). **parameter spec** Layer 2, the invariant truth. **render** Layer 3, the cosmetic calligraphy. **resolution** retained harmonics = recovered dimensions = precision (the slide rule).

---

## 11 · Provenance

Designed 2026-08-10 with Claude (Fable 5) at the operator's direction, out of a multi-turn brainstorm that walked from *Arrival*'s Heptapod B, through embedding geometry (word2vec displacements), the slide-rule/quantization intuition, the 2-D-vs-3-D decision, and the zigzag/multi-hop path idea that unified analog nuance (legs) with discrete grammar (turns) into one closed continuous stroke. Sibling doctrine in the estate: STELE/XENOSTELE (self-describing sovereign records), BRAIN ("estimate analog, commit digital"), the free-tail razor (render what the forward pass already computed). Staked before the build, dated, structured to lose honestly — the F-battery is the contract.

*End — MEANDER spec v0.1.*
