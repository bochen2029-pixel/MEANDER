"""The space E (spec §2, Layer 0) and its sense-disambiguation problem.

v0.2 §2 is blunt about the trap: static embeddings SENSE-BLEND (bank/crane), and
a blended vector corrupts the F-METRIC test itself — the map would be asked to
place a point that means two things near both, which no continuous map can do.
So the v1 lexicon must be sense-disambiguated before a single glyph is drawn.

Strategy pinned in meander.lock: MONOSEMOUS FILTER. Keep only words with exactly
one WordNet synset; a static vector cannot blend what has only one sense. It is
stricter and far cheaper than sense-tagging a polysemous corpus. The cost is a
smaller and slightly peculiar lexicon (monosemous words skew concrete and
technical), which is an honest v1 limit and is reported, not hidden.

Nothing real is on disk yet. `load()` reports exactly what is missing instead of
quietly falling back, and the synthetic space carries `synthetic=True` all the
way through to the results file so no number produced from it can be quoted as
a MEANDER result.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Lexicon", "load_synthetic", "load_glove", "load"]


@dataclass
class Lexicon:
    words: list[str]
    vectors: np.ndarray            # (N, d), L2-normalised
    source: str
    synthetic: bool
    disambiguation: str
    notes: list[str] = field(default_factory=list)
    groups: np.ndarray | None = None   # ground-truth cluster ids, synthetic only

    def __len__(self):
        return len(self.words)

    @property
    def dim(self):
        return self.vectors.shape[1]


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n < 1e-12, 1e-12, n)


# --------------------------------------------------------------- synthetic

def load_synthetic(n_groups=20, n_sub=5, n_items=5, dim=300, seed=1729,
                   r_sub=0.55, r_item=0.28) -> Lexicon:
    """A hierarchical stand-in with GRADED similarity — the property under test.

    Three levels (field / subfield / item) so near-neighbour triples exist at two
    difficulty grades: same-subgroup (hard) and same-group-different-subgroup
    (medium). A flat cluster space would make F-METRIC trivially easy and the
    harness would report a comfortable lie.

    HARNESS-ONLY. This is not E. It proves the pipeline runs and lets the nulls
    be measured with zero network; it cannot answer whether MEANDER is alive.
    """
    rng = np.random.default_rng(seed)
    words, vecs, groups = [], [], []
    for g in range(n_groups):
        c_g = rng.normal(size=dim)
        c_g /= np.linalg.norm(c_g)
        for s in range(n_sub):
            off = rng.normal(size=dim)
            off -= off @ c_g * c_g
            off /= np.linalg.norm(off)
            c_s = c_g + r_sub * off
            c_s /= np.linalg.norm(c_s)
            for i in range(n_items):
                off2 = rng.normal(size=dim)
                off2 -= off2 @ c_s * c_s
                off2 /= np.linalg.norm(off2)
                v = c_s + r_item * off2
                vecs.append(v / np.linalg.norm(v))
                words.append(f"g{g:02d}s{s}i{i}")
                groups.append(g * n_sub + s)
    return Lexicon(
        words=words,
        vectors=_l2(np.array(vecs, dtype=np.float32)),
        source="synthetic-hierarchical-v1",
        synthetic=True,
        disambiguation="n/a (synthetic points are monosemous by construction)",
        notes=["HARNESS-ONLY: not E. No number from this space is a MEANDER result."],
        groups=np.array(groups),
    )


# --------------------------------------------------------------- real

def _monosemous_filter(words: list[str]) -> tuple[list[str], str]:
    """Keep monosemous COMMON NOUNS. Returns (kept, note).

    Three conditions, and the second and third are not fussiness:

      exactly one synset   a static vector cannot sense-blend what has one sense
      part of speech = n   'within', 'sometimes', 'mainly', 'non' are monosemous
                           and are not concepts; their geometry encodes syntactic
                           distribution, and asking phi to render them as shapes
                           tests a paradigmatic structure they do not have
      not an instance      'italy', 'michael', 'malaysia', 'nigeria' each have
                           exactly one synset and would sail through a naive
                           filter. A first pass produced a 500-word lexicon
                           roughly 40% proper nouns, which would have measured
                           entity co-occurrence clustering and reported it as
                           semantic metric preservation. WordNet marks these with
                           instance_hypernyms (Italy IS-AN-INSTANCE-OF country,
                           as against dog IS-A mammal), so they are cheap to
                           exclude.
    """
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test")
    except Exception as exc:                                   # noqa: BLE001
        return words, (f"MONOSEMOUS FILTER NOT APPLIED ({type(exc).__name__}). "
                       "WordNet unavailable - the lexicon is sense-BLENDED and "
                       "F-METRIC run against it is invalid per v0.2 §2.")
    kept, n_poly, n_pos, n_inst = [], 0, 0, 0
    for w in words:
        syns = wn.synsets(w)
        if len(syns) != 1:
            n_poly += 1
            continue
        s = syns[0]
        if s.pos() != "n":
            n_pos += 1
            continue
        if s.instance_hypernyms():
            n_inst += 1
            continue
        kept.append(w)
    return kept, (f"monosemous_common_noun_wordnet: {len(kept)}/{len(words)} kept "
                  f"(dropped {n_poly} polysemous, {n_pos} non-noun, "
                  f"{n_inst} proper-noun instances)")


def load_glove(path: str, size_target: int = 500, min_rank: int = 200,
               max_rank: int = 20000, seed: int = 7) -> Lexicon:
    """Load a GloVe/fastText text-format vector file and disambiguate it.

    Skips the first `min_rank` entries: the very top of the frequency list is
    function words ("the", "of", "and"), whose embeddings encode syntax rather
    than concepts and would be tested for a semantic geometry they do not have.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    words, vecs = [], []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= max_rank:
                break
            if i < min_rank:
                continue
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue                          # header line of a .vec file
            w = parts[0]
            if not w.isalpha() or len(w) < 3:
                continue
            words.append(w)
            vecs.append(np.asarray(parts[1:], dtype=np.float32))

    kept, note = _monosemous_filter(words)
    keep = set(kept)
    idx = [i for i, w in enumerate(words) if w in keep]
    rng = np.random.default_rng(seed)
    if len(idx) > size_target:
        idx = sorted(rng.choice(idx, size=size_target, replace=False).tolist())

    return Lexicon(
        words=[words[i] for i in idx],
        vectors=_l2(np.array([vecs[i] for i in idx], dtype=np.float32)),
        source=os.path.basename(path),
        synthetic=False,
        disambiguation="monosemous_wordnet",
        notes=[note, f"rank window [{min_rank}, {max_rank})"],
    )


def load(lock: dict, data_dir: str = "data") -> Lexicon:
    """Real E if it is on disk, else the synthetic stand-in — never silently."""
    candidates = [
        os.path.join(data_dir, "glove.6B.300d.txt"),
        os.path.join(data_dir, "crawl-300d-2M.vec"),
        os.path.join(data_dir, "cc.en.300.vec"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_glove(path, size_target=lock["lexicon"]["size_target"])
    n = lock["lexicon"]["size_target"]
    n_groups = max(2, n // 25)
    lex = load_synthetic(n_groups=n_groups, dim=lock["lexicon"]["harness_only"]["dim"])
    lex.notes.append(
        "No embedding file found in ./" + data_dir + " - looked for: "
        + ", ".join(os.path.basename(c) for c in candidates))
    return lex
