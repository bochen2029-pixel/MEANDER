"""The maps under test, and the strong nulls (M-NULL).

    random_phi                  BANNED. Beating it proves nothing; v0.2 retired
                                v0.1's "correlation beats random-phi" because it
                                returned a false "alive".
    roundtrip_only_phi          the SAME network as phi, denied only the
                                continuity/separation/nesting objectives.
    pca_2d_render               a linear 2-D squash, lifted back into the full
                                shape space by a fixed affine map so it renders
                                shapes of the SAME complexity as phi and is only
                                constrained in RANK. Strictly stronger than
                                feeding 2 numbers into 2 shape parameters.
    umap_2d_render              optional; SKIPPED loudly if umap-learn is absent.
    discrete_code_matched_bits  arbitrary codebook at matched capacity: a normal
                                writing system. Perfect separation, zero metric.
    discrete_semantic_ordered   NOT in the spec's list, added by B0 because it is
                                the adversary a sceptic reaches for first: sort
                                the clusters along their principal axis and hand
                                out codebook shapes IN THAT ORDER. A syllabary
                                that cheated. It gets metric structure with no
                                continuity anywhere in the map. If phi cannot
                                beat a sorted alphabet, phi is not earning its keep.
"""

from __future__ import annotations

import numpy as np

from . import efd, phi as phi_mod

__all__ = ["ShapeMap", "LearnedPhi", "PCA2D", "UMAP2D", "DiscreteCode", "build_all"]


class ShapeMap:
    kind = "null"
    name = "base"

    def __init__(self):
        self.available = True
        self.note = ""
        self.m = None

    def fit(self, vectors, m, ctx):
        raise NotImplementedError

    def encode(self, vectors) -> np.ndarray:      # -> (N, p) free parameters
        raise NotImplementedError

    def decode(self, params) -> np.ndarray | None:
        return None

    def coeffs(self, vectors) -> np.ndarray:      # -> (N, m, 4)
        z = self.encode(vectors)
        return np.stack([efd.vector_to_coeffs(row, self.m) for row in z])


# --------------------------------------------------------------- learned

class LearnedPhi(ShapeMap):
    def __init__(self, name, weights: phi_mod.LossWeights, kind="null", epochs=3000,
                 seed=0):
        super().__init__()
        self.name = name
        self.kind = kind
        self.weights = weights
        self.epochs = epochs
        self.seed = seed
        self.net = None
        self.history = None

    def fit(self, vectors, m, ctx):
        self.m = m
        self.net, self.history = phi_mod.train(
            vectors, m, self.weights, ctx["harmonic_w"],
            delta=ctx["delta"], epochs=self.epochs, seed=self.seed)
        return self

    def encode(self, vectors):
        import torch
        with torch.no_grad():
            e = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
            return self.net.encode(e).numpy()

    def decode(self, params):
        import torch
        with torch.no_grad():
            z = torch.from_numpy(np.asarray(params, dtype=np.float32))
            return self.net.decode(z).numpy()


# --------------------------------------------------------------- 2-D squashes

class _Rank2Lift(ShapeMap):
    """Squash to 2 dims, then lift affinely into the full free-parameter space."""

    def __init__(self, name, seed=3):
        super().__init__()
        self.name = name
        self.seed = seed
        self.z2 = None
        self.W = None
        self.b = None
        self._fit_vectors = None

    def _embed(self, vectors):
        raise NotImplementedError

    def fit(self, vectors, m, ctx):
        self.m = m
        p = efd.n_free_params(m)
        z2 = np.asarray(self._embed(vectors), dtype=np.float32)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-8)
        rng = np.random.default_rng(self.seed)
        W = rng.normal(size=(2, p)).astype(np.float32)
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
        self.W, self.b = W, np.zeros(p, dtype=np.float32)
        self.z2, self._fit_vectors = z2, np.asarray(vectors, dtype=np.float32)
        self.scale = phi_mod.param_scale(m)
        self.offset = phi_mod.param_offset(m)
        return self

    def encode(self, vectors):
        v = np.asarray(vectors, dtype=np.float32)
        if len(v) == len(self._fit_vectors) and np.allclose(v, self._fit_vectors):
            z2 = self.z2
        else:                                     # out-of-sample: nearest fitted
            sim = v @ self._fit_vectors.T
            z2 = self.z2[np.argmax(sim, axis=1)]
        return np.tanh(z2 @ self.W + self.b) * self.scale + self.offset


class PCA2D(_Rank2Lift):
    def __init__(self, seed=3):
        super().__init__("pca_2d_render", seed)

    def _embed(self, vectors):
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(vectors)


class UMAP2D(_Rank2Lift):
    def __init__(self, seed=3):
        super().__init__("umap_2d_render", seed)
        try:
            import umap                                            # noqa: F401
            self.available = True
        except Exception as exc:                                   # noqa: BLE001
            self.available = False
            self.note = (f"SKIPPED - umap-learn not installed ({type(exc).__name__}). "
                         "Listed in meander.lock as optional; recorded as skipped, "
                         "never silently dropped.")

    def _embed(self, vectors):
        import umap
        return umap.UMAP(n_components=2, random_state=0).fit_transform(vectors)


# --------------------------------------------------------------- discrete

class DiscreteCode(ShapeMap):
    """A finite inventory of drawn shapes: i.e. an ordinary writing system."""

    def __init__(self, ordered: bool, capacity_bits: float, seed=5):
        super().__init__()
        self.ordered = ordered
        self.name = "discrete_semantic_ordered" if ordered else "discrete_code_matched_bits"
        self.capacity_bits = float(capacity_bits)
        self.seed = seed
        self.centroids = None
        self.codebook = None
        self.n_codes = None

    def _make_codebook(self, n_codes, p, scale, offset, rng):
        if self.ordered:
            # A smooth 1-D path through shape space: adjacent codes -> adjacent
            # shapes. Combined with the semantic sort in fit(), nearby meanings
            # get nearby marks with no continuous map anywhere in sight.
            t = np.linspace(0.0, 1.0, n_codes, dtype=np.float32)[:, None]
            book = np.zeros((n_codes, p), dtype=np.float32)
            for k in range(1, 4):
                ph = rng.uniform(0, 2 * np.pi, size=p).astype(np.float32)
                book += np.cos(2 * np.pi * k * t + ph) / k
            book /= np.abs(book).max() + 1e-8
            return book * scale + offset
        # Arbitrary: greedy farthest-point spread, so the discrete null gets the
        # BEST separation a codebook of this size can have.
        pool = rng.uniform(-1, 1, size=(max(n_codes * 8, 64), p)).astype(np.float32)
        chosen = [int(rng.integers(len(pool)))]
        d = np.linalg.norm(pool - pool[chosen[0]], axis=1)
        while len(chosen) < n_codes:
            i = int(np.argmax(d))
            chosen.append(i)
            d = np.minimum(d, np.linalg.norm(pool - pool[i], axis=1))
        return pool[chosen] * scale + offset

    def fit(self, vectors, m, ctx):
        from sklearn.cluster import KMeans

        self.m = m
        p = efd.n_free_params(m)
        v = np.asarray(vectors, dtype=np.float32)
        rng = np.random.default_rng(self.seed)

        self.n_codes = int(min(2 ** self.capacity_bits, len(v)))
        km = KMeans(n_clusters=self.n_codes, n_init=4, random_state=0).fit(v)
        cent = km.cluster_centers_.astype(np.float32)

        if self.ordered:
            from sklearn.decomposition import PCA
            order = np.argsort(PCA(n_components=1, random_state=0)
                               .fit_transform(cent)[:, 0])
            cent = cent[order]

        self.centroids = cent
        self.codebook = self._make_codebook(self.n_codes, p, phi_mod.param_scale(m),
                                            phi_mod.param_offset(m), rng)
        if self.n_codes >= len(v):
            self.note = (f"capacity {self.capacity_bits:.1f} bits allows "
                         f"{2 ** self.capacity_bits:.0f} codes >= lexicon N={len(v)}: "
                         "the bit constraint is NOT binding, so this null runs at "
                         "its strongest (one code per word).")
        return self

    def encode(self, vectors):
        sim = np.asarray(vectors, dtype=np.float32) @ self.centroids.T
        return self.codebook[np.argmax(sim, axis=1)]

    def decode(self, params):
        d = np.linalg.norm(np.asarray(params, dtype=np.float32)[:, None, :]
                           - self.codebook[None, :, :], axis=2)
        return self.centroids[np.argmin(d, axis=1)]


# --------------------------------------------------------------- registry

def build_all(capacity_bits: float, epochs: int = 3000, include_candidate=True):
    """Every map the harness will run. Order is report order."""
    maps = []
    if include_candidate:
        maps.append(LearnedPhi(
            "phi_learned", phi_mod.LossWeights(1.0, 1.0, 0.5, 0.5),
            kind="candidate", epochs=epochs))
    maps += [
        LearnedPhi("roundtrip_only_phi", phi_mod.LossWeights(1.0, 0, 0, 0),
                   epochs=epochs),
        PCA2D(),
        UMAP2D(),
        DiscreteCode(ordered=False, capacity_bits=capacity_bits),
        DiscreteCode(ordered=True, capacity_bits=capacity_bits),
    ]
    return maps
