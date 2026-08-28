"""meander.lock access — M-PIN, enforced in code rather than in prose.

Two jobs:
  1. Load the pinned numbers and REFUSE to invent a missing one. A threshold that
     is absent must halt the run and name the law it disables (v0.2 §11: "every
     unset threshold names the law it disables"). Defaulting a floor silently is
     how a pre-registration turns into a post-hoc rationalisation.
  2. Fingerprint the code that produced a result. Version = the tuple of
     fingerprints; a results file that cannot say which code made it is not
     evidence.
"""

from __future__ import annotations

import hashlib
import os

import yaml

__all__ = ["load", "code_fingerprint", "require", "disabled_laws", "assert_no_peek"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEFAULT_LOCK = os.path.join(_ROOT, "meander.lock.yaml")


def load(path: str | None = None) -> dict:
    with open(path or DEFAULT_LOCK, encoding="utf-8") as fh:
        lock = yaml.safe_load(fh)
    for key in ("schema", "floors", "noise", "render", "perceptual", "budget"):
        if key not in lock:
            raise ValueError(f"meander.lock is missing required section {key!r}")
    return lock


def code_fingerprint(src_dir: str | None = None) -> str:
    """sha256 over the sorted contents of src/meander/*.py."""
    src_dir = src_dir or _HERE
    h = hashlib.sha256()
    for name in sorted(os.listdir(src_dir)):
        if name.endswith(".py"):
            with open(os.path.join(src_dir, name), "rb") as fh:
                h.update(name.encode())
                h.update(fh.read())
    return h.hexdigest()[:16]


def require(lock: dict, dotted: str):
    """Fetch a pinned value; raise if it is absent, null, or still a TODO."""
    node = lock
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"meander.lock: {dotted} is not pinned — the law that "
                           f"consumes it is disabled and must be reported as such")
        node = node[part]
    if node is None or (isinstance(node, str) and node.startswith("TODO")):
        raise ValueError(f"meander.lock: {dotted} is unpinned ({node!r})")
    return node


def disabled_laws(lock: dict) -> list[dict]:
    return list(lock.get("disabled_laws") or [])


PEEK_DISCLOSED = ("synthetic_only",)   # allowed, but surfaced on every run


def assert_no_peek(lock: dict) -> list[dict]:
    """Guard the one unrecoverable sin: a bar moved after REAL results were seen.

    Not every peek is fatal, and collapsing the distinction would make the guard
    useless — B0 exists precisely to shake the harness out on a stand-in space,
    and a change made there cannot contaminate a pre-registration for data that
    has never been touched. So `peeked: synthetic_only` passes but is RETURNED,
    and the harness prints it into every report. Anything else halts the run.
    """
    disclosed = []
    for rev in lock.get("revisions") or []:
        peeked = rev.get("peeked")
        if peeked in (None, False):
            continue
        if peeked in PEEK_DISCLOSED:
            disclosed.append(rev)
            continue
        raise RuntimeError(
            f"meander.lock revision {rev.get('key')!r} was made after a peek at "
            f"real results (peeked={peeked!r}). Every result downstream of it is "
            "post-hoc and may not be published.")
    return disclosed
