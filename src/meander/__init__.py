"""MEANDER — a continuous analog lexicon over an explicit discrete role-grammar.

Implementation of the falsifier harness staked in MEANDER-SPEC-v0.2.md.
Every threshold this package uses comes from meander.lock.yaml; nothing here is
allowed to invent a number (M-PIN).
"""

__version__ = "0.2.0-b0"

from . import capacity, efd, falsifiers, lexicon, lock, maps, perceptual, phi, render

__all__ = ["capacity", "efd", "falsifiers", "lexicon", "lock", "maps",
           "perceptual", "phi", "render"]
