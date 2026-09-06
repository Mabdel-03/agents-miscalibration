"""``python -m agents_scaling.study.seal`` — entry point of :mod:`agents_scaling.study.selection.seal` (WP5).

Architecture §1.13 names the CLI at the package root while the ownership table places the
implementation in ``selection/seal.py``; this module only re-exports it.
"""

from __future__ import annotations

import sys

from agents_scaling.study.selection.seal import build_parser, main, seal_pools, seal_selections

__all__ = ["build_parser", "main", "seal_pools", "seal_selections"]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
