"""``python -m agents_scaling.nextdyn <verb> ...`` (spec 17.5).  The dispatcher is ``cli.py``."""

from __future__ import annotations

import sys

from agents_scaling.nextdyn.cli import main

if __name__ == "__main__":
    sys.exit(main())
