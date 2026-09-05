"""agent_design_v4 on ORCD — self-contained study package (spec §11.2, flat layout).

WP0 (this contract): ``types``, ``identity``, ``config``, ``metrics_reference``,
``prompts`` (frozen templates + hashes) and ``freeze`` (amendment register, prior-exposure
manifest).  Later work packages add data/, inference/, parse/, selection/, resources/,
policies/, evaluation/ and the cell runner.  Import here is side-effect free.
"""

from __future__ import annotations

STUDY_PACKAGE_VERSION = "4.0-orcd"

__all__ = ["STUDY_PACKAGE_VERSION"]
