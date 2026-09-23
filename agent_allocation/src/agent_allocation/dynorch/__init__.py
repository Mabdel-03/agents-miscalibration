"""dynorch_v1: the metering proxy core (WP-A) for study ``openrouter_dynorch_v1``.

Implements ``docs/dynorch/00_contract.md`` §2 (records), §3 (``ProxyCore``), §4 (rendering + tool-call grammar)
and §5 (budget, upstream, journal). The wire layer (§6), tables (§7) and ops (§8) are other work packages.
``PROXY_VERSION`` (``config.py``) is stamped on every record; a post-freeze change that affects scoring or the cap
bumps it and forces re-execution of affected cells (plan §9, Gate 3).
"""
from agent_allocation.dynorch.config import PROXY_VERSION, DynorchSettings
from agent_allocation.dynorch.records import ProxyError

__all__ = ["PROXY_VERSION", "DynorchSettings", "ProxyError"]
