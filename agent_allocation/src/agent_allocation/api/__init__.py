"""OpenRouter adapter for openrouter_agent_allocation_v1 (agent 01).

Modules:

* ``request``   -- the one request body/headers contract (``build_request_body``, ``REQUEST_HEADERS``).
* ``normalize`` -- pure response classification and usage normalization (no I/O).
* ``client``    -- ``OpenRouterClient``: one persistent ``httpx.AsyncClient``, zero retries, never touches disk/ledger.
* ``endpoints`` -- ``select_standard_endpoint``: exact standard-tier endpoint lock from a saved endpoints listing.
* ``fake``      -- in-process OpenRouter fake (``fake_openrouter_transport``, ``fake_endpoint_lock``, ``FakeScript``).
* ``canary``    -- ``canary_report``: the Gate 1 capability summary per requested reasoning budget.

Nothing here is imported eagerly so that ``import agent_allocation.api.fake`` stays cheap for other agents' tests.
"""
