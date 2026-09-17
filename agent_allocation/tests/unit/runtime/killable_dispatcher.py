"""Subprocess body for the kill -9 test: runs a dispatcher whose fake requests never return, prints READY when
``--inflight`` requests are on the wire, then waits to be killed. Not collected by pytest."""
from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from pathlib import Path


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--inflight", type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root), str(root / "src")]
    from tests.unit.runtime.fakes import Behavior, Env, FakeClient, make_graph, plan_of

    env = Env.create(Path(args.tmp), concurrency_start=args.inflight, max_calls_per_item_in_flight=16)
    graph = make_graph("killed", bank=10, chain=1, team_slots=1, peer_sizes=(), deep=())

    def script(spec, attempt_id, index):
        if index + 1 == args.inflight:
            print("READY", flush=True)
        return Behavior(delay=3600)

    await env.dispatcher(FakeClient(script)).run(plan_of([graph]), allow_paid=False, max_usd=Decimal("5"))


if __name__ == "__main__":
    asyncio.run(main())
