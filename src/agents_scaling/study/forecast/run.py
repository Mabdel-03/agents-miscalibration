"""``python -m agents_scaling.study.forecast.run`` — the shadow-forecast CPU cell (tier 2b, §8.7).

::

    python -m agents_scaling.study.forecast.run --run-id study_v4 --seal <sha> \\
        --methods IND_VOTE,DEC,CEN_FLAT (--items-file <panel items> | --panel-per-domain 150) \\
        --shard K --num-shards N [--report-only]

For every (item, method) of the shard it compiles the FINAL_HANDOFF_REPORT from the sealed
artifacts (``report.compile_report``), renders the forecast request (``manifest``), writes
the report render with ``prompt_token_ids`` and the two anchor token indices to
``<run_root>/forecast/reports/<item>.<method>.json`` (the capture stage's input) and — unless
``--report-only`` — issues the one shadow-forecast request through the same
``inference.client`` / ``inference.store`` path the policies use (content-addressed,
aliased on re-run), parses it strictly and writes ``<run_root>/forecast/<item>.<method>.json``.

Resumable: existing outputs are skipped; a job that failed leaves
``forecast/errors/<item>.<method>.json`` and is retried on the next run.  Exit codes follow
``run_one``: 0 done, 2 no live endpoint, 3 some jobs incomplete (infrastructure), 4 a
protocol error occurred (needs a human; the other jobs still ran).  Never reads
``data/protected``; correctness joins happen only in ``aggregate``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling.study.config import StudyConfig, load_config
from agents_scaling.study.data.public import load_public_tasks
from agents_scaling.study.forecast import manifest as M
from agents_scaling.study.forecast import shadow as S
from agents_scaling.study.forecast.report import REPORT_METHODS, PRIMARY_POOL_KIND, Report, compile_report
from agents_scaling.study.inference.client import EndpointPool, VllmChatClient
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.inference.tokens import load_tokenizer
from agents_scaling.study.resources.oracle import FlopOracle, make_cost_fn
from agents_scaling.study.selection import seal as seals
from agents_scaling.study.types import (
    EXIT_DONE,
    EXIT_INCOMPLETE,
    EXIT_NO_SERVER,
    EXIT_SUSPENDED,
    SELECTOR_VOTE,
    Checkpoint,
    InfraFailure,
    Method,
    ProtocolError,
    PublicTask,
)

DEFAULT_N = 5
DEFAULT_MODULE = "A"
DEFAULT_EPISODE_REP = 0
DEFAULT_PARALLEL = 4
DEFAULT_WAIT_S = 300.0
CELL_PREFIX = "forecast"


@dataclass(frozen=True)
class ForecastJob:
    source_id: str
    method: Method
    cell_id: str

    @property
    def label(self) -> str:
        return f"{self.source_id}.{self.method.value}"


# --------------------------------------------------------------------------- inputs


def read_items_file(path: str | os.PathLike) -> list[str]:
    """Panel item ids: a JSON list / ``{"items": [...]}`` / JSONL rows with ``source_id``, or
    one id per text line (``#`` comments and blanks ignored).  Order is preserved, duplicates
    rejected."""
    raw = Path(path).read_text(encoding="utf-8")
    items: list[str] = []
    stripped = raw.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except ValueError:
            data = None
        if isinstance(data, dict):
            data = data.get("items")
        if isinstance(data, list):
            items = [str(x["source_id"]) if isinstance(x, dict) else str(x) for x in data]
        else:  # JSONL
            for line in stripped.splitlines():
                if line.strip():
                    row = json.loads(line)
                    items.append(str(row["source_id"]) if isinstance(row, dict) else str(row))
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(line)
    if len(set(items)) != len(items):
        raise ProtocolError(f"{path}: duplicate item ids")
    if not items:
        raise ProtocolError(f"{path}: no items")
    return items


def shard_items(items: Sequence[str], shard: int, num_shards: int) -> list[str]:
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} outside 0..{num_shards - 1}")
    return list(items[shard::num_shards])


def parse_methods(text: str) -> list[Method]:
    methods = [Method(m.strip()) for m in text.split(",") if m.strip()]
    bad = [m.value for m in methods if m not in REPORT_METHODS]
    if bad:
        raise ProtocolError(f"methods without a FINAL_HANDOFF_REPORT node: {bad}")
    if not methods:
        raise ProtocolError("no methods given")
    return methods


def resolve_cell_id(
    selections: Mapping[str, Any],
    source_id: str,
    method: Method,
    *,
    checkpoint: str,
    N: int,
    B: int,
    module: str,
    episode_rep: int,
) -> str:
    """The generate cell whose sealed primary-pool VOTE selection matches the filters."""
    kind = PRIMARY_POOL_KIND[method]
    cells = sorted(
        {
            str(s["cell_id"])
            for s in seals.selections_of_item(selections, source_id)
            if s.get("selector_id") == SELECTOR_VOTE
            and s.get("pool_kind") == kind
            and s.get("method") == method.value
            and str(s.get("checkpoint")) == checkpoint
            and int(s.get("N", -1)) == N
            and int(s.get("B", -1)) == B
            and str(s.get("module")) == module
            and int(s.get("episode_rep", -1)) == episode_rep
        }
    )
    if not cells:
        raise ProtocolError(
            f"{source_id}/{method.value}: no sealed {kind} VOTE selection for module={module} {checkpoint} N={N} B={B} e{episode_rep}"
        )
    if len(cells) > 1:
        raise ProtocolError(f"{source_id}/{method.value}: several sealed cells match {cells}")
    return cells[0]


# --------------------------------------------------------------------------- the cell


class ForecastRunner:
    """One shard of the forecast cell (see the module docstring).

    Injection points (tests): ``tokenizer`` (must be the server's), ``client_factory(pool,
    tokenizer, cost_fn)``, ``oracle_factory``, ``clock``.
    """

    def __init__(
        self,
        run_root: str | os.PathLike,
        server_run_root: str | os.PathLike,
        seal: str,
        methods: Sequence[Method],
        items: Sequence[str],
        *,
        cfg: StudyConfig | None = None,
        checkpoint: str | None = None,
        N: int = DEFAULT_N,
        B: int | None = None,
        module: str = DEFAULT_MODULE,
        episode_rep: int = DEFAULT_EPISODE_REP,
        report_only: bool = False,
        shard: int = 0,
        parallel: int = DEFAULT_PARALLEL,
        tokenizer: Any | None = None,
        client_factory: Callable[[EndpointPool, Any, Any], Any] | None = None,
        oracle_factory: Callable[[Checkpoint], FlopOracle] | None = None,
        wait_timeout_s: float = DEFAULT_WAIT_S,
        run_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.run_root = Path(run_root)
        self.server_run_root = Path(server_run_root)
        self.seal = seal
        self.methods = list(methods)
        self.items = list(items)
        self.cfg = cfg or load_config()
        self.checkpoint = self.cfg.checkpoint(checkpoint or self.cfg.flagship)
        self.N = int(N)
        self.B = int(B if B is not None else self.cfg.budget.primary)
        self.module = str(module)
        self.episode_rep = int(episode_rep)
        self.report_only = bool(report_only)
        self.shard = int(shard)
        self.parallel = max(1, int(parallel))
        self._tokenizer = tokenizer
        self._client_factory = client_factory
        self._oracle_factory = oracle_factory or FlopOracle.from_checkpoint
        self.wait_timeout_s = float(wait_timeout_s)
        self.run_id = run_id or self.run_root.name
        self.clock = clock
        self.store = RequestStore(self.run_root / "requests")
        self.events_path = S.forecast_dir(self.run_root) / f"events.s{self.shard:03d}.jsonl"
        self.lock = threading.Lock()
        self.tokenizer: Any = None
        self.client: Any = None
        self.selections: Mapping[str, Any] = {}
        self.tasks: dict[str, PublicTask] = {}
        self.stats = {"reports_written": 0, "reports_skipped": 0, "forecasts_written": 0, "forecasts_skipped": 0,
                      "aliased": 0, "generated": 0, "infra_failures": 0, "protocol_errors": 0, "parse_failures": 0}

    # ---- helpers ----------------------------------------------------------------------
    def event(self, kind: str, **payload: Any) -> None:
        io.append_jsonl(self.events_path, {"t": self.clock(), "event": kind, "shard": self.shard, **payload})

    def cell_id(self, method: Method) -> str:
        return f"{CELL_PREFIX}.{method.value}.{self.checkpoint.size}.x{self.seal[:8]}"

    def jobs(self) -> list[ForecastJob]:
        """Resolve every (item, method) to its sealed cell; an unresolvable pair is recorded
        as a protocol error for that job (``forecast/errors/``) and the others still run."""
        out: list[ForecastJob] = []
        for sid in self.items:
            for method in self.methods:
                try:
                    cell = resolve_cell_id(self.selections, sid, method, checkpoint=self.checkpoint.size, N=self.N, B=self.B,
                                           module=self.module, episode_rep=self.episode_rep)
                except ProtocolError as exc:
                    self._record_error(ForecastJob(sid, method, ""), "ProtocolError", exc, None)
                    sys.stderr.write(f"[forecast] {exc}\n")
                    with self.lock:
                        self.stats["protocol_errors"] += 1
                    continue
                out.append(ForecastJob(sid, method, cell))
        return out

    def _connect(self) -> None:
        pool = EndpointPool(self.server_run_root, self.checkpoint.profile, shard=self.shard)
        pool.wait(self.wait_timeout_s)  # TimeoutError → EXIT_NO_SERVER (caller)
        cost_fn = make_cost_fn({self.checkpoint.size: self._oracle_factory(self.checkpoint)})
        if self._client_factory is not None:
            self.client = self._client_factory(pool, self.tokenizer, cost_fn)
        else:
            self.client = VllmChatClient(pool, self.tokenizer, cost_fn=cost_fn, run_id=self.run_id)

    def _pending(self, job: ForecastJob) -> bool:
        target = S.report_path(self.run_root, job.source_id, job.method) if self.report_only else S.forecast_path(self.run_root, job.source_id, job.method)
        return not target.exists()

    # ---- one job ---------------------------------------------------------------------
    def compile(self, job: ForecastJob) -> Report:
        return compile_report(self.run_root, job.source_id, job.cell_id, self.tokenizer, seal=self.seal, cfg=self.cfg,
                              selections=self.selections, tasks=self.tasks)

    def run_job(self, job: ForecastJob) -> None:
        report = self.compile(job)
        render = M.report_render(report, self.tokenizer, self.cfg, self.checkpoint)
        rpath = S.report_path(self.run_root, job.source_id, job.method)
        if rpath.exists():
            existing = json.loads(rpath.read_text(encoding="utf-8"))
            if existing.get("report_id") != render["report_id"] or existing.get("request_id") != render["request_id"]:
                raise ProtocolError(f"{rpath} holds report {existing.get('report_id', '')[:12]}, recompiled {render['report_id'][:12]}: sealed inputs changed")
            with self.lock:
                self.stats["reports_skipped"] += 1
        else:
            S.write_report_render(self.run_root, render)
            with self.lock:
                self.stats["reports_written"] += 1
        if self.report_only:
            return
        rendered = M.render_forecast_request(report, render["manifest"])
        spec = M.forecast_spec(report, rendered, self.cfg, self.checkpoint)
        if spec.request_id != render["request_id"]:
            raise ProtocolError("forecast request identity diverged from the written report render")
        cell_id = self.cell_id(job.method)
        record, aliased = self.store.get_or_generate(spec, self.client, cell_id, on_event=lambda k, p: self.event(k, **p))
        parsed = S.parse_forecast(record.response.get("content"), finish_reason=record.response.get("finish_reason"), mask=render["manifest"]["mask"])
        payload = S.forecast_output(report, render["manifest"], record, parsed, aliased=aliased, produced_at=self.clock(), cell_id=cell_id)
        S.write_forecast(self.run_root, payload)
        with self.lock:
            self.stats["forecasts_written"] += 1
            self.stats["aliased" if aliased else "generated"] += 1
            if not parsed.valid:
                self.stats["parse_failures"] += 1

    def _run_one(self, job: ForecastJob) -> None:
        started = self.clock()
        epath = S.error_path(self.run_root, job.source_id, job.method)
        try:
            self.run_job(job)
        except InfraFailure as exc:
            self._record_error(job, "InfraFailure", exc, getattr(exc, "attempts", None))
            with self.lock:
                self.stats["infra_failures"] += 1
            return
        except ProtocolError as exc:
            self._record_error(job, "ProtocolError", exc, None)
            with self.lock:
                self.stats["protocol_errors"] += 1
            return
        except Exception as exc:  # an unexpected harness defect is a protocol error too
            self._record_error(job, f"ProtocolError({type(exc).__name__})", exc, traceback.format_exc()[-4000:])
            with self.lock:
                self.stats["protocol_errors"] += 1
            return
        io.remove_file(epath)
        self.event("job_finished", job=job.label, cell_id=job.cell_id, wall_s=self.clock() - started)

    def _record_error(self, job: ForecastJob, kind: str, exc: BaseException, detail: Any) -> None:
        io.write_json(
            S.error_path(self.run_root, job.source_id, job.method),
            {"source_id": job.source_id, "method": job.method.value, "cell_id": job.cell_id, "kind": kind, "error": str(exc)[:4000],
             "detail": detail, "at": self.clock(), "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        )
        self.event("job_failed", job=job.label, failure=kind, error=str(exc)[:2000])

    # ---- main ------------------------------------------------------------------------
    def run(self) -> int:
        S.forecast_dir(self.run_root).mkdir(parents=True, exist_ok=True)
        try:
            self.selections = seals.load_selections(self.run_root, self.seal)
            self.tasks = {t.source_id: t for t in load_public_tasks(self.run_root)}
            jobs = self.jobs()
        except ProtocolError as exc:
            self.event("suspended", error=str(exc)[:2000])
            sys.stderr.write(f"[forecast] {exc}\n")
            return EXIT_SUSPENDED
        todo = [j for j in jobs if self._pending(j)]
        with self.lock:
            self.stats["forecasts_skipped" if not self.report_only else "reports_skipped"] += len(jobs) - len(todo)
        self.event("started", n_jobs=len(jobs), n_todo=len(todo), report_only=self.report_only, methods=[m.value for m in self.methods])
        if todo:
            self.tokenizer = self._tokenizer if self._tokenizer is not None else load_tokenizer(self.checkpoint)
            if not self.report_only:
                try:
                    self._connect()
                except TimeoutError as exc:
                    self.event("no_server", error=str(exc))
                    return EXIT_NO_SERVER
            with ThreadPoolExecutor(max_workers=min(self.parallel, len(todo)), thread_name_prefix="forecast") as pool:
                list(pool.map(self._run_one, todo))
        summary = {"shard": self.shard, "seal": self.seal, "n_jobs": len(jobs), "n_todo": len(todo), **self.stats}
        self.event("finished", **summary)
        print(json.dumps(summary, sort_keys=True))
        if self.stats["protocol_errors"]:
            return EXIT_SUSPENDED
        if self.stats["infra_failures"]:
            return EXIT_INCOMPLETE
        return EXIT_DONE


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.study.forecast.run", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--seal", required=True, help="sha256 of the sealed generate manifest (seals/<sha>/SELECTIONS.json)")
    p.add_argument("--methods", default="IND_VOTE,DEC,CEN_FLAT")
    panel = p.add_mutually_exclusive_group(required=True)
    panel.add_argument("--items-file", default=None, help="explicit item ids (json list / jsonl / one per line)")
    panel.add_argument("--panel-per-domain", type=int, default=None,
                       help="derive the items from the public export: PublicTask.rank < N per superdomain on 'main' (neural.panel; the N1/N3 rule)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--report-only", action="store_true", help="write the report renders only (no forecast request)")
    p.add_argument("--server-run-id", default=None, help="run id whose servers/ registry holds the endpoints (default: --run-id)")
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default=None, help="reader/generator checkpoint size (default: the flagship)")
    p.add_argument("--N", type=int, default=DEFAULT_N)
    p.add_argument("--B", type=int, default=None, help="budget multiplier of the sealed cells (default: budget.primary)")
    p.add_argument("--module", default=DEFAULT_MODULE)
    p.add_argument("--episode-rep", type=int, default=DEFAULT_EPISODE_REP)
    p.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL, help="concurrent jobs (each one request)")
    p.add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S, help="endpoint wait before exit 2")
    return p


def main(argv: Sequence[str] | None = None, **injected: Any) -> int:
    args = build_parser().parse_args(argv)
    results_root = Path(args.results_root)
    run_root = results_root / args.run_id
    server_run_root = results_root / (args.server_run_id or args.run_id)
    try:
        methods = parse_methods(args.methods)
        if args.items_file is not None:
            all_items = read_items_file(args.items_file)
        else:
            from agents_scaling.study.neural.panel import flatten_panel, panel_source_ids

            all_items = flatten_panel(panel_source_ids(run_root, int(args.panel_per_domain)))
        items = shard_items(all_items, args.shard, args.num_shards)
    except (ProtocolError, ValueError, OSError) as exc:
        sys.stderr.write(f"[forecast] {exc}\n")
        return EXIT_SUSPENDED
    runner = ForecastRunner(
        run_root, server_run_root, args.seal, methods, items,
        cfg=load_config(args.config), checkpoint=args.checkpoint, N=args.N, B=args.B, module=args.module, episode_rep=args.episode_rep,
        report_only=args.report_only, shard=args.shard, parallel=args.parallel, wait_timeout_s=args.wait_s, run_id=args.run_id,
        **injected,
    )
    code = runner.run()
    sys.stderr.write(f"[forecast] shard {args.shard}/{args.num_shards} exit {code}\n")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["ForecastJob", "ForecastRunner", "build_parser", "main", "parse_methods", "read_items_file", "resolve_cell_id", "shard_items"]
