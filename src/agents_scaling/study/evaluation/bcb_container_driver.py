"""BigCodeBench in-container driver (WP5).  Stdlib only; runs under the image's Python 3.10.

Spec §3.3 ("For code, evaluate each complete original-task candidate in a fresh official
test environment"), §10.6 (generated code runs in a bounded isolated environment with no
host credential or protected-mount access).  06_bcb_container.md: one ``apptainer exec``
per cell with an in-container loop; the fontconfig/matplotlib cache environment must be
set by this driver.  Corrections P1-5: ``solution.py = strip_one_fence(code)`` is done by
the harness before the job reaches this driver (one frozen rule shared with grouping);
the driver receives the program bytes verbatim.

Usage: ``python3 bcb_container_driver.py jobs.jsonl results.jsonl [--timeout S] [--mem BYTES]``

``jobs.jsonl`` lines: ``{"cid": str, "code": str, "test": str}``.  For every job a fresh
subdirectory receives ``solution.py = code + "\\n\\n" + test`` and
``python3 -m unittest -q solution`` runs in a fresh subprocess (own session, per-job
``TMPDIR``, ``RLIMIT_AS``, 10 MiB ``RLIMIT_STACK``, ``MPLBACKEND=Agg``).  ``pass`` iff the
return code is 0 and ``OK`` appears in stderr (unittest's verdict line); a wall-clock
overrun is ``timeout``; everything else is ``fail``; a driver-side exception is ``error``.
Results are appended line by line as they complete so a partial run is never lost.
"""

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time

STACK_LIMIT = 10 * 1024 * 1024
TAIL_BYTES = 4000
STATUSES = ("pass", "fail", "timeout", "error")


def _limits(mem_bytes):
    def apply():
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (STACK_LIMIT, STACK_LIMIT))
        except (ValueError, OSError):
            pass
        if mem_bytes and mem_bytes > 0:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass
        os.setsid()
    return apply


def _tail(data):
    if data is None:
        return ""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    return data[-TAIL_BYTES:]


def container_env(work):
    """The environment every candidate subprocess gets (06_bcb_container.md gotcha)."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": os.path.join(work, ".mpl"),
        "XDG_CACHE_HOME": os.path.join(work, ".cache"),
        "HOME": os.path.join(work, ".home"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
        if key in os.environ:
            env[key] = os.environ[key]
    for path in (env["MPLCONFIGDIR"], env["XDG_CACHE_HOME"], env["HOME"]):
        os.makedirs(path, exist_ok=True)
    return env


def run_job(job, work, timeout_s, mem_bytes, python):
    """Evaluate one job in a fresh subdirectory; returns the result dict."""
    cid = job["cid"]
    started = time.time()
    job_dir = os.path.join(work, "jobs", cid[:16] + "_" + str(int(started * 1000) % 1000000))
    os.makedirs(job_dir, exist_ok=False)
    tmp_dir = os.path.join(job_dir, "tmp")
    os.makedirs(tmp_dir)
    with open(os.path.join(job_dir, "solution.py"), "w", encoding="utf-8") as handle:
        handle.write(job["code"] + "\n\n" + job["test"])
    env = container_env(work)
    env["TMPDIR"] = tmp_dir
    env["TEMP"] = tmp_dir
    env["TMP"] = tmp_dir
    status = "error"
    stdout = stderr = b""
    returncode = None
    proc = None
    try:
        proc = subprocess.Popen(
            [python, "-m", "unittest", "-q", "solution"],
            cwd=job_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_limits(mem_bytes),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            returncode = proc.returncode
            ok = returncode == 0 and b"OK" in stderr
            status = "pass" if ok else "fail"
        except subprocess.TimeoutExpired:
            status = "timeout"
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except Exception:
                stdout, stderr = b"", b""
    except Exception as exc:  # driver-side failure, never a candidate verdict
        status = "error"
        stderr = (str(exc)).encode("utf-8")
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    return {
        "cid": cid,
        "status": status,
        "returncode": returncode,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "elapsed": time.time() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="BigCodeBench unittest driver (runs inside the evaluator container)")
    parser.add_argument("jobs")
    parser.add_argument("results")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--mem", type=int, default=8 * 1024 ** 3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--work", default=None, help="work directory (default: the directory of the jobs file)")
    args = parser.parse_args(argv)
    work = os.path.abspath(args.work or os.path.dirname(os.path.abspath(args.jobs)))
    os.makedirs(os.path.join(work, "jobs"), exist_ok=True)
    done = set()
    if os.path.exists(args.results):
        with open(args.results, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["cid"])
    n = 0
    with open(args.jobs, "r", encoding="utf-8") as jobs, open(args.results, "a", encoding="utf-8") as out:
        for line in jobs:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            if job["cid"] in done:
                continue
            result = run_job(job, work, args.timeout, args.mem, args.python)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            n += 1
    sys.stderr.write("bcb driver: %d jobs evaluated\n" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
