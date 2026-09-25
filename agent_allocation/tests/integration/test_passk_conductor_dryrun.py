"""WP-G P8: the conductor shell script, dry-run with stub sbatch/sacct/squeue/allocation/python (no Slurm, no API).

The stubs log every submission; sacct reports COMPLETED 0:0 except for the verb named in STUB_FAIL_VERB (FAILED 1:0).
Real bash, flock and the real venv python (for the conductor's inline heredocs) run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.forced_cfg import PROJECT

pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="flock not available")

STUB_SBATCH = r"""#!/bin/bash
n=$(cat "$STUBS/next" 2>/dev/null || echo 1000); echo $((n + 1)) > "$STUBS/next"
verb=$(echo "$*" | grep -oE 'VERB=[a-z0-9-]+' | head -1 | cut -d= -f2)
shard=$(echo "$*" | grep -oE 'SHARD=[a-z]+' | head -1 | cut -d= -f2)
stage=$(echo "$*" | grep -oE 'STAGE=[a-z0-9-]+' | head -1 | cut -d= -f2)
flags=$(echo "$*" | grep -oE 'FLAGS=[a-z0-9+-]+' | head -1 | cut -d= -f2)
rel=$(echo "$*" | grep -oE 'RELEASE=[a-z0-9+]+' | head -1 | cut -d= -f2)
echo "$n $verb $shard $stage" >> "$STUBS/submitted"
echo "$n $stage flags=$flags release=$rel" >> "$STUBS/exports"
echo "$n"
"""
STUB_SACCT = r"""#!/bin/bash
jid=$2
verb=$(awk -v j="$jid" '$1==j {print $2}' "$STUBS/submitted")
stage=$(awk -v j="$jid" '$1==j && $2=="analyze" {print $3}' "$STUBS/submitted")
field=${@: -1}
if [ -n "${STUB_FAIL_VERB:-}" ] && [ "$verb" = "$STUB_FAIL_VERB" ]; then st=FAILED; rc="1:0"
elif [ -n "${STUB_FAIL_STAGE:-}" ] && [ "$stage" = "$STUB_FAIL_STAGE" ]; then st=FAILED; rc="1:0"
elif [ "$verb" = grade-follow ]; then st=RUNNING; rc="0:0"
elif [ "$verb" = run ] && [ -n "${STUB_RUN_RUNNING:-}" ]; then st=RUNNING; rc="0:0"
else st=COMPLETED; rc="0:0"; fi
case "$field" in
  State) echo "$st";;
  ExitCode) echo "$rc";;
  JobID,State,ExitCode) echo "$jid|$st|$rc";;
  JobID,ElapsedRaw) echo "$jid|60";;
esac
"""
STUB_SQUEUE = r"""#!/bin/bash
name=""; while [ $# -gt 0 ]; do [ "$1" = -n ] && name=$2; shift; done
case "$name" in
  passk-grade-follow-*) awk -v s="${name#passk-grade-follow-}" '$2=="grade-follow" && $3==s {print $1}' \
                          "$STUBS/submitted" 2>/dev/null ;;
esac
exit 0
"""
STUB_ALLOCATION = r"""#!/bin/bash
echo "allocation $*" >> "$STUBS/inline"
exit 0
"""
STUB_PYTHON = r"""#!/bin/bash
if [ "$1" = "-m" ]; then
  echo "python $*" >> "$STUBS/inline"
  case "$2 $3" in
    "agent_allocation.passk.gates gate0") exit ${STUB_GATE0_RC:-0} ;;
    "agent_allocation.passk.checkpoints gate2") exit 4 ;;
    "agent_allocation.passk.checkpoints sample") exit 4 ;;
    "agent_allocation.passk.checkpoints gate4")
      [ -n "${STUB_GATE4_FAIL_SPLIT:-}" ] && echo "$*" | grep -q -- "--split $STUB_GATE4_FAIL_SPLIT" && exit 3
      exit 0 ;;
    *) exit 0 ;;
  esac
fi
if [ "$1" = "-" ] && [ "${2:-}" = "$STUDY" ] && [ -n "${STUB_STAGE1:-}" ]; then cat >/dev/null; echo yes; exit 0; fi
exec "$REAL_PY" "$@"
"""


def _setup(tmp_path: Path):
    code = tmp_path / "code"
    (code / "slurm").mkdir(parents=True)
    (code / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(PROJECT / "slurm" / "passk_conductor.sh", code / "slurm" / "passk_conductor.sh")
    (code / "slurm" / "env.sh").write_text("")
    (code / "slurm" / "passk_cont.sbatch").write_text("#!/bin/bash\n")
    stubs = tmp_path / "stubs"
    (stubs / "bin").mkdir(parents=True)
    for name, text in (("sbatch", STUB_SBATCH), ("sacct", STUB_SACCT), ("squeue", STUB_SQUEUE)):
        (stubs / "bin" / name).write_text(text)
    (code / ".venv" / "bin" / "allocation").write_text(STUB_ALLOCATION)
    (code / ".venv" / "bin" / "python").write_text(STUB_PYTHON)
    for path in [*(stubs / "bin").iterdir(), *(code / ".venv" / "bin").iterdir()]:
        path.chmod(0o755)
    study = tmp_path / "passk_cont_v1"
    for s in ("mancer", "parasail"):
        (study / "artifacts" / "shards" / s / "control").mkdir(parents=True)
    (study / "artifacts" / "shards" / "assignment.json").write_text("{}")
    env = {**os.environ, "CODE": str(code), "STUDY": str(study), "STUBS": str(stubs), "POLL": "1",
           "WAIT_POLL": "0", "SUBMIT_RETRY_S": "0", "LEASE_WAIT": "0",
           "REAL_PY": sys.executable, "PATH": f"{stubs / 'bin'}:{os.environ['PATH']}",
           "PYTHONPATH": str(PROJECT / "src"), "SLURM_JOB_ID": "999",
           "CFG": str(PROJECT / "configs" / "passk_cont_v1.json")}
    return code, study, stubs, env


def _run(code, env, **extra):
    return subprocess.run(["bash", str(code / "slurm" / "passk_conductor.sh")], env={**env, **extra},
                          capture_output=True, text=True, timeout=300)


def _submitted(stubs):
    return [line.split()[1:] for line in (stubs / "submitted").read_text().splitlines()]


def test_conductor_full_chain_dry_run(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    proc = _run(code, env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert (study / "conductor.state").read_text().strip() == "done"
    status = (study / "STATUS.md").read_text()
    assert "state: `done`" in status
    subs = _submitted(stubs)
    verbs = [s[0] for s in subs]
    assert verbs[:2] == ["canary", "canary"]
    assert verbs.count("smoke") == 2 and verbs.count("grade") == 2
    assert verbs.count("run") == 2 and verbs.count("grade-follow") == 2
    stages = [s[-1] for s in subs if s[0] == "analyze"]
    post = ["release", "pairs", "extract", "score", "report"]
    assert stages == ["extract", "pairs", "extract", "e1-lock", "e2-fit", "e5-fit", "seal", *post, *post]
    exports = [line.split()[1:] for line in (stubs / "exports").read_text().splitlines()]
    analyze = [e for e in exports if e[0] in stages]
    fits = {e[0]: e[1] for e in analyze if e[0] in ("e2-fit", "e5-fit")}
    assert fits == {"e2-fit": "flags=--sandbox", "e5-fit": "flags=--sandbox"}      # Gate P: the bwrap fence
    assert not any("strace" in e[1] for e in analyze)
    releases = [e[2] for e in analyze if e[0] == "release"]
    assert releases == ["release=v48", "release=t200"]                              # one job per split
    assert all(e[2] == "release=" for e in analyze if e[0] != "release")
    inline = (stubs / "inline").read_text()
    assert "passk.gates gate0" in inline and "--check-key" in inline and "--run-tests" in inline
    assert inline.count("canary") >= 2 and "--evaluate-only" in inline
    assert inline.count("--adopt-legacy-lock") == 2
    assert "shard-plan" not in inline                            # assignment.json exists: not re-planned
    for s in ("mancer", "parasail"):
        assert (study / "artifacts" / "shards" / s / "control" / "STOP").exists()


def test_conductor_stops_on_gate0_failure(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    proc = _run(code, env, STUB_GATE0_RC="3")
    assert proc.returncode == 3
    assert (study / "conductor.state").read_text().startswith("stopped:gate0")
    assert not (stubs / "submitted").exists()                   # nothing paid was submitted


def test_conductor_stops_on_semantic_canary_failure(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    gate = study / "artifacts" / "shards" / "mancer" / "gates"
    gate.mkdir(parents=True)
    detail = ("GateResult(criteria=(GateCriterion(name='logit_bias_honoured_banana', observed='{\"status\": \"ok\", "
              "\"banana_count\": 1}', threshold='>= 5', passed=False, evidence='x'),))")
    (gate / "gate0_endpoint.json").write_text(json.dumps({"passed": False, "detail": detail}))
    proc = _run(code, env, STUB_FAIL_VERB="canary")
    assert proc.returncode == 3, proc.stdout[-2000:]
    assert (study / "conductor.state").read_text().startswith("stopped:gate1")
    assert "semantically" in proc.stdout
    assert "run" not in [s[0] for s in _submitted(stubs)]


def test_conductor_refuses_a_second_instance(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    study.mkdir(parents=True, exist_ok=True)
    lock = study / "conductor.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "30"])
    try:
        import time
        time.sleep(1)
        proc = _run(code, env)
        assert proc.returncode == 1 and "another conductor" in proc.stdout
    finally:
        holder.kill()


def test_conductor_resumes_at_a_post_seal_step(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    proc = _run(code, env, START_AT="score-t200")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert [s[-1] for s in _submitted(stubs) if s[0] == "analyze"] == ["score", "report"]


def test_conductor_resumes_at_seal(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    proc = _run(code, env, START_AT="seal")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    stages = [s[-1] for s in _submitted(stubs) if s[0] == "analyze"]
    assert stages[0] == "seal" and stages.count("release") == 2
    assert [s[0] for s in _submitted(stubs)].count("grade-follow") == 2       # resume ensures the graders run


def test_background_analysis_failure_stops_during_generation(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    proc = _run(code, env, START_AT="main", STUB_FAIL_STAGE="e1-lock", STUB_STAGE1="1", STUB_RUN_RUNNING="1")
    assert proc.returncode == 3, proc.stdout[-3000:]
    assert (study / "conductor.state").read_text().startswith("stopped:analysis:e1-lock")
    assert "ANALYSIS FAILED" in (study / "STATUS.md").read_text()
    for s in ("mancer", "parasail"):
        assert (study / "artifacts" / "shards" / s / "control" / "DRAIN").exists()


def test_conductor_rejects_unknown_start_at(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    proc = _run(code, env, START_AT="bogus")
    assert proc.returncode == 2
    assert not (stubs / "submitted").exists()


def test_gate4_failure_of_one_split_releases_the_other(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    proc = _run(code, env, START_AT="gate4", STUB_GATE4_FAIL_SPLIT="v48")
    assert proc.returncode == 3, proc.stdout[-3000:]
    assert (study / "conductor.state").read_text().startswith("stopped:gate4")
    exports = [line.split()[1:] for line in (stubs / "exports").read_text().splitlines()]
    assert [e[0] for e in exports if e[0] in ("release", "pairs", "extract", "score", "report")] == \
        ["release", "pairs", "extract", "score", "report"]
    assert [e[2] for e in exports if e[0] == "release"] == ["release=t200"]


def test_conductor_resumes_at_smoke_without_canaries(tmp_path):
    code, study, stubs, env = _setup(tmp_path)
    (study / "artifacts" / "shards" / "mancer" / "control" / "DRAIN").touch()     # a previous stop's leftover
    proc = _run(code, env, START_AT="smoke")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    verbs = [s[0] for s in _submitted(stubs)]
    assert "canary" not in verbs and verbs[:2] == ["smoke", "smoke"] and verbs.count("run") == 2
