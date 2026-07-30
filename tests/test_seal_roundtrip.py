from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_evidence_reference_shapes as audit
from scripts import seal_schema5_r9_prelaunch_failure as r9


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _sealer_names() -> list[str]:
    return [path.name for path in audit.sealer_sources(SCRIPTS_ROOT)]


def test_sealer_discovery_covers_the_whole_release_history() -> None:
    # The audit must widen by itself when a release is added, so it globs rather than
    # naming modules.  Guard the glob so an accidental rename cannot silently empty it.
    names = _sealer_names()
    assert len(names) >= 9
    assert "seal_schema5_r9_prelaunch_failure.py" in names


@pytest.mark.parametrize("module", _sealer_names())
def test_every_sealer_verifies_references_in_the_shape_it_writes(module: str) -> None:
    mismatches = audit.audit_path(SCRIPTS_ROOT / module)
    assert mismatches == [], "\n".join(str(mismatch) for mismatch in mismatches)


def test_r9_transcript_reference_is_built_by_one_shared_constructor() -> None:
    # The r12 release failed because this reference was written through _file_ref and
    # verified through a hand-written literal.  One constructor now serves both paths.
    assert hasattr(r9, "_recorder_transcript_ref")
    source = (SCRIPTS_ROOT / "seal_schema5_r9_prelaunch_failure.py").read_text(
        encoding="utf-8"
    )
    assert source.count("_recorder_transcript_ref(") >= 3  # definition + both callers


# The exact shape that r12 published and then rejected: the writer merged a complete
# file reference with a transcript id, the verifier rebuilt four of the six fields.
R12_DEFECT_SOURCE = '''
def _execute_proof(evidence, recovery):
    return {
        "historical_recorder_transcript": _file_ref(
            transcript_path, description="exact r9 recorder reproduction transcript"
        )
        | {"transcript_id": recorder_transcript["transcript_id"]},
    }


def _verify_proof(evidence, recovery, proof):
    if proof.get("historical_recorder_transcript") != {
        "path": str(transcript_path),
        "sha256": hashlib.sha256(transcript_raw).hexdigest(),
        "size": len(transcript_raw),
        "transcript_id": transcript_id,
    }:
        raise RuntimeError("exact r9 recorder reproduction binding drifted")
'''


def test_audit_detects_the_r12_transcript_reference_defect() -> None:
    mismatches = audit.audit_source(R12_DEFECT_SOURCE, module="r12_defect.py")

    assert len(mismatches) == 1
    mismatch = mismatches[0]
    assert mismatch.field == "historical_recorder_transcript"
    assert mismatch.delta == ("link_count", "mode")
    assert "missing ['link_count', 'mode']" in str(mismatch)


R12_REPAIRED_SOURCE = '''
def _recorder_transcript_ref(transcript_path, transcript_id):
    return _file_ref(
        transcript_path, description="exact r9 recorder reproduction transcript"
    ) | {"transcript_id": transcript_id}


def _execute_proof(evidence, recovery):
    return {
        "historical_recorder_transcript": _recorder_transcript_ref(
            transcript_path, recorder_transcript["transcript_id"]
        ),
    }


def _verify_proof(evidence, recovery, proof):
    if proof.get("historical_recorder_transcript") != _recorder_transcript_ref(
        transcript_path, transcript_id
    ):
        raise RuntimeError("exact r9 recorder reproduction binding drifted")
'''


def test_audit_accepts_the_shared_constructor_repair() -> None:
    assert audit.audit_source(R12_REPAIRED_SOURCE, module="r12_repaired.py") == []


def test_audit_ignores_payload_dicts_that_are_not_references() -> None:
    # Ordinary payload keys must not be mistaken for file references.
    source = '''
def _execute_proof(evidence, recovery):
    return {"counts": {"files": 3, "directories": 1}}


def _verify_proof(evidence, recovery, proof):
    if proof.get("counts") != {"files": 3, "directories": 1, "symlinks": 0}:
        raise RuntimeError("drifted")
'''
    assert audit.audit_source(source, module="payload.py") == []


def test_audit_cli_reports_a_clean_release_history(capsys) -> None:
    assert audit.main(["--scripts-root", str(SCRIPTS_ROOT)]) == 0
    assert "0 mismatches" in capsys.readouterr().out
