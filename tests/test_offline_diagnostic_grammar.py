from __future__ import annotations

from typing import Callable

import pytest

from scripts import seal_recovery_evidence as evidence
from scripts import seal_schema5_r9_prelaunch_failure as r9


# Two modules parse the same Conda offline-probe diagnostic: the r9 sealer checks the
# equivalent reproduction, and seal_recovery_evidence checks the r3 probe.  Each spells
# the grammar out separately, so a delimiter rule fixed in one did not reach the other
# -- r11 repaired the r9 copy and r13 then failed on the identical defect in the r3
# copy.  These tests drive both copies with the same corpus and require the same
# verdict, so the two can no longer drift apart.

BASE = "/results/recovery/schema5-v1/toolchains/r14/conda/base"
DESTINATION = "/tmp/schema5-r3-prelaunch-tester/offline-clone-destination"

# The progress body between "...working..." and " done" must be non-empty and drawn
# from the spinner alphabet both parsers accept.
CANONICAL_STDOUT = (
    f"Source:      {BASE}\n"
    f"Destination: {DESTINATION}\n"
    "Packages: 89\n"
    "Files: 3\n\n"
    "Downloading and Extracting Packages: ...working..."
    "\r|\r-\r|\r-"
    " done\n"
)

_URL = (
    "https://conda.anaconda.org/conda-forge/linux-64/"
    "libgcc-ng-14.2.0-h69a702a_1.conda"
)
CANONICAL_BLOCK = (
    f"OfflineError: EnforceUnusedAdapter called with url {_URL}.\n"
    "This command is using a remote connection in offline mode.\n"
)


def _r9_verdict(stdout: str, stderr: str) -> bool:
    contract = {"argv": [None] * 5 + [BASE], "destination": DESTINATION}
    try:
        r9._validate_structural_diagnostic(
            stdout=stdout, stderr=stderr, contract=contract
        )
    except Exception:
        return False
    return True


def _evidence_verdict(stdout: str, stderr: str) -> bool:
    contract = {"toolchain_base": BASE, "offline_destination": DESTINATION}
    try:
        evidence._validate_r3_probe_failure_signature(
            classification="offline_clone_unseeded_release_local_cache",
            returncode=1,
            stdout=stdout,
            stderr=stderr,
            contract=contract,
        )
    except Exception:
        return False
    return True


PARSERS: dict[str, Callable[[str, str], bool]] = {
    "r9_equivalent_reproduction": _r9_verdict,
    "r3_probe_signature": _evidence_verdict,
}

STDERR_CORPUS = {
    "one_block": CANONICAL_BLOCK,
    "two_blocks": CANONICAL_BLOCK * 2,
    "one_leading_blank": "\n" + CANONICAL_BLOCK,
    "one_trailing_blank": CANONICAL_BLOCK + "\n",
    "one_leading_and_one_trailing_blank": "\n" + CANONICAL_BLOCK + "\n",
    "two_leading_blanks": "\n\n" + CANONICAL_BLOCK,
    "two_trailing_blanks": CANONICAL_BLOCK + "\n\n",
    "empty": "",
    "unmatched_prose": "something else entirely\n",
    "block_then_prose": CANONICAL_BLOCK + "trailing prose\n",
}


@pytest.mark.parametrize("name", sorted(STDERR_CORPUS))
def test_both_offline_diagnostic_parsers_agree(name: str) -> None:
    stderr = STDERR_CORPUS[name]
    verdicts = {
        parser: decide(CANONICAL_STDOUT, stderr)
        for parser, decide in PARSERS.items()
    }
    assert len(set(verdicts.values())) == 1, (
        f"offline-diagnostic parsers disagree on {name!r}: {verdicts}"
    )


@pytest.mark.parametrize("parser", sorted(PARSERS))
@pytest.mark.parametrize(
    "name",
    [
        "one_block",
        "two_blocks",
        "one_leading_blank",
        "one_trailing_blank",
        "one_leading_and_one_trailing_blank",
    ],
)
def test_canonical_streams_are_accepted(parser: str, name: str) -> None:
    # Conda 25.11 delimits the block stream with at most one leading and one trailing
    # blank line.  Rejecting those is exactly what stalled r9, r11, and r13.
    assert PARSERS[parser](CANONICAL_STDOUT, STDERR_CORPUS[name]) is True


@pytest.mark.parametrize("parser", sorted(PARSERS))
@pytest.mark.parametrize(
    "name",
    ["two_leading_blanks", "two_trailing_blanks", "empty", "unmatched_prose",
     "block_then_prose"],
)
def test_non_canonical_streams_are_rejected(parser: str, name: str) -> None:
    assert PARSERS[parser](CANONICAL_STDOUT, STDERR_CORPUS[name]) is False
