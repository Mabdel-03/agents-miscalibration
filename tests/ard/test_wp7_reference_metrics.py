"""The bundle's ``test_reference_metrics.py``, vendored (plan §10 "bundle
``test_reference_metrics.py`` vendored").

Everything below the parity test is the bundle file verbatim except its import line, which is
re-pointed at :mod:`agents_scaling.ard.reference_metrics` (the evaluator-namespace vendoring of
``Agents_Round_Dynamics_36h/reference_metrics.py``).  The first test pins the vendored module to
the bundle **byte for byte**, so the copy cannot drift from the artifact the spec ships.
"""

from __future__ import annotations

import unittest
import zipfile
from pathlib import Path

from agents_scaling.ard import reference_metrics as RM
from agents_scaling.ard.reference_metrics import (
    correct_count_pmf,
    heterogeneous_correct_count_pmf,
    iid_correct_count_pmf,
    pass_at_k,
    plurality_expected_correctness,
    request_digest,
    transition_counts,
    within_item_auc,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "expermt_agent_specs" / "AgentsRoundV1" / "Agents_Round_Dynamics_36h_Bundle.zip"
BUNDLE_ENTRY = "Agents_Round_Dynamics_36h/reference_metrics.py"


def test_vendored_module_is_the_bundle_file_byte_for_byte():
    with zipfile.ZipFile(BUNDLE) as archive:
        original = archive.read(BUNDLE_ENTRY)
    assert Path(RM.__file__).read_bytes() == original


def test_vendored_module_keeps_the_runtime_prohibition_in_its_docstring():
    """The bundle docstring is this module's own firewall statement (contract §0.2)."""
    assert (
        "must not be imported into runtime controller feature extraction with gold data."
        in RM.__doc__
    )


# --------------------------------------------------------------------------- bundle verbatim


class MetricChecks(unittest.TestCase):
    def test_pass_is_coverage_not_count(self):
        self.assertEqual(pass_at_k(5,1,5), 1.0)
        self.assertEqual(pass_at_k(5,4,5), 1.0)
        self.assertAlmostEqual(pass_at_k(5,1,1), 0.2)

    def test_hypergeometric(self):
        pmf = correct_count_pmf(16,4,5)
        self.assertAlmostEqual(sum(pmf), 1.0)
        self.assertAlmostEqual(1-pmf[0], pass_at_k(16,4,5))
        self.assertAlmostEqual(sum(i*p for i,p in enumerate(pmf)), 5*4/16)

    def test_iid_example(self):
        pmf = iid_correct_count_pmf(0.2,5)
        self.assertAlmostEqual(1-pmf[0], 0.67232)
        self.assertAlmostEqual(pmf[1], 0.4096)
        self.assertAlmostEqual(sum(pmf[3:]), 0.05792)

    def test_heterogeneous(self):
        self.assertEqual(heterogeneous_correct_count_pmf([0,1]), [0,1,0])
        a = heterogeneous_correct_count_pmf([0.2]*5)
        b = iid_correct_count_pmf(0.2,5)
        for x,y in zip(a,b):
            self.assertAlmostEqual(x,y)

    def test_round_accounting(self):
        self.assertEqual(transition_counts([1,0,0,1,0],[0,1,0,1,1]),
                         {'CC':1,'CW':1,'WC':2,'WW':1})

    def test_within_item_auc(self):
        self.assertEqual(within_item_auc([.7,.7], [0,1]), .5)
        self.assertEqual(within_item_auc([.9,.2], [1,0]), 1.0)
        self.assertIsNone(within_item_auc([.9,.2], [0,0]))

    def test_plurality_fragmentation_and_k1(self):
        self.assertEqual(plurality_expected_correctness(['A'],[1]), 1)
        self.assertEqual(plurality_expected_correctness([None],[0]), 0)
        self.assertEqual(plurality_expected_correctness(['A','B','B','B','B'],[1,0,0,0,0]), 0)
        self.assertAlmostEqual(plurality_expected_correctness(['A','B','C','D','E'],[1,0,0,0,0]), .2)
        with self.assertRaises(ValueError):
            plurality_expected_correctness(['A','A'],[1,0])

    def test_seed_changes_identity(self):
        payload = {'checkpoint_revision':'v','tokenizer_hash':'h','prompt_token_ids':[1,2],
                   'sampler':{'temperature':.6}, 'thinking_budget':1024,'final_budget':256,'seed':1}
        a = request_digest(payload)
        payload['seed'] = 2
        self.assertNotEqual(a, request_digest(payload))

    def test_monotonicity_and_invalid_bank(self):
        values = [pass_at_k(16,4,k) for k in range(1,17)]
        self.assertEqual(values, sorted(values))
        with self.assertRaises(ValueError):
            pass_at_k(5,1,6)


if __name__ == '__main__':
    unittest.main()
