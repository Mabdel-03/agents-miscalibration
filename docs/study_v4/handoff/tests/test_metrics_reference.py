"""Reference invariants; these tests do not validate a deployed experiment."""

from dataclasses import replace
import importlib.util
import inspect
import itertools
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "metrics_reference.py"
SPEC = importlib.util.spec_from_file_location("agent_design_v4_metrics_reference", MODULE_PATH)
assert SPEC and SPEC.loader
metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


class PassAtKTests(unittest.TestCase):
    def test_known_values_and_boundary_cases(self):
        self.assertAlmostEqual(metrics.pass_at_k(10, 2, 5), 7 / 9)
        self.assertEqual(metrics.pass_at_k(10, 0, 5), 0.0)
        self.assertEqual(metrics.pass_at_k(10, 10, 5), 1.0)
        self.assertEqual(metrics.pass_at_k(10, 1, 10), 1.0)
        self.assertEqual(metrics.pass_at_k(1, 0, 1), 0.0)

    def test_unbiasedness_under_binomial_complete_attempts(self):
        for n in (1, 5, 10):
            for k in range(1, n + 1):
                for p in (0.1, 0.3, 0.8):
                    expectation = sum(
                        math.comb(n, c) * p**c * (1 - p)**(n - c) * metrics.pass_at_k(n, c, k)
                        for c in range(n + 1)
                    )
                    self.assertAlmostEqual(expectation, 1 - (1 - p)**k, places=12)

    def test_task_averaging_preserves_heterogeneity_and_failed_banks(self):
        rows = [[False] * 10, [True] * 10]
        self.assertEqual(metrics.mean_task_pass_at_k(rows, 5), 0.5)
        self.assertNotAlmostEqual(metrics.mean_task_pass_at_k(rows, 5), 1 - 0.5**5)

    def test_invalid_inputs_are_not_silently_repaired(self):
        for values in [(0, 0, 1), (3, 4, 1), (3, 1, 4), (3, 1, 0), (True, 0, 1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                metrics.pass_at_k(*values)
        with self.assertRaises(ValueError):
            metrics.mean_task_pass_at_k([[True, None]], 1)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = {"source_id": "task-1", "tie_seed": b"sealed-tie-seed"}

    def test_equal_answers_keep_multiple_votes(self):
        pool = [metrics.PublicCandidate("a", "A"), metrics.PublicCandidate("b", "A"),
                metrics.PublicCandidate("c", "B")]
        selection = metrics.plurality_vote(pool, **self.kwargs)
        self.assertIn(selection.candidate_id, {"a", "b"})
        self.assertEqual(selection.winning_vote_count, 2)
        self.assertEqual(selection.tied_winning_classes, 1)

    def test_ties_are_deterministic_and_order_independent(self):
        pool = [metrics.PublicCandidate("a", "A"), metrics.PublicCandidate("b", "B"),
                metrics.PublicCandidate("c", "C")]
        selections = {metrics.plurality_vote(order, **self.kwargs).candidate_id
                      for order in itertools.permutations(pool)}
        self.assertEqual(len(selections), 1)
        relabeled = [replace(candidate, vote_key=f"new-{candidate.vote_key}") for candidate in pool]
        self.assertEqual(metrics.plurality_vote(pool, **self.kwargs).candidate_id,
                         metrics.plurality_vote(relabeled, **self.kwargs).candidate_id)

    def test_duplicate_ids_are_not_duplicate_votes(self):
        candidate = metrics.PublicCandidate("same-opportunity", "A")
        with self.assertRaises(ValueError):
            metrics.plurality_vote([candidate, candidate], **self.kwargs)

    def test_all_invalid_remains_failed_item_and_full_denominator(self):
        pool = [metrics.PublicCandidate(f"bad-{j}", None, False) for j in range(5)]
        selection = metrics.plurality_vote(pool, **self.kwargs)
        result = metrics.evaluate_sealed_selection(pool, selection, {})
        self.assertIsNone(selection.candidate_id)
        self.assertEqual(result.planned_count, 5)
        self.assertEqual(result.valid_count, 0)
        self.assertEqual(result.candidate_mean_correctness, 0.0)
        self.assertFalse(result.oracle_coverage)
        self.assertFalse(result.selected_correctness)

    def test_invalid_opportunities_are_retained_in_candidate_mean(self):
        pool = [metrics.PublicCandidate("good", "A"), metrics.PublicCandidate("bad", None, False)]
        selected = metrics.plurality_vote(pool, **self.kwargs)
        result = metrics.evaluate_sealed_selection(pool, selected, {"good": True})
        self.assertEqual(result.candidate_mean_correctness, 0.5)

    def test_selector_cannot_use_protected_correctness_argument(self):
        for function in (metrics.plurality_vote, metrics.judge_best):
            self.assertNotIn("protected_correctness", inspect.signature(function).parameters)
        pool = [metrics.PublicCandidate("wrong-1", "A"), metrics.PublicCandidate("wrong-2", "A"),
                metrics.PublicCandidate("correct", "B")]
        selected = metrics.plurality_vote(pool, **self.kwargs)
        result = metrics.evaluate_sealed_selection(
            pool, selected, {"wrong-1": False, "wrong-2": False, "correct": True})
        self.assertTrue(result.oracle_coverage)
        self.assertFalse(result.selected_correctness)
        self.assertEqual(result.selection_gap, 1)
        with self.assertRaises(TypeError):
            metrics.plurality_vote(pool, protected_correctness={"correct": True}, **self.kwargs)

    def test_judge_ranking_is_separate_and_requires_public_scores(self):
        pool = [metrics.PublicCandidate("a", "A"), metrics.PublicCandidate("b", "B")]
        result = metrics.judge_best(pool, {"a": 0.1, "b": 0.9}, **self.kwargs)
        self.assertEqual(result.candidate_id, "b")
        self.assertIsNone(result.winning_vote_count)
        with self.assertRaises(ValueError):
            metrics.judge_best(pool, {"a": True, "b": False}, **self.kwargs)
        with self.assertRaises(ValueError):
            metrics.judge_best(pool, {"a": 0.9}, **self.kwargs)

    def test_evaluation_rejects_selector_output_outside_its_bank(self):
        pool = [metrics.PublicCandidate("a", "A")]
        illegal = metrics.Selection("oracle-replacement", 1, 1, None, None)
        with self.assertRaises(ValueError):
            metrics.evaluate_sealed_selection(pool, illegal, {"a": False})

    def test_judge_rejects_out_of_contract_scores(self):
        pool = [metrics.PublicCandidate("a", "A")]
        for value in [-0.1, 1.1, "0.9", float("nan"), float("inf"), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                metrics.judge_best(pool, {"a": value}, **self.kwargs)


class ExactAliasTests(unittest.TestCase):
    def setUp(self):
        self.request = metrics.FrozenRequest(
            source_id="task-1", study_id="agent_design_v4", model_revision="model-rev",
            tokenizer_revision="tokenizer-rev", chat_template_hash="template-hash",
            engine_digest="engine-digest", input_utf8=b"neutral task prompt",
            input_token_ids=(10, 20, 30), canonical_decoding=b'{"temperature":0.6}',
            semantic_seed=17, canonical_caps=b'{"max_tokens":8192}', precision="bfloat16",
            hook_hash="no-hooks")

    def test_process_and_agent_labels_alias_same_ordered_bank(self):
        canonical = [metrics.BankEntry(f"agent-{j}", replace(self.request, semantic_seed=j)) for j in range(5)]
        requested = [metrics.BankEntry(f"reset-{j}", entry.request) for j, entry in enumerate(canonical)]
        self.assertEqual(metrics.exact_bank_alias(canonical, requested),
                         {f"reset-{j}": f"agent-{j}" for j in range(5)})

    def test_changed_prompt_seed_cap_or_runtime_never_aliases(self):
        canonical = [metrics.BankEntry("canonical", self.request)]
        changes = [
            {"input_utf8": b"You are a team member. neutral task prompt"},
            {"semantic_seed": 18},
            {"canonical_caps": b'{"max_tokens":4096}'},
            {"engine_digest": "different-engine"},
            {"input_token_ids": (10, 20, 31)},
            {"hook_hash": "activation-edit"},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                metrics.exact_bank_alias(canonical, [metrics.BankEntry("alias", replace(self.request, **change))])

    def test_alias_requires_same_prefix_order(self):
        a = metrics.BankEntry("a", self.request)
        b = metrics.BankEntry("b", replace(self.request, semantic_seed=18))
        with self.assertRaises(ValueError):
            metrics.exact_bank_alias([a, b], [b, a])

    def test_unresolved_pins_cannot_prove_exact_request_identity(self):
        with self.assertRaises(ValueError):
            replace(self.request, model_revision=None)
        with self.assertRaises(ValueError):
            replace(self.request, engine_digest="")


if __name__ == "__main__":
    unittest.main()
