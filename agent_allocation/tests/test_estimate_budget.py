import importlib.util
import sys
import unittest
from decimal import Decimal
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'tools' / 'estimate_budget.py'
spec = importlib.util.spec_from_file_location('estimate_budget', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class BudgetTests(unittest.TestCase):
    def test_full_main_counts(self):
        data = module.estimate('balanced150')
        row = data['blocks'][0]
        self.assertEqual(row['calls'], 50_000)
        self.assertEqual(row['completion_cap_tokens'], 125_849_600)
        self.assertEqual(Decimal(row['estimated_usd']), Decimal('62.839840'))
    def test_high_counts(self):
        row = module.estimate('balanced150')['blocks'][1]
        self.assertEqual(row['calls'], 96 * 123)
    def test_total_counts(self):
        self.assertEqual(module.estimate('balanced150')['generation_calls_excluding_pilot'], 67_040)
    def test_both_presets_fit(self):
        for preset in ('balanced150','lean100'):
            self.assertTrue(module.estimate(preset)['fits_credit_ceiling_under_assumptions'])
    def test_input_sensitivity(self):
        a = Decimal(module.estimate('balanced150',2500)['planned_total_usd'])
        b = Decimal(module.estimate('balanced150',5000)['planned_total_usd'])
        self.assertGreater(b,a)
    def test_invalid_average(self):
        with self.assertRaises(ValueError): module.estimate('balanced150',0)

if __name__ == '__main__': unittest.main()
