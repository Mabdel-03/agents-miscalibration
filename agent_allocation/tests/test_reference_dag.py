import importlib.util
import sys
import unittest
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'tools'/'reference_dag.py'
spec=importlib.util.spec_from_file_location('reference_dag',path)
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)

class PlanTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(m.build_plan().nodes),125)
        self.assertEqual(len(m.build_plan(a=4096,high=True).nodes),123)
        self.assertEqual(len(m.build_plan(reduced=True).nodes),57)
        self.assertEqual(len(m.build_plan(extension=True).nodes),138)
    def test_ancestor_costs(self):
        p=m.build_plan(extension=True)
        for cp,count in [('sample/25',25),('private_single/25',25),('peer/5/4',25),
                         ('peer/4/4',20),('private_team/4/4',20),('cp/3',20),('pooled/4',5)]:
            self.assertEqual(len(p.ancestors(cp)),count,cp)
    def test_roots_shared_explicitly(self):
        p=m.build_plan()
        self.assertEqual(p.checkpoints['sample/5'],p.checkpoints['peer/5/0'])
        self.assertEqual(p.checkpoints['sample/5'],p.checkpoints['private_team/5/0'])
    def test_no_future_or_same_round_peer(self):
        p=m.build_plan()
        for n in (2,3,4,5):
            for r in range(1,5):
                for key in p.checkpoints[f'peer/{n}/{r}']:
                    self.assertEqual(p.nodes[key].parents,p.checkpoints[f'peer/{n}/{r-1}'])
    def test_independent_draws_distinct(self):
        p=m.build_plan()
        self.assertEqual(len(set(p.checkpoints['sample/25'])),25)
    def test_cp_private_after_first_peer(self):
        p=m.build_plan(extension=True)
        for h in range(1,4):
            for j,key in enumerate(p.checkpoints[f'cp/{h}']):
                self.assertEqual(p.nodes[key].parents,(p.checkpoints[f'cp/{h-1}'][j],))
    def test_budget_not_clipped(self):
        self.assertEqual(m.build_plan(a=4096,high=True).nodes['deep/4'].reasoning_budget,16384)
    def test_illegal_profiles(self):
        with self.assertRaises(ValueError): m.build_plan(high=True,reduced=True)

if __name__=='__main__': unittest.main()
