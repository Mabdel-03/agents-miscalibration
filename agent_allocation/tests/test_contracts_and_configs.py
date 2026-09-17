import importlib.util
import sys
import unittest
import json
from dataclasses import FrozenInstanceError, replace, fields
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'shared'/'contracts.py'
spec=importlib.util.spec_from_file_location('allocation_contracts',p)
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)

class ContractTests(unittest.TestCase):
    def node(self):
        return c.GenerationSpec(run_id='test',partition_id='main',item_id='q1',
            benchmark='math500',kind=c.GenerationKind.ROOT,model='test-model',
            endpoint_lock_hash='endpoint',input_manifest_hash='manifest',
            decoding_config_hash='decode',prompt_version='v1',reasoning_budget=1024,
            completion_headroom=1024,root_slot=0,team_size=0,update_round=0,draw_id=0)
    def test_identity_contains_draw_and_kind(self):
        n=self.node()
        self.assertNotEqual(n.node_id,replace(n,draw_id=1).node_id)
        self.assertNotEqual(n.node_id,replace(n,kind=c.GenerationKind.PRIVATE).node_id)
    def test_identity_contains_decoding_and_manifest(self):
        n=self.node()
        self.assertNotEqual(n.node_id,replace(n,decoding_config_hash='new').node_id)
        self.assertNotEqual(n.node_id,replace(n,input_manifest_hash='new').node_id)
    def test_immutable(self):
        with self.assertRaises(FrozenInstanceError): self.node().draw_id=2
    def test_public_task_has_no_private_fields(self):
        names={f.name for f in fields(c.PublicTask)}
        self.assertFalse(names & {'answer','answer_index','gold','canonical_solution','plus_input','private_tests'})
    def test_headroom_and_invalid_budget(self):
        self.assertEqual(self.node().max_completion_tokens,2048)
        with self.assertRaises(ValueError): replace(self.node(),reasoning_budget=0)
    def test_panel_balance(self):
        for name in ('balanced150','lean100'):
            cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text())
            for count in cfg['panels'].values(): self.assertEqual(count%16,0)
            self.assertEqual(len(cfg['benchmarks']),4)
            self.assertTrue(cfg['execution']['require_allow_paid_flag'])
            self.assertFalse(cfg['provider']['allow_fallbacks'])
            self.assertEqual(cfg['execution']['default_mode'],'dry_run')

if __name__=='__main__': unittest.main()
