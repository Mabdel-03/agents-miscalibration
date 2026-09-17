"""Artificial full-upstream EvalPlus comparison parity (the 20 fixtures).

Vendored from ``learnagents/data/evalplus_validation.py`` (agent_methods_bench@2af2879, sha256 0d3c0576…).
``fixtures()`` is unchanged. ``validate()`` runs the official upstream ``untrusted_check`` of the pinned EvalPlus
source in bwrap (no seccomp: upstream forks a child per batch; fixtures are artificial trusted programs, still never
run on the host) and the vendored wrapper ``grade_evalplus`` at the supplied sandbox profile, then compares status
and passed counts per fixture. Changes: certified interpreter prefix instead of ``sys.prefix``; packaged upstream
root; psutil taken read-only from the pinned agent_methods_bench evalplus venv (upstream imports it); private record
built without learnagents ``adapt_evalplus``; profile arguments; report returned (the caller writes it); reference
certification drivers dropped (see ``grading/code/certify.py``).
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from agent_allocation._vendor.learnagents_io import content_hash
from .evalplus_scorer import ROOT,REVISION,PINS,TRANSPORT,pack_native,native_record,grade_evalplus
from .sandbox import CERTIFIED_PYTHON_PREFIX

EVALPLUS_EXTRAS = Path("/orcd/data/tpoggio/001/mabdel03/agent_methods_bench/.venvs/evalplus/lib/python3.11/site-packages")


def fixtures():
    return [
      {'name':'exact','dataset':'humaneval','entry':'f','source':'def f(x):return x+1','inputs':[[1]],'expected':[2],'atol':0},
      {'name':'wrong','dataset':'humaneval','entry':'f','source':'def f(x):return x+2','inputs':[[1]],'expected':[2],'atol':0},
      {'name':'float_auto_tolerance','dataset':'humaneval','entry':'f','source':'def f(x):return 1.00000001','inputs':[[1]],'expected':[1.0],'atol':0},
      {'name':'float_container_type','dataset':'humaneval','entry':'f','source':'def f(x):return (1.0,)','inputs':[[1]],'expected':[[1.0]],'atol':0},
      {'name':'set_equivalence','dataset':'mbpp','entry':'similar_elements','source':'def similar_elements(x):return [2,1]','inputs':[[1]],'expected':[[1,2]],'atol':0},
      {'name':'sum_div_alternative','dataset':'mbpp','entry':'sum_div','source':'def sum_div(x):return 0','inputs':[[1]],'expected':[9],'atol':0},
      {'name':'are_equivalent_native_exception','dataset':'mbpp','entry':'are_equivalent','source':'def are_equivalent(x):return False','inputs':[[1]],'expected':[True],'atol':0},
      {'name':'none_projection','dataset':'mbpp','entry':'check_str','source':'def check_str(x):return None','inputs':[[1]],'expected':[False],'atol':0},
      {'name':'boolean_not_none_exception','dataset':'mbpp','entry':'check_str','source':'def check_str(x):return False','inputs':[[1]],'expected':[True],'atol':0},
      {'name':'find_zero_alternative','dataset':'humaneval','entry':'find_zero','source':'def find_zero(xs):return 1.0','inputs':[[[1,-1]]],'expected':[999.0],'atol':1e-6},
      {'name':'find_zero_mutated_input_native_behavior','dataset':'humaneval','entry':'find_zero','source':'def find_zero(xs):\n xs[0]=0\n return 0.0','inputs':[[[1,1]]],'expected':[-1.0],'atol':1e-6},
      {'name':'surface_area_alternative','dataset':'mbpp','entry':'surface_Area','source':'def surface_Area(a,b):return 16','inputs':[[2,3]],'expected':[999],'atol':0},
      {'name':'digit_distance_padding','dataset':'mbpp','entry':'digit_distance_nums','source':'def digit_distance_nums(a,b):return 7','inputs':[[12,3]],'expected':[999],'atol':0},
      {'name':'complex','dataset':'humaneval','entry':'f','source':'def f(x):return complex(1,2)','inputs':[[1]],'expected':[complex(1,2)],'atol':0},
      {'name':'persistent_globals_within_batch','dataset':'humaneval','entry':'f','source':'count=0\ndef f(x):\n global count\n count+=1\n return count','inputs':[[0],[0]],'expected':[1,2],'atol':0},
      {'name':'large_integer_transport','dataset':'humaneval','entry':'f','source':'def f(x):return 2**x','inputs':[[20000]],'expected':[2**20000],'atol':0},
      {'name':'large_string_tuple_rows_exact','dataset':'mbpp','entry':'f','source':'def f(x):return [(str(i%81),str((i+1)%81)) for i in range(x)]','inputs':[[17000]],'expected':[[(str(i%81),str((i+1)%81)) for i in range(17000)]],'atol':0},
      {'name':'large_string_tuple_rows_wrong','dataset':'mbpp','entry':'f','source':'def f(x):return [(str(i%81),str((i+2)%81)) for i in range(x)]','inputs':[[17000]],'expected':[[(str(i%81),str((i+1)%81)) for i in range(17000)]],'atol':0},
      {'name':'large_numeric_tuple_rows_int_float_equality','dataset':'mbpp','entry':'f','source':'def f(x):return [(float(i%10),float(i%3)) for i in range(x)]','inputs':[[17000]],'expected':[[(i%10,i%3) for i in range(17000)]],'atol':0},
      {'name':'large_numeric_tuple_rows_tolerance','dataset':'mbpp','entry':'f','source':'def f(x):return [(float(i%10)+1e-7,float(i%3)) for i in range(x)]','inputs':[[17000]],'expected':[[(i%10,i%3) for i in range(17000)]],'atol':1e-6},
    ]


def official_results(cases, *, python_prefix=CERTIFIED_PYTHON_PREFIX, extras=EVALPLUS_EXTRAS):
    """Official upstream untrusted_check verdicts for the fixtures, computed inside bwrap."""
    payload=[{**f,'inputs':pack_native(f['inputs']),'expected':pack_native(f['expected'])} for f in cases]
    worker=TRANSPORT+'''\nimport json,sys\nsys.path.insert(0,'/upstream')\nsys.path.insert(0,'/extras')\nfrom evalplus.eval import untrusted_check\nresults=[]\nfor case in json.load(sys.stdin):\n inputs=unpack_native(case['inputs']);expected=unpack_native(case['expected'])\n status,details=untrusted_check(case['dataset'],case['source'],inputs,case['entry'],expected,case['atol'],[.001]*len(inputs),fast_check=False)\n results.append({'name':case['name'],'status':status,'details':[bool(x) for x in details]})\nprint('NATIVE_RESULT='+json.dumps(results))\n'''
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'worker.py';path.write_text(worker)
        prefix=str(Path(python_prefix).resolve())
        args=[shutil.which('bwrap'),'--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
              '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
              '--ro-bind',prefix,'/runtime','--ro-bind',str(ROOT),'/upstream','--ro-bind',str(path),'/worker.py',
              '--ro-bind',str(Path(extras).resolve()),'/extras',
              '--proc','/proc','--dev','/dev','--tmpfs','/dev/shm','--tmpfs','/tmp','--tmpfs','/work','--chdir','/work',
              '/runtime/bin/python','-I','/worker.py']
        result=subprocess.run(args,input=json.dumps(payload),text=True,capture_output=True,timeout=180,
                    env={'PATH':'/runtime/bin:/usr/bin:/bin','LD_LIBRARY_PATH':'/runtime/lib','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        if result.returncode:raise RuntimeError('Full native artificial fixture process failed: '+(result.stderr.splitlines() or ['?'])[-1])
    markers=[line.removeprefix('NATIVE_RESULT=') for line in result.stdout.splitlines() if line.startswith('NATIVE_RESULT=')]
    if len(markers)!=1:raise RuntimeError('Missing unique full-native fixture result')
    return {r['name']:r for r in json.loads(markers[0])}


def fixture_private(case):
    native_id='Mbpp/900' if case['dataset']=='mbpp' else 'HumanEval/900'
    row={'task_id':native_id,'prompt':'','canonical_solution':case['source'],'entry_point':case['entry'],
         'base_input':case['inputs'],'plus_input':[],'atol':case['atol']}
    return {'problem_id':'fixture_'+case['name'],'native_scorer':'evalplus','native_record':row}


def validate(*, python_prefix=CERTIFIED_PYTHON_PREFIX, memory_bytes=4*1024**3, output_bytes=64*1024,
             stack_bytes=None, suite_wall_seconds=60.0):
    cases=fixtures()
    official=official_results(cases,python_prefix=python_prefix);rows=[]
    for case in cases:
        private=fixture_private(case)
        decoded,_=native_record(private)
        oracle={'restricted_record_sha256':content_hash(private),'native_revision':REVISION,'batches':{
                 'base':{'inputs_sha256':content_hash(pack_native(decoded['base_input'])),'expected':[pack_native(x) for x in case['expected']], 'reference_times':[.001]*len(case['inputs'])},
                 'plus':{'inputs_sha256':content_hash(pack_native([])),'expected':[],'reference_times':[]}}}
        wrapped=grade_evalplus(case['source'],private,oracle,python_prefix=python_prefix,memory_bytes=memory_bytes,
                               output_bytes=output_bytes,stack_bytes=stack_bytes or memory_bytes,
                               suite_wall_seconds=suite_wall_seconds)
        target=official[case['name']]
        rows.append({'name':case['name'],'official_status':target['status'],'official_passed':sum(target['details']),
                     'wrapper_success':wrapped.success,'wrapper_passed':wrapped.details['base']['passed'],
                     'match':wrapped.success==(target['status']=='pass') and wrapped.details['base']['passed']==sum(target['details'])})
    return {'status':'passed' if all(r['match'] for r in rows) else 'blocked_native_metric_mismatch','fixtures_are_artificial':True,
            'count':len(rows),'results':rows,'native_revision':REVISION,'source_hashes':PINS,
            'profile':{'memory_bytes':memory_bytes,'output_bytes':output_bytes,'stack_bytes':stack_bytes or memory_bytes,
                       'suite_wall_seconds':suite_wall_seconds},
            'candidate_processes_receive_expected_outputs':False}
