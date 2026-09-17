"""Pinned EvalPlus comparisons with all expected outputs outside candidate processes.

Vendored from ``learnagents/data/evalplus_scorer.py`` (agent_methods_bench@2af2879, sha256 d13aa077…, the wrapper
hash recorded by ``manifests/evalplus_reference_certification_v4.json``). The comparison code is still taken by AST
from the hash-pinned upstream EvalPlus ``26d6d00b`` sources, now shipped in ``sandbox/upstream/``. The transport
functions that enter the candidate process (``StringTupleRows``, ``pack_native``, ``unpack_native``,
``_invoke_native``, ``_load_native``) are unchanged byte for byte. Changes:

* upstream root is the packaged copy; ``Grade`` is a local dataclass; ``prepare_oracle`` dropped (oracles are pinned
  and certified, never regenerated here);
* ``grade_evalplus`` takes the sandbox profile (memory, stdout, stack, suite ceiling, interpreter) as arguments
  instead of the hard-coded 4 GiB profile;
* sandbox failures are classified instead of all becoming ``restricted_execution_timeout_or_failure``: a failure
  outside candidate code (sandbox start, trusted transport bootstrap) raises ``EvaluatorOutage``; inside candidate
  code a wall/CPU limit is ``timeout``, and an unexplained process death/protocol break is ``candidate_crash``
  (``crash`` flag set so the caller can confirm with a canary and one fresh retry).
"""
from __future__ import annotations
import ast
import copy
import dataclasses
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import inspect
import math
from pathlib import Path
import time
from typing import Any
from agent_allocation._vendor.learnagents_io import file_hash,content_hash,read_json
from agent_allocation._vendor.evalplus_external import decode_evalplus_record
from .sandbox import (CERTIFIED_PYTHON_PREFIX, PROCESS_EXITED, SCRATCH_LIMIT, WALL_TIMEOUT, Sandbox, SandboxError,
                      SandboxLimits)

REVISION='26d6d00bb1fd0fa37f39c99d5290da67891d1c5e'
ROOT=Path(__file__).resolve().parent/'upstream/evalplus'/REVISION
PINS={'evalplus/eval/__init__.py':'3c833c39b842e33f251c83db4347e0a95191909f23b390c12d73ab29a28a4daf',
      'evalplus/eval/_special_oracle.py':'92b126b907ee493121b55de06f6a34058b6e18adc8cf1c48737eedcd83f24cdd',
      'evalplus/data/mbpp.py':'186e09b7b14dcf12ed259dfca1d6644a1c266c1bbac41c917fd3c0449a734b24',
      'evalplus/config.py':'d27099cc275ae6e64fb2c440643fc3b5d0952d1633db646e206b538517ffe6f2'}
SCORER_ID='evalplus-native-'+REVISION[:12]
SIGXCPU_EXIT=128+24   # bwrap reports a signalled child as 128+signo; RLIMIT_CPU soft limit sends SIGXCPU


class EvaluatorOutage(RuntimeError):
    """The evaluator itself failed (sandbox start, trusted bootstrap): unknown verdict, never a wrong answer."""

    def __init__(self, message, *, kind=None, phase=None, returncode=None):
        super().__init__(message)
        self.kind, self.phase, self.returncode = kind, phase, returncode


@dataclass
class Grade:
    success: bool
    native_score: float
    scorer: str
    failure_class: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def native_functions():
    import numpy as np
    for name,digest in PINS.items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Pinned EvalPlus native source changed')
    namespace={'math':math,'np':np}
    exec(compile((ROOT/'evalplus/eval/_special_oracle.py').read_text(),'<pinned-special-oracles>','exec'),namespace)
    tree=ast.parse((ROOT/'evalplus/eval/__init__.py').read_text())
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    exec(compile(ast.Module(body=[functions['is_floats']],type_ignores=[]),'<pinned-is-floats>','exec'),namespace)
    loops=[node for node in ast.walk(functions['unsafe_execute']) if isinstance(node,ast.For) and
           isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name) and node.iter.func.id=='enumerate']
    if len(loops)!=1:raise ValueError('Unexpected pinned native comparison structure')
    case_try=loops[0].body[0]
    if not isinstance(case_try,ast.Try) or not isinstance(case_try.body[0],ast.With):raise ValueError('Unexpected native case execution')
    body=copy.deepcopy(case_try.body[1:])
    if not isinstance(body[0],ast.Assign) or body[0].targets[0].id!='exp':raise ValueError('Missing native expected assignment')
    body=body[1:]  # expected value is already a private caller argument.
    class ProjectBookkeeping(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='details' for t in node.targets):return None
            return self.generic_visit(node)
        def visit_AugAssign(self,node):
            if isinstance(node.target,ast.Attribute) and isinstance(node.target.value,ast.Name) and node.target.value.id=='progress':return None
            return self.generic_visit(node)
        def visit_Continue(self,node):
            return ast.Return(ast.Tuple(elts=[ast.Constant(True),ast.Name(id='atol',ctx=ast.Load())],ctx=ast.Load()))
    function=ast.parse('def compare_result(dataset,entry_point,inp,out,exp,atol):\n try:\n  return True,atol\n except BaseException:\n  return False,atol\n').body[0]
    function.body[0].body=[ProjectBookkeeping().visit(n) for n in body]+function.body[0].body
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(module,'<pinned-native-comparisons-private>','exec'),namespace)
    source=ast.parse((ROOT/'evalplus/data/mbpp.py').read_text())
    deserializer=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='mbpp_deserialize_inputs')
    exec(compile(ast.Module(body=[deserializer],type_ignores=[]),'<pinned-mbpp-deserializer>','exec'),namespace)
    return namespace


class StringTupleRows:
    """Lossless bounded scalar-table representation for native sequence equality."""
    def __init__(self, outer, width, vocabulary, indices):
        self.outer, self.width, self.vocabulary, self.indices = outer, width, vocabulary, indices

    def materialize(self):
        rows = [tuple(self.vocabulary[index] for index in self.indices[start:start+self.width])
                for start in range(0, len(self.indices), self.width)]
        return rows if self.outer == 'list' else tuple(rows)

    def __eq__(self, other):
        if isinstance(other, StringTupleRows):
            return (self.outer, self.width, self.vocabulary, self.indices) == (other.outer, other.width, other.vocabulary, other.indices)
        # Correctly typed equal native outputs necessarily have the same large
        # dimensions and therefore take the same registered transport branch.
        return False

    def __array__(self, dtype=None, copy=None):
        import numpy as np
        result=np.asarray(self.vocabulary,dtype=dtype)[np.frombuffer(self.indices,dtype=np.uint8)].reshape(-1,self.width)
        return result.copy() if copy else result


def pack_native(value):
    import math
    if type(value) in (list, tuple) and len(value) >= 16384 and all(type(row) is tuple for row in value):
        width = len(value[0])
        if 1 <= width <= 256 and all(len(row) == width for row in value):
            unique = set()
            for row in value:
                try:unique.update(row)
                except TypeError:
                    unique=None;break
            def scalar(item):
                return (item is None or type(item) in (int,bool) or type(item) is str and len(item)<=1024
                        or type(item) is float and math.isfinite(item))
            if unique is not None and len(unique) <= 256 and all(scalar(item) for item in unique):
                import zlib, base64
                vocabulary = sorted(unique,key=lambda item:(0,item) if type(item) in (int,float,bool) else (1,'') if item is None else (2,item))
                indices = {item:bytes([index]) for index,item in enumerate(vocabulary)}
                compressor = zlib.compressobj(level=1); chunks=[]
                for start in range(0, len(value), 4096):
                    block=b''.join(b''.join(map(indices.__getitem__,row)) for row in value[start:start+4096])
                    chunks.append(compressor.compress(block))
                chunks.append(compressor.flush())
                return ['scalar_tuple_rows', {'outer':type(value).__name__, 'width':width, 'count':len(value),
                          'vocabulary':[pack_native(item) for item in vocabulary], 'zlib_base64':base64.b64encode(b''.join(chunks)).decode('ascii')}]
    if type(value) is int and value.bit_length() > 12000:
        return ['integer_hex', hex(value)]
    # Large native combinatorial outputs contain millions of fixed-size tuples.
    # Factor their shared type envelope instead of allocating a tagged object
    # for every scalar; retain exact outer/inner Python container types.
    if isinstance(value, (list, tuple)) and len(value) >= 1024 and all(type(row) is tuple for row in value):
        def plain(item):
            return (item is None or type(item) in (str, bool) or
                    type(item) is int and item.bit_length() <= 12000 or
                    type(item) is float and math.isfinite(item))
        if all(len(row) <= 64 and all(plain(item) for item in row) for row in value):
            return ['tuple_rows', {'outer': type(value).__name__, 'rows': value}]
    if isinstance(value,dict):return ['dict',[[pack_native(k),pack_native(v)] for k,v in value.items()]]
    if isinstance(value,(list,tuple,set,frozenset)):
        items=[pack_native(v) for v in value]
        if isinstance(value,(set,frozenset)):
            import json
            items.sort(key=lambda v:json.dumps(v,sort_keys=True,separators=(',',':')))
        return [type(value).__name__,items]
    if isinstance(value,complex):return ['complex',[pack_native(value.real),pack_native(value.imag)]]
    if type(value) is float and not math.isfinite(value):return ['nonfinite','nan' if math.isnan(value) else 'inf' if value>0 else '-inf']
    if value is None or type(value) in (str,int,float,bool):return ['scalar',value]
    if type(value).__module__.split('.')[0]=='numpy':
        import numpy as np
        if isinstance(value,np.ndarray):return ['ndarray',{'dtype':str(value.dtype),'shape':list(value.shape),'items':pack_native(value.tolist())}]
        if isinstance(value,np.generic):return ['numpy_scalar',{'dtype':str(value.dtype),'item':pack_native(value.item())}]
    raise TypeError('Native transport cannot encode '+type(value).__name__)


def unpack_native(value, depth=0, lazy_rows=False):
    if depth>64:raise ValueError('Native transport nesting exceeds64')
    if type(value) is not list or len(value)!=2:raise ValueError('Malformed native transport envelope')
    kind,payload=value
    if kind in {'string_tuple_rows','scalar_tuple_rows'}:
        import base64,zlib
        width,count,vocabulary=payload['width'],payload['count'],payload['vocabulary']
        if (type(width) is not int or type(count) is not int or not 1<=width<=256 or not 1<=count<=2_000_000
                or width*count>256*1024**2 or payload['outer'] not in {'list','tuple'}):
            raise ValueError('Native string tuple dimensions exceed bounded transport')
        if type(vocabulary) is not list or not 1<=len(vocabulary)<=256:
            raise ValueError('Malformed native canonical scalar table')
        if kind=='scalar_tuple_rows':vocabulary=[unpack_native(item,depth+1) for item in vocabulary]
        import math
        if any(item is not None and not (type(item) in (int,bool) or type(item) is str and len(item)<=1024 or
                       type(item) is float and math.isfinite(item)) for item in vocabulary):
            raise ValueError('Malformed native canonical scalar table')
        ordered=sorted(set(vocabulary),key=lambda item:(0,item) if type(item) in (int,float,bool) else (1,'') if item is None else (2,item))
        if vocabulary!=ordered:raise ValueError('Native scalar table is not canonical')
        if type(payload['zlib_base64']) is not str or len(payload['zlib_base64'])>64*1024**2:
            raise ValueError('Native compressed tuple stream exceeds64MiB')
        stream=zlib.decompressobj();indices=stream.decompress(base64.b64decode(payload['zlib_base64'],validate=True),width*count+1)
        if not stream.eof or stream.unused_data or stream.unconsumed_tail or len(indices)!=width*count or max(indices)>=len(vocabulary):
            raise ValueError('Native compressed tuple stream disagrees with declared dimensions')
        result=StringTupleRows(payload['outer'],width,vocabulary,indices)
        return result if lazy_rows else result.materialize()
    if kind=='scalar':return payload
    if kind=='nonfinite':return float(payload)
    if kind=='integer_hex':
        if type(payload) is not str or len(payload)>8*1024**2:
            raise ValueError('Native integer exceeds bounded hexadecimal transport')
        return int(payload,16)
    if kind=='tuple_rows':
        rows=payload['rows']
        if payload['outer'] not in {'list','tuple'} or type(rows) not in (list,tuple) or len(rows)>2_000_000:
            raise ValueError('Native tuple rows exceed transport bounds')
        if any(type(row) not in (list,tuple) or len(row)>64 or
               any(item is not None and type(item) not in (str,int,float,bool) for item in row) for row in rows):
            raise ValueError('Malformed compact native tuple rows')
        if sum(map(len,rows))>8_000_000:raise ValueError('Native tuple row scalar inventory exceeds8million')
        result=[tuple(row) for row in rows]
        return result if payload['outer']=='list' else tuple(result)
    if kind=='dict':return {unpack_native(k,depth+1):unpack_native(v,depth+1) for k,v in payload}
    if kind in {'list','tuple','set','frozenset'}:
        items=[unpack_native(v,depth+1) for v in payload]
        return {'list':list,'tuple':tuple,'set':set,'frozenset':frozenset}[kind](items)
    if kind=='complex':return complex(*[unpack_native(v,depth+1) for v in payload])
    if kind in {'ndarray','numpy_scalar'}:
        import numpy as np
        if type(payload['dtype']) is not str or len(payload['dtype'])>32:
            raise ValueError('Native dtype must be a bounded scalar descriptor')
        try:dtype=np.dtype(payload['dtype'])
        except (TypeError,ValueError):raise ValueError('Malformed native array type') from None
        if dtype.hasobject or dtype.kind not in 'biufcSU' or dtype.itemsize>256:
            raise ValueError('Unsupported or oversized native array element type')
        if kind=='ndarray':
            import math
            shape=payload['shape']
            if len(shape)>8 or any(type(n) is not int or n<0 for n in shape) or math.prod(shape)>4_000_000:
                raise ValueError('Native array shape exceeds transport bound')
            return np.array(unpack_native(payload['items'],depth+1),dtype=dtype).reshape(shape)
        return dtype.type(unpack_native(payload['item'],depth+1))
    raise ValueError('Unknown native transport type')


def _invoke_native(value):
    import copy,time,signal
    arguments=copy.deepcopy(unpack_native(value['input']))
    started=time.time()
    def expired(*_):raise TimeoutError('Native candidate execution time exceeded')
    previous=signal.signal(signal.SIGALRM,expired)
    if value.get('native_timeout') is not None:signal.setitimer(signal.ITIMER_REAL,value['native_timeout'])
    try:result=_native_module[value['entry_point']](*arguments)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    elapsed=time.time()-started
    if value['output_not_none']:
        result=result if isinstance(result,bool) else result is not None
    return {'output':pack_native(result),'post_input':pack_native(arguments),'reference_wall_seconds':elapsed}

def _load_native(value):
    exec(compile(value['source'],'<native-candidate>','exec'),_native_module)
    return True


TRANSPORT='_native_module={}\n'+'\n'.join(inspect.getsource(f) for f in [StringTupleRows,pack_native,unpack_native,_invoke_native,_load_native])


def load_program(sandbox,source):
    bootstrap=sandbox.execute(TRANSPORT,phase='trusted')
    if not bootstrap.get('ok'):
        raise EvaluatorOutage('Trusted EvalPlus transport bootstrap failed',kind='bootstrap_failed',phase='trusted')
    result=sandbox.invoke({'source':source},'_load_native')
    result['_resource_usage']['sandbox_cpu_seconds']+=bootstrap.get('_resource_usage',{}).get('sandbox_cpu_seconds',0)
    return result


def native_record(private):
    row=copy.deepcopy(decode_evalplus_record(private))
    dataset='mbpp' if row['task_id'].startswith('Mbpp/') else 'humaneval' if row['task_id'].startswith('HumanEval/') else None
    if dataset is None:raise ValueError('Unrecognized native EvalPlus task identity')
    if dataset=='mbpp':
        for split in ['base','plus']:row[split+'_input']=native_functions()['mbpp_deserialize_inputs'](row['task_id'],row[split+'_input'])
    return row,dataset


def classify_sandbox_error(error):
    """Map a SandboxError to (failure_class, crash) or raise EvaluatorOutage for evaluator-side faults."""
    if getattr(error,'phase',None)!='candidate':
        raise EvaluatorOutage('Sandbox failed outside candidate code: '+str(error)[:300],
                              kind=getattr(error,'kind',None),phase=getattr(error,'phase',None),
                              returncode=getattr(error,'returncode',None)) from error
    if error.kind==WALL_TIMEOUT:
        return 'timeout',False
    if error.kind==PROCESS_EXITED and error.returncode==SIGXCPU_EXIT:
        return 'cpu_limit',False
    if error.kind==SCRATCH_LIMIT:
        return 'scratch_limit',False
    return 'candidate_crash',True


THREAD_DENIED_MARKERS=("can't start new thread",)


def sandbox_denial(message):
    """Name the seccomp denial behind a candidate exception, if any: 'thread' or 'eperm' (fork/socket/...)."""
    text=(message or {}).get('error') or ''
    if any(marker in text for marker in THREAD_DENIED_MARKERS):return 'thread'
    if '[Errno 1] Operation not permitted' in text:return 'eperm'
    return None


def grade_evalplus(source,private,oracle,*,python_prefix=None,memory_bytes=4*1024**3,output_bytes=64*1024,
                   stack_bytes=64*1024**2,suite_wall_seconds=60.0,row=None,scratch_bytes=64*1024**2,file_bytes=16*1024**2):
    if not isinstance(source,str):return Grade(False,0,'evalplus-'+REVISION[:12],'non_string_program')
    if isinstance(oracle,(str,Path)):oracle=read_json(oracle)
    if oracle['restricted_record_sha256']!=content_hash(private) or oracle['native_revision']!=REVISION:
        raise ValueError('Native reference oracle differs from the frozen task')
    if row is None:row,dataset=native_record(private)
    else:dataset='mbpp' if row['task_id'].startswith('Mbpp/') else 'humaneval'
    native=native_functions();batches={};cpu=0.0;crash=False;stdout_truncated=False;denied=set()
    not_none=dataset=='mbpp' and row['entry_point'] in native['MBPP_OUTPUT_NOT_NONE_TASKS']
    for split in ['base','plus']:
        inputs=row[split+'_input'];batch=oracle['batches'][split]
        if batch['inputs_sha256']!=content_hash(pack_native(inputs)) or len(inputs)!=len(batch['expected']) or len(inputs)!=len(batch['reference_times']):
            raise ValueError('Native input/expected/time inventories disagree')
        limits=[max(4.0,4.0*t) for t in batch['reference_times']]
        if any(not math.isfinite(t) or t<=0 for t in limits):raise ValueError('Invalid native reference timing')
        wall=min(float(suite_wall_seconds),sum(limits))+3;deadline=time.monotonic()+wall
        passed=[];atol=row['atol'];failure=None;timeouts=0
        try:
            with Sandbox({},python_prefix=python_prefix or CERTIFIED_PYTHON_PREFIX,
                         limits=SandboxLimits(cpu_seconds=math.ceil(wall)+1,wall_seconds=wall,memory_bytes=memory_bytes,
                                              output_bytes=output_bytes,stack_bytes=stack_bytes,file_bytes=file_bytes,
                                              scratch_bytes=scratch_bytes,max_protocol_bytes=64*1024**2)) as sandbox:
                loaded=load_program(sandbox,source);cpu+=loaded.get('_resource_usage',{}).get('sandbox_cpu_seconds',0)
                if not loaded.get('ok'):
                    failure='compile_or_initialization_failure'
                    if sandbox_denial(loaded):denied.add(sandbox_denial(loaded))
                else:
                    for index,inp in enumerate(inputs):
                        remaining=deadline-time.monotonic()
                        if remaining<=0:failure='native_batch_timeout';break
                        sandbox.limits=dataclasses.replace(sandbox.limits,wall_seconds=min(remaining,limits[index]+60))
                        result=sandbox.invoke({'input':pack_native(inp),'entry_point':row['entry_point'],'output_not_none':not_none,'native_timeout':limits[index]},'_invoke_native')
                        cpu+=result.get('_resource_usage',{}).get('sandbox_cpu_seconds',0)
                        stdout_truncated=stdout_truncated or bool(result.get('stdout_truncated'))
                        if result.get('ok'):
                            try:
                                actual=unpack_native(result['value']['output'],lazy_rows=True);expected=unpack_native(batch['expected'][index],lazy_rows=True)
                                mutated_input=unpack_native(result['value']['post_input'])
                                if isinstance(actual,StringTupleRows) and isinstance(expected,StringTupleRows) and actual.outer!=expected.outer:
                                    correct=False
                                else:correct,atol=native['compare_result'](dataset,row['entry_point'],mutated_input,actual,expected,atol)
                            except Exception:correct=False
                        else:
                            correct=False
                            if sandbox_denial(result):denied.add(sandbox_denial(result))
                            if 'Native candidate execution time exceeded' in (result.get('error') or ''):timeouts+=1
                        passed.append(bool(correct))
        except SandboxError as error:
            failure,split_crash=classify_sandbox_error(error)
            crash=crash or split_crash
        batches[split]={'passed':sum(passed),'attempted':len(passed),'total':len(inputs),
                        'success':len(passed)==len(inputs) and all(passed),'failure_class':failure,'input_timeouts':timeouts,
                        'wall_seconds':round(time.monotonic()-(deadline-wall),3),'batch_ceiling_seconds':round(wall,3)}
    success=all(value['success'] for value in batches.values())
    return Grade(success,float(success),SCORER_ID,None if success else 'native_tests_failed',
                 {'base':batches['base'],'plus':batches['plus'],'_resource_usage':{'sandbox_cpu_seconds':cpu},
                  'candidate_crash':crash,'stdout_truncated':stdout_truncated,'sandbox_denied':sorted(denied),
                  'native_comparison_source_pins':PINS,
                  'runtime_profile':{'memory_bytes':memory_bytes,'output_bytes':output_bytes,'stack_bytes':stack_bytes,
                                     'suite_wall_seconds':suite_wall_seconds,'per_input':'max(4 s, 4 x reference time)'}})
