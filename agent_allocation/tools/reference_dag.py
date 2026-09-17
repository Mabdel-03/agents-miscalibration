"""Executable structural reference for protocol/reuse tests. No API calls.

Node names here are readable local references, not production semantic IDs.
The main implementation should create GenerationSpec instances and policy records.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    name: str
    kind: str
    parents: tuple[str, ...]
    reasoning_budget: int

@dataclass
class Plan:
    nodes: dict[str, Node]
    checkpoints: dict[str, tuple[str, ...]]

    def ancestors(self, checkpoint: str) -> set[str]:
        found: set[str] = set()
        todo = list(self.checkpoints[checkpoint])
        while todo:
            key = todo.pop()
            if key not in found:
                found.add(key)
                todo.extend(self.nodes[key].parents)
        return found

    def validate(self) -> None:
        complete: set[str] = set()
        active: set[str] = set()
        def visit(name: str) -> None:
            if name in active: raise ValueError(f'Cycle: {name}')
            if name in complete: return
            if name not in self.nodes: raise ValueError(f'Missing node: {name}')
            active.add(name)
            for p in self.nodes[name].parents: visit(p)
            active.remove(name)
            complete.add(name)
        for n in self.nodes: visit(n)
        for cp in self.checkpoints.values():
            for n in cp:
                if n not in self.nodes: raise ValueError(f'Unknown checkpoint candidate: {n}')

def build_plan(a: int = 1024, *, high: bool = False,
               reduced: bool = False, extension: bool = False) -> Plan:
    if high and reduced: raise ValueError('High and reduced profiles are distinct.')
    if extension and (high or reduced): raise ValueError('Extension is a main-profile panel.')
    bank = 16 if reduced else 25
    rounds = 2 if reduced else 4
    sizes = (3,5) if reduced else (2,3,4,5)
    multipliers = (4,16) if reduced else ((2,4) if high else (2,4,8,16))
    nodes: dict[str, Node] = {}
    cps: dict[str, tuple[str, ...]] = {}
    def add(name: str, kind: str, parents: tuple[str,...], budget: int = a) -> str:
        if name in nodes: raise ValueError('Duplicate generation node')
        nodes[name] = Node(name, kind, parents, budget)
        return name
    roots = tuple(add(f'root/{j}', 'root', ()) for j in range(bank))
    for k in range(1, bank+1): cps[f'sample/{k}'] = roots[:k]
    private: dict[tuple[int,int],str] = {}
    for j in range(5): private[j,0] = roots[j]
    cps['private_single/1'] = (roots[0],)
    for r in range(1, bank):
        name=add(f'private/0/{r}', 'private', (private[0,r-1],))
        private[0,r]=name
        cps[f'private_single/{r+1}']=(name,)
    for j in range(1,5):
        for r in range(1,rounds+1):
            private[j,r]=add(f'private/{j}/{r}', 'private', (private[j,r-1],))
    for n in sizes:
        for r in range(rounds+1):
            cps[f'private_team/{n}/{r}']=tuple(private[j,r] for j in range(n))
    for n in sizes:
        prev=roots[:n]
        cps[f'peer/{n}/0']=prev
        for r in range(1,rounds+1):
            current=tuple(add(f'peer/{n}/{r}/{j}', 'peer', prev) for j in range(n))
            cps[f'peer/{n}/{r}']=current
            prev=current
    for m in multipliers:
        node=add(f'deep/{m}', 'deep', (), a*m)
        cps[f'deep/{m}']=(node,)
    if extension:
        prev=cps['peer/4/1']
        cps['cp/0']=prev
        for h in range(1,4):
            current=tuple(add(f'cp/{h}/{j}', 'cp', (prev[j],)) for j in range(4))
            cps[f'cp/{h}']=current
            prev=current
        pooled=add('pooled/4', 'pooled', roots[:4])
        cps['pooled/4']=(pooled,)
    result=Plan(nodes,cps)
    result.validate()
    return result

if __name__ == '__main__':
    for name,kwargs in [('main',{}),('high',{'a':4096,'high':True}),
                        ('replication',{'reduced':True}),('main_with_extension',{'extension':True})]:
        p=build_plan(**kwargs)
        print(name, len(p.nodes), 'generation nodes', len(p.checkpoints), 'checkpoints')
