#!/usr/bin/env python3
"""Offline planning calculator. Does not contact OpenRouter or run experiments.

Prices are planning snapshots, not guaranteed future endpoint prices.
Completion caps are assumed saturated; input consumption is an explicit estimate.
Run from anywhere: python estimate_budget.py --preset balanced150
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MILLION = Decimal(1_000_000)

@dataclass(frozen=True)
class Block:
    name: str
    items: int
    calls: int
    input_tokens: int
    completion_cap_tokens: int
    estimated_usd: str
    optional: bool = False

def calculate_block(name: str, items: int, base_calls: int, reasoning: int,
                    deep_multipliers: list[int], input_per_call: int,
                    input_price: str, output_price: str,
                    optional: bool = False) -> Block:
    if items < 0 or items % 4:
        raise ValueError('Each block must contain a nonnegative multiple of four items.')
    if min(base_calls, reasoning, input_per_call) < 0:
        raise ValueError('Inputs must be nonnegative.')
    # Three text benchmark families and one code family, equally weighted.
    text_n, code_n = 3 * items // 4, items // 4
    text_h, code_h = 1024, 2048
    calls_per_item = base_calls + len(deep_multipliers)
    def cap_per_item(headroom: int) -> int:
        return base_calls * (reasoning + headroom) + sum(
            reasoning * m + headroom for m in deep_multipliers)
    caps = text_n * cap_per_item(text_h) + code_n * cap_per_item(code_h)
    calls = items * calls_per_item
    inputs = calls * input_per_call
    price = (Decimal(inputs) * Decimal(input_price) +
             Decimal(caps) * Decimal(output_price)) / MILLION
    return Block(name, items, calls, inputs, caps, f'{price:.6f}', optional)

def estimate(preset: str = 'balanced150', avg_input_tokens: int = 2500) -> dict:
    if avg_input_tokens <= 0:
        raise ValueError('avg_input_tokens must be positive')
    cfg = json.loads((ROOT / 'configs' / f'{preset}.json').read_text())
    p = cfg['panels']
    a = cfg['reasoning']['main_a']
    ah = cfg['reasoning']['high_a']
    lite = cfg['planning_prices']['primary']
    flash = cfg['planning_prices']['replication']
    blocks = [
        calculate_block('main_full', p['main_items'], 121, a, [2,4,8,16],
                        avg_input_tokens, **lite),
        calculate_block('high_full', p['high_items'], 121, ah, [2,4],
                        avg_input_tokens, **lite),
        calculate_block('communicate_then_private', p['extension_items'], 12, a, [],
                        avg_input_tokens, **lite),
        calculate_block('pooled_synthesis', p['extension_items'], 1, a, [],
                        avg_input_tokens, **lite),
        calculate_block('second_model_reduced', p['replication_items'], 55, a, [4,16],
                        avg_input_tokens, **flash, optional=True),
    ]
    api = sum(Decimal(b.estimated_usd) for b in blocks)
    pilot = Decimal(str(cfg['budget']['pilot_allowance_usd']))
    reserve = Decimal(str(cfg['budget']['contingency_usd']))
    return {
        'preset': preset,
        'assumptions': {'average_input_tokens_per_call': avg_input_tokens,
                        'completion_caps_saturated': True,
                        'code_fraction': 0.25,
                        'cache_discounts': False,
                        'excludes_purchase_fees_and_external_compute': True,
                        'prices_require_live_preflight': True},
        'blocks': [asdict(b) for b in blocks],
        'generation_calls_excluding_pilot': sum(b.calls for b in blocks),
        'api_estimate_excluding_pilot_usd': f'{api:.6f}',
        'pilot_allowance_usd': str(pilot),
        'contingency_usd': str(reserve),
        'planned_total_usd': f'{api + pilot + reserve:.6f}',
        'credit_ceiling_usd': str(cfg['budget']['credit_ceiling_usd']),
        'fits_credit_ceiling_under_assumptions': api + pilot + reserve <= Decimal(str(cfg['budget']['credit_ceiling_usd'])),
    }

FORCED_PRESETS = ('forced27b',)


def forced_node_budgets(cfg: dict) -> list[int]:
    """Reasoning budget of every node of one main_extension item (the forced grid): root bank, private chains,
    peer rounds and CP/POOLED at A, DEEP at m*A."""
    p = cfg['protocols']
    a = int(cfg['reasoning']['main_a'])
    n_max = max(p['team_sizes'])
    single = (p['root_bank_size'] + (p['private_total_cycles'] - 1)
              + (n_max - 1) * max(p['private_team_update_rounds'])
              + sum(p['team_sizes']) * max(p['peer_update_rounds'])
              + p['cp_team_size'] * p['cp_private_rounds'] + 1)
    return [a] * single + [a * m for m in p['deep_main_multipliers']]


def estimate_forced(preset: str = 'forced27b', avg_input_tokens: int = 2500, answer_tokens: int | None = None,
                    natural_tokens: int = 8000) -> dict:
    """Forced two-call cost model per node: input = 2P + budget (THINK prompt, then ANSWER prompt re-feeding the
    reasoning), output = budget + answer (answer defaults to the saturated headroom cap). One natural (unforced)
    reference root per item adds P input and ``natural_tokens`` + answer output. Per shard = share/sum(share)."""
    if avg_input_tokens <= 0 or natural_tokens < 0:
        raise ValueError('avg_input_tokens must be positive and natural_tokens nonnegative')
    cfg = json.loads((ROOT / 'configs' / f'{preset}.json').read_text())
    headroom = int(cfg['reasoning']['text_completion_headroom'])
    answer = headroom if answer_tokens is None else int(answer_tokens)
    if answer < 0:
        raise ValueError('answer_tokens must be nonnegative')
    prices = cfg['planning_prices']['primary']
    price_in, price_out = Decimal(prices['input_price']), Decimal(prices['output_price'])
    budgets = forced_node_budgets(cfg)
    items = int(cfg['panels']['main_items'])
    p = avg_input_tokens
    in_item = sum(2 * p + b for b in budgets) + p
    out_item = sum(b + answer for b in budgets) + natural_tokens + answer
    per_item = (Decimal(in_item) * price_in + Decimal(out_item) * price_out) / MILLION
    main = per_item * items
    pilot = Decimal(str(cfg['budget']['pilot_allowance_usd']))
    reserve = Decimal(str(cfg['budget']['contingency_usd']))
    ceiling = Decimal(str(cfg['budget']['credit_ceiling_usd']))
    shares = [(s['id'], Decimal(str(s['share']))) for s in cfg.get('shards', [])] or [('all', Decimal(1))]
    total_share = sum(w for _, w in shares)
    shard_rows = []
    for sid, w in shares:
        frac = w / total_share
        shard_rows.append({'shard_id': sid, 'fraction': f'{frac:.4f}', 'items': f'{items * frac:.1f}',
                           'main_usd': f'{main * frac:.2f}', 'credit_ceiling_usd': f"{(ceiling * frac).quantize(Decimal('0.01'), rounding=ROUND_DOWN)}",
                           'fits': main * frac + (pilot + reserve) * frac <= ceiling * frac})
    total = main + pilot + reserve
    return {
        'preset': preset,
        'assumptions': {'average_prompt_tokens_P': p, 'answer_tokens_per_call': answer,
                        'answer_saturated_at_headroom': answer_tokens is None, 'natural_reference_tokens': natural_tokens,
                        'cost_model': 'input 2P+budget, output budget+answer per forced node',
                        'prices_per_million': {'input': str(price_in), 'output': str(price_out)}},
        'nodes_per_item': len(budgets) + 1, 'forced_nodes_per_item': len(budgets),
        'forced_reasoning_tokens_per_item': sum(budgets),
        'reasoning_a_units_per_item': sum(budgets) // int(cfg['reasoning']['main_a']),
        'input_tokens_per_item': in_item, 'output_tokens_per_item': out_item,
        'usd_per_item': f'{per_item:.4f}', 'main_items': items, 'main_usd': f'{main:.2f}',
        'pilot_allowance_usd': str(pilot), 'contingency_usd': str(reserve), 'planned_total_usd': f'{total:.2f}',
        'credit_ceiling_usd': str(ceiling), 'fits_credit_ceiling_under_assumptions': total <= ceiling,
        'shards': shard_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['balanced150','lean100', *FORCED_PRESETS], default='balanced150')
    parser.add_argument('--avg-input-tokens', type=int, default=2500)
    parser.add_argument('--answer-tokens', type=int, default=None,
                        help='forced presets: expected ANSWER tokens per call (default: saturated headroom cap)')
    parser.add_argument('--natural-tokens', type=int, default=8000,
                        help='forced presets: reasoning tokens of the natural reference root per item')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.preset in FORCED_PRESETS:
        try:
            result = estimate_forced(args.preset, args.avg_input_tokens, args.answer_tokens, args.natural_tokens)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        if args.json:
            print(json.dumps(result, indent=2))
            return
        a = result['assumptions']
        print(f"Preset: {result['preset']} | P={a['average_prompt_tokens_P']} | answer tokens/call="
              f"{a['answer_tokens_per_call']}{' (saturated cap)' if a['answer_saturated_at_headroom'] else ''}")
        print(f"Nodes/item: {result['nodes_per_item']} ({result['forced_nodes_per_item']} forced + 1 natural); "
              f"forced reasoning/item: {result['forced_reasoning_tokens_per_item']:,} tokens "
              f"({result['reasoning_a_units_per_item']} x A)")
        print(f"Per item: {result['input_tokens_per_item']:,} in / {result['output_tokens_per_item']:,} out = "
              f"${Decimal(result['usd_per_item']):.4f}")
        print(f"{'Shard':12} {'Fraction':>8} {'Items':>7} {'Main $':>9} {'Ceiling $':>10} Fits")
        for r in result['shards']:
            print(f"{r['shard_id']:12} {r['fraction']:>8} {r['items']:>7} {r['main_usd']:>9} "
                  f"{r['credit_ceiling_usd']:>10} {r['fits']}")
        print(f"Total: main ${result['main_usd']} + pilot {result['pilot_allowance_usd']} + contingency "
              f"{result['contingency_usd']} = ${result['planned_total_usd']} (ceiling ${result['credit_ceiling_usd']})")
        print('Planning estimate only; realized answer length is measured by the pilot.')
        print('Fits assumed ceiling:', result['fits_credit_ceiling_under_assumptions'])
        return
    try:
        result = estimate(args.preset, args.avg_input_tokens)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Preset: {result['preset']} | assumed input tokens/call: {args.avg_input_tokens}")
    print(f"{'Block':31} {'Items':>6} {'Calls':>8} {'Estimate':>11}")
    for b in result['blocks']:
        print(f"{b['name']:31} {b['items']:6d} {b['calls']:8d} ${Decimal(b['estimated_usd']):10.2f}")
    print(f"Generation calls (pilot excluded): {result['generation_calls_excluding_pilot']:,}")
    print(f"API estimate + pilot + contingency: ${Decimal(result['planned_total_usd']):.2f}")
    print(f"Credit ceiling: ${result['credit_ceiling_usd']}")
    print('Planning estimate only. Local grading, external compute and credit-purchase fees are excluded.')
    print('Fits assumed ceiling:', result['fits_credit_ceiling_under_assumptions'])

if __name__ == '__main__':
    main()
