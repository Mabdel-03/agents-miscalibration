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
from decimal import Decimal
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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['balanced150','lean100'], default='balanced150')
    parser.add_argument('--avg-input-tokens', type=int, default=2500)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
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
