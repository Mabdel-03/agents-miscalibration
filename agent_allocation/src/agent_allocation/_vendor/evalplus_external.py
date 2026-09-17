"""``decode_evalplus_record`` copied from ``learnagents/data/external.py`` (agent_methods_bench@2af2879).

The frozen MBPP+ restricted rows store the native EvalPlus record; one row (Mbpp/404-style non-finite floats)
uses the typed ``typed-json-with-nonfinite-floats-v1`` encoding and must be decoded before use.
"""
from __future__ import annotations


def decode_evalplus_record(private):
    if "native_record_encoding" not in private: return private["native_record"]
    if private["native_record_encoding"] != "typed-json-with-nonfinite-floats-v1":
        raise ValueError("Unknown native record encoding")
    def decode(value):
        kind, payload = value
        if kind == "dict": return {k: decode(v) for k, v in payload}
        if kind == "list": return [decode(v) for v in payload]
        if kind == "nonfinite" and payload in {"nan", "inf", "-inf"}: return float(payload)
        if kind == "scalar": return payload
        raise ValueError("Malformed native record encoding")
    return decode(private["native_record"])
