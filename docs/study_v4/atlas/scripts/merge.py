"""Merge the per-script fragments into the single provenance-carrying atlas_data.json."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

merged, origin = {}, {}
for p in sorted(glob.glob(f"{C.DATA}/_frag_*.json")):
    frag = json.load(open(p))
    for k, v in frag.items():
        if k in merged:
            raise KeyError(f"key {k} emitted by both {origin[k]} and {os.path.basename(p)}")
        merged[k] = v; origin[k] = os.path.basename(p)
with open(f"{C.DATA}/atlas_data.json", "w") as f:
    json.dump(merged, f, indent=1, sort_keys=True)
by_script = {}
for k, v in merged.items(): by_script.setdefault(v["script"], []).append(k)
print(f"merged {len(merged)} keys from {len(by_script)} scripts -> {C.DATA}/atlas_data.json")
for s in sorted(by_script): print(f"  {s:22s} {len(by_script[s]):3d} keys")
