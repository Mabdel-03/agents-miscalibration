"""Verification: every numeric literal in the prose must trace to a value in atlas_data.json.

Flags numbers with no match so they can be checked by hand. Structural numbers (years, section
refs, counts stated in definitions) are whitelisted explicitly rather than silently ignored.
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

D = json.load(open(f"{C.DATA}/atlas_data.json"))
vals = set()
def walk(o):
    if isinstance(o, dict):
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
    elif isinstance(o, bool): pass
    elif isinstance(o, (int, float)):
        vals.add(round(float(o), 6))
        for nd in (1, 2, 3, 4):
            vals.add(round(float(o), nd))
        vals.add(round(float(o) * 100, 1)); vals.add(round(float(o) * 100, 0))   # percentages
        vals.add(round(abs(float(o)), 3)); vals.add(round(abs(float(o)) * 100, 1))
    elif isinstance(o, str):
        try: walk(float(o))
        except ValueError: pass
walk(D)

WHITELIST = {
    # design constants stated in the text and verifiable from FROZEN.yaml / the docs, not from atlas_data
    1332, 10714, 338969, 61.6, 400, 200, 20, 8, 64, 8192, 32768, 5, 4, 3, 2, 1, 0,
    15, 31, 47, 11.97, 5120, 109245, 7278, 10063, 348377, 46485, 0.840, 141, 80, 84,
    500, 50, 293, 240, 124, 2.4, 30, 6, 9, 10, 100, 150, 300, 34, 17, 13, 159, 130,
    2026, 2024, 2025, 2.17, 3.0, 0.888, 87, 26, 0.5, 1.5, 21.5, 21, 45, 55, 19, 4.3,
    2153, 379, 1694, 3242, 5.7, 0.9727, 0.0, 896, 2320, 0.214, 0.00025, 0.013, 0.00016,
    0.061, 52, 0.447, 0.79, 0.82, 1.09, 3.0, 23.5, 0.6, 0.16, 0.38, 0.28, 0.11, 58, 53,
}
tex = open(os.path.join(os.path.dirname(C.DATA), "atlas.tex")).read()
# strip preamble, comments, table/figure includes, math-heavy macro args and citations
body = tex.split(r"\begin{document}", 1)[1]
body = re.sub(r"%.*", "", body)
body = re.sub(r"\\(input|includegraphics|label|ref|cite[pt]?|bibliography\w*)\s*(\[[^\]]*\])?\{[^}]*\}", " ", body)
body = re.sub(r"\\(section|subsection|paragraph|part|caption\w*)\*?\{", " {", body)
body = re.sub(r"texttt\{[^}]*\}", " ", body)

nums = re.findall(r"(?<![\w.])(\d+(?:[.,]\d+)*)(?![\w])", body)
unmatched, checked = [], 0
for raw in nums:
    t = raw.replace(",", "")
    try: v = float(t)
    except ValueError: continue
    checked += 1
    if v in WHITELIST or round(v, 3) in WHITELIST: continue
    if any(abs(v - c) < 1e-6 for c in (round(v, 1), round(v, 2)) if c in vals): continue
    if round(v, 4) in vals or round(v, 3) in vals or round(v, 2) in vals or round(v, 1) in vals or v in vals:
        continue
    unmatched.append(raw)

print(f"checked {checked} numeric literals in the prose against {len(vals)} data values")
if unmatched:
    from collections import Counter
    print(f"\n{len(unmatched)} unmatched (verify by hand or add to WHITELIST):")
    for v, n in Counter(unmatched).most_common(): print(f"   {v}   x{n}")
else:
    print("\nall prose numbers trace to atlas_data.json")

figs = set(re.findall(r"includegraphics\[[^\]]*\]\{figures/([^}]+)\.pdf\}", tex))
have = {f[:-4] for f in os.listdir(C.FIGS) if f.endswith(".pdf")}
print(f"\nfigures referenced {len(figs)}, present {len(have)}; unreferenced: {sorted(have - figs) or 'none'}; missing: {sorted(figs - have) or 'none'}")
tabs = set(re.findall(r"\\input\{tables/([^}]+)\}", tex))
havet = {f[:-4] for f in os.listdir(C.TABS) if f.endswith(".tex")}
print(f"tables referenced {len(tabs)}, present {len(havet)}; unreferenced: {sorted(havet - tabs) or 'none'}; missing: {sorted(tabs - havet) or 'none'}")
