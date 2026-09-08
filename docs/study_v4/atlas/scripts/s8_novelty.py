"""Generate the novelty-assessment table from the literature workflow's verdicts."""
import sys, os, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

nov = json.load(open(f"{C.DATA}/novelty.json"))
CLAIMS = {
 "C1": ("\\ax{C1}", "Independent+vote beats the centralised hub at matched compute; peer revision beats neither."),
 "C2": ("\\ax{C3}", "Peer revision raises agreement and confidence monotonically while pool coverage falls."),
 "C3": ("\\ax{C2}", "A lone correct member is overturned 75--86\\% of the time at $N\\ge5$."),
 "C4": ("\\ax{C3}", "Round 1 delivers the whole correctness gain; rounds 2--8 cost $4.3\\times$ and buy agreement."),
 "C5": ("\\ax{C1}", "Private revision without peers is a no-op; peer content is the active ingredient."),
 "C6": ("\\ax{C2}", "Team size buys pool coverage ($+0.13$) but not delivered accuracy ($+0.04$)."),
 "C7": ("\\ax{C1}", "The orchestrator rarely delegates; multi-call episodes are output-cap retry loops."),
 "C8": ("\\ax{C4}", "pass@$k$ climbs while vote@$k$ and judge@$k$ stay flat."),
 "C9": ("\\ax{C4}", "Nominal budget is inert once call/round caps bind; 17--45\\% of the allowance is spent."),
 "C10": ("\\ax{C4}", "59 nominal cells collapse to 40 distinct outcome vectors under content-addressed reuse."),
 "C11": ("\\ax{C3}", "Sequential compute is dominated by parallel compute at equal FLOPs."),
 "C12": ("\\bx{B3}", "Oracle coverage 0.542 against 0.328 delivered: a 21-point selection gap."),
 "C13": ("\\ax{C7}", "Plurality adds $+0.026$ over a random valid draw; a pointwise judge adds nothing."),
 "C14": ("\\ax{C7}", "On code, syntactic grouping makes voting structurally unavailable in 77\\% of failures."),
 "C15": ("\\ax{C7}", "On the exam, 91\\% of covered-but-wrong pools are a correct minority under a wrong consensus."),
 "C16": ("\\ax{C7}", "Confidence-aware selectors land within 1.5 points of plain plurality."),
 "C17": ("\\bx{B5}", "Confidence is ordinal between items and empty within them (within-item AUC $\\approx$0.50)."),
 "C18": ("\\ax{C1}", "Information sharing inflates certainty without improving it (Brier 0.478$\\to$0.583)."),
 "C19": ("\\bx{B5}", "The explicit self-forecast copies the read confidence verbatim in 896/896 cases."),
 "C20": ("\\bx{B5}", "Post-hoc recalibration cuts Brier 0.491$\\to$0.216, dwarfing every other calibration effect."),
 "C21": ("\\ax{C5}", "8B matches 32B at matched FLOPs; the scale ordering reverses by domain."),
 "C22": ("\\ax{C5}", "The 4B deficit is substantially a format-compliance failure (0.67 valid)."),
 "C23": ("\\ax{C5}", "Stated confidence rises with scale while accuracy does not."),
 "C24": ("\\ax{C5}", "Small checkpoints emit a literal 0.50 that functions as abstention, not a default."),
 "C25": ("\\mx{M1}", "Peer-exposed agents commit from a lower-rank answer state (PR 2.36$\\to$1.84)."),
 "C26": ("\\mx{M2}", "That drop is anisotropy: mean separation unchanged, variance share on $\\lambda_1$ 0.61$\\to$0.74."),
 "C27": ("\\mx{M5}", "The effect is anchor-specific: absent mid-generation, present only at the committed answer."),
 "C28": ("\\mx{M2}", "Unanimous episodes have \\emph{more} dispersed member states arranged along fewer directions."),
 "C29": ("\\mx{M3}", "On code the consensus signal is in the norm ($\\rho=+0.389$), invisible to the scale-free metric."),
 "C30": ("\\mx{M7}", "A reader's state identifies the harness at 0.989 and answer correctness at 0.694 (0.649 from task alone)."),
 "C31": ("\\mx{M7}", "Item difficulty is decodable from the prompt state before any token is generated."),
 "C32": ("\\mx{M7}", "Within an item, majority membership is more decodable than correctness."),
 "C33": ("\\mx{M7}", "The stated confidence value is not represented mid-generation ($R^2$ 0.10) but is at the answer token (0.63)."),
 "C34": ("\\ax{C10}", "Bimodal item difficulty gates every axis; pooled averages understate conditional effects $\\sim4\\times$."),
}
TAG = {"replication": "\\vrep", "extension": "\\vext", "novel": "\\vnov", "contradicts": "\\vcon"}
order = ["novel", "contradicts", "extension", "replication"]
rows = []
for verdict in order:
    ids = [c for c in CLAIMS if nov.get(c, {}).get("verdict") == verdict]
    ids.sort(key=lambda x: int(x[1:]))
    for cid in ids:
        ax, txt = CLAIMS[cid]
        v = nov[cid]
        keys = [k for k in v.get("keys", [])][:3]
        cite = "\\citep{" + ",".join(keys) + "}" if keys else "--"
        rows.append(f"{ax} & {txt} & {TAG[verdict]} & {cite} \\\\")
body = ("\\begin{longtable}{@{}p{0.9cm}p{8.8cm}p{1.9cm}p{3.2cm}@{}}\n\\toprule\n"
        "Axis & Claim & Verdict & Closest prior work \\\\\n\\midrule\n\\endfirsthead\n"
        "\\toprule\nAxis & Claim & Verdict & Closest prior work \\\\\n\\midrule\n\\endhead\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}")
with open(os.path.join(C.TABS, "t_novelty.tex"), "w") as f: f.write(body)
print(f"  table -> t_novelty.tex ({len(rows)} claims)")
# per-verdict justification list for the text
just = {c: dict(verdict=nov[c]["verdict"], just=nov[c]["just"], closest=nov[c]["closest"]) for c in CLAIMS if c in nov}
json.dump(just, open(f"{C.DATA}/novelty_text.json", "w"), indent=1, sort_keys=True)
