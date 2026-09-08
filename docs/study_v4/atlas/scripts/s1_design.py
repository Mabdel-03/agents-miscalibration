"""A1: the design as executed — scale, coverage, realised levels, and nominal-vs-realised cell collapse."""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import _common as C

E = C.Emitter("s1_design.py")
sel, ep = C.selections(), C.episodes()
main = sel[sel.split == "main"]

# ---- scale ----
rec = json.load(open(f"{C.RUN_ROOT}/reconcile.json"))
E.emit("scale.cells", rec.get("n_cells") or rec.get("cells") or len(sel.cell_id.unique()), source="reconcile.json")
for k in ("n_generated_requests", "n_aliased_requests", "flops_total", "n_episodes"):
    if k in rec: E.emit(f"scale.{k}", rec[k], source="reconcile.json")
E.emit("scale.episode_rows", len(ep), source="tables/episodes.parquet")
E.emit("scale.selection_rows", len(sel), source="tables/selections.parquet")
E.emit("scale.items_main", main.source_id.nunique(), source="tables/selections.parquet")
E.emit("scale.items_by_domain", main.groupby("domain").source_id.nunique().to_dict())

# ---- module coverage: what each module varies ----
cov = {}
for mod, d in ep[ep.split == "main"].groupby("module"):
    cov[mod] = dict(
        items=int(d.source_id.nunique()), episodes=int(len(d)),
        methods=sorted(d.method.unique().tolist()),
        checkpoints=sorted(d.checkpoint.astype(str).unique().tolist()),
        N=sorted(int(x) for x in d.N.dropna().unique()),
        B=sorted(int(x) for x in d.B.dropna().unique()),
        framings=sorted(d.framing.astype(str).unique().tolist()),
        degrees=sorted(int(x) for x in d.degree.dropna().unique()) if d.degree.notna().any() else [],
        reps=sorted(int(x) for x in d.episode_rep.dropna().unique()),
    )
E.emit("design.module_coverage", cov, source="tables/episodes.parquet")

# ---- axis level sets actually realised ----
axes = {
    "C1_topology": sorted(ep[ep.split == "main"].method.unique().tolist()),
    "C2_team_size": sorted(int(x) for x in ep.N.dropna().unique()),
    "C3_degree": sorted(int(x) for x in ep.degree.dropna().unique()),
    "C4_budget": sorted(int(x) for x in ep.B.dropna().unique()),
    "C5_scale": sorted(ep.checkpoint.astype(str).unique().tolist()),
    "C6_framing": sorted(ep.framing.astype(str).unique().tolist()),
    "C7_selector": sorted(sel.selector_id.astype(str).unique().tolist()),
    "C8_repetition": sorted(int(x) for x in ep.episode_rep.dropna().unique()),
    "C9_domain": sorted(sel.domain.unique().tolist()),
    "C9b_answer_format": sorted(sel.answer_format.dropna().astype(str).unique().tolist()),
    "C9c_stratum": sorted(sel.stratum.dropna().astype(str).unique().tolist()),
}
E.emit("design.axis_levels", axes, source="tables/*.parquet")

# ---- nominal vs realised: do distinct nominal cells produce distinct episodes? ----
# Signature of a cell = its item -> (correctness, spent_flops, calls) vector on shared items.
m = ep[(ep.split == "main") & (~ep.module.isin(["D", "E"]))].copy()
m["nominal"] = (m.module + "|" + m.method + "|" + m.checkpoint.astype(str) + "|N" +
                m.N.astype("Int64").astype(str) + "|B" + m.B.astype("Int64").astype(str))
panel = m.groupby("source_id").nominal.nunique()
shared = set(panel[panel >= 8].index)          # items covered by most nominal cells
sig_corr, sig_spent = {}, {}
for nom, d in m.groupby("nominal"):
    d = d[d.source_id.isin(shared)].drop_duplicates("source_id").sort_values("source_id")
    if len(d) < 40: continue
    sig_corr[nom] = hash(tuple(zip(d.source_id, d.acc)))
    sig_spent[nom] = hash(tuple(zip(d.source_id, d.spent_flops.round(0))))
E.emit("design.nominal_cells", len(sig_corr), note="module/method/checkpoint/N/B cells on the shared panel (D,E excluded)")
E.emit("design.distinct_correctness_vectors", len(set(sig_corr.values())))
E.emit("design.distinct_compute_vectors", len(set(sig_spent.values())))
groups = collections.defaultdict(list)
for nom, h in sig_corr.items(): groups[h].append(nom)
dupes = {str(i): sorted(v) for i, v in enumerate(g for g in groups.values() if len(g) > 1)}
E.emit("design.identical_cell_groups", dupes, note="nominal cells that are the same physical episode set")
E.emit("design.largest_identical_group", max((len(v) for v in dupes.values()), default=0))

# ---- aliasing: how much generation was actually reused ----
al = ep[ep.split == "main"].groupby("method").agg(
    aliased=("aliased_calls", "mean"), generated=("generated_calls", "mean"), calls=("calls_admitted", "mean"))
E.emit("design.alias_by_method", al.round(2).to_dict("index"), source="tables/episodes.parquet")

# ---- difficulty band structure on the main panel ----
diff = C.difficulty().merge(main[["source_id", "domain"]].drop_duplicates("source_id"), on="source_id", how="inner")
E.emit("design.difficulty_bands", diff.band.value_counts().reindex(["never", "hard", "contested", "easy"]).to_dict(),
       n=len(diff), source="tables/banks.parquet framing=00 ckpt=32B, main split")
E.emit("design.difficulty_bands_by_domain",
       {d: g.band.value_counts().reindex(["never", "hard", "contested", "easy"]).to_dict()
        for d, g in diff.groupby("domain")})
E.emit("design.difficulty_band_pass1",
       diff.groupby("band", observed=True).pass_at_1.mean().round(4).to_dict())
E.emit("design.never_band_pass10_max", float(diff[diff.band == "never"].pass_at_10.max()),
       note="ceiling of the never-solved band over 10 neutral draws")

E.save("design")
