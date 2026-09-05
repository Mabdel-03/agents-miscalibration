# Compute-demand model for scope tiers. Tokens/gen: HLE thinking avg ~4500 (cap 8192), BCB ~2500 -> blend 3500 for 32B; small models similar.
TOK = 3500
TP2_32B = 900      # gen tok/s per 32B-long server (2 GPUs) planning
PER_GPU = {"32B": TP2_32B/2, "14B": 950, "8B": 1100, "4B": 1200}
def hours(gens, gpus, size): return gens*TOK/(PER_GPU[size]*gpus)/3600
# calls per item per module (planning, B4 ~ 20 solver calls/episode for 5-slot methods; N=9 DEC up to 64)
mods = {
 # name: (gens per item, note)
 "F 2x2x10 banks":            (40,  "4 cells x 10; S_FRESH/IND first-10 alias"),
 "A six methods @B4":         (5*20 - 20, "S_FRESH+S_HISTORY+IND+DEC+CEN_FLAT ~20 calls each; minus aliases from F (IND 5 roots+S_FRESH 10)"),
 "A + judge/selector calls":  (3*20, "JUDGE_BEST pointwise <=1 call/candidate for IND/S_FRESH/S_HISTORY banks (1024-tok, cheap x0.3)"),
 "C shadow forecast":         (5, "1 forecast per method-episode (256 tok, cheap)"),
 "N panel N=1,2,3,5,9 x3":    (3*(4+8+12+20+64) - 20, "IND/DEC/CEN at 00 framing; N=9 DEC to 64-call cap"),
 "D degree module":           (5*5, "9 roots alias F00; 5 degrees x (1 rev + 4 children)"),
 "E repeated episodes":       (9*4*20, "9 extra episodes x 4 methods x ~20 calls"),
 "A-budget B1,B2,B8":         (5*(5+10+40), "5 methods; B8 ~40 calls"),
 "M ckpt panel per model":    (3*20, "IND/DEC/CEN_FLAT @N=5 B4 (RLM excluded)"),
}
for n,(g,note) in mods.items(): print(f"{n:28s} {g:5d} gens/item   {note}")
print()
scen = {"N400": dict(F=400, A=400, C=400, N=100, D=100, E=40, B=100, M=150),
        "N300": dict(F=300, A=300, C=300, N=100, D=100, E=40, B=60,  M=150),
        "N600": dict(F=600, A=600, C=600, N=150, D=150, E=60, B=150, M=200)}
for sn, s in scen.items():
    g32 = (40*s["F"] + 80*s["A"] + 0.3*60*s["A"] + 0.1*5*s["C"] + 304*s["N"] + 25*s["D"] + 720*s["E"] + 275*s["B"])
    print(f"--- {sn}: 32B gens={g32:,.0f}  tokens={g32*TOK/1e6:,.0f}M")
    for gpus in (12, 14, 16, 20):
        print(f"    32B on {gpus} GPUs: {hours(g32,gpus,'32B'):5.1f} h")
    for size,gpus in (("14B",2),("8B",2),("4B",1)):
        gm = 60*s["M"]
        print(f"    {size} ckpt panel on {gpus} GPU: {hours(gm,gpus,size):5.1f} h  ({gm:,} gens)")
