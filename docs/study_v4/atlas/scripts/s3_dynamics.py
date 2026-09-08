"""A8/A9: within-episode dynamics from the item files — peer-revision rounds and hub cycles."""
import sys, os, json, glob, hashlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import _common as C

E = C.Emitter("s3_dynamics.py")
R = C.RUN_ROOT

# ---------- candidate correctness ----------
bcb = {}
for p in glob.glob(f"{R}/eval/bcb/*.jsonl"):
    for line in open(p):
        try: q = json.loads(line)
        except Exception: continue
        bcb[q["candidate_id"]] = (q.get("status") == "pass")
_hle = {}
def hle_j(sid):
    if sid not in _hle:
        p = f"{R}/eval/hle/{sid}.json"
        _hle[sid] = (json.load(open(p)).get("judgements") or {}) if os.path.exists(p) else {}
    return _hle[sid]
def correct(sid, c):
    if sid.startswith("bcb"): return bcb.get(c.get("candidate_id"))
    js = hle_j(sid); fa = str((c.get("candidate") or {}).get("final_answer", ""))
    for k in (c.get("candidate_id"), c.get("candidate_sha256"), hashlib.sha256(fa.encode()).hexdigest()):
        if k in js: return bool(js[k].get("correct"))
    return None

def item_files(pattern):
    for cell in sorted(glob.glob(f"{R}/cells/{pattern}")):
        for p in sorted(glob.glob(f"{cell}/items/*.json")) + sorted(glob.glob(f"{cell}/items/*/*.json")):
            yield p

def answer_key(c):
    """Cross-round grouping key. `vote_key` is only populated in the sealed final round, so the
    per-round trajectory uses exact-answer identity (candidate_sha256). This over-splits on code
    exactly as documented for AST grouping; the official vote_key is reported at the final round."""
    return c.get("candidate_sha256")


def conf_of(c):
    try: return float((c.get("candidate") or {}).get("confidence"))
    except Exception: return None

STAGES = ["root"] + [f"round{i}" for i in range(1, 9)]

# ================= peer revision, round by round =================
rows, switch_rows = [], []
for p in item_files("A.DEC.32B.N5.B4.Fnat.e0.s*"):
    d = json.load(open(p))
    if d.get("split") != "main": continue
    sid = d["source_id"]; ep = d.get("episode") or {}
    by = collections.defaultdict(dict)
    for c in ep.get("candidates") or []:
        if c.get("stage") and c.get("slot") is not None: by[c["stage"]][c["slot"]] = c
    order = [s for s in STAGES if s in by]
    prev = None
    for ri, st in enumerate(order):
        members = by[st]
        valid = {s: c for s, c in members.items() if c.get("valid")}
        keys = [answer_key(c) for c in valid.values() if answer_key(c)]
        cnt = collections.Counter(keys)
        labs = {s: correct(sid, c) for s, c in valid.items()}
        known = [v for v in labs.values() if v is not None]
        rows.append(dict(sid=sid, domain="bcb" if sid.startswith("bcb") else "hle", round=ri, stage=st,
                         n_valid=len(valid), valid_frac=len(valid) / max(len(members), 1),
                         agreement=(max(cnt.values()) / len(keys)) if keys else np.nan,
                         unanimous=float(len(cnt) == 1) if keys else np.nan,
                         conf=np.mean([c for c in (conf_of(c) for c in valid.values()) if c is not None]) if valid else np.nan,
                         cand_correct=float(np.mean(known)) if known else np.nan,
                         oracle=float(any(known)) if known else np.nan,
                         label_cov=len(known) / max(len(valid), 1)))
        if prev is not None:
            maj = cnt.most_common(1)[0][0] if cnt else None
            for s, c in valid.items():
                pc = prev.get(s)
                if pc is None or not pc.get("valid"): continue
                a, b = answer_key(pc), answer_key(c)
                if a is None or b is None: continue
                ca, cb = correct(sid, pc), correct(sid, c)
                switch_rows.append(dict(sid=sid, domain="bcb" if sid.startswith("bcb") else "hle", round=ri,
                                        switched=int(a != b), to_majority=int(a != b and b == maj),
                                        was_correct=ca, now_correct=cb,
                                        conf_before=conf_of(pc), conf_after=conf_of(c),
                                        in_minority_before=int(a != maj) if maj else None))
        prev = members
dec = pd.DataFrame(rows); sw = pd.DataFrame(switch_rows)
for col in ("valid_frac", "agreement", "unanimous", "conf", "cand_correct", "oracle", "label_cov"):
    dec[col] = pd.to_numeric(dec[col], errors="coerce")
for col in ("switched", "to_majority", "conf_before", "conf_after", "in_minority_before"):
    if col in sw: sw[col] = pd.to_numeric(sw[col], errors="coerce")
for col in ("was_correct", "now_correct"):
    if col in sw: sw[col] = sw[col].map({True: 1.0, False: 0.0})
E.emit("dec.episodes", int(dec.sid.nunique()), n=int(len(dec)), source="A.DEC main item files")
agg = dec.groupby("round").agg(items=("sid", "nunique"), valid=("valid_frac", "mean"), agreement=("agreement", "mean"),
                               unanimous=("unanimous", "mean"), conf=("conf", "mean"),
                               cand_correct=("cand_correct", "mean"), oracle=("oracle", "mean"),
                               label_cov=("label_cov", "mean")).round(4)
E.emit("dec.by_round", agg.to_dict("index"), note="round 0 = independent roots, before any peer content")
E.emit("dec.by_round_domain", {dom: g.groupby("round").agg(agreement=("agreement", "mean"), conf=("conf", "mean"),
                                                           cand_correct=("cand_correct", "mean"),
                                                           oracle=("oracle", "mean")).round(4).to_dict("index")
                               for dom, g in dec.groupby("domain")})
# herding accounting
if len(sw):
    s2 = sw.dropna(subset=["was_correct", "now_correct"]).copy()
    s2["was_correct"] = s2.was_correct.astype(bool); s2["now_correct"] = s2.now_correct.astype(bool)
    E.emit("dec.switching", dict(transitions=int(len(sw)), switch_rate=float(sw.switched.mean()),
                                 to_majority_given_switch=float(sw[sw.switched == 1].to_majority.mean()),
                                 minority_switch_rate=float(sw[sw.in_minority_before == 1].switched.mean()),
                                 majority_switch_rate=float(sw[sw.in_minority_before == 0].switched.mean())))
    E.emit("dec.flip_accounting", dict(
        correct_to_wrong=int((s2.was_correct & ~s2.now_correct).sum()),
        wrong_to_correct=int((~s2.was_correct & s2.now_correct).sum()),
        correct_kept=int((s2.was_correct & s2.now_correct).sum()),
        wrong_kept=int((~s2.was_correct & ~s2.now_correct).sum()),
        labelled_transitions=int(len(s2))))
    E.emit("dec.confidence_after_switch",
           dict(switchers_before=float(sw[sw.switched == 1].conf_before.mean(skipna=True)),
                switchers_after=float(sw[sw.switched == 1].conf_after.mean(skipna=True)),
                stayers_before=float(sw[sw.switched == 0].conf_before.mean(skipna=True)),
                stayers_after=float(sw[sw.switched == 0].conf_after.mean(skipna=True))))
# official sealed-pool agreement at the final round (vote_key grouping)
off = []
for p2 in item_files("A.DEC.32B.N5.B4.Fnat.e0.s*"):
    d = json.load(open(p2))
    if d.get("split") != "main": continue
    ep2 = d.get("episode") or {}
    by2 = collections.defaultdict(dict)
    for c in ep2.get("candidates") or []:
        if c.get("stage") and c.get("slot") is not None: by2[c["stage"]][c["slot"]] = c
    order2 = [s2 for s2 in STAGES if s2 in by2]
    if not order2: continue
    ks = [c.get("vote_key") for c in by2[order2[-1]].values() if c.get("valid") and c.get("vote_key")]
    if ks:
        cnt2 = collections.Counter(ks)
        off.append(dict(sid=d["source_id"], agreement=max(cnt2.values()) / len(ks), unanimous=float(len(cnt2) == 1)))
if off:
    O = pd.DataFrame(off)
    E.emit("dec.final_agreement_votekey", dict(agreement=float(O.agreement.mean()), unanimous=float(O.unanimous.mean()),
                                               n=int(len(O))),
           note="final round only, official vote_key grouping (AST-canonical on code)")
    O.to_parquet(f"{C.DATA}/dec_final_agreement.parquet")

# lone-correct survival: episodes where exactly one member was right at roots
lone = []
for sid, g in dec.groupby("sid"):
    g = g.sort_values("round")
    if g.iloc[0].n_valid >= 4 and g.iloc[0].cand_correct is not None and not np.isnan(g.iloc[0].cand_correct):
        n_correct0 = g.iloc[0].cand_correct * g.iloc[0].n_valid
        if 0.5 <= n_correct0 <= 1.5:
            lone.append(dict(sid=sid, final_oracle=g.iloc[-1].oracle, final_correct=g.iloc[-1].cand_correct))
if lone:
    L = pd.DataFrame(lone)
    E.emit("dec.lone_correct", dict(episodes=int(len(L)), lost_by_final_round=float(1 - L.final_oracle.mean()),
                                    mean_final_share_correct=float(L.final_correct.mean())),
           note="episodes with exactly one correct member at the root round")

# ================= hub cycles =================
hrows, err_codes, tok_by_index, trans = [], collections.Counter(), collections.defaultdict(list), collections.Counter()
for p in item_files("A.CEN_FLAT.32B.N5.B4.Fnat.e0.s*"):
    d = json.load(open(p))
    if d.get("split") != "main": continue
    sid = d["source_id"]; ep = d.get("episode") or {}
    calls = ep.get("calls") or []; ctr = ep.get("counters") or {}
    by_role = ctr.get("calls_by_role") or {}
    errs = ep.get("hub_action_errors") or []
    for e in errs:
        code = e.get("code") or e.get("failure_code") or e.get("error") or (e if isinstance(e, str) else "UNKNOWN")
        err_codes[str(code)] += 1
    fr = [c.get("finish_reason") for c in calls]
    for i, c in enumerate(calls):
        if c.get("completion_tokens") is not None:
            tok_by_index[min(i, 5)].append((c["completion_tokens"], c.get("finish_reason") == "length"))
    for a, b in zip(fr, fr[1:]):
        trans[f"{a}->{b}"] += 1
    n_trunc = sum(1 for x in fr if x == "length")
    hrows.append(dict(sid=sid, domain="bcb" if sid.startswith("bcb") else "hle",
                      calls=ctr.get("hub_calls") or by_role.get("hub") or len(calls),
                      admitted=(ep.get("ledger") or {}).get("calls_admitted"),
                      worker_calls=int(by_role.get("worker", 0)),
                      delegation_cycles=int(ctr.get("delegation_cycles") or 0),
                      delegated=int((ctr.get("delegation_cycles") or 0) > 0),
                      hub_action_errors=int(ctr.get("hub_action_errors") or len(errs)),
                      truncated=n_trunc, trunc_frac=n_trunc / max(len(calls), 1),
                      n_calls=len(calls), stop=ep.get("stop_reason")))
hub = pd.DataFrame(hrows)
E.emit("hub.episodes", int(len(hub)), source="A.CEN_FLAT main item files")
E.emit("hub.call_distribution", hub.calls.value_counts().sort_index().to_dict())
E.emit("hub.delegation_rate", float(hub.delegated.mean()), n=int(len(hub)),
       note="episodes with at least one delegation cycle (counters.delegation_cycles > 0)")
E.emit("hub.by_domain", hub.groupby("domain").agg(calls=("calls", "mean"), delegated=("delegated", "mean"),
                                                  trunc=("trunc_frac", "mean"), errors=("hub_action_errors", "mean"),
                                                  n=("sid", "size")).round(4).to_dict("index"))
E.emit("hub.action_error_codes", dict(err_codes.most_common()), n=int(sum(err_codes.values())),
       note="typed coordinator_action errors; each consumes a hub call and a cycle")
E.emit("hub.truncation", dict(mean_trunc_frac=float(hub.trunc_frac.mean()),
                              episodes_any_truncation=float((hub.truncated > 0).mean()),
                              episodes_all_truncated=float(((hub.truncated == hub.n_calls) & (hub.n_calls > 1)).mean())))
E.emit("hub.tokens_by_call_index",
       {str(i): dict(mean_tokens=float(np.mean([t for t, _ in v])), share_at_cap=float(np.mean([f for _, f in v])),
                     mean_tokens_uncapped=float(np.mean([t for t, f in v if not f])) if any(not f for _, f in v) else None,
                     n=len(v))
        for i, v in sorted(tok_by_index.items())},
       note="index 5 pools calls 6+; 'uncapped' excludes calls that hit the 8192 output cap, which is the honest version of 'reasons longer each retry'")
E.emit("hub.finish_reason_transitions", dict(trans.most_common(8)),
       note="consecutive hub calls; length->length is the re-prompt loop")
hub["bucket"] = pd.cut(hub.calls, [0, 1, 2, 4, 999], labels=["1", "2", "3-4", ">=5"])
ep_all = C.episodes(); acc_map = ep_all[(ep_all.module == "A") & (ep_all.method == "CEN_FLAT") &
                                        (ep_all.checkpoint == "32B")].set_index("source_id").acc
hub["acc"] = hub.sid.map(acc_map)
diff = C.difficulty().set_index("source_id")
hub["band"] = hub.sid.map(diff.band); hub["p1"] = hub.sid.map(diff.pass_at_1)
E.emit("hub.by_call_bucket", hub.groupby("bucket", observed=True).agg(
    n=("sid", "size"), acc=("acc", "mean"), p1=("p1", "mean"), trunc=("trunc_frac", "mean"),
    errors=("hub_action_errors", "mean")).round(4).to_dict("index"),
    note="item difficulty (p1) is shown beside accuracy because long episodes sit on much harder items")
E.emit("hub.call_bucket_band_mix", pd.crosstab(hub.bucket, hub.band).to_dict())
# difficulty-matched: within the never-solved band only, does call count still predict failure?
nb = hub[(hub.band == "never") & hub.acc.notna()]
if len(nb) > 30:
    E.emit("hub.within_never_band", nb.groupby("bucket", observed=True).agg(n=("sid", "size"), acc=("acc", "mean")).round(4).to_dict("index"),
           note="difficulty held constant: the call-count gradient largely disappears")
one = hub[hub.calls == 1]
bank = C.difficulty()[["source_id", "pass_at_1"]].set_index("source_id").pass_at_1
paired = one.dropna(subset=["acc"]).assign(draw=lambda x: x.sid.map(bank)).dropna(subset=["draw"])
E.emit("hub.single_call_vs_single_draw", C.boot_paired(paired.acc.values, paired.draw.values),
       note="hub episodes that never delegated vs the neutral 10-draw bank pass@1 on the same items")

pd.concat([dec.assign(kind="dec")], ignore_index=True).to_parquet(f"{C.DATA}/dec_rounds.parquet")
hub.to_parquet(f"{C.DATA}/hub_episodes.parquet")
E.save("dynamics")
