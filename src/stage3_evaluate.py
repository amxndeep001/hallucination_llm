"""Stage 3 evaluation: does the first unsupported step point at the human first wrong step?

Predicted first error:
  Stage 3   earliest step with support_min <= t, for t = 0.0, 0.2 (the default in s3_first_low_step), 0.4
  Stage 4   first_fail_step of the SymPy verifier
  union     min(Stage 3, Stage 4) over the predictions that exist
Metrics as in Stage 4's evaluate(): share of erroneous solutions (label != -1) flagged; exact match with the
human first wrong step; flagged earlier / later; false alarms on correct solutions (label == -1); precision of
a flag; ceiling = erroneous solutions whose first wrong step has a signal. Chance for exact match = predicting
a uniformly random step of each erroneous solution (mean of 1 / number of steps).

Usage:
  python src/stage3_evaluate.py data/processed/s3_gsm8k.jsonl data/processed/s4_gsm8k.jsonl \
         data/processed/pb_gsm8k.jsonl data/raw/processbench_gsm8k.jsonl [--results-dir results]
Writes results/stage3_summary_<split>.csv.
"""
import argparse, csv, os, re
from stage3_align import read_jsonl

THRESHOLDS = [0.0, 0.2, 0.4]
POSITIONS = [("step 0", lambda l: l == 0), ("step 1", lambda l: l == 1), ("step 2", lambda l: l == 2),
             ("step 3+", lambda l: l >= 3)]


def first_low(rec, t):
    return min((s["idx"] for s in rec["steps"] if s["support_min"] is not None and s["support_min"] <= t), default=None)


def union(*preds):
    p = [x for x in preds if x is not None]
    return min(p) if p else None


def metrics(recs, pred, ceiling=None):
    """pred: id -> predicted first wrong step or None (not flagged)."""
    err = [r for r in recs if r["label"] != -1]
    ok = [r for r in recs if r["label"] == -1]
    flagged = [r for r in err if pred[r["id"]] is not None]
    exact = [r for r in err if pred[r["id"]] == r["label"]]
    fp = sum(pred[r["id"]] is not None for r in ok)
    pct = lambda n, d: round(100 * n / d, 1) if d else float("nan")
    m = {"n_erroneous": len(err), "n_correct": len(ok),
         "flagged_pct": pct(len(flagged), len(err)), "exact_match_pct": pct(len(exact), len(err)),
         "earlier": sum(pred[r["id"]] < r["label"] for r in flagged),
         "later": sum(pred[r["id"]] > r["label"] for r in flagged),
         "false_alarm_pct": pct(fp, len(ok)), "precision_pct": pct(len(flagged), len(flagged) + fp)}
    if ceiling is not None:
        m["ceiling_pct"] = pct(sum(ceiling(r) for r in err), len(err))
    for name, f in POSITIONS:
        sub = [r for r in err if f(r["label"])]
        m[f"exact_{name.replace(' ', '')}_pct"] = pct(sum(pred[r["id"]] == r["label"] for r in sub), len(sub))
        m[f"n_{name.replace(' ', '')}"] = len(sub)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("s3")
    ap.add_argument("s4")
    ap.add_argument("pb")
    ap.add_argument("raw")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    split = re.sub(r"^s3_", "", os.path.splitext(os.path.basename(a.s3))[0])
    s3 = read_jsonl(a.s3)
    s4 = {r["id"]: r for r in read_jsonl(a.s4)}
    pb = {r["id"]: r for r in read_jsonl(a.pb)}
    raw = {r["id"]: r for r in read_jsonl(a.raw)}
    missing = [r["id"] for r in s3 if r["id"] not in s4 or r["id"] not in pb or r["id"] not in raw]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} ids missing from s4/pb/raw, e.g. {missing[:5]}")
    bad = [r["id"] for r in s3 if not (r["label"] == raw[r["id"]]["label"] == s4[r["id"]]["label"])]
    if bad:
        raise SystemExit(f"ERROR: labels differ between s3/s4/raw for {len(bad)} ids, e.g. {bad[:5]}")

    s4_pred = {i: r["sympy"]["first_fail_step"] for i, r in s4.items()}
    has_signal = lambda r: r["steps"][r["label"]]["support_min"] is not None
    has_eq = lambda r: bool(pb[r["id"]]["steps"][r["label"]]["equations"])
    methods = []
    for t in THRESHOLDS:
        methods.append((f"Stage 3 (support_min <= {t})", t, {r["id"]: first_low(r, t) for r in s3}, has_signal))
    methods.append(("Stage 4 (SymPy first fail)", "", s4_pred, has_eq))
    for t in THRESHOLDS:
        methods.append((f"union Stage 3 (<= {t}) + Stage 4", t,
                        {r["id"]: union(first_low(r, t), s4_pred[r["id"]]) for r in s3},
                        lambda r: has_signal(r) or has_eq(r)))

    err = [r for r in s3 if r["label"] != -1]
    chance = 100 * sum(1 / len(r["steps"]) for r in err) / len(err)
    print(f"===== Stage 3 step-level evaluation: {split}, {len(s3)} problems, {len(err)} with a process error, "
          f"{len(s3) - len(err)} correct")
    print(f"chance exact match (uniformly random step of each erroneous solution): {chance:.1f}%")
    print(f"chance precision of a flag (= share of erroneous solutions): {100*len(err)/len(s3):.1f}%\n")
    rows = []
    print("| method | flagged % | exact match % | earlier | later | false alarm % | precision % | ceiling % |")
    print("|---|---|---|---|---|---|---|---|")
    for name, t, pred, ceil in methods:
        m = metrics(s3, pred, ceil)
        rows.append({"split": split, "method": name, "threshold": t, **m, "chance_exact_pct": round(chance, 1)})
        print(f"| {name} | {m['flagged_pct']} | {m['exact_match_pct']} | {m['earlier']} | {m['later']} | "
              f"{m['false_alarm_pct']} | {m['precision_pct']} | {m['ceiling_pct']} |")
    print("(ceiling: erroneous solutions whose human first wrong step has a key number (Stage 3) / an equation (Stage 4))")

    print(f"\n## Exact match % by position of the human first wrong step (n per bucket)\n")
    print("| method | " + " | ".join(f"{n} (n={rows[0]['n_' + n.replace(' ', '')]})" for n, _ in POSITIONS) + " |")
    print("|---|" + "---|" * len(POSITIONS))
    for r in rows:
        print(f"| {r['method']} | " + " | ".join(str(r[f"exact_{n.replace(' ', '')}_pct"]) for n, _ in POSITIONS) + " |")
    ch = {n: [r for r in err if f(r["label"])] for n, f in POSITIONS}
    print("| chance (random step) | " + " | ".join(f"{100*sum(1/len(r['steps']) for r in v)/len(v):.1f}" for v in ch.values()) + " |")

    hard = [r for r in s3 if raw[r["id"]]["final_answer_correct"] is True and r["label"] != -1]
    print(f"\n## HARD: final answer correct but a wrong step: n = {len(hard)}"
          f"{'  -- NOT RELIABLE (< 30)' if len(hard) < 30 else ''}")
    for t in THRESHOLDS:
        fl = [r for r in hard if first_low(r, t) is not None]
        ex = [r for r in hard if first_low(r, t) == r["label"]]
        print(f"  Stage 3 (<= {t}): flagged {len(fl)} of {len(hard)}, exact first wrong step {len(ex)}")
    print(f"  Stage 4: flagged {sum(s4_pred[r['id']] is not None for r in hard)} of {len(hard)}, "
          f"exact {sum(s4_pred[r['id']] == r['label'] for r in hard)}")

    os.makedirs(a.results_dir, exist_ok=True)
    out_csv = os.path.join(a.results_dir, f"stage3_summary_{split}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out_csv}")

    print(f"\n## 10 most confident Stage 3 flags (support_min <= 0.2) that are NOT the human first wrong step\n"
          f"(confidence: lower support_min, then lower support_mean, then more key numbers)\n")
    cand = []
    for r in s3:
        k = first_low(r, 0.2)
        if k is not None and k != r["label"]:
            st = r["steps"][k]
            cand.append(((st["support_min"], st["support_mean"], -st["n_key"], r["id"]), r, k, st))
    cand.sort(key=lambda c: c[0])
    for _, r, k, st in cand[:10]:
        text = pb[r["id"]]["steps"][k]["text"].replace("\n", " ")
        print(f"- {r['id']} | label {r['label']} | flagged step {k} | key numbers {st['key_numbers']} | "
              f"support_min {st['support_min']:.1f} mean {st['support_mean']:.2f}\n    {text[:400]}")
    print(f"({len(cand)} such flags in total; {sum(c[0][0] == 0 for c in cand)} with support_min 0, tied in support -> "
          f"ordered by more key numbers, then id)")
