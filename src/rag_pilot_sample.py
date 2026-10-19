"""RAG pilot sample: 150 solutions from the sampled problems, split once into DEV and TEST halves.

Per split (only problems with Stage 2 samples, i.e. the ids in s7_<split>.jsonl):
  GSM8K 40, MATH 40, OlympiadBench 35, Omni-MATH 35 solutions; about half correct (label = -1) and half
  erroneous (label >= 0). The erroneous ones include HARD cases (correct final answer, wrong step):
  N_HARD per split, or all that exist if fewer (GSM8K has only 7). This over-represents hard cases relative to
  the splits (by design, so the hard subset can be scored at all); the rest of the erroneous ones are random.
DEV / TEST: stratified by (split, stratum = correct / hard / other_error), shuffled with the same seed; within a
stratum TEST gets the larger half for hard cases (to reach >= 15 hard TEST cases) and the smaller half otherwise.
Thresholds and choices may be tuned on DEV only; everything is reported on TEST.
The label is used here only to draw the sample and to evaluate; no signal reads this file's label columns.

Usage:
  python src/rag_pilot_sample.py [--seed 0] [--out results/rag_pilot_sample.csv]
"""
import argparse, csv, json, random
from collections import Counter

PLAN = {"gsm8k": 40, "math": 40, "olympiadbench": 35, "omnimath": 35}
N_HARD = {"gsm8k": 5, "math": 8, "olympiadbench": 8, "omnimath": 8}
P = "data/processed"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def draw(seed):
    rows = []
    for sp, n in PLAN.items():
        rng = random.Random(f"{seed}-{sp}")
        ids = [r["id"] for r in read_jsonl(f"{P}/s7_{sp}.jsonl")]
        raw = {r["id"]: r for r in read_jsonl(f"data/raw/processbench_{sp}.jsonl")}
        stratum = lambda i: ("correct" if raw[i]["label"] == -1 else
                             "hard" if raw[i]["final_answer_correct"] is True else "other_error")
        pools = {s: sorted(i for i in ids if stratum(i) == s) for s in ("correct", "hard", "other_error")}
        n_err = (n + 1) // 2 if len(pools["correct"]) < len(pools["hard"]) + len(pools["other_error"]) else n // 2
        n_hard = min(N_HARD[sp], len(pools["hard"]))
        take = {"correct": n - n_err, "hard": n_hard, "other_error": n_err - n_hard}
        for s, k in take.items():
            chosen = rng.sample(pools[s], k)
            n_test = (k + 1) // 2 if s == "hard" else k // 2
            for j, i in enumerate(chosen):                 # chosen is already in random order
                rows.append({"id": i, "split": sp, "stratum": s, "label": raw[i]["label"],
                             "final_answer_correct": raw[i]["final_answer_correct"],
                             "n_steps": len(raw[i]["steps"]), "half": "TEST" if j < n_test else "DEV"})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rag_pilot_sample.csv")
    a = ap.parse_args()
    rows = draw(a.seed)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = Counter((r["split"], r["half"], r["stratum"]) for r in rows)
    print(f"{len(rows)} solutions -> {a.out}")
    print(f"{'split':14s} {'half':5s} {'correct':>8s} {'hard':>5s} {'other_err':>9s} {'total':>6s}")
    for sp in PLAN:
        for h in ("DEV", "TEST"):
            k = [c[(sp, h, s)] for s in ("correct", "hard", "other_error")]
            print(f"{sp:14s} {h:5s} {k[0]:8d} {k[1]:5d} {k[2]:9d} {sum(k):6d}")
    for h in ("DEV", "TEST"):
        k = [sum(v for (sp, hh, s), v in c.items() if hh == h and s == st) for st in ("correct", "hard", "other_error")]
        print(f"{'ALL':14s} {h:5s} {k[0]:8d} {k[1]:5d} {k[2]:9d} {sum(k):6d}")
