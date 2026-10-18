"""Stage 4 audit: dump the Stage 4 decisions worth checking by hand into a CSV.

Rows (one per equation):
  early        - every solution whose first failing step is EARLIER than the human label
                 (the failing equations of that step)
  false_alarm  - a random sample of failing equations in solutions labelled all-correct (label == -1)
  skip         - a random sample of skipped equations from any solution

Fill in the blank 'my_verdict' column, e.g. real_error / stage1_extraction / stage4_parser / ok.

Usage:
  python src/stage4_audit.py data/processed/s4_math.jsonl results/stage4_audit_math.csv [--n 30] [--seed 0]
"""
import argparse, csv, json, random

FIELDS = ["category", "id", "human_label", "first_fail_step", "step_idx",
          "equation", "status", "reason", "step_text", "my_verdict"]


def row(category, rec, step_idx, eq):
    return {"category": category, "id": rec["id"], "human_label": rec["label"],
            "first_fail_step": rec["sympy"]["first_fail_step"], "step_idx": step_idx,
            "equation": eq["equation"], "status": eq["status"], "reason": eq["reason"],
            "step_text": rec["steps"][step_idx]["text"], "my_verdict": ""}


def audit_rows(recs, n, seed):
    rng = random.Random(seed)
    early, false_alarms, skips = [], [], []
    for r in recs:
        ff, label = r["sympy"]["first_fail_step"], r["label"]
        for st in r["sympy"]["steps"]:
            for eq in st["equations"]:
                if eq["status"] == "fail" and label not in (-1, None) and ff is not None \
                        and ff < label and st["idx"] == ff:
                    early.append(row("early", r, st["idx"], eq))
                elif eq["status"] == "fail" and label == -1:
                    false_alarms.append(row("false_alarm", r, st["idx"], eq))
                elif eq["status"] == "skip":
                    skips.append(row("skip", r, st["idx"], eq))
    sample = lambda xs: rng.sample(xs, min(n, len(xs)))
    return early + sample(false_alarms) + sample(skips), len(early), len(false_alarms), len(skips)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--n", type=int, default=30, help="sample size for false alarms and for skips")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(a.input, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    rows, n_early, n_fa, n_skip = audit_rows(recs, a.n, a.seed)
    with open(a.output, "w", encoding="utf-8-sig", newline="") as out:   # utf-8-sig so Excel reads it
        w = csv.DictWriter(out, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{a.output}: {len(rows)} rows  (early: all {n_early} equations; "
          f"false alarms: {min(a.n, n_fa)} of {n_fa}; skips: {min(a.n, n_skip)} of {n_skip})")
