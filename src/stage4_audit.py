"""Stage 4 audit: dump the Stage 4 decisions worth checking by hand into a CSV.

Default mode, one split (rows are equations):
  early        - every solution whose first failing step is EARLIER than the human label
                 (the failing equations of that step)
  false_alarm  - a random sample of failing equations in solutions labelled all-correct (label == -1)
  skip         - a random sample of skipped equations from any solution
  Fill in 'my_verdict', e.g. real_error / stage1_extraction / stage4_parser / ok.

--fresh-sample mode, all splits (rows are solutions): an unbiased sample for measuring precision.
  From each s4_<split>.jsonl, 30 random solutions that Stage 4 flagged (n_fail > 0), excluding every
  solution id already in results/stage4_audit_<split>.csv. One row per solution with its first failing
  equation. Fill in 'my_verdict' with real_error / false_alarm / extraction_bug, then run
  src/stage4_precision.py on the file.

Usage:
  python src/stage4_audit.py data/processed/s4_math.jsonl results/stage4_audit_math.csv [--n 30] [--seed 0]
  python src/stage4_audit.py --fresh-sample results/stage4_fresh_audit.csv [--n 30] [--seed 123]
"""
import argparse, csv, json, os, random

SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]
FIELDS = ["category", "id", "human_label", "first_fail_step", "step_idx",
          "equation", "status", "reason", "step_text", "my_verdict"]
FRESH_FIELDS = ["split", "id", "human_label", "label_is_correct", "first_fail_step",
                "n_fail", "equation", "reason", "step_text", "my_verdict"]


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


def fresh_rows(split, recs, exclude, n, seed):
    """n random flagged solutions not already audited; one row each, with its first failing equation."""
    pool = [r for r in recs if r["sympy"]["n_fail"] > 0 and r["id"] not in exclude]
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    rows = []
    for r in picked:
        ff = r["sympy"]["first_fail_step"]
        eq = next(e for st in r["sympy"]["steps"] if st["idx"] == ff
                  for e in st["equations"] if e["status"] == "fail")
        rows.append({"split": split, "id": r["id"], "human_label": r["label"],
                     "label_is_correct": r["label"] == -1, "first_fail_step": ff,
                     "n_fail": r["sympy"]["n_fail"], "equation": eq["equation"], "reason": eq["reason"],
                     "step_text": r["steps"][ff]["text"], "my_verdict": ""})
    return rows, len(pool)


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as out:      # utf-8-sig so Excel reads it
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="s4_<split>.jsonl (default mode)")
    ap.add_argument("output", nargs="?", help="audit CSV (default mode)")
    ap.add_argument("--fresh-sample", metavar="CSV", help="write the all-splits precision sample to CSV")
    ap.add_argument("--n", type=int, default=30, help="sample size per category / per split")
    ap.add_argument("--seed", type=int, default=None, help="default 0, or 123 with --fresh-sample")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--audit-dir", default="results", help="where the earlier stage4_audit_<split>.csv files are")
    a = ap.parse_args()

    if a.fresh_sample:
        seed = 123 if a.seed is None else a.seed
        rows = []
        for split in SPLITS:
            prev = os.path.join(a.audit_dir, f"stage4_audit_{split}.csv")
            exclude = set()
            if os.path.exists(prev):
                with open(prev, encoding="utf-8-sig", newline="") as f:
                    exclude = {r["id"] for r in csv.DictReader(f)}
            recs = read_jsonl(os.path.join(a.processed_dir, f"s4_{split}.jsonl"))
            split_rows, n_pool = fresh_rows(split, recs, exclude, a.n, seed)
            rows += split_rows
            print(f"{split}: {len(split_rows)} of {n_pool} flagged solutions not yet audited "
                  f"({len(exclude)} ids excluded)")
        write_csv(a.fresh_sample, FRESH_FIELDS, rows)
        print(f"{a.fresh_sample}: {len(rows)} rows")
    else:
        if not (a.input and a.output):
            ap.error("give INPUT and OUTPUT, or --fresh-sample CSV")
        rows, n_early, n_fa, n_skip = audit_rows(read_jsonl(a.input), a.n, 0 if a.seed is None else a.seed)
        write_csv(a.output, FIELDS, rows)
        print(f"{a.output}: {len(rows)} rows  (early: all {n_early} equations; "
              f"false alarms: {min(a.n, n_fa)} of {n_fa}; skips: {min(a.n, n_skip)} of {n_skip})")
