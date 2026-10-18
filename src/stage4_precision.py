"""Stage 4 precision from a hand-labelled audit sample.

Reads results/stage4_fresh_audit.csv (written by `stage4_audit.py --fresh-sample`) after the
'my_verdict' column has been filled in with real_error / false_alarm / extraction_bug, and prints
per split:  precision = real_error / (real_error + false_alarm + extraction_bug)
with a 95% Wilson score interval. Rows with an empty verdict are counted as not yet labelled.

Usage:
  python src/stage4_precision.py results/stage4_fresh_audit.csv
"""
import argparse, csv, math
from collections import Counter, defaultdict

VERDICTS = ("real_error", "false_alarm", "extraction_bug")


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="results/stage4_fresh_audit.csv")
    a = ap.parse_args()

    counts, unlabelled, unknown = defaultdict(Counter), Counter(), Counter()
    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = r["my_verdict"].strip().lower()
            if not v:
                unlabelled[r["split"]] += 1
            elif v in VERDICTS:
                counts[r["split"]][v] += 1
            else:
                unknown[(r["split"], v)] += 1

    splits = list(dict.fromkeys(list(counts) + list(unlabelled)))
    total = Counter()
    print(f"{'split':15s} {'real':>5s} {'false':>6s} {'extr':>5s} {'n':>4s}  precision  95% Wilson CI   unlabelled")
    for s in splits + ["ALL"]:
        c = total if s == "ALL" else counts[s]
        n = sum(c[v] for v in VERDICTS)
        lo, hi = wilson(c["real_error"], n)
        prec = f"{100 * c['real_error'] / n:8.1f}%" if n else "      n/a"
        ci = f"[{100 * lo:5.1f}%, {100 * hi:5.1f}%]" if n else "      n/a      "
        left = sum(unlabelled.values()) if s == "ALL" else unlabelled[s]
        print(f"{s:15s} {c['real_error']:5d} {c['false_alarm']:6d} {c['extraction_bug']:5d} {n:4d}  {prec}  {ci}  {left:6d}")
        if s != "ALL":
            total.update(c)
    for (s, v), k in unknown.items():
        print(f"WARNING: {k} row(s) in {s} have an unknown verdict {v!r} (expected one of {', '.join(VERDICTS)})")
