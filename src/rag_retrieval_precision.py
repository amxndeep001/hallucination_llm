"""Source D retrieval precision from the hand-labelled audit (results/rag_retrieval_audit.csv).

Each row is one knowledge_claim step and one retriever (bm25 or bge) with its top-5 ProofWiki passages.
Fill 'my_verdict' with the ranks of the passages that are RELEVANT to the claim (state the fact the step uses,
or one that decides whether the step is right), comma-separated, e.g. '1,3'; write 0 (or 'none') if none is.
Prints per retriever (and for all rows):
  precision@1 = rows whose top-1 passage is relevant / labelled rows
  hit@5       = rows with at least one relevant passage in the top 5 / labelled rows
  precision@5 = relevant passages / (5 x labelled rows)   (the 5 passages of a row are not independent, so its
                Wilson interval is too narrow; read it as approximate)
with 95% Wilson score intervals (as src/stage4_precision.py). Rows with an empty verdict are not yet labelled.

Usage:
  python src/rag_retrieval_precision.py [results/rag_retrieval_audit.csv]
"""
import argparse, csv, re
from collections import defaultdict
from stage4_precision import wilson


def parse_verdict(v):
    """'' -> None (unlabelled); '0' / 'none' -> empty set; '1, 3' -> {1, 3}. Raises on anything else."""
    v = v.strip().lower()
    if not v:
        return None
    if v in ("0", "none", "-"):
        return set()
    ranks = {int(x) for x in re.split(r"[,;\s]+", v) if x}
    if not ranks or not ranks <= {1, 2, 3, 4, 5}:
        raise ValueError(v)
    return ranks


def fmt(k, n):
    if not n:
        return "      n/a                    "
    lo, hi = wilson(k, n)
    return f"{100 * k / n:6.1f}% [{100 * lo:5.1f}%, {100 * hi:5.1f}%]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="results/rag_retrieval_audit.csv")
    a = ap.parse_args()
    rows, unlabelled, bad = defaultdict(list), defaultdict(int), []
    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                v = parse_verdict(r["my_verdict"])
            except ValueError:
                bad.append((r["id"], r["step"], r["retriever"], r["my_verdict"]))
                continue
            if v is None:
                unlabelled[r["retriever"]] += 1
            else:
                rows[r["retriever"]].append(v)
    print(f"{'retriever':10s} {'n':>4s}  {'precision@1 [95% CI]':29s}  {'hit@5 [95% CI]':29s}  "
          f"{'precision@5 [95% CI]':29s}  unlabelled")
    for ret in sorted(set(rows) | set(unlabelled)) + ["ALL"]:
        vs = [v for vv in rows.values() for v in vv] if ret == "ALL" else rows[ret]
        n = len(vs)
        left = sum(unlabelled.values()) if ret == "ALL" else unlabelled[ret]
        print(f"{ret:10s} {n:4d}  {fmt(sum(1 in v for v in vs), n)}  {fmt(sum(bool(v) for v in vs), n)}  "
              f"{fmt(sum(len(v) for v in vs), 5 * n)}  {left:6d}")
    for b in bad:
        print(f"WARNING: unreadable verdict {b[3]!r} for {b[0]} step {b[1]} ({b[2]}); expected ranks like '1,3' or 0")
