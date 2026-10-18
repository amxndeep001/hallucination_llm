"""Stage 7: answer consistency and semantic entropy features, one record per problem.

For every problem, from its N sampled solutions (Stage 2) and the original ProcessBench solution:
  majority_share       size of the largest answer cluster / samples with an answer
  agree_with_original  share of samples whose answer is equivalent to the original solution's answer
                       (here only, X% also matches X and X/100)
  n_clusters, cluster_sizes, semantic_entropy
                       answers clustered by MATHEMATICAL equivalence: the same normalize_answer string,
                       or SymPy finds the difference exactly 0 (0.5 = 1/2 = \\frac{1}{2} = 0.50; 10\\% = 1/10,
                       so 10\\% = 0.1 but 10\\% != 10). An answer SymPy cannot parse, or one containing a
                       word ('6 red, 4 blue'), matches by string only.
                       semantic_entropy = -sum p_k ln p_k over the cluster shares.
  no_answer_rate       share of samples with no extractable answer
  mean_logprob_mean / _min, mean_entropy_mean / _max, n_tokens_mean
  truncated_rate       share of samples whose n_tokens reached the max_new_tokens cap
  n_fail, n_checked, first_fail_step, sympy_flag, n_equations     copied from Stage 4
The human label and final_answer_correct are copied as targets only, never used for a feature.

Usage:
  python src/stage7_consistency.py data/processed/samples_gsm8k.jsonl data/processed/s4_gsm8k.jsonl \
         data/raw/processbench_gsm8k.jsonl data/processed/s7_gsm8k.jsonl [--max-new-tokens 768]
Without --max-new-tokens the cap is taken as the largest n_tokens in the samples file.
"""
import argparse, json, math, re, sys
from collections import Counter
import sympy as sp
from stage2_generate import normalize_answer
from stage2_check import original_answer
from stage4_sympy_verify import to_sympy, _kind, SYMPY_ERRORS

_parsed = {}


def parse_answer(a):
    """Normalized answer -> SymPy expression, or None if it cannot be parsed."""
    if a not in _parsed:
        words = re.search(r"[A-Za-z]{2,}", re.sub(r"\\[A-Za-z]+", "", a))   # '4blue6red' is text, not 4*b*l*u*e*...
        e = to_sympy(a) if a and not words else None
        _parsed[a] = e if e is not None and not isinstance(e, (tuple, list)) and not _kind(e) else None
    return _parsed[a]


def equivalent(a, b):
    """a, b: normalized answers. Same string, or SymPy says the difference is exactly 0."""
    if a == b:
        return a != ""
    x, y = parse_answer(a), parse_answer(b)
    if x is None or y is None:
        return False
    try:
        d = x - y
        if d.free_symbols:                                     # '\frac{5}{8}A + 45' vs '\frac{5A + 360}{8}'
            return sp.expand(d) == 0
        if abs(complex(sp.N(d))) > 1e-9:                       # cheap numeric rejection before simplify
            return False
        return d == 0 or sp.simplify(d) == 0                   # exact equality only, no tolerance
    except SYMPY_ERRORS:
        return False


def agrees(a, orig):
    """agree_with_original only: equivalent, or one side is 'X%' and the other is X or X/100.
    (The clustering keeps 10% != 10, so competition answers are unaffected.)"""
    if equivalent(a, orig):
        return True
    for x, y in ((a, orig), (orig, a)):
        m = re.fullmatch(r"(.+?)\\?%", x)
        if m and y and (equivalent(normalize_answer(m[1]), y) or equivalent(f"({m[1]})/100", y)):
            return True
    return False


def cluster(answers):
    """Cluster non-empty normalized answers by equivalence (transitive closure). Sizes, largest first."""
    answers = [a for a in answers if a != ""]
    parent = list(range(len(answers)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            if find(i) != find(j) and equivalent(answers[i], answers[j]):
                parent[find(j)] = find(i)
    return sorted(Counter(find(i) for i in range(len(answers))).values(), reverse=True)


def semantic_entropy(sizes):
    n = sum(sizes)
    return max(0.0, -sum(k / n * math.log(k / n) for k in sizes)) if n else 0.0     # max: no -0.0


def features(sample_rec, s4_rec, raw_rec, cap):
    samples = sample_rec["samples"]
    answers = [normalize_answer(s["final_answer"]) for s in samples]
    orig = original_answer(raw_rec)
    sizes = cluster(answers)
    n_answered = sum(sizes)
    lp = [s["mean_logprob"] for s in samples]
    ent = [s["mean_entropy"] for s in samples]
    toks = [s["n_tokens"] for s in samples]
    sy = s4_rec["sympy"]
    return {
        "id": sample_rec["id"], "label": raw_rec["label"], "final_answer_correct": raw_rec["final_answer_correct"],
        "majority_share": sizes[0] / n_answered if n_answered else 0.0,
        "agree_with_original": sum(agrees(a, orig) for a in answers) / len(answers),
        "n_clusters": len(sizes), "cluster_sizes": sizes, "semantic_entropy": semantic_entropy(sizes),
        "no_answer_rate": sum(a == "" for a in answers) / len(answers),
        "mean_logprob_mean": sum(lp) / len(lp), "mean_logprob_min": min(lp),
        "mean_entropy_mean": sum(ent) / len(ent), "mean_entropy_max": max(ent),
        "n_tokens_mean": sum(toks) / len(toks), "truncated_rate": sum(t >= cap for t in toks) / len(toks),
        "n_fail": sy["n_fail"], "n_checked": sy["n_checked"], "first_fail_step": sy["first_fail_step"],
        "sympy_flag": sy["n_fail"] > 0, "n_equations": sy["n_equations"],
        "original_answer": orig, "sample_answers": answers,                    # for inspection, not features
    }


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("s4")
    ap.add_argument("raw")
    ap.add_argument("output")
    ap.add_argument("--max-new-tokens", type=int, help="generation cap of the Stage 2 run (default: max n_tokens)")
    a = ap.parse_args()

    samples, s4, raw = read_jsonl(a.samples), read_jsonl(a.s4), read_jsonl(a.raw)
    s4, raw = {r["id"]: r for r in s4}, {r["id"]: r for r in raw}
    missing = [r["id"] for r in samples if r["id"] not in s4 or r["id"] not in raw]
    if missing:
        sys.exit(f"ERROR: {len(missing)} sample ids missing from s4/raw, e.g. {missing[:5]}")
    if a.max_new_tokens:
        cap, src = a.max_new_tokens, "--max-new-tokens"
    else:
        cap, src = max(s["n_tokens"] for r in samples for s in r["samples"]), "largest n_tokens in the samples file"
    print(f"{len(samples)} problems; truncation cap = {cap} tokens (from {src})")

    out = [features(r, s4[r["id"]], raw[r["id"]], cap) for r in samples]
    with open(a.output, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n = len(out)
    print(f"-> {a.output}")
    print(f"  mean majority_share {sum(r['majority_share'] for r in out)/n:.3f} | "
          f"mean agree_with_original {sum(r['agree_with_original'] for r in out)/n:.3f} | "
          f"mean semantic_entropy {sum(r['semantic_entropy'] for r in out)/n:.3f}")
    print(f"  n_clusters distribution: {dict(sorted(Counter(r['n_clusters'] for r in out).items()))}")
    print(f"  samples truncated: {sum(r['truncated_rate'] for r in out)/n:.3%} | "
          f"no answer: {sum(r['no_answer_rate'] for r in out)/n:.3%} | "
          f"answers matched by string only (unparsable or text): {sum(v is None for k, v in _parsed.items() if k)} of {sum(1 for k in _parsed if k)} distinct")
