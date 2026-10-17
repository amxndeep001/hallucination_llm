"""Stage 2 sanity check: did sampling work, and is the agreement signal useful?
Usage:  python src/stage2_check.py data/processed/samples_gsm8k.jsonl data/raw/processbench_gsm8k.jsonl
Prints: formatting/length stats, how often the samples agree with each other, how often they agree
with the ORIGINAL ProcessBench solution, and a first AUROC for 'low agreement => solution is wrong'.
"""
import json, sys
from collections import Counter
from stage2_generate import extract_answer, normalize_answer


def auroc(scores, labels):
    """P(score of a positive > score of a negative); ties count 1/2. labels: 1 = positive."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


samples_path, raw_path = sys.argv[1], sys.argv[2]
raw = {}
with open(raw_path, encoding="utf-8") as f:
    for l in f:
        r = json.loads(l); raw[r["id"]] = r
recs = [json.loads(l) for l in open(samples_path, encoding="utf-8") if l.strip()]
print(f"{len(recs)} problems, {sum(len(r['samples']) for r in recs)} samples")

empty = toks = total = 0
self_agree, orig_agree, has_err, ans_wrong = [], [], [], []
for r in recs:
    answers = [normalize_answer(s["final_answer"]) for s in r["samples"]]
    total += len(answers)
    empty += sum(a == "" for a in answers)
    toks += sum(s["n_tokens"] for s in r["samples"])
    self_agree.append(Counter(answers).most_common(1)[0][1] / len(answers))
    o = raw[r["id"]]
    orig = normalize_answer(extract_answer(o["steps"][-1]))
    orig_agree.append(sum(a == orig for a in answers) / len(answers))
    has_err.append(int(o["label"] not in (-1, None)))
    ans_wrong.append(int(o.get("final_answer_correct") is False))

print(f"samples with no extractable answer : {100*empty/total:.1f}%   (high => truncated or format problem)")
print(f"mean generated tokens per sample   : {toks/total:.0f}")
print(f"mean self-agreement (majority share): {sum(self_agree)/len(self_agree):.2f}")
print(f"mean agreement with ORIGINAL answer : {sum(orig_agree)/len(orig_agree):.2f}")
for name, lab in [("has a process error (label != -1)", has_err), ("final answer is wrong", ans_wrong)]:
    a = [x for x, l in zip(orig_agree, lab) if l]; b = [x for x, l in zip(orig_agree, lab) if not l]
    print(f"\n[{name}] n={sum(lab)} vs {len(lab)-sum(lab)}")
    if a and b:
        print(f"  agreement with original: {sum(a)/len(a):.2f} (wrong) vs {sum(b)/len(b):.2f} (fine)")
    print(f"  AUROC of (1 - agreement with original): {auroc([1-x for x in orig_agree], lab):.3f}   (0.5 = no signal)")
