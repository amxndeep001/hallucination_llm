"""Stage 2 sanity check: did sampling work, and is the agreement signal useful?
Usage:  python src/stage2_check.py data/processed/samples_gsm8k.jsonl data/raw/processbench_gsm8k.jsonl
        [--truncated-no-answer --max-new-tokens 1024]
Prints: formatting/length stats, how often the samples agree with each other, how often they agree
with the ORIGINAL ProcessBench solution, and a first AUROC for 'low agreement => solution is wrong'.
"""
import argparse, json, re
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


_SPAN = r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]"


def unboxed_answer(step):
    """Answer of a last step without \\boxed{}, where the last-number fallback would cut a LaTeX answer
    ('$16\\pi$' -> '16', '\\(\\frac{4}{35}\\)' -> '35'). Returns '' when neither rule applies:
      - 'The final answer is $X$' -> X
      - the step ends with a math span holding a pure number expression (no '=', no letters outside
        LaTeX commands) -> that span: 'is \\(-\\frac{2\\sqrt{3}}{3}\\).'"""
    if "\\boxed{" in step:
        return ""
    m = re.search(rf"final answer is:?\s*(?:{_SPAN})", step, re.S | re.I)
    if m:
        return next(g for g in m.groups() if g is not None).strip()
    spans = list(re.finditer(_SPAN, step, re.S))
    if spans and re.fullmatch(r"\s*[.!]?\s*", step[spans[-1].end():]):
        s = next(g for g in spans[-1].groups() if g is not None).strip()
        if "=" not in s and re.search(r"\d", s) and not re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", "", s)):
            return s
    return ""


def original_answer(raw_rec):
    """Normalized final answer of the original ProcessBench solution (from its last step).
    'expression = number' (e.g. '150 \\div 3 = 50') -> the number after the last '=', unless the answer is
    a list or text ('a = 2.08, b = 0.28'), where the last value is not the whole answer."""
    raw = unboxed_answer(raw_rec["steps"][-1]) or extract_answer(raw_rec["steps"][-1])
    a = normalize_answer(raw)
    tail = a.rsplit("=", 1)[-1]
    if "=" in a and "," not in raw and "\\text" not in raw and re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?", tail):
        a = normalize_answer(tail)
    return a


def agreement(answers, orig):
    """Share of normalized sample answers equal to the original answer."""
    return sum(a == orig for a in answers) / len(answers)


def sample_answer(s, cap=None):
    """Normalized final answer of one sample. With cap (the Stage 2 max_new_tokens): a sample that reached
    the cap without a \\boxed{} answer has NO answer ('') instead of the last number of its cut-off text."""
    if cap is not None and s["n_tokens"] >= cap and "\\boxed{" not in s["text"]:
        return ""
    return normalize_answer(s["final_answer"])


def main(samples_path, raw_path, cap=None):
    raw = {}
    with open(raw_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l); raw[r["id"]] = r
    recs = [json.loads(l) for l in open(samples_path, encoding="utf-8") if l.strip()]
    print(f"{len(recs)} problems, {sum(len(r['samples']) for r in recs)} samples"
          + (f"; truncated at {cap} tokens without \\boxed = no answer" if cap else ""))

    empty = toks = total = 0
    self_agree, orig_agree, has_err, ans_wrong = [], [], [], []
    for r in recs:
        answers = [sample_answer(s, cap) for s in r["samples"]]
        total += len(answers)
        empty += sum(a == "" for a in answers)
        toks += sum(s["n_tokens"] for s in r["samples"])
        self_agree.append(Counter(answers).most_common(1)[0][1] / len(answers))
        o = raw[r["id"]]
        orig_agree.append(agreement(answers, original_answer(o)))
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("raw")
    ap.add_argument("--truncated-no-answer", action="store_true",
                    help="a sample that hit the token cap without \\boxed{} has no answer (needs --max-new-tokens)")
    ap.add_argument("--max-new-tokens", type=int, help="Stage 2 cap: 768 for GSM8K, 1024 for the Colab splits")
    a = ap.parse_args()
    if a.truncated_no_answer and not a.max_new_tokens:
        ap.error("--truncated-no-answer needs --max-new-tokens")
    main(a.samples, a.raw, a.max_new_tokens if a.truncated_no_answer else None)
