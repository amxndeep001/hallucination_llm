"""Stage 3: number-based alignment of the ORIGINAL solution's steps with the N sampled solutions.

A step is SUPPORTED if the new quantities it computes also appear in the independently sampled solutions.
Alignment is by numbers, not by text: every number is compared as an exact Fraction.

Per step:
  key numbers  if the step has equations (Stage 1 field): the result on the right-hand side of each equation
               (the value after the last '=', when that side is a single number: '= 150 medium-sized' -> 150,
               but '= 4x - 8' or '= x/2' is not a result). If no equation gives a result, the step falls back to:
               derived = the numbers in the step text that are not problem numbers and not trivial constants
               (0, 1, 2, 10, 100). A trivial constant does count when it is an equation result.
               A problem number never counts as a key number.
  support      per key number, the share of samples whose text contains it (25% matches 25 or 1/4);
               support_mean / support_min over the step's key numbers. No key numbers -> None (no signal).
Per solution (steps with a signal only):
  s3_min_support     min support_min (1.0 if no step has a signal)
  s3_first_low_step  earliest step with support_min <= 0.2, else -1
  s3_n_low_steps     steps with support_min <= 0.2
  s3_mean_support    mean support_mean (1.0 if no step has a signal)
  s3_n_steps_with_signal, s3_frac_steps_with_signal
The human label is copied as the target only; label and final_answer_correct are never used for a feature.

Usage:
  python src/stage3_align.py data/processed/pb_gsm8k.jsonl data/processed/samples_gsm8k.jsonl \
         data/processed/s3_gsm8k.jsonl
"""
import argparse, json, re, sys
from collections import Counter
from fractions import Fraction

TRIVIAL = {Fraction(v) for v in (0, 1, 2, 10, 100)}
LOW = 0.2                                                    # s3_first_low_step threshold on support_min

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"         # 1,250 | 12 | 12.50   (signs are ignored)
_TOKEN = re.compile(
    rf"\\[dt]?frac\{{\s*(?P<fn>{_NUM})\s*\}}\{{\s*(?P<fd>{_NUM})\s*\}}"            # \frac{3}{4}
    rf"|(?<![\w.,])(?P<w>\d+)[ \t]+(?P<mn>\d+)\s*/\s*(?P<md>\d+)(?![\d.])"             # 1 1/3
    rf"|(?<![\w.,])(?P<n>\d+)\s*/\s*(?P<d>\d+)(?![\d.])"                            # 3/4, 12 / 2
    rf"|(?<![\w.,])(?P<v>{_NUM})(?![\d])(?P<pct>\s*(?:%|percent\b))?")              # 25, $5,125, 25%
_LIST_NUMBER = re.compile(r"(?m)^[ \t>#*-]*\d+[.)](?=\s|\*)")                      # '1. ', '**2.**', '### 3)'
_STEP_LABEL = re.compile(r"(?i)\bstep\s*\d+")                                       # 'Step 3'
_OPERATOR = re.compile(r"\s*(?:[-+*/^=]|\\times|\\div|\\cdot)")


def _frac(s):
    return Fraction(s.replace(",", ""))


def _clean(text):
    text = text.replace("\\%", "%").replace("\\$", "$").replace("{,}", ",").replace("\\!", "")
    text = re.sub(r"(\\[A-Za-z]+)(?=\d)", r"\1 ", text)                  # '\times18' -> '\times 18'
    return _STEP_LABEL.sub(" ", _LIST_NUMBER.sub(" ", text))


def number_items(text):
    """Numeric mentions in text, in order: (match, value, candidates, components).
    value: the number as written (25% -> 25); candidates: what it may equal (25% -> {25, 1/4});
    components: the literals inside a fraction (12/2 -> 12, 2), which a later text may mention on their own."""
    out = []
    for m in _TOKEN.finditer(_clean(text)):
        g = m.groupdict()
        if g["fn"] is not None:
            a, b = _frac(g["fn"]), _frac(g["fd"])
            if b:
                out.append((m, a / b, {a / b}, {a, b}))
        elif g["w"] is not None:
            w, a, b = int(g["w"]), int(g["mn"]), int(g["md"])
            if 0 < a < b:                                     # proper fraction: a mixed number
                out.append((m, w + Fraction(a, b), {w + Fraction(a, b)}, {Fraction(w), Fraction(a, b), Fraction(a), Fraction(b)}))
            else:                                             # '3 12/2': a number, then a fraction
                out.append((m, Fraction(w), {Fraction(w)}, set()))
                if b:
                    out.append((m, Fraction(a, b), {Fraction(a, b)}, {Fraction(a), Fraction(b)}))
        elif g["n"] is not None:
            a, b = int(g["n"]), int(g["d"])
            if b:
                out.append((m, Fraction(a, b), {Fraction(a, b)}, {Fraction(a), Fraction(b)}))
        else:
            v = _frac(g["v"])
            out.append((m, v, {v, v / 100} if g["pct"] else {v}, set()))
    return out


def extract_numbers(text):
    """All numeric values in text as a set of Fractions (percents give both 25 and 1/4)."""
    return {x for _, _, cand, comp in number_items(text) for x in cand | comp}


def fmt_number(v, pct=False):
    """Fraction -> '12', '12.5', '1/3' (exact decimals shown as decimals)."""
    d = v.denominator
    while d % 2 == 0: d //= 2
    while d % 5 == 0: d //= 5
    s = str(v) if d != 1 else (str(v.numerator) if v.denominator == 1 else f"{float(v):.10g}")
    return s + "%" if pct else s


def equation_result(eq):
    """The single number on the right of the last '=' of an equation, or None ('= 4x - 8', '= x/2')."""
    if "=" not in eq:
        return None
    rhs = _clean(eq.rsplit("=", 1)[1])
    items = number_items(rhs)
    if not items:
        return None
    m, v, cand, _ = items[0]
    before, after = rhs[:m.start()], rhs[m.end():]
    if re.search(r"[A-Za-z0-9]", re.sub(r"\\[A-Za-z]+", "", before)) or _OPERATOR.match(after) or re.match(r"[A-Za-z](?![A-Za-z])", after):
        return None                                           # an expression or a variable, not a result
    return v, cand, m.group().strip().endswith(("%", "percent"))


def key_numbers(step, problem_numbers):
    """(keys, source, stats). keys: list of (label, candidates), one per distinct value."""
    def is_problem(cand):
        return bool(cand & problem_numbers)

    keys, seen = [], set()
    def add(v, cand, pct):
        if v not in seen:
            seen.add(v); keys.append((fmt_number(v, pct), cand))

    for eq in step.get("equations") or []:
        r = equation_result(eq)
        if r and not is_problem(r[1]):
            add(*r)
    source = "rhs"
    items = number_items(step["text"])
    stats = {"n_numbers": len(items), "n_problem": sum(is_problem(c) for _, _, c, _ in items),
             "n_trivial": sum(not is_problem(c) and v in TRIVIAL for _, v, c, _ in items)}
    if not keys:
        source = "derived"
        for m, v, cand, _ in items:
            if not is_problem(cand) and v not in TRIVIAL:
                add(v, cand, m.group().strip().endswith(("%", "percent")))
    return keys, source, stats


def align(pb_rec, samples):
    problem_numbers = extract_numbers(pb_rec["problem"])
    sample_numbers = [extract_numbers(s["text"]) for s in samples]
    steps, stats = [], []
    for st in pb_rec["steps"]:
        keys, source, s = key_numbers(st, problem_numbers)
        share = [sum(bool(cand & sn) for sn in sample_numbers) / len(sample_numbers) for _, cand in keys]
        steps.append({"idx": st["idx"], "key_numbers": [k for k, _ in keys],
                      "support_mean": sum(share) / len(share) if share else None,
                      "support_min": min(share) if share else None, "n_key": len(keys)})
        stats.append({**s, "source": source, "type": st["type"]})
    return steps, stats


def solution_features(steps, low=LOW):
    sig = [s for s in steps if s["support_min"] is not None]
    lows = [s["idx"] for s in sig if s["support_min"] <= low]
    return {"s3_min_support": min((s["support_min"] for s in sig), default=1.0),
            "s3_first_low_step": min(lows, default=-1),
            "s3_n_low_steps": len(lows),
            "s3_mean_support": sum(s["support_mean"] for s in sig) / len(sig) if sig else 1.0,
            "s3_n_steps_with_signal": len(sig),
            "s3_frac_steps_with_signal": len(sig) / len(steps) if steps else 0.0}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pb")
    ap.add_argument("samples")
    ap.add_argument("output")
    a = ap.parse_args()

    pb, samples = read_jsonl(a.pb), {r["id"]: r for r in read_jsonl(a.samples)}
    missing = [r["id"] for r in pb if r["id"] not in samples]
    if missing:
        sys.exit(f"ERROR: {len(missing)} Stage 1 ids have no samples, e.g. {missing[:5]}")
    out, allstats = [], []
    for r in pb:
        steps, stats = align(r, samples[r["id"]]["samples"])
        out.append({"id": r["id"], "label": r["label"], **solution_features(steps), "steps": steps})
        allstats += stats
    with open(a.output, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    allsteps = [s for r in out for s in r["steps"]]
    n = len(allsteps)
    print(f"{len(out)} problems, {n} steps -> {a.output}")
    nn = sum(s["n_numbers"] for s in allstats)
    print(f"numbers in original steps: {nn}; excluded as problem numbers: {sum(s['n_problem'] for s in allstats)} "
          f"({100*sum(s['n_problem'] for s in allstats)/nn:.1f}%); trivial constants (0,1,2,10,100) not counted unless "
          f"an equation result: {sum(s['n_trivial'] for s in allstats)} ({100*sum(s['n_trivial'] for s in allstats)/nn:.1f}%)")
    print(f"key numbers from: {dict(Counter(s['source'] for s, st in zip(allstats, allsteps) if st['n_key']))} "
          f"(rhs = equation results, derived = numbers in the text)")
    print("steps with no signal (no key number), by Stage 1 step type:")
    for t, c in sorted(Counter(s["type"] for s in allstats).items()):
        none = sum(s["type"] == t and st["n_key"] == 0 for s, st in zip(allstats, allsteps))
        print(f"  {t:16s} {none:5d} of {c:5d} ({100*none/c:.1f}%)")
    none = sum(st["n_key"] == 0 for st in allsteps)
    print(f"  {'all':16s} {none:5d} of {n:5d} ({100*none/n:.1f}%)")
    sig = [s for s in allsteps if s["support_min"] is not None]
    print(f"support_min distribution over steps with a signal: "
          f"{dict(sorted(Counter(round(s['support_min'], 2) for s in sig).items()))}")
    print(f"solutions with a low step (support_min <= {LOW}): {sum(r['s3_first_low_step'] != -1 for r in out)} of {len(out)}")
