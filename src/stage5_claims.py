"""Stage 5 (RAG pilot): rule-based claim extraction from the steps of the ORIGINAL solution.

For each step (Stage 1 text, data/processed/pb_<split>.jsonl):
  1. split into sentences (newlines, '.', '!', '?' followed by a space and a capital / digit / bracket;
     decimals like 3.5 and 'e.g.' are not split), then at ';'
  2. split 'A, so B' (also thus / therefore / hence / which means / which gives / which is) into two claims
     when both parts have at least 3 tokens
  3. keep a sentence if it states a quantity, a result or a relation: it contains a number, a variable,
     an equation / relation sign, or one of the keywords (so, therefore, thus, since, because, is, are, ...).
     Dropped: questions, fragments with fewer than 3 tokens, and planning sentences ('Let's ...',
     'We need to find ...', 'To solve ...') that contain no number and no equation.
  4. the claim keeps its LaTeX form ('latex', used by the numeric spot check of Stage 6 C) and a plain-text
     form ('text', LaTeX converted: \\frac{a}{b} -> a/b, \\times -> *, \\le -> <=, ...; used as the NLI
     hypothesis); leading connectives ('So,', 'Thus,') are stripped from 'text'.
  5. claim type: Stage 1's tag_step on the sentence (computational / knowledge_claim / reasoning).
Nothing here reads the human label or final_answer_correct (tests/test_rag_pilot.py checks it).

Usage:
  python src/stage5_claims.py                   # claims for the pilot sample + sanity ids
        [--sample results/rag_pilot_sample.csv] [--out data/processed/rag_pilot_claims.jsonl]
Writes the claims JSONL and prints/saves the report to results/logs/rag_stage5_claims.txt.
"""
import argparse, csv, json, os, random, re, sys
from collections import defaultdict
from stage1_preprocess import KNOWLEDGE_CUES, extract_equations, tag_step

SANITY_IDS = ["gsm8k-0", "gsm8k-34", "gsm8k-202"]
SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]
P = "data/processed"

# ------------------------------------------------------------------ LaTeX -> plain text
_CMD = {r"\times": " * ", r"\cdot": " * ", r"\div": " / ", r"\leq": " <= ", r"\le": " <= ", r"\geq": " >= ",
        r"\ge": " >= ", r"\neq": " != ", r"\ne": " != ", r"\approx": " approx ", r"\pm": " +- ", r"\infty": " infinity ",
        r"\ldots": "...", r"\cdots": "...", r"\dots": "...", r"\qquad": " ", r"\quad": " ", r"\implies": " implies ",
        r"\Rightarrow": " implies ", r"\rightarrow": " -> ", r"\to": " -> ", r"\equiv": " equiv ", r"\pmod": " mod ",
        r"\in": " in ", r"\%": "%", r"\$": "$", r"\{": "(", r"\}": ")", r"\ds": " ", r"\displaystyle": " "}
_UNWRAP = re.compile(r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|mathit|mathbb|operatorname|boxed|ensuremath|paren)\s*\{([^{}]*)\}")


def latex_to_plain(s: str) -> str:
    s = re.sub(r"\\[\[\]()]", " ", s)                                       # \[ \] \( \) math delimiters
    s = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", s).replace("&", " ")      # align / cases environments
    s = s.replace("\\\\", " ").replace("\\,", " ").replace("\\;", " ").replace("\\!", "").replace("\\ ", " ")
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b\s*([()\[\]|.]|\\[{}])?",
               lambda m: "" if m[1] in (None, ".") else m[1], s)
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", " degrees", s)
    for _ in range(6):                                                    # innermost first, nested ones next round
        old = s
        s = _UNWRAP.sub(r"\1", s)
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
                   lambda m: f"{m[1]}/{m[2]}" if re.fullmatch(r"\s*-?[\w.]+\s*", m[1]) and re.fullmatch(r"\s*[\w.]+\s*", m[2])
                   else f"({m[1]})/({m[2]})", s)
        s = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", r"\1/\2", s)            # \frac12
        s = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", r"root\1(\2)", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\[dt]?binom\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"C(\1, \2)", s)
        s = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: "^" + (m[1] if re.fullmatch(r"\w+", m[1]) else f"({m[1]})"), s)
        s = re.sub(r"_\s*\{([^{}]*)\}", lambda m: "_" + (m[1] if re.fullmatch(r"\w+", m[1]) else f"({m[1]})"), s)
        if s == old:
            break
    for k in sorted(_CMD, key=len, reverse=True):
        s = re.sub(re.escape(k) + r"(?![A-Za-z])", lambda m, v=_CMD[k]: v, s)
    s = re.sub(r"\\([A-Za-z]+)", r" \1 ", s)                              # \sin -> sin, \alpha -> alpha, \angle -> angle
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = re.sub(r"\*\*", "", s)                                            # markdown bold
    s = re.sub(r"\s+([,.;:?)]|!(?!=))", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------ sentences
_SENT_END = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<=[.!?])\s+(?=[A-Z0-9(\[\\$\-])")
_CONNECTIVE_SPLIT = re.compile(r",\s+(?:and\s+)?(?:so|thus|therefore|hence|which means(?: that)?|which gives|which is|"
                               r"giving|meaning(?: that)?)\s+", re.I)
_LEAD = re.compile(r"^(?:(?:so|thus|therefore|hence|then|now|next|finally|first|second|third|also|and|but|"
                   r"consequently|as a result)\b[,:]?\s*)+", re.I)
_KEYWORDS = re.compile(r"\b(so|therefore|thus|since|because|is|are|was|were|hence|equals?|gives|implies|means|must|"
                       r"has|have|had|cannot|becomes?)\b", re.I)
_PLANNING = re.compile(r"^(?:let'?s|let us|we need|we want|we will|we'll|we must|we have to|we can|now,? we|next,? we|"
                       r"to (?:find|solve|determine|calculate|compute|figure)|in order to|i need|i will)\b", re.I)
_RELATION = re.compile(r"[=<>]|\\(?:le|ge|leq|geq|neq|ne|approx|equiv|times|cdot|div|frac|sqrt)\b|[+*/^]")
_VARIABLE = re.compile(r"(?<![A-Za-z'\\])(?:[b-zB-HJ-Z]|[aA](?=\s*[=+\-*/^)]|_))(?![A-Za-z'])")


def split_sentences(text: str):
    out = []
    for line in re.split(r"\n+", text):
        for s in _SENT_END.split(line):
            out += [p.strip() for p in s.split(";")]
    return [s for s in out if s]


def _split_connectives(s: str):
    parts = _CONNECTIVE_SPLIT.split(s, maxsplit=1)
    if len(parts) == 2 and all(len(p.split()) >= 3 for p in parts):
        return [parts[0].strip()] + _split_connectives(parts[1].strip())
    return [s]


def is_claim(s: str) -> bool:
    toks = s.split()
    if len(toks) < 3 or s.rstrip().endswith("?"):
        return False
    has_num, has_rel = bool(re.search(r"\d", s)), bool(_RELATION.search(s))
    if _PLANNING.match(s.strip("*# ")) and not has_num and not has_rel:
        return False
    return has_num or has_rel or bool(_VARIABLE.search(s)) or bool(_KEYWORDS.search(s))


def claim_type(latex: str) -> str:
    return tag_step(latex, extract_equations(latex))


def extract_claims(problem: str, steps):
    """steps: Stage 1 step dicts (idx, text, ...). Returns claim dicts (step, claim_idx, text, latex, type).
    Takes only the problem and the steps: never the record (with its label)."""
    claims = []
    for st in steps:
        k = 0
        for sent in split_sentences(st["text"]):
            for part in _split_connectives(sent):
                part = part.strip(" ,:*#-")
                if not is_claim(part):
                    continue
                text = _LEAD.sub("", latex_to_plain(part)).strip(" ,:")
                if len(text.split()) < 2:
                    continue
                if len(text) > 400:                                       # keep hypotheses short for the NLI model
                    text = text[:400].rsplit(" ", 1)[0] + " ..."
                claims.append({"step": st["idx"], "claim_idx": k, "text": text, "latex": part, "type": claim_type(part)})
                k += 1
    return claims


# ------------------------------------------------------------------ pilot I/O
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def split_of(sid: str) -> str:
    return sid.rsplit("-", 1)[0]


def pilot_ids(sample_csv):
    with open(sample_csv, encoding="utf-8", newline="") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    return ids + [i for i in SANITY_IDS if i not in ids]


def load_pb(ids):
    want = defaultdict(set)
    for i in ids:
        want[split_of(i)].add(i)
    pb = {}
    for sp, s in want.items():
        pb.update({r["id"]: r for r in read_jsonl(f"{P}/pb_{sp}.jsonl") if r["id"] in s})
    return pb


def report(claims_by_id, pb, ids, sample_ids, log):
    """Per split (pilot sample only, sanity extras excluded): claims per step, steps without claims,
    claim types, and 10 random examples (seed 0)."""
    rng = random.Random(0)
    for sp in SPLITS:
        sids = [i for i in ids if split_of(i) == sp and i in sample_ids]
        if not sids:
            continue
        n_steps = sum(len(pb[i]["steps"]) for i in sids)
        cl = [c for i in sids for c in claims_by_id[i]]
        per_step = defaultdict(int)
        for i in sids:
            for c in claims_by_id[i]:
                per_step[(i, c["step"])] += 1
        zero = n_steps - len(per_step)
        types = defaultdict(int)
        for c in cl:
            types[c["type"]] += 1
        log(f"\n## {sp}: {len(sids)} solutions, {n_steps} steps, {len(cl)} claims")
        log(f"claims per step: {len(cl) / n_steps:.2f}   steps with zero claims: {zero} ({100 * zero / n_steps:.1f}%)")
        log("claim types: " + ", ".join(f"{t} {n} ({100 * n / len(cl):.1f}%)" for t, n in sorted(types.items())))
        log("10 random examples:")
        for i, c in rng.sample([(i, c) for i in sids for c in claims_by_id[i]], min(10, len(cl))):
            log(f"  - {i} step {c['step']} [{c['type']}] {c['text']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="results/rag_pilot_sample.csv")
    ap.add_argument("--out", default=f"{P}/rag_pilot_claims.jsonl")
    ap.add_argument("--log", default="results/logs/rag_stage5_claims.txt")
    a = ap.parse_args()
    ids = pilot_ids(a.sample)
    with open(a.sample, encoding="utf-8", newline="") as f:
        sample_ids = {r["id"] for r in csv.DictReader(f)}
    pb = load_pb(ids)
    missing = [i for i in ids if i not in pb]
    if missing:
        raise SystemExit(f"ERROR: ids not in pb_*.jsonl: {missing[:5]}")
    claims_by_id = {i: extract_claims(pb[i]["problem"], pb[i]["steps"]) for i in ids}
    with open(a.out, "w", encoding="utf-8") as f:
        for i in ids:
            for c in claims_by_id[i]:
                f.write(json.dumps({"id": i, "split": split_of(i), **c}) + "\n")
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    lines = []
    log = lambda s="": (print(s), lines.append(s))
    log(f"===== Stage 5 claims: {len(ids)} solutions ({len(sample_ids)} pilot + {len(ids) - len(sample_ids)} sanity extras), "
        f"{sum(map(len, claims_by_id.values()))} claims -> {a.out}")
    report(claims_by_id, pb, ids, sample_ids, log)
    log("\n## sanity cases (all claims)")
    for i in SANITY_IDS:
        log(f"\n{i} ({'in the pilot sample' if i in sample_ids else 'not in the sample'})")
        for c in claims_by_id[i]:
            log(f"  step {c['step']}.{c['claim_idx']} [{c['type']}] {c['text']}")
    with open(a.log, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
