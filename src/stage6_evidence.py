"""Stage 6 (RAG pilot): evidence for each Stage 5 claim from four sources.

A   problem-statement NLI. premise = problem text, hypothesis = claim (both LaTeX -> plain text).
    Premise rule (one rule, used everywhere): if problem + claim fit in --max-length tokens, the premise is the
    full problem; otherwise it is the 3 problem sentences most similar to the claim by BM25 (in their original
    order), then ' ... ', then the full problem, and the tokenizer truncates the premise only.
A2  as A, premise = problem + the earlier steps of the same original solution (strictly earlier steps, so step 0
    equals A). If too long: top-3 BM25 problem sentences + top-3 BM25 sentences of the earlier steps + the
    immediately preceding step, then the full text, truncated.
B   samples as evidence (SelfCheck-NLI style). For each of the 5 sampled solutions the premise is its 3 sentences
    most similar to the claim by BM25 (original order); per claim: mean p_contradict and max p_entail over the
    samples that have text.
C   numeric spot check of equations with free symbols (which Stage 4 skips): see numeric_check().
D   retrieval over ProofWiki theorem statements for steps that Stage 1 tagged knowledge_claim: BM25 and a small
    embedding model (bge-small), top-5 each, then NLI with each retrieved passage as premise and each claim of the
    step as hypothesis (mean p_contradict, max p_entail over the 5 passages, per retriever).
NLI outputs p_entail, p_neutral, p_contradict (softmax over the model's three labels, mapped by its config).
Nothing in this file reads the human label or final_answer_correct: it sees only problem text, steps, samples
and claims (tests/test_rag_pilot.py checks it).

Usage (from the project root; every NLI run is resumable: scored claims are skipped):
  python src/stage6_evidence.py nli --source A   [--model M] [--batch-size 16] [--max-length 512] [--device cpu]
  python src/stage6_evidence.py nli --source A2
  python src/stage6_evidence.py nli --source B
  python src/stage6_evidence.py numeric           # pilot claims + all sampled problems (counts, failure list)
  python src/stage6_evidence.py build-index       # ProofWiki statements -> data/processed/rag_index/ (git-ignored)
  python src/stage6_evidence.py retrieve          # top-5 per knowledge_claim step, NLI, audit CSV
  python src/stage6_evidence.py benchmark         # time A, A2, B, C alone on every 10th claim (idle CPU)
Outputs: data/processed/rag_pilot_{nli_A,nli_A2,nli_B,numeric,retrieval,nli_D}.jsonl,
         results/rag_pilot_runtime.csv, results/rag_numeric_failures.csv, results/rag_retrieval_audit.csv
"""
import argparse, csv, json, math, os, random, re, sys, time, zlib
from collections import Counter, defaultdict
from urllib.parse import unquote
import sympy as sp
from stage1_preprocess import KNOWLEDGE_CUES, _clean_equation, _is_math_token, _split_juxtaposed, _trim_sides, extract_equations
from stage4_sympy_verify import SYMPY_ERRORS, _split_top_commas, to_sympy
from stage5_claims import (SANITY_IDS, SPLITS, extract_claims, latex_to_plain, load_pb, pilot_ids, read_jsonl,
                           split_of, split_sentences)

P = "data/processed"
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY = "Represent this sentence for searching relevant passages: "
PROOFWIKI_CSV = "data/external/rag_corpus/proofwiki_theorems/proof_wiki_theorems.csv"
INDEX_DIR = f"{P}/rag_index"
SEED = 0

# ------------------------------------------------------------------ BM25 (also picks premise sentences)
_TOK = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
STOP = set("a an the of to in on at for by with and or is are was were be been it its this that these those as from "
           "we you they he she i so then than there their which what how can will would has have had do does did "
           "into each per our us let lets".split())


def tokenize(s):
    return [t for t in _TOK.findall(s.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1) or 1.0
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query):
        q = tokenize(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out.append(sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf))
        return out


def top_sentences(query, sents, k):
    """The k sentences with the highest BM25 score (ties: earlier first), returned in their original order."""
    if len(sents) <= k:
        return list(sents)
    sc = BM25(sents).scores(query)
    best = sorted(range(len(sents)), key=lambda i: (-sc[i], i))[:k]
    return [sents[i] for i in sorted(best)]


# ------------------------------------------------------------------ NLI model
class NLI:
    """Any MNLI-style sequence classifier with entailment / neutral / contradiction labels."""

    def __init__(self, model=DEFAULT_MODEL, batch_size=16, max_length=512, device="cpu", threads=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch, self.name, self.batch_size, self.max_length, self.device = torch, model, batch_size, max_length, device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
        names = {i: n.lower() for i, n in self.model.config.id2label.items()}
        find = lambda key: next(i for i, n in names.items() if n.startswith(key))
        self.idx = (find("entail"), find("neutral"), find("contradict"))

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def fits(self, premise, hypothesis):
        return self.n_tokens(premise) + self.n_tokens(hypothesis) + 4 <= self.max_length

    def score(self, pairs):
        """pairs: [(premise, hypothesis)] -> [(p_entail, p_neutral, p_contradict)], batched by length."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = [None] * len(pairs)
        with self.torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], return_tensors="pt",
                               padding=True, truncation="only_first", max_length=self.max_length).to(self.device)
                prob = self.model(**enc).logits.float().softmax(-1).cpu().tolist()
                for i, p in zip(idx, prob):
                    out[i] = tuple(round(p[j], 5) for j in self.idx)
        return out


# ------------------------------------------------------------------ premises (pure functions; fits = NLI.fits)
def plain_sentences(raw):
    """Split the raw (LaTeX) text into sentences first, then convert each: the conversion collapses newlines."""
    return [t for t in (latex_to_plain(x) for x in split_sentences(raw or "")) if t]


def premise_a(prob_sents, claim, fits):
    full = " ".join(prob_sents)
    if fits(full, claim):
        return full
    return " ".join(top_sentences(claim, prob_sents, 3)) + " ... " + full


def premise_a2(prob_sents, earlier, claim, fits, return_mode=False):
    """earlier: the plain sentences of each step before the claim's step. The claim is never cut.
      full            problem + all earlier steps fit
      dropped_oldest  problem + the nearest earlier steps: the oldest steps are dropped one by one until it fits
      cut             not even problem + the nearest step fits: the problem (A's rule if it alone is too long) + the
                      nearest step, and the tokenizer cuts the end of the premise (the nearest step's tail)"""
    prob = " ".join(prob_sents)
    steps = [" ".join(e) for e in earlier]
    full = " ".join([prob] + steps)
    mode = "full"
    if not fits(full, claim):
        for k in range(1, len(steps)):
            cand = " ".join([prob] + steps[k:])
            if fits(cand, claim):
                full, mode = cand, "dropped_oldest"
                break
        else:
            full, mode = " ".join([premise_a(prob_sents, claim, fits)] + steps[-1:]), "cut"
    return (full, mode) if return_mode else full


def a2_mode(rec, claim, fits):
    """full / dropped_oldest / cut for one claim's A2 premise (rec: Stage 1 problem + steps only)."""
    earlier = [plain_sentences(st["text"]) for st in rec["steps"][:claim["step"]]]
    return premise_a2(plain_sentences(rec["problem"]), earlier, claim["text"], fits, return_mode=True)[1]


def premises_b(sample_sents, claim):
    """sample_sents: per sample, its plain sentences. One premise per sample with text."""
    return [(j, " ".join(top_sentences(claim, ss, 3))) for j, ss in enumerate(sample_sents) if ss]


def pairs_for(source, problem, steps, claims, fits, sample_sents=None):
    """(key, sample_or_None, premise, hypothesis) for every claim of one solution."""
    prob = plain_sentences(problem)
    step_sents = [plain_sentences(st["text"]) for st in steps]
    out = []
    for c in claims:
        key = f"{c['step']}.{c['claim_idx']}"
        if source == "A":
            out.append((key, None, premise_a(prob, c["text"], fits), c["text"]))
        elif source == "A2":
            out.append((key, None, premise_a2(prob, step_sents[:c["step"]], c["text"], fits), c["text"]))
        elif source == "B":
            out += [(key, j, prem, c["text"]) for j, prem in premises_b(sample_sents, c["text"])]
        else:
            raise ValueError(source)
    return out


def aggregate_b(rows):
    """rows: [(sample, (pe, pn, pc))] of one claim -> mean p_contradict, max p_entail over the samples."""
    if not rows:
        return {"p_contradict": None, "p_entail": None, "p_neutral": None, "n_samples": 0}
    pe, pn, pc = zip(*[p for _, p in rows])
    return {"p_contradict": round(sum(pc) / len(pc), 5), "p_entail": round(max(pe), 5),
            "p_neutral": round(sum(pn) / len(pn), 5), "n_samples": len(rows),
            "per_sample": {str(j): p for j, p in rows}}


def sample_sentences(split, ids):
    path = f"{P}/samples_{split}.jsonl"
    out = {}
    for r in read_jsonl(path):
        if r["id"] in ids:
            out[r["id"]] = [plain_sentences(x.get("text")) for x in r["samples"]]
    return out


# ------------------------------------------------------------------ Source C: numeric spot check
CONDITION = re.compile(r"\b(if|when|whenever|suppose|assume|assuming|such that|where|for (?:all|some|every|any|each)|"
                       r"solve|solving|solution|let|contradiction|impossible|cannot|no solution)\b|[<>≤≥≠]|"
                       r"\\(?:le|ge|leq|geq|neq|ne|lt|gt|approx|equiv|pmod|bmod|mod|in|pm|mp)(?![A-Za-z])", re.I)
REWRITE = re.compile(r"\b(simplif\w*|expand\w*|factor\w*|rewrit\w*|identity|equivalent\w*|distribut\w*|"
                     r"combin\w* like terms)\b", re.I)
INTEGER_NAMES = set("nmkijNMK")
ALLOWED_FUNCS = (sp.binomial, sp.factorial)
N_POINTS, MAX_TRIES, MAX_EQ_CHARS = 5, 60, 160


def _lone_name(side):
    return bool(re.fullmatch(r"\s*[A-Za-z](?:_\{?\w+\}?)?\s*(?:\([^()]*\))?\s*", side))


def _juxtaposed(side):
    """'23b^3 b^3': two terms next to each other across a space, usually two expressions run together."""
    toks = side.split()
    return any(re.search(r"[\w)}]$", a) and re.match(r"[\w(\\]", b) and not re.match(r"\\(?:times|cdot|div)", b)
               and not re.search(r"\\(?:times|cdot|div|frac|sqrt|left|right)$", a) for a, b in zip(toks, toks[1:]))


def _point(syms, rng):
    return {s: sp.Integer(rng.randint(2, 12)) if s.name in INTEGER_NAMES
            else sp.Rational(rng.randint(1, 40), rng.randint(1, 9)) for s in syms}


def _value(e, pt):
    v = sp.N(e.subs(pt), 30)
    if v.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or not v.is_number:
        return None
    re_, im_ = v.as_real_imag()
    if abs(im_) > 1e-12 * max(1, abs(re_)):
        return None
    return float(re_)


def _check_pair(a, b, rng):
    """One link of an equation chain: 'pass', ('const', c), 'differ', ('skip', why) or None (no free symbols)."""
    syms = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    if not syms:
        return None                                                    # plain numbers: Stage 4's job
    vals, tries = [], 0
    while len(vals) < N_POINTS and tries < MAX_TRIES:
        tries += 1
        pt = _point(syms, rng)
        try:
            va, vb = _value(a, pt), _value(b, pt)
        except SYMPY_ERRORS:
            va = vb = None
        if va is not None and vb is not None:
            vals.append((va, vb))
    if len(vals) < N_POINTS:
        return "skip", f"domain issue: only {len(vals)} valid points in {MAX_TRIES} tries"
    differ = [abs(va - vb) > 1e-9 * max(1.0, abs(va), abs(vb)) for va, vb in vals]
    if not any(differ):
        return "pass"
    if not all(differ):
        return "skip", f"domain issue: {sum(differ)} of {N_POINTS} points differ"
    diffs = [va - vb for va, vb in vals]
    if all(abs(d - diffs[0]) <= 1e-9 * max(1.0, abs(d)) for d in diffs):
        return "const", diffs[0]
    return "differ"


def numeric_check_single(eq, context, rng):
    """A chain s0 = s1 = ... is checked link by link. A link that differs at all 5 points by the same constant is
    true for no values: fail. A link that differs by a varying amount may be the condition being solved
    ('5x + 9 = (x+3)^2 = x^2 + 6x + 9'); a chain holds at most one condition, so it fails when two or more links
    differ, or one differs and the claim says it is a rewrite (simplify / expand / factor ...)."""
    sides = eq.split("=")
    if len(sides) < 2 or any(not s.strip() for s in sides):
        return "skip", "not an equation"
    while len(sides) > 2 and _lone_name(sides[0]):                     # 'f(x) = (x+1)^2 = x^2 + 2x + 1'
        sides.pop(0)
    if len(eq) > MAX_EQ_CHARS:                                         # SymPy can stall on these (guard)
        return "skip", f"guard: equation longer than {MAX_EQ_CHARS} characters"
    if re.search(r"\^\s*\{?\s*\(?\s*\d{3,}", eq):
        return "skip", "guard: exponent of 100 or more"
    exprs = [to_sympy(s) for s in sides]
    if any(e is None for e in exprs):
        return "skip", "could not parse"
    if any(not isinstance(e, sp.Expr) for e in exprs):
        return "skip", "inequality or non-scalar value"
    if not any(e.free_symbols for e in exprs):
        return None
    for e in exprs:
        if any(not isinstance(f, ALLOWED_FUNCS) for f in e.atoms(sp.Function)):
            return "skip", "unsupported function"
    links = []
    for i in range(len(exprs) - 1):
        try:
            r = _check_pair(exprs[i], exprs[i + 1], rng)
        except SYMPY_ERRORS as e:
            r = ("skip", f"error: {type(e).__name__}")
        if r is not None:
            links.append((i, r))
    if not links:
        return None
    for i, r in links:
        if isinstance(r, tuple) and r[0] == "const":
            return "fail", f"sides differ by the constant {r[1]:.6g} at every point (true for no values)"
    differ = [i for i, r in links if r == "differ"]
    if differ and any(_juxtaposed(sides[k]) for i in differ for k in (i, i + 1)):
        return "skip", "terms run together across a space (probably two expressions)"
    if len(differ) >= 2:
        return "fail", f"{len(differ)} links of the chain differ at all {N_POINTS} points (at most one can be a condition)"
    if len(differ) == 1:
        i = differ[0]
        if not exprs[i].free_symbols or not exprs[i + 1].free_symbols:
            return "skip", "condition: one side is a constant (an equation to solve)"
        if _lone_name(sides[i]) or _lone_name(sides[i + 1]):
            return "skip", "definition / assignment (one side is a single name)"
        if REWRITE.search(context):
            return "fail", f"claimed rewrite differs at all {N_POINTS} points"
        return "skip", "equation in the unknowns (may be a condition to solve)"
    skips = [r for _, r in links if r != "pass"]
    return ("pass", f"equal at {N_POINTS} points") if not skips else skips[0]


def claim_equations(latex):
    """Equations in a claim: Stage 1's extractor plus maximal runs of math tokens around '=' (Stage 1 drops
    '(x+1)^2 = x^2 + 2x + 1' because its left side is one symbolic token). Substrings of a longer one are dropped."""
    toks = latex.replace("=", " = ").split()
    runs, cur = [], []
    for t in toks + [None]:
        if t is not None and _is_math_token(t):
            cur.append(t)
            continue
        if "=" in cur:
            runs.append(_clean_equation(" ".join(cur)).strip(" ,;:."))
        cur = []
    found = list(extract_equations(latex))
    for r in runs:
        for piece in _split_juxtaposed(r):
            eq = _trim_sides(piece.strip(" ,;:."))
            if eq and re.search(r"[A-Za-z\d]", eq):
                found.append(eq)
    norm = lambda e: re.sub(r"\s+", "", e)
    uniq = list(dict.fromkeys(found))
    return [e for e in uniq if not any(norm(e) != norm(o) and norm(e) in norm(o) for o in uniq)]


def numeric_check(claim_latex):
    """Equations of the claim (claim_equations) that contain free symbols, each evaluated with SymPy at 5 random
    rational points (seeded by the equation text; n, m, k, i, j, N, M, K get integers 2..12, other symbols
    p/q with p in 1..40, q in 1..9, so positive and away from 0; points giving a pole, an undefined or complex
    value are redrawn, at most 60 tries).
      pass  every link of the equation is equal at all 5 points
      fail  a link differs at all 5 points by the same constant (true for no values of the unknowns), or the chain
            has 2+ links that differ at all 5 points, or 1 such link in a claim that calls it a rewrite
            (simplify / expand / factor / rewrite / identity / equivalent / distribute / combine like terms)
      skip  with a reason: condition words or relations in the claim (if, when, suppose, let, solve, where, <, >,
            \\neq, \\equiv, mod, contradiction ...), inequality, parse failure, unsupported function (anything but
            binomial / factorial), domain issue (fewer than 5 valid points, or only some points differ: sporadic,
            not a flag), a constant side or a single-name side (equation to solve, definition), or a single
            differing link that may be the condition being solved.
    Equations without free symbols are left to Stage 4 (not counted). Returns [(equation, status, reason)]."""
    out = []
    cond = CONDITION.search(claim_latex)
    for eq in claim_equations(claim_latex):
        for part in _split_top_commas(eq):
            if "=" not in part:
                continue
            rng = random.Random(zlib.crc32(f"{SEED}|{part}".encode()))
            if cond:
                exprs = [to_sympy(s) for s in part.split("=")] if len(part) <= MAX_EQ_CHARS else [None]
                if all(e is not None for e in exprs) and not any(getattr(e, "free_symbols", None) for e in exprs):
                    continue                                           # numeric: Stage 4's job
                out.append((part, "skip", f"condition in the claim ('{cond.group(0).strip()}')"))
                continue
            r = numeric_check_single(part, claim_latex, rng)
            if r is not None:
                out.append((part, *r))
    return out


def claim_status(results):
    st = [s for _, s, _ in results]
    return "none" if not st else "fail" if "fail" in st else "pass" if all(s == "pass" for s in st) else "skip"


# ------------------------------------------------------------------ Source D: ProofWiki index + retrieval
TITLE_KEEP = re.compile(
    r"\b(triangle|circle|angle|square|rectangle|polygon|quadrilateral|parallelogram|trapezoid|rhombus|hexagon|"
    r"pentagon|area|volume|perimeter|circumference|chord|tangent|secant|inscribed|circumscribed|pythagor\w*|"
    r"similar|congruent|median|altitude|bisector|centroid|incircle|circumcircle|sphere|cylinder|cone|cube|"
    r"prime|divisor|divisib\w*|divides|multiple|gcd|lcm|greatest common|lowest common|factor\w*|congruen\w*|"
    r"modulo|residue|remainder|euclid\w*|fermat|euler|wilson|chinese remainder|digit\w*|parity|even|odd|"
    r"integer|rational|irrational|fraction|decimal|percent\w*|ratio|proportion|average|mean|median|"
    r"sum|product|series|sequence|arithmetic|geometric|harmonic|progression|fibonacci|telescop\w*|"
    r"polynomial|quadratic|cubic|root|roots|vieta|viete|discriminant|binomial|coefficient|combination\w*|"
    r"permutation\w*|factorial|choose|counting|pigeonhole|inclusion|exclusion|probability|expectation|"
    r"inequality|am-gm|cauchy|schwarz|triangle inequality|absolute value|logarithm\w*|exponent\w*|power|powers|"
    r"square root|cube root|trigonometric|sine|cosine|tangent|law of|identity|distance|midpoint|slope|line|"
    r"parabola|ellipse|hyperbola|vector|dot product|cross product|determinant|complex number|modulus|"
    r"floor|ceiling|function|inverse|composition|graph)\b", re.I)
TITLE_DROP = re.compile(
    r"\b(primitive|integral|derivative|differential|topolog\w*|group|subgroup|ring|ideal|field extension|module|"
    r"banach|hilbert|measure|lebesgue|category|sigma-algebra|ordinal|cardinal|lattice|boolean|manifold|metric space|"
    r"homomorphism|isomorphism|morphism|functor|sheaf|scheme|galois|semigroup|monoid|zorn|axiom|set theory|"
    r"ultrafilter|compact|hausdorff|continuous|limit|convergen\w*|laplace|fourier|transform|zeta|gamma function|"
    r"bessel|legendre polynomial|hermite|chebyshev polynomial|elliptic|hyperbolic|arcsine|arccosine|arctangent|"
    r"definite|indefinite|matrix|matrices|operator|eigen\w*|tensor|quaternion|projective|affine|orthogonal group)\b", re.I)
PW_MACROS = [(r"\\paren\s*\{", "("), (r"\\size\s*\{", "|"), (r"\\set\s*\{", "{"), (r"\\map\s*\{([^{}]*)\}\s*\{", r"\1("),
             (r"\\ds\b", ""), (r"\\rd\b", "d"), (r"\\leadsto", "implies"), (r"\\floor\s*\{", "floor("),
             (r"\\ceiling\s*\{", "ceil("), (r"\\gcd\s*\{", "gcd("), (r"\\lcm\s*\{", "lcm("), (r"\\divides", "divides"),
             (r"\\card\s*\{", "card("), (r"\\bsqbrk\s*\{", "["), (r"\\tuple\s*\{", "(")]


def _pw_plain(s):
    """ProofWiki macros -> plain. The opening macro brace becomes '(' and its closing '}' is dropped later by
    latex_to_plain; the bracket balance in passages is therefore approximate, which is fine for retrieval."""
    for pat, rep in PW_MACROS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"This article is incomplete\..*?(?:You can help[^.]*\.|$)", " ", s.replace("$", " "))
    return latex_to_plain(s)


def build_index(csv_path, out_dir, embed_model, batch_size, max_passages):
    csv.field_size_limit(10 ** 9)
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seen, passages = set(), []
    for r in rows:
        title = unquote(r["links"].rstrip("/").split("/wiki/")[-1]).replace("_", " ").strip()
        if "/" in title or not TITLE_KEEP.search(title) or TITLE_DROP.search(title):
            continue
        stmt = _pw_plain(r["thms"] or "")
        stmt = re.sub(r"\s+", " ", stmt).strip()
        if not 20 <= len(stmt) <= 1200 or title.lower() in seen:
            continue
        seen.add(title.lower())
        passages.append({"pid": len(passages), "title": title, "text": f"{title}: {stmt}", "url": r["links"]})
    if max_passages and len(passages) > max_passages:
        rng = random.Random(SEED)
        passages = sorted(rng.sample(passages, max_passages), key=lambda p: p["pid"])
        for i, p in enumerate(passages):
            p["pid"] = i
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/passages.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    import numpy as np
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    emb = SentenceTransformer(embed_model, device="cpu").encode([p["text"] for p in passages], batch_size=batch_size,
                                                                normalize_embeddings=True, show_progress_bar=True)
    np.save(f"{out_dir}/bge_embeddings.npy", emb.astype("float32"))
    print(f"{len(passages)} of {len(rows)} ProofWiki statements kept -> {out_dir}/passages.jsonl; "
          f"embeddings {emb.shape} in {time.time() - t0:.0f} s")


def knowledge_queries(pb, claims_by_id, ids):
    """One query per Stage 1 knowledge_claim step: its knowledge-cue claims, else the step text (plain, 400 chars)."""
    out = []
    for i in ids:
        for st in pb[i]["steps"]:
            if st["type"] != "knowledge_claim":
                continue
            cl = [c for c in claims_by_id.get(i, []) if c["step"] == st["idx"]]
            cue = [c["text"] for c in cl if KNOWLEDGE_CUES.search(c["latex"])]
            q = " ".join(cue) if cue else latex_to_plain(st["text"])
            out.append({"id": i, "step": st["idx"], "query": q[:400], "claims": cl})
    return out


# ------------------------------------------------------------------ runners
def done_keys(path):
    return {r["key"] for r in read_jsonl(path)} if os.path.exists(path) else set()


def log_runtime(source, model, n_claims, n_pairs, seconds, batch_size, note=""):
    path = "results/rag_pilot_runtime.csv"
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["source", "model", "n_claims", "n_pairs", "seconds", "sec_per_1000_claims", "batch_size",
                        "cpu_threads", "date", "note"])
        import torch
        w.writerow([source, model, n_claims, n_pairs, round(seconds, 1),
                    round(1000 * seconds / n_claims, 1) if n_claims else "", batch_size, torch.get_num_threads(),
                    time.strftime("%Y-%m-%d %H:%M"), note])


def load_claims(path):
    by_id = defaultdict(list)
    for c in read_jsonl(path):
        by_id[c["id"]].append(c)
    return by_id


def run_nli(a):
    claims = load_claims(a.claims)
    ids = list(claims)
    pb = load_pb(ids)
    out = f"{P}/rag_pilot_nli_{a.source}.jsonl"
    done = done_keys(out)
    todo = [i for i in ids if any(f"{i}|{c['step']}.{c['claim_idx']}" not in done for c in claims[i])]
    print(f"source {a.source}: {len(ids)} solutions, {sum(map(len, claims.values()))} claims, "
          f"{len(ids) - len(todo)} solutions already done")
    if not todo:
        return
    nli = NLI(a.model, a.batch_size, a.max_length, a.device, a.threads)
    samples = {}
    if a.source == "B":
        for spl in SPLITS:
            samples.update(sample_sentences(spl, {i for i in todo if split_of(i) == spl}))
    t_total, n_claims, n_pairs = 0.0, 0, 0
    with open(out, "a", encoding="utf-8") as f:
        for k, i in enumerate(todo):
            cl = [c for c in claims[i] if f"{i}|{c['step']}.{c['claim_idx']}" not in done]
            t0 = time.time()
            pairs = pairs_for(a.source, pb[i]["problem"], pb[i]["steps"], cl, nli.fits, samples.get(i))
            probs = nli.score([(p, h) for _, _, p, h in pairs])
            t_total += time.time() - t0
            n_claims += len(cl)
            n_pairs += len(pairs)
            by_key = defaultdict(list)
            for (key, j, _, _), p in zip(pairs, probs):
                by_key[key].append((j, p))
            for c in cl:
                key = f"{c['step']}.{c['claim_idx']}"
                if a.source == "B":
                    rec = aggregate_b(by_key.get(key, []))
                else:
                    pe, pn, pc = by_key[key][0][1]
                    rec = {"p_entail": pe, "p_neutral": pn, "p_contradict": pc}
                    if a.source == "A2":
                        rec["premise_mode"] = a2_mode(pb[i], c, nli.fits)
                f.write(json.dumps({"key": f"{i}|{key}", "id": i, "step": c["step"], "claim_idx": c["claim_idx"],
                                    **rec}) + "\n")
            f.flush()
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                print(f"  {k + 1}/{len(todo)} solutions, {n_claims} claims, {n_pairs} pairs, {t_total:.0f} s "
                      f"({1000 * t_total / max(n_claims, 1):.0f} s per 1000 claims)", flush=True)
    log_runtime(a.source, a.model, n_claims, n_pairs, t_total, a.batch_size)


def run_numeric(a):
    claims = load_claims(a.claims)
    t0 = time.time()
    n = 0
    with open(f"{P}/rag_pilot_numeric.jsonl", "w", encoding="utf-8") as f:
        for i, cl in claims.items():
            for c in cl:
                res = numeric_check(c["latex"])
                n += 1
                f.write(json.dumps({"key": f"{i}|{c['step']}.{c['claim_idx']}", "id": i, "step": c["step"],
                                    "claim_idx": c["claim_idx"], "status": claim_status(res),
                                    "equations": [{"equation": e, "status": s, "reason": r} for e, s, r in res]}) + "\n")
    dt = time.time() - t0
    log_runtime("C", "sympy", n, 0, dt, "")
    print(f"pilot claims: {n} checked in {dt:.1f} s -> {P}/rag_pilot_numeric.jsonl")

    # all sampled problems: counts per split and a failure list for inspection (no labels used or shown)
    with open(a.sample, encoding="utf-8", newline="") as f:
        pilot = {r["id"]: r["half"] for r in csv.DictReader(f)}
    lines, fails = [], []
    log = lambda s="": (print(s), lines.append(s))
    log(f"\n===== Source C (numeric spot check) on all sampled problems; counts are over equations with free symbols")
    for spl in SPLITS:
        ids = {r["id"] for r in read_jsonl(f"{P}/s7_{spl}.jsonl")}
        pb = [r for r in read_jsonl(f"{P}/pb_{spl}.jsonl") if r["id"] in ids]
        cnt, why, n_cl, sf, guard, slow, t_split = Counter(), Counter(), 0, [], Counter(), [], time.time()
        for r in pb:
            for c in extract_claims(r["problem"], r["steps"]):
                n_cl += 1
                t = time.time()
                res = numeric_check(c["latex"])
                if time.time() - t > 2:
                    slow.append((round(time.time() - t, 1), r["id"], c["step"], c["latex"][:120]))
                for reason in {w for _, s, w in res if w.startswith("guard:")}:
                    guard[reason] += 1                                 # claims, not equations
                for e, s, w in res:
                    cnt[s] += 1
                    if s == "skip":
                        why[re.sub(r"\d+ of \d+|only \d+|\('.*'\)|-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "#", w)] += 1
                    if s == "fail":
                        sf.append({"split": spl, "id": r["id"], "step": c["step"], "claim": c["text"], "equation": e,
                                   "reason": w, "pilot_half": pilot.get(r["id"], "")})
        tot = sum(cnt.values())
        log(f"\n{spl}: {len(pb)} solutions, {n_cl} claims, {tot} symbolic equations: pass {cnt['pass']}, "
            f"fail {cnt['fail']}, skip {cnt['skip']}  ({time.time() - t_split:.0f} s)")
        log("  skip reasons: " + "; ".join(f"{w} {k}" for w, k in why.most_common(8)))
        log(f"  claims skipped by the size guard: {sum(guard.values())}" +
            ("".join(f"; {w[7:]} {k}" for w, k in guard.most_common()) if guard else ""))
        for t_, i_, st_, lx in slow:
            log(f"  slow claim ({t_} s): {i_} step {st_}: {lx}")
        rng = random.Random(SEED)
        fails += rng.sample(sf, min(15, len(sf)))
    with open("results/rag_numeric_failures.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "id", "step", "claim", "equation", "reason", "pilot_half", "my_verdict"])
        w.writeheader()
        w.writerows({**r, "my_verdict": ""} for r in fails)
    log(f"\n{len(fails)} failures (up to 15 random per split) -> results/rag_numeric_failures.csv")
    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/rag_numeric_check.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_retrieve(a):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    passages = read_jsonl(f"{INDEX_DIR}/passages.jsonl")
    emb = np.load(f"{INDEX_DIR}/bge_embeddings.npy")
    claims = load_claims(a.claims)
    ids = list(claims)
    pb = load_pb(ids)
    queries = knowledge_queries(pb, claims, ids)
    print(f"{len(queries)} knowledge_claim steps in {len(ids)} solutions; index: {len(passages)} passages")
    t0 = time.time()
    bm = BM25([p["text"] for p in passages])
    t_bm_build = time.time() - t0
    enc = SentenceTransformer(EMBED_MODEL, device="cpu")
    t_bm = t_emb = 0.0
    for q in queries:
        t = time.time()
        sc = bm.scores(q["query"])
        q["bm25"] = sorted(range(len(sc)), key=lambda k: (-sc[k], k))[:5]
        t_bm += time.time() - t
        t = time.time()
        qv = enc.encode([BGE_QUERY + q["query"]], normalize_embeddings=True)[0]
        q["bge"] = [int(k) for k in np.argsort(-(emb @ qv), kind="stable")[:5]]
        t_emb += time.time() - t
    n_cl = sum(len(q["claims"]) for q in queries)
    nli = NLI(a.model, a.batch_size, a.max_length, a.device, a.threads)
    t = time.time()
    with open(f"{P}/rag_pilot_nli_D.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            for ret in ("bm25", "bge"):
                prem = [passages[k]["text"] for k in q[ret]]
                for c in q["claims"]:
                    probs = nli.score([(p, c["text"]) for p in prem])
                    pe, pn, pc = zip(*probs)
                    f.write(json.dumps({"key": f"{q['id']}|{c['step']}.{c['claim_idx']}", "id": q["id"], "step": c["step"],
                                        "claim_idx": c["claim_idx"], "retriever": ret, "p_contradict": round(sum(pc) / 5, 5),
                                        "p_entail": round(max(pe), 5), "p_neutral": round(sum(pn) / 5, 5),
                                        "per_passage": probs}) + "\n")
    t_nli = time.time() - t
    with open(f"{P}/rag_pilot_retrieval.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"id": q["id"], "step": q["step"], "query": q["query"],
                                "bm25": q["bm25"], "bge": q["bge"]}) + "\n")
    log_runtime("D-retrieval-bm25", "bm25", n_cl, len(queries), t_bm + t_bm_build, "", f"{len(queries)} queries incl. index build")
    log_runtime("D-retrieval-bge", EMBED_MODEL, n_cl, len(queries), t_emb, "", f"{len(queries)} queries")
    log_runtime("D-nli", a.model, n_cl, 10 * n_cl, t_nli, a.batch_size, "5 passages x 2 retrievers per claim")
    write_audit(queries, passages, a)


def run_benchmark(a):
    """Time each source alone on every 10th pilot claim (fixed subset); nothing is written except runtime rows.
    Run it when no other job uses the CPU."""
    claims = load_claims(a.claims)
    pb = load_pb(list(claims))
    flat = [c for cl in claims.values() for c in cl][::10]
    by_id = defaultdict(list)
    for c in flat:
        by_id[c["id"]].append(c)
    samples = {}
    for spl in SPLITS:
        samples.update(sample_sentences(spl, {i for i in by_id if split_of(i) == spl}))
    t = time.time()
    for c in flat:
        numeric_check(c["latex"])
    log_runtime("C", "sympy", len(flat), 0, time.time() - t, "", "benchmark: every 10th pilot claim")
    nli = NLI(a.model, a.batch_size, a.max_length, a.device, a.threads)
    for src in ("A", "A2", "B"):
        t, n_pairs = time.time(), 0
        for i, cl in by_id.items():
            pairs = pairs_for(src, pb[i]["problem"], pb[i]["steps"], cl, nli.fits, samples.get(i))
            nli.score([(p, h) for _, _, p, h in pairs])
            n_pairs += len(pairs)
        dt = time.time() - t
        log_runtime(src, a.model, len(flat), n_pairs, dt, a.batch_size, "benchmark: every 10th pilot claim")
        print(f"{src}: {len(flat)} claims, {n_pairs} pairs, {dt:.0f} s ({1000 * dt / len(flat):.0f} s per 1000 claims)")


def write_audit(queries, passages, a):
    """~50 rows: one per (knowledge step, retriever), steps from the pilot sample first; if the pilot has fewer than
    25 such steps, knowledge steps of other sampled problems (seed 0) fill up to 25 (column in_pilot)."""
    with open(a.sample, encoding="utf-8", newline="") as f:
        sample_ids = {r["id"] for r in csv.DictReader(f)}
    rows = [(q, True) for q in queries if q["id"] in sample_ids]
    if len(rows) < a.audit_steps:
        extra_ids = []
        for spl in SPLITS:
            extra_ids += [r["id"] for r in read_jsonl(f"{P}/s7_{spl}.jsonl") if r["id"] not in sample_ids]
        pb = load_pb(extra_ids)
        cl = {i: extract_claims(pb[i]["problem"], pb[i]["steps"]) for i in extra_ids}
        extra = knowledge_queries(pb, cl, extra_ids)
        rng = random.Random(SEED)
        extra = rng.sample(extra, min(len(extra), a.audit_steps - len(rows)))
        if extra:
            bm = BM25([p["text"] for p in passages])
            import numpy as np
            from sentence_transformers import SentenceTransformer
            emb = np.load(f"{INDEX_DIR}/bge_embeddings.npy")
            enc = SentenceTransformer(EMBED_MODEL, device="cpu")
            for q in extra:
                sc = bm.scores(q["query"])
                q["bm25"] = sorted(range(len(sc)), key=lambda k: (-sc[k], k))[:5]
                qv = enc.encode([BGE_QUERY + q["query"]], normalize_embeddings=True)[0]
                q["bge"] = [int(k) for k in np.argsort(-(emb @ qv), kind="stable")[:5]]
        rows += [(q, False) for q in extra]
    elif len(rows) > a.audit_steps:
        rows = random.Random(SEED).sample(rows, a.audit_steps)
    out = [{"id": q["id"], "step": q["step"], "claim": q["query"], "retriever": ret, "in_pilot": inp,
            **{f"top{k + 1}": passages[pid]["text"][:300] for k, pid in enumerate(q[ret])}, "my_verdict": ""}
           for q, inp in rows for ret in ("bm25", "bge")]
    random.Random(SEED).shuffle(out)
    with open("results/rag_retrieval_audit.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} audit rows ({len(rows)} steps x 2 retrievers, {sum(inp for _, inp in rows)} steps from the "
          f"pilot) -> results/rag_retrieval_audit.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["nli", "numeric", "build-index", "retrieve", "benchmark"])
    ap.add_argument("--source", choices=["A", "A2", "B"])
    ap.add_argument("--claims", default=f"{P}/rag_pilot_claims.jsonl")
    ap.add_argument("--sample", default="results/rag_pilot_sample.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-passages", type=int, default=5000)
    ap.add_argument("--audit-steps", type=int, default=25)
    a = ap.parse_args()
    if a.command == "nli":
        if not a.source:
            ap.error("nli needs --source A, A2 or B")
        run_nli(a)
    elif a.command == "numeric":
        run_numeric(a)
    elif a.command == "build-index":
        build_index(PROOFWIKI_CSV, INDEX_DIR, EMBED_MODEL, a.batch_size * 4, a.max_passages)
    elif a.command == "benchmark":
        run_benchmark(a)
    else:
        run_retrieve(a)
