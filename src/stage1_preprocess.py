import json, re, sys

STEP_SPLIT = re.compile(r"\n\s*\n|^\s*(?:Step\s*\d+[:.)]|\d+[.)])\s*", re.M)
NUM = re.compile(r"-?\d+(?:\.\d+)?")
KNOWLEDGE_CUES = re.compile(
    r"\b(theorem|formula|identity|law|rule|by definition|property|lemma|"
    r"we know that|recall that|according to)\b", re.I)

def normalize(text: str, keep_dollar: bool = False) -> str:
    """keep_dollar=True for GSM8K, where '$' is currency ($18), not a LaTeX math delimiter."""
    text = text.replace("\u2212", "-").replace("\u00d7", "*").replace("\u00f7", "/")
    if keep_dollar:
        text = text.replace("\\$", "$")                           # '\$18' -> '$18'
        text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)         # strip \( \) \[ \] only
    else:
        text = re.sub(r"\\\(|\\\)|\\\[|\\\]|\$\$?", "", text)  # strip math delimiters
    text = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)            # strip \text{...}
    text = re.sub(r"\\(?:textbf|textit|textrm|textit)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right)", "", text)                # strip \left \right
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)           # strip \boxed{...}
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def split_steps(solution: str):
    return [p.strip() for p in STEP_SPLIT.split(solution) if p and p.strip()]

UNIT_WORDS = {
    "pounds", "pound", "lbs", "lb", "kg", "g", "grams", "gram",
    "books", "book", "gallons", "gallon", "gal", "liters", "liter", "l",
    "meters", "meter", "m", "feet", "foot", "ft", "inches", "inch", "in",
    "dollars", "dollar", "cents", "cent", "euros", "euro",
    "hours", "hour", "hrs", "hr", "minutes", "minute", "mins", "min", "seconds", "second", "secs", "sec",
    "miles", "mile", "km", "kilometers", "kilometer",
    "items", "item", "pieces", "piece", "pcs", "pc",
    "students", "student", "people", "person", "children", "child",
    "apples", "apple", "oranges", "orange", "flamingos", "flamingo",
    "clips", "clip", "boxes", "box", "cars", "car",
}

MATH_TOKEN = re.compile(r"^[\\(){}\[\]\d\.\+\-\*/\^=<>_,%$]*[\d\.\+\-\*/\^=<>%]?[\\(){}\[\]\d\.\+\-\*/\^=<>_,%a-zA-Z]*$")

def _is_unit(tok: str) -> bool:
    return tok.strip(",;:.").lower() in UNIT_WORDS and len(tok.strip(",;:.")) > 1

def _is_number(tok: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?|\d+/\d+", tok.strip(",;:.")))

def _is_math_token(tok: str) -> bool:
    """Numbers, operators, single-letter variables, LaTeX-ish tokens. Unit words are
    handled separately (see extract_equations) so they never act as a span boundary."""
    tok = tok.strip(",;:.")
    if not tok:
        return False
    if re.fullmatch(r"[A-Za-z]", tok):              # single-letter variable
        return True
    if _is_unit(tok):
        return False
    if re.fullmatch(r"[A-Za-z]{2,}", tok):          # ordinary word -> not math
        return False
    return bool(MATH_TOKEN.match(tok))


def _balance_parens(s: str) -> str:
    """Remove '(' / ')' characters that have no partner."""
    stack, drop = [], set()
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                stack.pop()
            else:
                drop.add(i)
    drop.update(stack)
    return "".join(ch for i, ch in enumerate(s) if i not in drop)


def _clean_equation(eq: str) -> str:
    """Strip units and non-math words from an equation for SymPy."""
    eq = re.sub(r"\([^()\d]*\)", "", eq)                      # drop text-only notes like '(women)', keep '4(3)'
    eq = eq.replace("$", "")                                   # currency sign is not part of the math
    eq = re.sub(r"(?<=\d),(?=\d{3}\b)", "", eq)              # 14,000 -> 14000
    words = eq.split()
    cleaned = []
    for w in words:
        w_clean = w.strip(",;:.)(")
        low = w_clean.lower()
        if (low in UNIT_WORDS or re.match(r"^[a-z]+/[a-z]+$", low)
                or re.fullmatch(r"[A-Za-z]{2,}", w_clean)):   # drop words, KEEP variables like x, y
            parens = "".join(ch for ch in w if ch in "()")    # but keep their brackets
            if parens:
                cleaned.append(parens)
            continue
        cleaned.append(w)
    return _balance_parens(" ".join(cleaned).strip(" ,;:.")).strip()


BINARY_START = re.compile(r"^\s*(?:[+*/^]|-\s|\\times|\\div|\\cdot)")
BINARY_END = re.compile(r"(?:[+\-*/^]|\\times|\\div|\\cdot)\s*$")

def _trim_sides(eq: str):
    """Drop incomplete sides at either end of a chain ('- 4 = 10 - 4 = 6' -> '10 - 4 = 6').
    Such fragments appear when a word operand was stripped out. Returns None if < 2 sides remain."""
    sides = eq.split("=")
    while sides and (not sides[0].strip() or BINARY_START.match(sides[0]) or BINARY_END.search(sides[0])):
        sides.pop(0)
    while sides and (not sides[-1].strip() or BINARY_START.match(sides[-1]) or BINARY_END.search(sides[-1])):
        sides.pop()
    if len(sides) < 2 or any(not s.strip() for s in sides):
        return None
    return " = ".join(s.strip() for s in sides)


def _split_juxtaposed(eq: str):
    """'71 + 72 = 143 143 + 73 = 216' is two equations written back to back:
    split wherever two numbers stand next to each other with no operator."""
    toks, pieces, cur = eq.split(), [], []
    for i, t in enumerate(toks):
        if t in ("\\quad", "\\qquad"):                       # explicit separator between equations
            pieces.append(" ".join(cur)); cur = []
            continue
        cur.append(t)
        if i + 1 < len(toks) and re.search(r"\d$", t) and (
                re.match(r"\d", toks[i + 1])                                        # '143 143'
                or (re.fullmatch(r"[A-Za-z]", toks[i + 1]) and toks[i + 2:i + 3] == ["="])):  # 'n-1 n = 14'
            pieces.append(" ".join(cur)); cur = []
    pieces.append(" ".join(cur))
    return pieces


def extract_equations(step: str):
    OPERATORS = {"+", "-", "*", "/", "^", "\\times", "\\div", "\\cdot"}
    step = re.sub(r"(\d(?:\.\d+)?\s*\\?%)\s+of\s+", r"\1 * ", step)   # '25% of 1.60' -> '25% * 1.60'
    toks = step.replace("=", " = ").split()
    eqs = []
    eq_positions = [i for i, t in enumerate(toks) if t == "="]
    
    for eq_idx in eq_positions:
        # RIGHT: a unit is accepted only directly after a number (e.g. "= 6 pounds")
        r = eq_idx
        has_operator = False
        while r + 1 < len(toks):
            if r > eq_idx and toks[r].endswith((".", ":", ";")):   # sentence ended
                break
            nxt = toks[r + 1]
            ok = _is_math_token(nxt) or (_is_unit(nxt) and _is_number(toks[r]))
            if not ok:
                break
            r += 1
            if toks[r] in OPERATORS:
                has_operator = True

        # LEFT: a unit is accepted only if an operator or '=' follows it (e.g. "2 pounds * 3")
        l = eq_idx
        left_has_operator = False
        while l - 1 >= 0:
            prv = toks[l - 1]
            if prv.endswith((".", ":", ";")):                      # previous sentence / label 'n:'
                break
            ok =_is_math_token(prv) or (_is_unit(prv) and (toks[l] in OPERATORS or toks[l] == "="))
            if not ok:
                break
            l -= 1
            if toks[l] in OPERATORS:
                left_has_operator = True
        
        if left_has_operator and l < eq_idx and r > eq_idx:
            span = " ".join(toks[l:r + 1]).strip(" ,;:.")
            eqs.append(_clean_equation(span))
        elif has_operator and r > eq_idx:
            span = " ".join(toks[eq_idx + 1:r + 1]).strip(" ,;:.")
            eqs.append(_clean_equation(span))
        elif l < eq_idx and r > eq_idx:
            span = " ".join(toks[l:r + 1]).strip(" ,;:.")
            eqs.append(_clean_equation(span))
    
    seen = set()
    unique_eqs = []
    for piece in (p for eq in eqs for p in _split_juxtaposed(eq)):
        eq = _trim_sides(piece)
        if eq is None:
            continue
        if re.search(r"[\+\-\*/\^]|\\times|\\div|\\cdot", eq) and eq not in seen:
            seen.add(eq)
            unique_eqs.append(eq)
    return unique_eqs

def tag_step(step: str, equations):
    if equations and len(NUM.findall(step)) >= 2:
        return "computational"
    if KNOWLEDGE_CUES.search(step):
        return "knowledge_claim"
    return "reasoning"

def parse_gsm8k_answer(answer: str):
    """Parse GSM8K answer into steps, extracting calculations from << >> markers."""
    parts = re.split(r"<<[^<>]+>>", answer)
    calcs = re.findall(r"<<([^<>]+)>>", answer)
    steps = []
    for i, (text, calc) in enumerate(zip(parts, calcs)):
        if text.strip():
            steps.append(text.strip())
        steps.append(calc.strip())
    if len(parts) > len(calcs) and parts[-1].strip():
        steps.append(parts[-1].strip())
    return [s for s in steps if s]


def preprocess_record(rec: dict) -> dict:
    # ProcessBench fields: problem, steps (list), label (earliest error idx, -1 = all correct)
    # GSM8K fields: question, answer
    keep = str(rec.get("id", "")).startswith("gsm8k") or "question" in rec
    if "problem" in rec:
        problem = normalize(rec["problem"], keep)
        raw_steps = rec.get("steps") or split_steps(rec.get("solution", ""))
        label = rec.get("label")
    elif "question" in rec:
        problem = normalize(rec["question"], keep)
        raw_steps = parse_gsm8k_answer(rec["answer"])
        label = None
    else:
        raise ValueError(f"Unknown record format: {rec.keys()}")
    steps = []
    for i, s in enumerate(raw_steps):
        s = normalize(s, keep)
        eqs = extract_equations(s)
        steps.append({"idx": i, "text": s, "equations": eqs, "type": tag_step(s, eqs)})
    return {"id": rec.get("id"), "problem": problem, "steps": steps,
            "label": label}

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            out.write(json.dumps(preprocess_record(json.loads(line))) + "\n"); n += 1
    print(f"processed {n} records -> {dst}")
