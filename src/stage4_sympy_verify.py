"""Stage 4: symbolic / arithmetic verification of the equations extracted in Stage 1.

For every equation (e.g. '1/3 \\times 18 = 6') we answer one of three things:
  pass  - both sides are mathematically equal (rounding to the stated decimals is allowed)
  fail  - both sides are numbers and they are NOT equal  => a calculation error
  skip  - we cannot judge (parse problem, unsupported LaTeX, or an equation with variables that
          may be a condition to solve, such as 2x = 4). 'skip' is NOT 'pass'.

Usage:
  python src/stage4_sympy_verify.py data/processed/pb_gsm8k.jsonl data/processed/s4_gsm8k.jsonl \
         [--summary results/stage4_summary.csv]
If the input has ProcessBench labels, a small evaluation against the human-marked first wrong step is printed
(and, with --summary, saved as this split's row of the CSV).
"""
import argparse, csv, json, os, re
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                         implicit_multiplication_application, rationalize)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, rationalize)
# single letters are always plain variables (otherwise E -> Euler's e, I -> imaginary unit, S, N, Q, O ...)
LETTERS = {c: sp.Symbol(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"}


# ------------------------------------------------------------------ LaTeX -> python string
def _match_brace(s, i):
    """s[i] == '{'. Return index of the matching '}' or -1."""
    depth = 0
    for j in range(i, len(s)):
        depth += (s[j] == "{") - (s[j] == "}")
        if depth == 0:
            return j
    return -1


def _replace_cmd(s, cmd, nargs, fmt):
    """Replace \\cmd{a}{b}... using fmt (with {0},{1}) handling nested braces."""
    key = "\\" + cmd
    while True:
        i = s.find(key)
        if i == -1:
            return s
        j, args = i + len(key), []
        for _ in range(nargs):
            while j < len(s) and s[j] == " ":
                j += 1
            if j >= len(s) or s[j] != "{":
                return None
            k = _match_brace(s, j)
            if k == -1:
                return None
            args.append(s[j + 1:k]); j = k + 1
        s = s[:i] + fmt.format(*args) + s[j:]


def latex_to_py(expr: str):
    s = expr.replace("$", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("\\pi", "pi")
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s)                         # degrees: 45^\circ -> 45
    if re.search(r"\d_\{?\d", s):                                         # base notation 1111_2: not a decimal number
        return None
    s = re.sub(r"(\d+(?:\.\d+)?)\s*\\?%", r"(\1/100)", s)                 # 20% -> (20/100)
    s = re.sub(r"(?<![\d.*/^])(\d+)\s*\\frac\{(\d+)\}\{(\d+)\}",           # mixed number 3\frac{4}{5} -> (3+4/5)
               lambda m: f"({m[1]}+{m[2]}/{m[3]})" if int(m[2]) < int(m[3]) else m[0], s)
    for cmd, n, fmt in (("frac", 2, "(({0})/({1}))"), ("sqrt", 1, "sqrt({0})"),
                        ("binom", 2, "binomial({0},{1})")):
        s = _replace_cmd(s, cmd, n, fmt)
        if s is None:
            return None
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", s)
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    if "\\" in s:                                                          # unsupported LaTeX command
        return None
    # f(0), F(1), C(9, 7): a function value, not f*0 -> one opaque symbol per call ('_' stops symbol splitting)
    s = re.sub(r"(?<![A-Za-z_])([A-Za-z])\(\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*)\s*\)",
               lambda m: f"{m[1]}_at_" + re.sub(r"\s*,\s*", "_", m[2]).replace("-", "m").replace(".", "p"), s)
    return s.strip()


def to_sympy(expr: str):
    py = latex_to_py(expr)
    if not py:
        return None
    if re.search(r"\*\*\s*\(?\s*\d{4,}", py):                              # avoid gigantic powers
        return None
    if re.search(r"\d\.?\s+\.?\d", py):                                    # '143 143' would become 143*143
        return None
    try:
        return parse_expr(py, local_dict=LETTERS, transformations=TRANSFORMS, evaluate=True)
    except Exception:
        return None


# ------------------------------------------------------------------ checking
def _decimals(side: str) -> int:
    m = re.fullmatch(r"\s*-?\d+\.(\d+)\s*", side)
    return len(m.group(1)) if m else 0


SYMPY_ERRORS = (TypeError, ValueError, AttributeError, NotImplementedError,
                ZeroDivisionError, OverflowError, RecursionError)


def _kind(e):
    """Why a parsed side cannot be compared as a number, or None if it can."""
    if not isinstance(e, sp.Expr):                                         # Symbol is also a Boolean, so test Expr first
        if isinstance(e, (sp.logic.boolalg.Boolean, sp.core.relational.Relational)):
            return "contains an inequality"
        return f"unsupported value ({type(e).__name__})"
    if e.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return "undefined value (division by zero or infinity)"
    return None


def _compare(a, b, left, right):
    """Compare two scalar SymPy expressions."""
    for e in (a, b):
        why = _kind(e)
        if why:
            return "skip", why
    diff = a - b
    if not a.free_symbols and not b.free_symbols:
        if sp.simplify(diff) == 0:
            return "pass", "equal"
        d = _decimals(right)
        err = float(abs(sp.N(diff)))                                       # abs() also handles complex values
        if d > 0 and err <= 10 ** (-d):                                    # rounded / truncated to stated decimals
            return "pass", f"equal up to {d} decimals"
        if "%" in left + right:                                            # '0.2 * 100 = 20%' (percent notation)
            a2, b2 = to_sympy(left.replace("\\%", "").replace("%", "")), to_sympy(right.replace("\\%", "").replace("%", ""))
            if (a2 is not None and b2 is not None and not _kind(a2) and not _kind(b2)
                    and not (a2 - b2).free_symbols and sp.simplify(a2 - b2) == 0):
                return "pass", "equal in percent notation"
        m = re.fullmatch(r"\s*(-?\d+)\s*\\div\s*(\d+)\s*", left)              # '410 \div 333 = 1' (remainder 77)
        if m and re.fullmatch(r"\s*-?\d+\s*", right) and int(m[2]):
            n, dv = int(m[1]), int(m[2])
            if int(right) in (n // dv, int(n / dv)):
                return "pass", "integer quotient (remainder dropped)"
        return "fail", f"{sp.N(a, 8)} != {sp.N(b, 8)}"
    if sp.expand(diff) == 0:
        return "pass", "identity"
    return "skip", "has variables (may be a condition to solve)"


def check_pair(left: str, right: str):
    a, b = to_sympy(left), to_sympy(right)
    if a is None or b is None:
        return "skip", "could not parse"
    try:
        seq_a, seq_b = isinstance(a, (tuple, list)), isinstance(b, (tuple, list))
        if seq_a and seq_b:                                                # points / tuples: compare element-wise
            if len(a) != len(b):
                return "skip", "tuples of different length"
            res = [_compare(x, y, "", "") for x, y in zip(a, b)]
            if any(st == "fail" for st, _ in res):
                return "fail", f"tuple element {next(why for st, why in res if st == 'fail')}"
            if all(st == "pass" for st, _ in res):
                return "pass", "equal tuples"
            return "skip", next(why for st, why in res if st == "skip")
        if seq_a or seq_b:
            return "skip", "tuple compared with a single value"
        return _compare(a, b, left, right)
    except SYMPY_ERRORS as e:
        return "skip", f"error: {type(e).__name__}: {str(e)[:80]}"


def _split_top_commas(eq: str):
    """'a = 1, b = 2' -> ['a = 1', 'b = 2']: split at commas outside any brackets."""
    parts, depth, cur = [], 0, ""
    for ch in eq:
        depth += (ch in "({[") - (ch in ")}]")
        if ch == "," and depth == 0 and not cur.endswith("\\"):           # '\,' is a LaTeX space, not a comma
            parts.append(cur); cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in (p.strip().rstrip("\\").strip() for p in parts) if p]   # drop trailing '\\' line breaks only


def check_equation(eq: str):
    parts = _split_top_commas(eq)
    if len(parts) > 1:                                                     # a list of equations
        res = [check_single(p) for p in parts if "=" in p]
        if not res:
            return "skip", "not an equation"
        for st, why in res:
            if st == "fail":
                return "fail", why
        if all(st == "pass" for st, _ in res):
            return "pass", res[0][1]
        return "skip", next(why for st, why in res if st == "skip")
    return check_single(eq)


def check_single(eq: str):
    sides = [s for s in eq.split("=")]
    if len(sides) < 2 or any(not s.strip() for s in sides):
        return "skip", "not an equation"
    if any(s.count("(") != s.count(")") or s.count("{") != s.count("}") for s in sides):
        return "skip", "unbalanced brackets"
    if re.fullmatch(r"\s*[A-Za-z]\s*\([^()]*\)\s*", sides[0]):               # 'f(0) = ...' is a function value
        sides.pop(0)
    while len(sides) > 2 and re.fullmatch(r"\s*[A-Za-z]\s*", sides[0]):     # 'S = 4(3) - 5 = 7': S is a name
        sides.pop(0)
    if len(sides) < 2:
        return "skip", "not an equation"
    results = [check_pair(sides[i], sides[i + 1]) for i in range(len(sides) - 1)]
    for st, why in results:
        if st == "fail":
            return "fail", why
    if all(st == "pass" for st, _ in results):
        return "pass", results[0][1]
    return "skip", next(why for st, why in results if st == "skip")


def verify_record(rec: dict) -> dict:
    steps_out, first_fail = [], None
    n_eq = n_checked = n_fail = 0
    for s in rec["steps"]:
        eq_res = []
        for eq in s["equations"]:
            st, why = check_equation(eq)
            eq_res.append({"equation": eq, "status": st, "reason": why})
            n_eq += 1
            n_checked += st in ("pass", "fail")
            n_fail += st == "fail"
        statuses = [e["status"] for e in eq_res]
        step_status = "fail" if "fail" in statuses else ("pass" if "pass" in statuses else "none")
        if step_status == "fail" and first_fail is None:
            first_fail = s["idx"]
        steps_out.append({"idx": s["idx"], "sympy_status": step_status, "equations": eq_res})
    out = dict(rec)
    out["sympy"] = {"steps": steps_out, "n_equations": n_eq, "n_checked": n_checked,
                    "n_fail": n_fail, "first_fail_step": first_fail}
    return out


# ------------------------------------------------------------------ evaluation vs human labels
def evaluate(records):
    err = [r for r in records if r.get("label") not in (-1, None)]
    ok = [r for r in records if r.get("label") == -1]
    if not err or not ok:
        return
    flagged_err = [r for r in err if r["sympy"]["n_fail"] > 0]
    exact = [r for r in err if r["sympy"]["first_fail_step"] == r["label"]]
    early = [r for r in flagged_err if r["sympy"]["first_fail_step"] < r["label"]]
    late = [r for r in flagged_err if r["sympy"]["first_fail_step"] > r["label"]]
    ceiling = [r for r in err if r["steps"][r["label"]]["equations"]]
    flagged_ok = [r for r in ok if r["sympy"]["n_fail"] > 0]
    tp, fp = len(flagged_err), len(flagged_ok)
    print("\n--- SymPy verifier vs human labels ---")
    print(f"solutions with a process error : {len(err)}   without: {len(ok)}")
    print(f"flagged (any failed equation)  : {tp} of {len(err)} erroneous ({100*tp/len(err):.1f}%)")
    print(f"   first failure = human first wrong step : {len(exact)} ({100*len(exact)/len(err):.1f}%)")
    print(f"   first failure earlier than human label : {len(early)}   later: {len(late)}")
    print(f"ceiling (first wrong step has an equation): {len(ceiling)} ({100*len(ceiling)/len(err):.1f}%)")
    print(f"false alarms on correct solutions: {fp} of {len(ok)} ({100*fp/len(ok):.1f}%)")
    print(f"precision of 'flagged': {100*tp/max(tp+fp,1):.1f}%")
    allsteps = [e for r in records for st in r["sympy"]["steps"] for e in st["equations"]]
    from collections import Counter
    c = Counter(e["status"] for e in allsteps)
    print(f"all equations: {dict(c)}")
    skip = Counter(e["reason"] for e in allsteps if e["status"] == "skip")
    print("most common skip reasons:", skip.most_common(4))
    pct = lambda n, d: round(100 * n / d, 1)
    return {"n_erroneous": len(err), "n_correct": len(ok),
            "flagged_pct": pct(tp, len(err)), "exact_first_step_pct": pct(len(exact), len(err)),
            "earlier": len(early), "later": len(late), "ceiling_pct": pct(len(ceiling), len(err)),
            "false_alarm_pct": pct(fp, len(ok)), "precision_pct": pct(tp, max(tp + fp, 1)),
            "eq_pass": c["pass"], "eq_fail": c["fail"], "eq_skip": c["skip"]}


def write_summary(path, split, metrics):
    """Insert or replace this split's row in a per-split summary CSV."""
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            rows = {r["split"]: r for r in csv.DictReader(f)}
    rows[split] = {"split": split, **metrics}
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", *metrics])
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--summary", help="CSV to update with this split's metrics, e.g. results/stage4_summary.csv")
    a = ap.parse_args()
    recs = []
    with open(a.input, encoding="utf-8") as f:
        for line in f:
            recs.append(verify_record(json.loads(line)))
    with open(a.output, "w", encoding="utf-8") as out:
        for r in recs:
            out.write(json.dumps(r) + "\n")
    print(f"verified {len(recs)} solutions -> {a.output}")
    metrics = evaluate(recs)
    if a.summary and metrics:
        split = re.sub(r"^(s4_|pb_)", "", os.path.splitext(os.path.basename(a.input))[0])
        write_summary(a.summary, split, metrics)
        print(f"summary row '{split}' -> {a.summary}")
