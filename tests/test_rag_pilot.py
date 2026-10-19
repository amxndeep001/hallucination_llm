"""RAG pilot (Stage 5 claims, Stage 6 evidence): unit tests and leakage tests.  Run:  python tests/test_rag_pilot.py
No model is loaded: the NLI code path is exercised with a deterministic fake scorer."""
import ast, os, re, sys, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from stage5_claims import extract_claims, latex_to_plain
from stage6_evidence import aggregate_b, claim_status, numeric_check, pairs_for, premise_a, premise_a2

FLAMINGO_PROBLEM = ("Sue lives in a fun neighborhood. On Friday morning, the neighbors placed 18 pink plastic flamingos out "
                    "on Sue's front yard. On Saturday morning, the neighbors took back one third of the flamingos, painted "
                    "them white, and put these newly painted white flamingos back out on Sue's front yard.")
FLAMINGO_STEPS = [
    {"idx": 0, "text": "First, on Friday, the neighbors start with 18 pink plastic flamingos.", "equations": [], "type": "reasoning"},
    {"idx": 1, "text": "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, 1/3 \\times 18 = 6 "
                       "flamingos are taken back. So, they have 18 - 6 = 12 flamingos left in their possession. Then, they paint "
                       "these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink "
                       "flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has 12 + 6 = 18 pink flamingos and "
                       "6 white flamingos.", "equations": ["1/3 \\times 18 = 6", "18 - 6 = 12", "12 + 6 = 18"],
     "type": "computational"}]
SAMPLES = [["On Saturday 6 flamingos are painted white.", "So 12 pink flamingos remain."], [],
           ["There are 18 + 18 = 36 pink flamingos.", "The answer is 24."]]


def fake_fits(premise, hypothesis):
    return len(premise) + len(hypothesis) < 300


def fake_score(pairs):
    """Deterministic stand-in for NLI.score: depends only on the premise and hypothesis text."""
    out = []
    for p, h in pairs:
        v = zlib.crc32(f"{p}||{h}".encode()) % 1000 / 1000
        out.append((round(v, 3), round((1 - v) / 2, 3), round((1 - v) / 2, 3)))
    return out


# ------------------------------------------------------------------ claim extraction
def test_flamingo_step_yields_the_wrong_claim():
    claims = extract_claims(FLAMINGO_PROBLEM, FLAMINGO_STEPS)
    texts = [c["text"] for c in claims if c["step"] == 1]
    assert any("12 + 6 = 18 pink" in t for t in texts), texts
    assert any("1/3 * 18 = 6" in t for t in texts), texts                 # LaTeX converted to plain text
    assert all(set(c) == {"step", "claim_idx", "text", "latex", "type"} for c in claims)
    assert [c["claim_idx"] for c in claims if c["step"] == 1] == list(range(len(texts)))


def test_planning_sentences_and_questions_are_not_claims():
    steps = [{"idx": 0, "text": "Let's solve this step by step. What is the total? The total is 7 + 5 = 12 apples.",
              "equations": [], "type": "computational"}]
    texts = [c["text"] for c in extract_claims("Tom has 7 apples and buys 5.", steps)]
    assert texts == ["The total is 7 + 5 = 12 apples."], texts


def test_so_splits_into_two_claims():
    steps = [{"idx": 0, "text": "The price is 3 times 4 dollars, so the cost is 12 dollars.", "equations": [], "type": "reasoning"}]
    texts = [c["text"] for c in extract_claims("", steps)]
    assert texts == ["The price is 3 times 4 dollars", "the cost is 12 dollars."], texts


def test_latex_to_plain():
    assert latex_to_plain(r"\frac{1}{2} \times 18 = 9") == "1/2 * 18 = 9"
    assert latex_to_plain(r"\left( \frac{a+b}{2} \right)^{2} \geq ab") == "((a+b)/(2))^2 >= ab"
    assert latex_to_plain(r"\sqrt{x_{n+1}} \neq 45^\circ") == "sqrt(x_(n+1)) != 45 degrees"


# ------------------------------------------------------------------ numeric spot check
def test_true_identity_passes():
    for claim in [r"So (x+1)^2 = x^2 + 2x + 1.", r"\frac{x^2-1}{x-1} = x + 1", r"\binom{n}{2} = \frac{n(n-1)}{2}"]:
        res = numeric_check(claim)
        assert res and claim_status(res) == "pass", (claim, res)


def test_false_identity_fails():
    for claim in [r"We simplify: 2(x+3) = 2x + 3", r"f(x) = (x+1)^2 = x^2 + 2x + 2",
                  r"5x + 9 = (x+3)^2 = x^2 + 6x + 10 = x^2 + 5x"]:
        res = numeric_check(claim)
        assert claim_status(res) == "fail", (claim, res)


def test_conditions_are_skipped_not_flagged():
    for claim in [r"If x + 1 = 3 then x = 2", r"x^2 - 5x + 6 = 0", r"2x + 3 = x + 7", r"y = 3x + 2",
                  r"5x + 9 = (x+3)^2 = x^2 + 6x + 9", r"Suppose a + b = 2ab",
                  # false flags in the first version (results/rag_numeric_failures_v1.csv):
                  r"2^{3+x} = 2^{61}", r"f(f(2)) = f(4) = 2", r"OP_1 = OP_2 = 1",
                  r"we subtract 156.6 from both sides: 6 - 156.6 = 0.3n -150.6 = 0.3n",
                  r"Simplifying, we get x_1^2 + x_2^2 + bx_1 + bx_2 = x_3^2 + bx_3."]:
        res = numeric_check(claim)
        assert claim_status(res) == "skip", (claim, res)


def test_numbers_only_are_left_to_stage4():
    assert numeric_check("12 + 6 = 18 pink flamingos") == []


def test_text_notes_are_not_variables():
    """Bug fixed in the pilot: '9 (red) + 14 (blue)' was read as 9*r*e*d + 14*b*l*u*e, i.e. a symbolic equation."""
    for claim in ["Total number of sticks = 9 (red) + 14 (blue) + 11 (yellow) = 34.",
                  "Total = 5 (apples) + 3 (pears) = 9 (fruit)"]:
        assert numeric_check(claim) == [], (claim, numeric_check(claim))     # numbers only: Stage 4's job
    from stage6_evidence import claim_equations
    assert all(not re.search(r"red|blue|yellow", e) for e in claim_equations("Total = 9 (red) + 14 (blue) = 23"))


def test_size_guard_skips_with_a_reason():
    long_eq = " + ".join(f"x^{k}" for k in range(1, 40)) + " = y"
    res = numeric_check(long_eq)
    assert res and all(s == "skip" and w.startswith("guard: equation longer") for _, s, w in res), res
    res = numeric_check(r"(x+1)^{200} = x^{200} + 1")
    assert res and res[0][1] == "skip" and res[0][2] == "guard: exponent of 100 or more", res


def test_numeric_check_is_deterministic():
    assert numeric_check(r"(a+b)^3 = a^3 + 3a^2b + 3ab^2 + b^3") == numeric_check(r"(a+b)^3 = a^3 + 3a^2b + 3ab^2 + b^3")


# ------------------------------------------------------------------ NLI premises
def test_premise_rule():
    sents = ["Tom has 3 boxes.", "Each box has 4 apples.", "The sky is blue.", "Birds sing.", "Apples are red."]
    assert premise_a(sents, "Tom has 12 apples.", lambda p, h: True) == " ".join(sents)
    p = premise_a(sents, "Each box has 4 apples.", lambda p, h: False)        # too long: top-3 BM25 + full text
    head, _, tail = p.partition(" ... ")
    assert tail == " ".join(sents) and "Each box has 4 apples." in head and head.count(".") == 3


def test_a2_drops_oldest_steps_first_and_keeps_the_problem():
    prob = ["Tom has 3 boxes."]
    earlier = [["Step zero text is long."], ["Step one text."], ["Step two, the nearest."]]
    limit = lambda n: (lambda p, h: len(p) <= n)
    full = "Tom has 3 boxes. Step zero text is long. Step one text. Step two, the nearest."
    assert premise_a2(prob, earlier, "claim", limit(999), return_mode=True) == (full, "full")
    p, mode = premise_a2(prob, earlier, "claim", limit(60), return_mode=True)
    assert mode == "dropped_oldest" and p == "Tom has 3 boxes. Step one text. Step two, the nearest."
    p, mode = premise_a2(prob, earlier, "claim", limit(20), return_mode=True)
    assert mode == "cut" and p.startswith("Tom has 3 boxes.") and p.endswith("Step two, the nearest.")
    assert premise_a2(prob, [], "claim", limit(999)) == "Tom has 3 boxes."    # step 0: same premise as A


# ------------------------------------------------------------------ leakage
FORBIDDEN = {"label", "final_answer_correct"}


def test_claim_and_evidence_code_never_touch_target_fields():
    """No identifier, attribute, keyword or string key named label / final_answer_correct in the signal code."""
    for name in ("stage5_claims.py", "stage6_evidence.py"):
        tree = ast.parse(open(os.path.join(HERE, "..", "src", name), encoding="utf-8").read())
        for node in ast.walk(tree):
            hit = ((isinstance(node, ast.Constant) and node.value in FORBIDDEN)
                   or (isinstance(node, ast.Name) and node.id in FORBIDDEN)
                   or (isinstance(node, ast.Attribute) and node.attr in FORBIDDEN)
                   or (isinstance(node, ast.arg) and node.arg in FORBIDDEN)
                   or (isinstance(node, ast.keyword) and node.arg in FORBIDDEN))
            assert not hit, f"{name} line {node.lineno} uses a target field"


def test_claims_nli_and_numeric_outputs_ignore_label():
    outs = []
    for label, fac in [(-1, True), (1, False), (1, True), (0, None)]:
        rec = {"id": "gsm8k-0", "problem": FLAMINGO_PROBLEM, "steps": FLAMINGO_STEPS, "label": label,
               "final_answer_correct": fac}
        claims = extract_claims(rec["problem"], rec["steps"])
        nli = {}
        for src in ("A", "A2", "B"):
            pairs = pairs_for(src, rec["problem"], rec["steps"], claims, fake_fits, SAMPLES)
            probs = fake_score([(p, h) for _, _, p, h in pairs])
            nli[src] = list(zip([(k, j) for k, j, _, _ in pairs], probs))
        b = aggregate_b([(j, p) for (k, j), p in nli["B"] if k == "1.0"])
        num = [numeric_check(c["latex"]) for c in claims]
        outs.append((claims, nli, b, num))
    assert all(o == outs[0] for o in outs)
    assert outs[0][2]["n_samples"] == 2                                      # the empty sample is skipped


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
