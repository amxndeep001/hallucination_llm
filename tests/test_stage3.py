"""Unit tests for the Stage 3 number alignment.  Run from the project root:  python tests/test_stage3.py"""
import os, sys
from fractions import Fraction as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from stage3_align import align, equation_result, extract_numbers as nums, key_numbers, solution_features


def test_thousands_commas_and_currency():
    assert nums("1,250") == {F(1250)}
    assert nums("$5,125") == nums("\\$5,125") == {F(5125)}
    assert nums("costs \\$12.50 each") == {F(25, 2)}
    assert nums("1, 2, 3") == {F(1), F(2), F(3)}                     # a list, not thousands


def test_percent_gives_both_candidates():
    assert nums("25%") == nums("25\\%") == nums("25 percent") == {F(25), F(1, 4)}


def test_fractions_and_mixed_numbers():
    assert F(3, 4) in nums("3/4 of the cake") and F(3, 4) in nums("\\frac{3}{4}")
    assert nums("\\frac{18}{3} = 6") == {F(6), F(18), F(3)}          # value and the literals inside
    assert F(4, 3) in nums("1 1/3 cups")
    assert nums("12 / 2 + 1.5") == {F(6), F(12), F(2), F(3, 2)}      # 12/2: value 6 and the literals 12, 2


def test_list_numbering_and_step_labels_ignored():
    text = "1. **Initial:** 18 flamingos\n2. Then 6 more.\n### 3) Step 4: total 24."
    assert nums(text) == {F(18), F(6), F(24)}
    assert nums("The answer is 30.") == {F(30)}
    assert nums("3 \\times18") == {F(3), F(18)}


def test_equation_results():
    assert equation_result("3 \\times 4 = 12")[0] == 12
    assert equation_result("2 x 75 = 150 medium-sized")[0] == 150
    assert equation_result("0.7 \\times 100 = 70%")[1] == {F(70), F(7, 10)}
    assert equation_result("(4x - 13) + 5 = 4x - 8") is None           # expression, not a result
    assert equation_result("5000 = x/2") is None
    assert equation_result("x * 1.5 = 300 * (3/2)") is None


def test_problem_number_is_never_a_key_number():
    pn = nums("Tom has 3 boxes with 4 apples each. He eats 5 apples.")
    keys, _, _ = key_numbers({"text": "He eats 5 apples, so 1 \\times 5 = 5.", "equations": ["1 \\times 5 = 5"]}, pn)
    assert keys == []
    keys, src, _ = key_numbers({"text": "He has 12 apples and 2 bags.", "equations": []}, pn)
    assert [k for k, _ in keys] == ["12"] and src == "derived"         # 2 is a trivial constant
    keys, src, _ = key_numbers({"text": "4 - 2 = 2", "equations": ["4 - 2 = 2"]}, pn)
    assert [k for k, _ in keys] == ["2"] and src == "rhs"              # a trivial constant as an equation result counts


PROBLEM = {"problem": "Tom has 3 boxes with 4 apples each. He eats 5 apples. How many apples are left?"}
GOOD_SAMPLE = {"text": "1. Apples: 3 \\times 4 = 12.\n2. Left: 12 - 5 = 7.\nThe answer is \\boxed{7}."}


def toy(step2):
    steps = [{"idx": 0, "text": "Tom has 3 boxes with 4 apples.", "equations": [], "type": "reasoning"},
             {"idx": 1, "text": "He has 3 \\times 4 = 12 apples.", "equations": ["3 \\times 4 = 12"], "type": "computational"},
             {"idx": 2, "text": f"He eats 5, so 12 - 5 = {step2} are left.", "equations": [f"12 - 5 = {step2}"], "type": "computational"},
             {"idx": 3, "text": f"The answer is {step2}.", "equations": [], "type": "reasoning"}]
    steps, _ = align({**PROBLEM, "steps": steps}, [GOOD_SAMPLE] * 5)
    return steps, solution_features(steps)


def test_wrong_step_is_not_supported():
    steps, f = toy(8)                                                  # 12 - 5 = 8 is wrong; no sample has 8
    assert steps[0]["support_min"] is None and steps[0]["n_key"] == 0  # only problem numbers: no signal
    assert steps[1]["support_min"] == 1.0 and steps[1]["key_numbers"] == ["12"]
    assert steps[2]["support_min"] == 0.0 and steps[2]["key_numbers"] == ["8"]
    assert f["s3_first_low_step"] == 2 and f["s3_n_low_steps"] == 2 and f["s3_min_support"] == 0.0
    assert f["s3_n_steps_with_signal"] == 3 and f["s3_frac_steps_with_signal"] == 0.75


def test_all_steps_reproduced():
    steps, f = toy(7)
    assert [s["support_min"] for s in steps] == [None, 1.0, 1.0, 1.0]
    assert f["s3_first_low_step"] == -1 and f["s3_min_support"] == 1.0 and f["s3_n_low_steps"] == 0


def test_partial_support():
    steps = [{"idx": 0, "text": "3 \\times 4 = 12 and 12 - 5 = 7", "equations": ["3 \\times 4 = 12", "12 - 5 = 7"], "type": "computational"}]
    samples = [GOOD_SAMPLE, GOOD_SAMPLE, {"text": "3 \\times 4 = 12, answer 12"}, {"text": "no numbers"}, {"text": "7"}]
    s, _ = align({**PROBLEM, "steps": steps}, samples)
    assert s[0]["support_min"] == 0.6 and abs(s[0]["support_mean"] - 0.6) < 1e-12   # 12: 3/5, 7: 3/5


def test_no_signal_solution():
    steps = [{"idx": 0, "text": "He has 3 boxes.", "equations": [], "type": "reasoning"}]
    s, _ = align({**PROBLEM, "steps": steps}, [GOOD_SAMPLE] * 5)
    f = solution_features(s)
    assert f["s3_min_support"] == 1.0 and f["s3_first_low_step"] == -1 and f["s3_n_steps_with_signal"] == 0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
