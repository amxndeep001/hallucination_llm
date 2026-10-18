"""Unit tests for the Stage 7 answer clustering.  Run from the project root:  python tests/test_stage7.py"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from stage2_generate import normalize_answer
from stage7_consistency import cluster, equivalent, semantic_entropy


def sizes(answers):
    return cluster([normalize_answer(a) for a in answers])


def test_equivalent_forms_form_one_cluster():
    assert sizes(["0.5", "1/2", "\\frac{1}{2}", "0.50", "2"]) == [4, 1]


def test_identical_answers_have_zero_entropy():
    s = sizes(["18", "18", "18", "18", "18"])
    assert s == [5]
    assert semantic_entropy(s) == 0.0


def test_five_different_answers_have_entropy_ln5():
    s = sizes(["1", "2", "3", "4", "5"])
    assert s == [1, 1, 1, 1, 1]
    assert math.isclose(semantic_entropy(s), math.log(5))


def test_unparsable_answer_does_not_crash():
    s = sizes(["\\mathbb{R}", "\\frac{1}{", "\\mathbb{R}", "3"])
    assert s == [2, 1, 1]
    assert not equivalent("\\frac{1}{", "1")


def test_exact_equality_only():
    assert equivalent(normalize_answer("12.50"), normalize_answer("12.5"))
    assert equivalent(normalize_answer("2/4"), normalize_answer("1/2"))
    assert equivalent(normalize_answer("10\\%"), normalize_answer("0.1"))       # 10% is exactly 1/10
    assert not equivalent(normalize_answer("10\\%"), normalize_answer("10"))
    assert not equivalent(normalize_answer("0.333"), normalize_answer("1/3"))  # no rounding tolerance


def test_text_answers_match_by_string_only():
    assert not equivalent(normalize_answer("6 \\text{ red, } 4 \\text{ blue}"),
                          normalize_answer("4 \\text{ blue, } 6 \\text{ red}"))
    assert equivalent(normalize_answer("\\frac{5}{8}A + 45"), normalize_answer("\\frac{5A + 360}{8}"))
    assert equivalent(normalize_answer("2\\sqrt{2}"), normalize_answer("\\sqrt{8}"))


def test_empty_answers_are_not_clustered():
    assert sizes(["", "", "7"]) == [1]
    assert sizes(["", ""]) == []
    assert semantic_entropy([]) == 0.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
