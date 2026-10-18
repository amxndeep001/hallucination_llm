"""Unit tests for normalize_answer.  Run from the project root:  python tests/test_normalize.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from stage2_generate import normalize_answer as norm
from stage2_check import original_answer


def test_currency():
    assert norm("\\$288.00") == "288"
    assert norm("\\$65,760") == "65760"
    assert norm("\\$70.00") == "70"
    assert norm("\\$256.70") == "256.7"
    assert norm("\\$ 40") == "40"
    assert norm("$18$") == "18"
    assert norm("\\$288.00") == norm("288")                    # 288.00 = 288


def test_stray_leading_backslash():
    assert norm("\\15") == "15"


def test_thousands_commas():
    assert norm("37,800") == "37800"
    assert norm("\\$31,800") == "31800"


def test_unit_words():
    assert norm("53 minutes") == "53"
    assert norm("5 hours") == "5"
    assert norm("84 pounds") == "84"
    assert norm("12 \\text{ hours}") == "12"
    assert norm("20.16 Mb/h") == "20.16"


def test_mixed_number():
    assert norm("1 \\text{ and } 1/3") == "4/3"
    assert norm("2 and 1/2") == "5/2"
    assert norm("1 and 4/3") == "1and4/3"                      # improper fraction: ambiguous, left alone


def test_text_answers_left_alone():
    assert norm("4 blue shoe boxes and 6 red shoe boxes") == "4blueshoeboxesand6redshoeboxes"


def test_competition_answers_unchanged():
    assert norm("\\frac{1}{2}") == "\\frac{1}{2}"
    assert norm("2\\sqrt{3}") == "2\\sqrt{3}"
    assert norm("(-\\infty, 2]") == "(-\\infty2]"               # commas were always dropped (old behaviour)
    assert norm("\\left( 3, \\ 6\\right)") == "\\left(3\\6\\right)"
    assert norm("\\begin{pmatrix} 6 \\\\ 3 \\\\ 0 \\end{pmatrix}") == "\\begin{pmatrix}6\\\\3\\\\0\\end{pmatrix}"
    assert norm("\\text{Devon}") == "Devon"
    assert norm("2x") == "2x"                                  # glued letter is algebra, not a unit
    assert norm("3a + 21") == "3a+21"


def test_original_answer_takes_value_after_last_equals():
    orig = lambda boxed: original_answer({"steps": [f"So the answer is \\boxed{{{boxed}}}."]})
    assert orig("150 \\div 3 = 50") == "50"
    assert orig("3 + 4 = 7.00") == "7"
    assert orig("x = \\frac{1}{2}") == "x=\\frac{1}{2}"            # not a plain number after '=': unchanged
    assert orig("\\$288.00") == "288"
    assert orig("a = 2.08, b = 0.28") == "a=2.08b=0.28"              # a list of values: unchanged


def test_whitespace_and_newlines():
    assert norm("\\begin{pmatrix}\n1 & 2\\\\\n3 & 4\n\\end{pmatrix}") == norm("\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}")


def test_huge_integer_does_not_crash():
    assert norm("7" * 400) == "7" * 400                         # float() gives inf: left as a string


def test_original_answer_unboxed_latex():
    orig = lambda last: original_answer({"steps": [last]})
    assert orig("Final Answer: The final answer is $16\\pi$. I hope it is correct.") == "16\\pi"
    assert orig("Final Answer: The final answer is \\(\\frac{4}{35}\\). I hope it is correct.") == "\\frac{4}{35}"
    assert orig("so the value is \\(-\\frac{2\\sqrt{3}}{3}\\).") == "-\\frac{2\\sqrt{3}}{3}"
    assert orig("The probability is \\(\\frac{11}{36}\\).") == "\\frac{11}{36}"
    # a closing span that is not a pure number expression keeps the old last-number fallback
    assert orig("so the answer is \\( x = 9 \\).") == "9"
    assert orig("Then \\(f(50) = 6\\)") == "6"
    assert orig("The answer is 7, as \\(n\\) is odd.") == "7"
    # boxed answers are untouched
    assert orig("\\(\\frac{1}{2}\\) so \\boxed{3}") == "3"


def test_plain_numbers():
    assert norm("18") == norm("18.0") == norm("\\text{18}") == "18"
    assert norm("0.50") == "0.5"
    assert norm("") == ""


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
