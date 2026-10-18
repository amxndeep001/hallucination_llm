"""Signals must not use the human label or final_answer_correct.  Run:  python tests/test_no_leakage.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from stage2_check import sample_answer
from stage3_align import align, solution_features
from stage7_consistency import features
from evaluate_all import FEATURE_SETS, TRANSFER_SETS, check_no_leakage

SAMPLES = {"id": "t-0", "samples": [
    {"text": "3 \\times 4 = 12, so \\boxed{12}", "final_answer": "12", "mean_logprob": -0.1, "mean_entropy": 0.2, "n_tokens": 50},
    {"text": "12 apples, \\boxed{12}", "final_answer": "12", "mean_logprob": -0.2, "mean_entropy": 0.3, "n_tokens": 60},
    {"text": "maybe 13 ... and then 7", "final_answer": "7", "mean_logprob": -0.5, "mean_entropy": 0.9, "n_tokens": 100}]}
S4 = {"sympy": {"n_fail": 1, "n_checked": 2, "first_fail_step": 1, "n_equations": 2}}
STEPS = [{"idx": 0, "text": "He has 3 \\times 4 = 12 apples.", "equations": ["3 \\times 4 = 12"], "type": "computational"},
         {"idx": 1, "text": "The answer is \\boxed{13}.", "equations": [], "type": "reasoning"}]


def test_feature_sets_have_no_target_fields():
    check_no_leakage(FEATURE_SETS + TRANSFER_SETS)
    for bad in ("label", "final_answer_correct", "original_answer"):
        try:
            check_no_leakage([("bad", ["agree_with_original", bad])])
        except AssertionError:
            continue
        raise AssertionError(f"{bad} was not caught")


def test_stage7_features_ignore_label_and_final_answer_correct():
    outs = []
    for label, fac in [(-1, True), (0, False), (1, True), (5, None)]:
        raw = {"id": "t-0", "label": label, "final_answer_correct": fac, "steps": [s["text"] for s in STEPS]}
        f = features(SAMPLES, S4, raw, cap=100)
        assert f["label"] == label and f["final_answer_correct"] == fac           # copied as targets
        outs.append({k: v for k, v in f.items() if k not in ("label", "final_answer_correct")})
    assert all(o == outs[0] for o in outs)


def test_stage3_features_ignore_label():
    outs = []
    for label, fac in [(-1, True), (0, False), (1, None)]:
        pb = {"id": "t-0", "problem": "Tom has 3 boxes with 4 apples each.", "label": label,
              "final_answer_correct": fac, "steps": STEPS}
        steps, _ = align(pb, SAMPLES["samples"])
        outs.append((steps, solution_features(steps)))
    assert all(o == outs[0] for o in outs)


def test_truncated_without_box_is_no_answer():
    cut = {"text": "so far we have 13 and", "final_answer": "13", "n_tokens": 1024}
    boxed = {"text": "\\boxed{7} and then more text", "final_answer": "7", "n_tokens": 1024}
    short = {"text": "the answer is 13", "final_answer": "13", "n_tokens": 300}
    assert sample_answer(cut) == "13"                                  # default: last-number fallback (old behaviour)
    assert sample_answer(cut, cap=1024) == ""                           # option: hit the cap, no box -> no answer
    assert sample_answer(boxed, cap=1024) == "7"                        # a boxed answer before the cap is kept
    assert sample_answer(short, cap=1024) == "13"                       # not truncated: unchanged


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
