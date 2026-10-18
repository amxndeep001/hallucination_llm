"""Run Stages 2b, 3 and 7 for one split on its sampled problems, with the existing CLIs.

  python src/run_split.py math            # or gsm8k, olympiadbench, omnimath

1. Checks data/processed/samples_<split>.jsonl before anything else and stops if it is not a real Stage 2 run:
   no duplicate ids, every id in pb_<split>, exactly 5 samples per problem, one model
   (Qwen/Qwen2.5-Math-1.5B-Instruct). The problem count is printed, not enforced (GSM8K 400, others 300).
2. stage2_check.py, with and without --truncated-no-answer
3. stage3_align.py --sampled-only -> s3_<split>.jsonl;  stage3_evaluate.py -> results/stage3_summary_<split>.csv
4. stage7_consistency.py -> s7_<split>.jsonl, and with --truncated-no-answer -> s7_<split>_truncnone.jsonl
5. stage7_evaluate.py s7_<split>.jsonl --s3 s3_<split>.jsonl -> results/stage7_ablation_<split>[_with_s3].csv
Stage 4 (s4_<split>.jsonl) does not depend on the samples and is not re-run; its records are matched by id.
Each step's output is also saved to results/logs/<split>_<step>.txt. Then run src/evaluate_all.py.
"""
import argparse, json, os, subprocess, sys
from collections import Counter

MODEL, N_SAMPLES = "Qwen/Qwen2.5-Math-1.5B-Instruct", 5
CAP = {"gsm8k": 768}                                        # Stage 2 max_new_tokens; the Colab splits used 1024
P = "data/processed"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_samples(split):
    """Stop with a message if samples_<split>.jsonl is not a real Stage 2 run."""
    recs = read_jsonl(f"{P}/samples_{split}.jsonl")
    pb = {r["id"] for r in read_jsonl(f"{P}/pb_{split}.jsonl")}
    ids = [r["id"] for r in recs]
    errors = []
    if len(set(ids)) != len(ids):
        errors.append(f"{len(ids) - len(set(ids))} duplicate ids")
    if set(ids) - pb:
        errors.append(f"{len(set(ids) - pb)} ids not in pb_{split}.jsonl")
    n = Counter(len(r["samples"]) for r in recs)
    if set(n) != {N_SAMPLES}:
        errors.append(f"samples per problem {dict(n)}, expected {N_SAMPLES}")
    models = Counter(r.get("model") for r in recs)
    if set(models) != {MODEL}:
        errors.append(f"models {dict(models)}, expected {MODEL}")
    if errors:
        sys.exit(f"ERROR: samples_{split}.jsonl is not a valid Stage 2 run: " + "; ".join(errors))
    print(f"samples_{split}.jsonl: {len(recs)} problems x {N_SAMPLES} samples, {MODEL}: ok")


def run(split, step, args):
    os.makedirs("results/logs", exist_ok=True)
    print(f"\n######## {step}: python {' '.join(args)}", flush=True)
    out = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding="utf-8")
    with open(f"results/logs/{split}_{step}.txt", "w", encoding="utf-8") as f:
        f.write(out.stdout + out.stderr)
    print(out.stdout[-3000:] + out.stderr, flush=True)
    if out.returncode:
        sys.exit(f"ERROR: {step} failed (exit {out.returncode})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["gsm8k", "math", "olympiadbench", "omnimath"])
    a = ap.parse_args()
    sp, cap = a.split, str(CAP.get(a.split, 1024))
    samples, raw, pb = f"{P}/samples_{sp}.jsonl", f"data/raw/processbench_{sp}.jsonl", f"{P}/pb_{sp}.jsonl"
    s3, s4, s7 = f"{P}/s3_{sp}.jsonl", f"{P}/s4_{sp}.jsonl", f"{P}/s7_{sp}.jsonl"

    check_samples(sp)
    run(sp, "stage2_check", ["src/stage2_check.py", samples, raw])
    run(sp, "stage2_check_truncnone", ["src/stage2_check.py", samples, raw, "--truncated-no-answer", "--max-new-tokens", cap])
    run(sp, "stage3_align", ["src/stage3_align.py", pb, samples, s3, "--sampled-only"])
    run(sp, "stage3_evaluate", ["src/stage3_evaluate.py", s3, s4, pb, raw])
    run(sp, "stage7_consistency", ["src/stage7_consistency.py", samples, s4, raw, s7, "--max-new-tokens", cap])
    run(sp, "stage7_consistency_truncnone", ["src/stage7_consistency.py", samples, s4, raw,
                                             f"{P}/s7_{sp}_truncnone.jsonl", "--max-new-tokens", cap, "--truncated-no-answer"])
    run(sp, "stage7_evaluate", ["src/stage7_evaluate.py", s7, "--s3", s3])
