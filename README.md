# Math Hallucination Detection (A1 + A3 + A2, RAG + SymPy)

Find the first wrong step in an LLM's step-by-step math solution by combining
symbolic checking (SymPy), retrieval (RAG) and agreement across sampled solutions.
Evaluated on [ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench), whose `label`
is the index of the first wrong step (`-1` = all steps correct).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # pinned versions, Python 3.9

# external reference repos (not tracked in this repo)
git clone https://github.com/DIYANAPV/SelfCheckAgent data/external/SelfCheckAgent
git clone https://github.com/openai/prm800k data/external/prm800k
```

On Colab (Stage 2), do not install `requirements.txt`: it pins torch 2.8.0, which breaks Colab's
preinstalled CUDA libraries and torchvision. Stage 2 needs only torch (Colab's own), transformers and tqdm:

```bash
pip install -q transformers==4.57.6 tqdm
```

## Pipeline

| Stage | Script | Output | Status |
|---|---|---|---|
| 0 Load data | `src/stage0_load_data.py` | `data/raw/*.jsonl` | done |
| 1 Preprocess (steps, equations, step types) | `src/stage1_preprocess.py` | `data/processed/pb_*.jsonl` | done |
| 2 Sample N solutions + check | `src/stage2_generate.py`, `src/stage2_check.py`, `colab/run_stage2.ipynb` | `data/processed/samples_*.jsonl` | GSM8K done; MATH, OlympiadBench, Omni-MATH ready to run on Colab |
| 3 Number-based step alignment + step-level evaluation | `src/stage3_align.py`, `src/stage3_evaluate.py`, `tests/test_stage3.py` | `data/processed/s3_*.jsonl`, `results/stage3_summary_*.csv` | GSM8K done; other splits run unchanged once their Stage 2 samples exist |
| 4 SymPy verification + audit | `src/stage4_sympy_verify.py`, `src/stage4_audit.py`, `src/stage4_precision.py` | `data/processed/s4_*.jsonl`, `results/stage4_*.csv` | done |
| 7 Answer consistency, semantic entropy, first combined classifier | `src/stage7_consistency.py`, `src/stage7_evaluate.py`, `tests/test_stage7.py` | `data/processed/s7_*.jsonl`, `results/stage7_ablation_*.csv` | GSM8K done; other splits run unchanged once their Stage 2 samples exist |

## Run order

Run every command from the project root. `<split>` is one of `gsm8k`, `math`, `olympiadbench`, `omnimath`.

```bash
# Stage 0: download ProcessBench + GSM8K (needs internet / Hugging Face)
python src/stage0_load_data.py

# Stage 1: split solutions into steps and extract equations
python src/stage1_preprocess.py data/raw/processbench_<split>.jsonl data/processed/pb_<split>.jsonl

# Stage 2a: sample 5 solutions per problem (GPU; resumable: finished ids are skipped)
#   GSM8K (done):
python src/stage2_generate.py --input data/processed/pb_gsm8k.jsonl \
    --output data/processed/samples_gsm8k.jsonl \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct --n 5 --batch-size 2
#   MATH / OlympiadBench / Omni-MATH: run colab/run_stage2.ipynb (installs only transformers + tqdm,
#   keeps Colab's torch; see Setup), which does for each split
#   (raw input = the original problem text with its $...$; pb_* strips it. Same ids and order.)
python src/stage2_generate.py --input data/raw/processbench_<split>.jsonl \
    --output data/processed/samples_<split>.jsonl --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --sample 300 --seed 0 --max-new-tokens 1024 --n 5 --temperature 0.7 --top-p 0.95 --batch-size 2
#   smoke test without a model: add --backend mock --limit 5
# Stage 2b: check the samples and the agreement signal
python src/stage2_check.py data/processed/samples_<split>.jsonl data/raw/processbench_<split>.jsonl

# Stage 3a: support of each original step = share of samples containing its key numbers
python src/stage3_align.py data/processed/pb_<split>.jsonl data/processed/samples_<split>.jsonl \
    data/processed/s3_<split>.jsonl
# Stage 3b: first unsupported step vs the human first wrong step; Stage 4 and the union for comparison
python src/stage3_evaluate.py data/processed/s3_<split>.jsonl data/processed/s4_<split>.jsonl \
    data/processed/pb_<split>.jsonl data/raw/processbench_<split>.jsonl   # -> results/stage3_summary_<split>.csv
python tests/test_stage3.py

# Stage 4a: verify every extracted equation with SymPy, print the evaluation, save the summary row
python src/stage4_sympy_verify.py data/processed/pb_<split>.jsonl data/processed/s4_<split>.jsonl \
    --summary results/stage4_summary.csv
# Stage 4b: CSV of cases to review by hand (early failures, sampled false alarms, sampled skips)
python src/stage4_audit.py data/processed/s4_<split>.jsonl results/stage4_audit_<split>.csv
# Stage 4c: fresh sample of 30 flagged solutions per split (seed 123, not yet audited), all splits in one CSV;
#           fill in my_verdict (real_error / false_alarm / extraction_bug), then compute precision
python src/stage4_audit.py --fresh-sample results/stage4_fresh_audit.csv
python src/stage4_precision.py results/stage4_fresh_audit.csv

# Stage 7a: consistency features per problem (needs samples_<split>, s4_<split>, the raw file)
#   --max-new-tokens = the Stage 2 cap (768 for GSM8K, 1024 for the Colab splits); without it the
#   largest n_tokens in the samples file is used
python src/stage7_consistency.py data/processed/samples_<split>.jsonl data/processed/s4_<split>.jsonl     data/raw/processbench_<split>.jsonl data/processed/s7_<split>.jsonl --max-new-tokens 768
# Stage 7b: single-feature AUROCs, logistic regression + random forest (5-fold x 10 repeats), ablation
python src/stage7_evaluate.py data/processed/s7_<split>.jsonl      # -> results/stage7_ablation_<split>.csv
#   with the Stage 3 features added (the file above is unchanged)  -> results/stage7_ablation_<split>_with_s3.csv
python src/stage7_evaluate.py data/processed/s7_<split>.jsonl --s3 data/processed/s3_<split>.jsonl
# tests for the answer clustering and normalization
python tests/test_stage7.py
python tests/test_normalize.py
```

## Stage 3 in short

Stage 7 says *whether* a solution disagrees with the samples; Stage 3 tries to say *at which step*.
A step of the original solution is **supported** if the new quantities it computes also appear in the
5 independently sampled solutions. The alignment is by numbers, not by text: every number is read as
an exact fraction (thousands commas, `$` and `\$`, decimals, `3/4`, `\frac{3}{4}`, mixed numbers like
`1 1/3`; `25%` matches 25 or 1/4; list numbering and "Step 3" labels are ignored).

- **key numbers** of a step: the result of each of its equations (Stage 1), or, if it has none, the
  numbers in its text that are not in the problem and not trivial constants (0, 1, 2, 10, 100).
  A number from the problem statement is never a key number.
- **support** of a key number: share of samples containing it; per step the mean and the minimum.
  A step without key numbers has no signal (`None`), not support 0.
- solution features: `s3_first_low_step` (first step with support_min ≤ 0.2, else −1), `s3_min_support`,
  `s3_n_low_steps`, `s3_mean_support`, and how many steps have a signal.

`stage3_evaluate.py` compares the first low-support step with the human first wrong step (as Stage 4's
evaluation does), at thresholds 0.0, 0.2 and 0.4, against Stage 4 and the union of both. A correct step
can still get low support when the samples reach the answer by another route (for example one
combined sum instead of two partial ones), so a flag is a hint to look, not proof of an error.

## Stage 4 in short

Each equation from Stage 1 is converted from LaTeX, parsed by SymPy and marked:

- **pass**: both sides are equal (rounding to the stated decimals, percent notation and integer quotients are allowed)
- **fail**: both sides are plain numbers and differ, which means a calculation error
- **skip**: cannot be judged; the reason says why (variables, inequality, unsupported LaTeX, unbalanced brackets …). Skip is not pass.

A step fails if any of its equations fails. The first failing step is the predicted first error.

## Stage 7 in short

For each problem, the 5 sampled final answers are clustered by mathematical equivalence: the same
`normalize_answer` string, or SymPy (via the Stage 4 parser) finds the difference exactly 0, so
`0.5`, `1/2`, `\frac{1}{2}` and `0.50` are one cluster. Answers SymPy cannot parse, or that contain
words, match by string only. Answers are first normalized by `normalize_answer`, which also strips
currency (`\$288.00` -> 288), a stray leading backslash, thousands commas, unit words separated
from the number (`53 minutes` -> 53) and proper mixed numbers (`1 and 1/3` -> 4/3); an original answer
`expression = number` becomes the number. For **agree_with_original** only, `X%` also matches `X`
and `X/100` (the clustering keeps `10% != 10`). From the clusters and the samples:

- **majority_share**, **n_clusters**, **semantic_entropy** (−Σ p ln p over cluster shares): do the samples agree with each other?
- **agree_with_original**: share of samples equivalent to the original ProcessBench solution's answer
- **no_answer_rate**, **truncated_rate**, token log-probability and entropy summaries, mean length
- the Stage 4 SymPy counts (n_fail, n_checked, first_fail_step, sympy_flag, n_equations)

`stage7_evaluate.py` scores each feature alone (AUROC), then trains logistic regression and a small
random forest with cross-validation, for two targets: *process error* (`label != -1`) and *wrong
final answer*. It also scores the hard subset: a correct final answer but a wrong step. The ablation
shows what each group of features adds. Human labels are used only as targets, never as features.

## Results so far

Stage 2 (GSM8K, 400 problems × 5 samples): low agreement between the samples and the original
answer detects a process error with **AUROC 0.946** (0.916 before the answer normalization fix).

Stage 4, against the ProcessBench labels (`results/stage4_summary.csv`; byte-identical across repeated runs and across Python 3.9 and 3.13 with the pinned SymPy 1.14.0):

| Split | Erroneous flagged | Exact first-step match | Ceiling | False alarms on correct | Precision |
|---|---|---|---|---|---|
| GSM8K (400) | 8.2% | 7.2% | 61.8% | 0.0% | 100.0% |
| MATH (1,000) | 16.3% | 7.9% | 56.6% | 0.5% | 98.0% |
| OlympiadBench (1,000) | 10.6% | 2.9% | 52.0% | 1.8% | 92.1% |
| Omni-MATH (1,000) | 11.1% | 4.9% | 44.4% | 2.1% | 94.4% |

*Ceiling* = share of erroneous solutions whose first wrong step contains an equation at all.
SymPy catches arithmetic slips with high precision but not reasoning errors, so it is one
signal to combine with the others, not a detector on its own.

## Layout

```
data/raw/          downloaded datasets (Stage 0; do not edit)
data/processed/    pb_* (Stage 1), samples_* (Stage 2), s3_* (Stage 3), s4_* (Stage 4), s7_* (Stage 7)
data/external/     cloned reference repos (git-ignored)
src/               pipeline scripts
tests/             unit tests (python tests/test_<name>.py)
results/           exploration outputs (explore_data.py), Stage 3 summary, Stage 4 summary/audit CSVs, Stage 7 ablation CSVs
```

## Next

- Stage 2 sampling on MATH, OlympiadBench, Omni-MATH (`colab/run_stage2.ipynb`, about 3 h on a T4)
- Answer normalization for competition answers: 37% of original ProcessBench final answers
  (fractions, radicals, tuples, intervals, text) do not reduce to a plain number with `normalize_answer`
- Stage 3 and Stage 7 on MATH, OlympiadBench, Omni-MATH once their samples arrive; then add retrieval (A2)
