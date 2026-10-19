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
| 2 Sample N solutions + check | `src/stage2_generate.py`, `src/stage2_check.py`, `colab/run_stage2.ipynb` | `data/processed/samples_*.jsonl` | all four splits done (GSM8K locally, the others on Colab) |
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
# All splits: validate samples_<split>.jsonl, then Stage 2b (both answer rules), 3 and 7 for its sampled problems
python src/run_split.py math            # gsm8k, math, olympiadbench, omnimath; logs in results/logs/
# Evaluation over every split that has s7_ and s3_ files, per split and pooled -> results/all_splits_*.csv
python src/evaluate_all.py
python src/evaluate_all.py --suffix _truncnone              # truncated without \boxed = no answer
python src/evaluate_all.py --suffix _truncnone --compare    # both side by side -> all_splits_compare_truncnone.csv
#   Omni-MATH: drop the new data/processed/samples_omnimath.jsonl in, run run_split.py omnimath, then the three
#   lines above; it joins every table (and the pooled rows) automatically.
# tests
python tests/test_stage7.py
python tests/test_normalize.py
python tests/test_stage3.py
python tests/test_no_leakage.py     # no feature uses label / final_answer_correct
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

### All splits (GSM8K, MATH, OlympiadBench, Omni-MATH)

Sampled problems only: GSM8K 400, MATH, OlympiadBench and Omni-MATH 300 each (seed 0, the same problem
indices in every competition split and the same problems for every stage), 5 samples each from Qwen2.5-Math-1.5B-Instruct (max 768 new tokens for GSM8K, 1024 for the others).
Source: `results/all_splits_*.csv`, written by `src/evaluate_all.py`.

**Protocol.** Logistic regression (standardised, C = 1) and a random forest (200 trees, depth 4),
stratified 5-fold cross-validation repeated 10 times (seed 0), no tuning. Each problem gets an out-of-fold
probability in each repeat, and these are averaged over the 10 repeats. The AUROC reported is the AUROC of
those averaged predictions. The 95% CI bootstraps the problems (2000 resamples) of the **same**
predictions, so the point estimate and the interval describe one statistic. (The per-split
`stage7_ablation_*.csv` files report the mean of the 10 per-repeat AUROCs instead; for near-chance feature
sets that mean can fall outside the CI, so it is kept only as `auroc_mean_over_repeats`.) Human labels
and `final_answer_correct` are targets only; `tests/test_no_leakage.py` checks that no feature uses them.

Target: process error (label ≠ −1), logistic regression, AUROC [95% CI]:

| Split | n | Erroneous (base rate) | Stage 4 only | Stage 3 only | Agreement only | Stage 4 + agreement | ALL (Stage 4 + 7) | ALL + Stage 3 |
|---|---|---|---|---|---|---|---|---|
| GSM8K | 400 | 207 (0.52) | 0.550 [0.496, 0.606] | 0.894 [0.859, 0.925] | 0.922 [0.889, 0.955] | 0.942 [0.915, 0.967] | 0.940 [0.912, 0.965] | 0.948 [0.921, 0.970] |
| MATH | 300 | 167 (0.56) | 0.538 [0.474, 0.604] | 0.774 [0.720, 0.827] | 0.857 [0.805, 0.909] | 0.904 [0.865, 0.941] | 0.915 [0.879, 0.949] | 0.912 [0.875, 0.947] |
| OlympiadBench | 300 | 189 (0.63) | 0.479 [0.411, 0.550] | 0.640 [0.574, 0.708] | 0.675 [0.600, 0.749] | 0.738 [0.676, 0.799] | 0.773 [0.715, 0.827] | 0.777 [0.722, 0.829] |
| Omni-MATH | 300 | 214 (0.71) | 0.648 [0.580, 0.713] | 0.757 [0.697, 0.815] | 0.713 [0.635, 0.786] | 0.791 [0.732, 0.850] | 0.835 [0.785, 0.884] | 0.851 [0.800, 0.897] |
| Pooled | 1300 | 777 (0.60) | 0.586 [0.556, 0.616] | 0.777 [0.751, 0.803] | 0.836 [0.810, 0.860] | 0.879 [0.858, 0.897] | 0.897 [0.877, 0.914] | 0.895 [0.874, 0.912] |

Target: wrong final answer, logistic regression (random forest results and both targets in full: `all_splits_auroc.csv`):

| Split | Wrong answers (base rate) | Agreement only | ALL (Stage 4 + 7) | ALL + Stage 3 |
|---|---|---|---|---|
| GSM8K | 200 (0.50) | 0.935 [0.905, 0.963] | 0.952 [0.929, 0.973] | 0.956 [0.932, 0.975] |
| MATH | 137 (0.46) | 0.863 [0.818, 0.911] | 0.916 [0.884, 0.945] | 0.903 [0.868, 0.935] |
| OlympiadBench | 137 (0.46) | 0.671 [0.608, 0.732] | 0.730 [0.674, 0.784] | 0.739 [0.683, 0.793] |
| Omni-MATH | 137 (0.46) | 0.674 [0.614, 0.735] | 0.714 [0.654, 0.770] | 0.734 [0.677, 0.788] |
| Pooled | 611 (0.47) | 0.818 [0.793, 0.841] | 0.871 [0.851, 0.888] | 0.870 [0.850, 0.888] |

On Omni-MATH the wrong-answer AUROC is clearly lower than the process-error one (0.71 vs 0.84). 77 of
its 214 erroneous solutions still end in the right answer, and agreement cannot see those.

**Hard subset** (correct final answer but a wrong step, against fully correct solutions), process error,
logistic regression (`all_splits_hard.csv`). **GSM8K has only 7 such cases (vs 193 fully correct),
which is too few to train or score a classifier, so it is not evaluated on its own**; its 7 cases are
included in the pooled row.

| Split | Hard / correct | Stage 4 only | Stage 3 only | Agreement only | Stage 7 only (agreement + entropy) | ALL (Stage 4 + 7) | ALL + Stage 3 |
|---|---|---|---|---|---|---|---|
| GSM8K | 7 / 193 | too few | too few | too few | too few | too few | too few |
| MATH | 30 / 133 | 0.446 [0.315, 0.576] | 0.661 [0.530, 0.785] | 0.630 [0.474, 0.773] | 0.724 [0.607, 0.835] | 0.783 [0.697, 0.863] | 0.760 [0.667, 0.848] |
| OlympiadBench | 52 / 111 | 0.375 [0.280, 0.474] | 0.522 [0.433, 0.613] | 0.599 [0.505, 0.692] | 0.709 [0.622, 0.792] | 0.709 [0.621, 0.792] | 0.692 [0.606, 0.774] |
| Omni-MATH | 77 / 86 | 0.634 [0.548, 0.718] | 0.727 [0.645, 0.803] | 0.617 [0.523, 0.711] | 0.776 [0.699, 0.845] | 0.794 [0.722, 0.861] | 0.819 [0.749, 0.880] |
| Pooled | 166 / 523 | 0.623 [0.573, 0.670] | 0.730 [0.686, 0.773] | 0.714 [0.664, 0.763] | 0.829 [0.790, 0.865] | 0.839 [0.803, 0.873] | 0.840 [0.802, 0.874] |

On the hard subset the signal comes from the Stage 7 sample features: agreement *plus* the
entropy/log-probability/length summaries. Agreement alone is weak there, as it should be, since the
original answer is right. Stage 3 adds some signal on Omni-MATH (0.727 alone, +0.025 on top of ALL), but
not on MATH or OlympiadBench. Stage 4 alone is below chance on MATH and OlympiadBench and modest on
Omni-MATH (0.634). Part of its pooled 0.623 comes from the splits differing in base rate and equation
counts rather than from detecting errors. MATH's 30 hard cases give wide intervals.

**Step level** (first flagged step vs the human first wrong step, `all_splits_steps.csv`):

| Split | Stage 3 exact match | Chance exact match | Stage 3 false alarms on correct | Steps with no key number | Stage 4 flagged / exact / false alarms / precision |
|---|---|---|---|---|---|
| GSM8K | 44.4% | 20.8% | 18.7% | 22.0% | 8.2% / 7.2% / 0.0% / 100% |
| MATH | 21.0% | 18.1% | 26.3% | 34.6% | 12.6% / 7.2% / 0.8% / 95.5% |
| OlympiadBench | 17.5% | 13.1% | 45.0% | 36.0% | 9.0% / 2.6% / 1.8% / 89.5% |
| Omni-MATH | 19.6% | 14.4% | 46.5% | 43.5% | 13.6% / 6.1% / 3.5% / 90.6% |
| Pooled | 26.0% | 16.6% | 30.8% | 34.4% | 10.8% / 5.8% / 1.1% / 93.3% |

*Chance* = picking a uniformly random step of each erroneous solution. On GSM8K, Stage 3 finds the
first wrong step about twice as often as chance. On MATH, OlympiadBench and Omni-MATH it is only 3–5
points above chance, with many false alarms, so **it does not localise errors on these splits**. There,
its features help only as a solution-level signal.

**Generalisation** (`all_splits_transfer.csv`, process error, ALL, logistic regression, fitted once on the
training splits):

| Test split | Trained on GSM8K only | Trained on the other three splits |
|---|---|---|
| GSM8K | — | 0.931 [0.904, 0.956] |
| MATH | 0.917 [0.884, 0.950] | 0.937 [0.908, 0.963] |
| OlympiadBench | 0.750 [0.691, 0.808] | 0.796 [0.741, 0.846] |
| Omni-MATH | 0.795 [0.735, 0.853] | 0.829 [0.773, 0.881] |

**Why OlympiadBench and Omni-MATH are weak** (`all_splits_overview.csv`):

| | OlympiadBench | Omni-MATH | GSM8K, for reference |
|---|---|---|---|
| Samples that hit the 1024-token cap | 21.7% | 18.8% | 1.0% (768 cap) |
| Problems with ≥ 3 of 5 samples truncated | 56 / 300 | 46 / 300 | 2 / 400 |
| Mean majority share (self-consistency) | 0.59 | 0.61 | 0.86 |
| Agreement with *correct* original solutions | 52% | 60% | 88% |
| Original answers SymPy cannot parse (string only) | 27 | 42 | 1 |
| Agreement AUROC: problems with no truncated sample | 0.83 (175) | 0.84 (176) | 0.95 (388) |
| Agreement AUROC: problems with ≥ 1 truncated sample | 0.50 (125) | 0.64 (124) | — (12) |

On problems where every sample finished, agreement works almost as well as on MATH. Truncation is the
main reason these two splits are weak.

**Truncated samples: two answer rules.** By default a sample cut off at the token cap takes the last
number of its text as its answer. With `--truncated-no-answer` (files `*_truncnone*`), a sample that
hit the cap without a `\boxed{}` answer has no answer. Nothing was re-sampled. Process error, logistic
regression, default → truncnone (every row: `all_splits_compare_truncnone.csv`):

| Split | No-answer samples | Stage 2 AUROC (1 − agreement) | ALL (Stage 4 + 7) | ALL + Stage 3 | Hard subset, ALL |
|---|---|---|---|---|---|
| GSM8K | 0.1% → 1.0% | 0.946 → 0.946 | 0.940 → 0.941 | 0.948 → 0.948 | too few |
| MATH | 0.0% → 8.0% | 0.910 → 0.910 | 0.915 → 0.928 | 0.912 → 0.926 | 0.783 → 0.809 |
| OlympiadBench | 0.0% → 21.4% | 0.730 → 0.737 | 0.773 → 0.776 | 0.777 → 0.780 | 0.709 → 0.728 |
| Omni-MATH | 0.0% → 18.1% | 0.744 → 0.745 | 0.835 → 0.833 | 0.851 → 0.850 | 0.794 → 0.789 |
| Pooled | 0.0% → 11.3% | — | 0.897 → 0.902 | 0.895 → 0.900 | 0.839 → 0.849 |

All differences are well inside the confidence intervals. Across every row of the comparison, logistic
regression moves by −0.006 to +0.026; the random forest moves by −0.016 to +0.032. Omni-MATH is flat. We recommend the truncnone rule for the
competition splits. The last number of a cut-off text is not an answer the model gave, and the default
rule counts it as one, which can manufacture agreement or disagreement. The truncnone rule is never
meaningfully worse for the main model. This is a choice for correctness of the feature, not a measured improvement.
The tables above use the default rule so that GSM8K's earlier outputs stay unchanged.

**What this means.**
- **GSM8K and MATH.** Disagreement between sampled solutions and the original answer is a strong, cheap
  signal of a wrong solution: AUROC about 0.91–0.95. It also carries over from GSM8K to MATH without retraining.
- **OlympiadBench and Omni-MATH.** The same signal is much weaker (about 0.77 and 0.84), mainly because
  a 1.5B model with 1024 tokens often cannot finish or agree with itself on olympiad problems. A more
  capable or longer-running sampler is the obvious fix, and we have not tested it.
- **Correct answer, wrong step.** These hard cases are detected only moderately (about 0.7–0.8 where
  measurable), mostly through sample entropy rather than any step check.
- **SymPy (Stage 4).** It is precise when it fires (90–100%), but it fires on only about 11% of
  erroneous solutions. Alone it ranks solutions near chance, except for a modest 0.65 on Omni-MATH.
- **Stage 3 (where the error is).** Step-level localisation works on GSM8K only; on the three
  competition splits Stage 3 is barely above chance.

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

- Answer normalization: answers with a variable prefix (`x = \sqrt{2}`, `I(6,3)=0`) are still compared
  as whole strings (the strict rule is kept on purpose)
- Retrieval (A2)
