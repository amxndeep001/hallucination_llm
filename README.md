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

## Pipeline

| Stage | Script | Output | Status |
|---|---|---|---|
| 0 Load data | `src/stage0_load_data.py` | `data/raw/*.jsonl` | done |
| 1 Preprocess (steps, equations, step types) | `src/stage1_preprocess.py` | `data/processed/pb_*.jsonl` | done |
| 2 Sample N solutions + check | `src/stage2_generate.py`, `src/stage2_check.py` | `data/processed/samples_*.jsonl` | GSM8K only |
| 3 Align | — | — | **not built yet** |
| 4 SymPy verification + audit | `src/stage4_sympy_verify.py`, `src/stage4_audit.py` | `data/processed/s4_*.jsonl`, `results/stage4_*.csv` | done |

## Run order

Run every command from the project root. `<split>` is one of `gsm8k`, `math`, `olympiadbench`, `omnimath`.

```bash
# Stage 0: download ProcessBench + GSM8K (needs internet / Hugging Face)
python src/stage0_load_data.py

# Stage 1: split solutions into steps and extract equations
python src/stage1_preprocess.py data/raw/processbench_<split>.jsonl data/processed/pb_<split>.jsonl

# Stage 2a: sample 5 solutions per problem (GPU, e.g. Colab; resumable)
python src/stage2_generate.py --input data/processed/pb_gsm8k.jsonl \
    --output data/processed/samples_gsm8k.jsonl \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct --n 5 --batch-size 2
#   smoke test without a model: add --backend mock --limit 5
# Stage 2b: check the samples and the agreement signal
python src/stage2_check.py data/processed/samples_gsm8k.jsonl data/raw/processbench_gsm8k.jsonl

# Stage 3: not built yet

# Stage 4a: verify every extracted equation with SymPy, print the evaluation, save the summary row
python src/stage4_sympy_verify.py data/processed/pb_<split>.jsonl data/processed/s4_<split>.jsonl \
    --summary results/stage4_summary.csv
# Stage 4b: CSV of cases to review by hand (early failures, sampled false alarms, sampled skips)
python src/stage4_audit.py data/processed/s4_<split>.jsonl results/stage4_audit_<split>.csv
```

## Stage 4 in short

Each equation from Stage 1 is converted from LaTeX, parsed by SymPy and marked:

- **pass**: both sides are equal (rounding to the stated decimals, percent notation and integer quotients are allowed)
- **fail**: both sides are plain numbers and differ, which means a calculation error
- **skip**: cannot be judged; the reason says why (variables, inequality, unsupported LaTeX, unbalanced brackets …). Skip is not pass.

A step fails if any of its equations fails. The first failing step is the predicted first error.

## Results so far

Stage 2 (GSM8K, 400 problems × 5 samples): low agreement between the samples and the original
answer detects a process error with **AUROC 0.916**.

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
data/processed/    pb_* (Stage 1), samples_* (Stage 2), s4_* (Stage 4)
data/external/     cloned reference repos (git-ignored)
src/               pipeline scripts
results/           exploration outputs (explore_data.py) and Stage 4 summary/audit CSVs
```

## Next

- Stage 3 alignment
- Stage 2 sampling on MATH, OlympiadBench, Omni-MATH
- Combine Stage 4 flags with the Stage 2 agreement signal and retrieval
