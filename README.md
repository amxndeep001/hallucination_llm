# Math Hallucination Detection (A1 + A3 + A2, RAG + SymPy)

Setup:
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Run order:
    python src/stage0_load_data.py                      # needs internet (Hugging Face)
    python src/stage1_preprocess.py data/raw/processbench_gsm8k.jsonl data/processed/pb_gsm8k.jsonl

Next: stage2_generate.py (sample N solutions), stage3 align, stage4_sympy_check.py ...
