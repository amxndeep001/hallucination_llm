"""Stage 0: download datasets locally (run on your machine / Colab, needs internet)."""
import json, pathlib
from datasets import load_dataset

RAW = pathlib.Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

def save(name, rows):
    with open(RAW / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows))

# ProcessBench: evaluation set (splits: gsm8k, math, olympiadbench, omnimath)
pb = load_dataset("Qwen/ProcessBench")
for split in pb:
    save(f"processbench_{split}", list(pb[split]))

# GSM8K: problems for generating your own solutions
gsm = load_dataset("openai/gsm8k", "main")
save("gsm8k_train", list(gsm["train"]))
save("gsm8k_test", list(gsm["test"]))

# PRM800K: clone separately -> git clone https://github.com/openai/prm800k
