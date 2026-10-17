"""Stage 1b: dataset statistics + charts + one annotated example (for your professor).
Usage:  python src/explore_data.py data/processed results
Reads  data/processed/pb_*.jsonl  (output of stage1_preprocess.py)
Writes results/summary.csv, problems.csv, steps.csv, results/*.png, results/example.md
"""
import glob, json, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(src_dir, "pb_*.jsonl")))
if not files:
    sys.exit(f"No pb_*.jsonl files found in '{os.path.abspath(src_dir)}'.\n"
             "Run this from the project root (not from src), or pass the full path to data/processed.")

rows, tag_rows, label_rows = [], [], []
prob_rows, step_rows = [], []
example = None
for path in files:
    split = os.path.basename(path)[3:-6]
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(recs)
    for r in recs:
        prob_rows.append({"split": split, "id": r["id"], "problem": r["problem"],
                          "num_steps": len(r["steps"]), "first_error_step": r["label"],
                          "has_process_error": r["label"] not in (-1, None)})
        for st in r["steps"]:
            step_rows.append({"split": split, "id": r["id"], "step_idx": st["idx"],
                              "step_text": st["text"], "equations": " | ".join(st["equations"]),
                              "step_type": st["type"],
                              "is_first_error": st["idx"] == r["label"]})
    err = [r for r in recs if r["label"] not in (-1, None)]
    n_steps = sum(len(r["steps"]) for r in recs)
    n_eq = sum(len(s["equations"]) for r in recs for s in r["steps"])
    types = pd.Series([s["type"] for r in recs for s in r["steps"]]).value_counts()
    # upper bound on what a symbolic checker could ever catch:
    # erroneous records whose first wrong step contains at least one equation
    cov = sum(1 for r in err if r["steps"][r["label"]]["equations"]) if err else 0
    rows.append({
        "split": split, "problems": n,
        "with_process_error_%": round(100 * len(err) / n, 1),
        "avg_steps": round(n_steps / n, 2),
        "equations_per_step": round(n_eq / max(n_steps, 1), 2),
        "first_wrong_step_has_equation_%": round(100 * cov / max(len(err), 1), 1),
    })
    for t, c in types.items():
        tag_rows.append({"split": split, "type": t, "share": c / n_steps})
    for r in err:
        label_rows.append({"split": split, "error_step": r["label"]})
    if example is None and err:
        example = (split, err[0])

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
print(df.to_string(index=False))
pd.DataFrame(prob_rows).to_csv(os.path.join(out_dir, "problems.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(step_rows).to_csv(os.path.join(out_dir, "steps.csv"), index=False, encoding="utf-8-sig")

# chart 1: share of step types per split
tag = pd.DataFrame(tag_rows).pivot(index="split", columns="type", values="share").fillna(0)
tag.plot(kind="bar", stacked=True, figsize=(7, 4), title="Step types per split")
plt.ylabel("share of steps"); plt.tight_layout()
plt.savefig(os.path.join(out_dir, "step_types.png"), dpi=150); plt.close()

# chart 2: where the first error occurs
lab = pd.DataFrame(label_rows)
if len(lab):
    lab.groupby(["error_step", "split"]).size().unstack(fill_value=0).plot(
        kind="bar", figsize=(8, 4), title="Position of first erroneous step (0-indexed)")
    plt.ylabel("problems"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_position.png"), dpi=150); plt.close()

# chart 3: symbolic-check upper bound
df.plot(x="split", y="first_wrong_step_has_equation_%", kind="bar", legend=False,
        figsize=(6, 4), title="Erroneous problems where the first wrong step has an equation (%)")
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "symbolic_upper_bound.png"), dpi=150); plt.close()

# annotated example
if example:
    split, r = example
    with open(os.path.join(out_dir, "example.md"), "w", encoding="utf-8") as f:
        f.write(f"# Example ({split}, id={r['id']})\n\n**Problem:** {r['problem']}\n\n")
        for s in r["steps"]:
            mark = "  <-- FIRST ERROR (human label)" if s["idx"] == r["label"] else ""
            f.write(f"**Step {s['idx']}** [{s['type']}]{mark}\n\n{s['text']}\n\n"
                    f"Equations found: {s['equations']}\n\n")
print("saved to", out_dir)