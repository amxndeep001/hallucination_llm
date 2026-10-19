"""Evaluation over all splits: per split and pooled, step-level Stage 3 / Stage 4, and generalisation.

A split is used when data/processed/s7_<split><suffix>.jsonl and s3_<split>.jsonl both exist (run
src/run_split.py <split> first), so Omni-MATH joins by dropping in its samples file and running that.
Only the sampled problems are evaluated (the ids in s7/s3; s4/pb/raw records are matched by id).

1. Overview per split + pooled: problems, erroneous (label >= 0), base rates, HARD subset (correct final
   answer but a wrong step), and diagnostics for weak splits (truncation, no answer, self-agreement,
   original answers SymPy cannot parse).
2. AUROC [bootstrap 95% CI] for Stage 4 only / agreement only / Stage 3 only / Stage 4 + agreement /
   ALL (Stage 4 + 7) / ALL + Stage 3, LR and RF, 5-fold CV x 10 repeats (same models, folds and bootstrap
   as stage7_evaluate.py), targets (a) process_error = label != -1 and (b) wrong_answer.
   The AUROC is that of the out-of-fold predictions averaged over the 10 repeats, and the CI bootstraps the
   problems of those same predictions, so both describe one statistic. (stage7_evaluate.py reports the mean
   of the 10 per-repeat AUROCs, kept here as auroc_mean_over_repeats.) HARD subset: the same for Stage 4 /
   Stage 3 / agreement / Stage 7 (agreement + entropy) / ALL / ALL + Stage 3, where both classes have >= 30.
3. Step level: Stage 3 (s3_first_low_step, support_min <= 0.2) and Stage 4 (SymPy first fail) vs the human
   first wrong step, with the chance exact-match level, and the share of steps with no key number.
4. Generalisation: train on GSM8K, test on each other split; leave one dataset out (train on the others).
   Models are fitted once on the training splits; AUROC and its bootstrap CI are on the test split.
Labels and final_answer_correct are targets only: check_no_leakage() asserts no feature set uses them.

Usage:
  python src/evaluate_all.py [--splits gsm8k math olympiadbench omnimath] [--suffix _truncnone]
  python src/evaluate_all.py --suffix _truncnone --compare     # both runs side by side
Writes results/all_splits_{overview,auroc,hard,steps,transfer}<suffix>.csv, all_splits_compare<suffix>.csv
"""
import argparse, csv, os, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from stage3_align import read_jsonl
from stage3_evaluate import metrics
from stage7_consistency import parse_answer
from stage7_evaluate import ALL, MODELS, N_SPLITS, S3, S4, boot_idx, evaluate_set, matrix

SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]
AGREE = ["agree_with_original"]
FEATURE_SETS = [("Stage 4 only", S4), ("agreement only", AGREE), ("Stage 3 only", S3),
                ("Stage 4 + agreement", S4 + AGREE), ("ALL (Stage 4 + 7)", ALL), ("ALL + Stage 3", ALL + S3)]
S7_ONLY = [f for f in ALL if f not in S4]                   # agreement, clusters, semantic entropy, logprob, length
HARD_SETS = [("Stage 4 only", S4), ("Stage 3 only", S3), ("agreement only", AGREE),
             ("Stage 7 only (agreement + entropy)", S7_ONLY), ("ALL (Stage 4 + 7)", ALL), ("ALL + Stage 3", ALL + S3)]
TRANSFER_SETS = [("agreement only", AGREE), ("Stage 4 + agreement", S4 + AGREE),
                 ("ALL (Stage 4 + 7)", ALL), ("ALL + Stage 3", ALL + S3)]
TARGETS = {"process_error": lambda r: r["label"] != -1, "wrong_answer": lambda r: r["final_answer_correct"] is False}
FORBIDDEN = {"id", "label", "final_answer_correct", "original_answer", "sample_answers", "cluster_sizes", "split"}
MIN_HARD = 30
P = "data/processed"


def check_no_leakage(feature_sets):
    for name, feats in feature_sets:
        bad = FORBIDDEN & set(feats)
        assert not bad, f"feature set '{name}' uses target or inspection fields {bad}"


def load(split, suffix):
    s7p, s3p = f"{P}/s7_{split}{suffix}.jsonl", f"{P}/s3_{split}.jsonl"
    if not (os.path.exists(s7p) and os.path.exists(s3p)):
        return None
    s7, s3 = read_jsonl(s7p), {r["id"]: r for r in read_jsonl(s3p)}
    bad = [r["id"] for r in s7 if r["id"] not in s3 or s3[r["id"]]["label"] != r["label"]]
    if bad or len(s3) != len(s7):
        raise SystemExit(f"ERROR: {split}: s7 and s3 ids/labels differ, e.g. {bad[:5]}")
    recs = [{**r, **{k: s3[r["id"]][k] for k in S3}, "split": split} for r in s7]
    s4 = {r["id"]: r for r in read_jsonl(f"{P}/s4_{split}.jsonl")}
    pb = {r["id"]: r for r in read_jsonl(f"{P}/pb_{split}.jsonl")}
    return {"recs": recs, "s3": [s3[r["id"]] for r in s7], "s4": s4, "pb": pb}


def y_of(recs, target):
    return np.array([TARGETS[target](r) for r in recs], dtype=int)


def overview(name, recs):
    ya, yb = y_of(recs, "process_error"), y_of(recs, "wrong_answer")
    hard = [r for r in recs if r["final_answer_correct"] is True]
    n_hard = sum(r["label"] != -1 for r in hard)
    ok = [r for r in recs if r["label"] == -1]
    origs = [r["original_answer"] for r in recs]
    tr = np.array([r["truncated_rate"] for r in recs])
    agree_auc = lambda m: (roc_auc_score(ya[m], -np.array([r["agree_with_original"] for r in recs])[m])
                           if 0 < ya[m].sum() < m.sum() else float("nan"))
    return {"group": name, "n": len(recs), "n_erroneous": int(ya.sum()), "base_rate": round(ya.mean(), 3),
            "n_wrong_answer": int(yb.sum()), "base_rate_wrong_answer": round(yb.mean(), 3),
            "n_hard": n_hard, "n_hard_correct_steps": len(hard) - n_hard,
            "hard_reliable": n_hard >= MIN_HARD and len(hard) - n_hard >= MIN_HARD,
            "truncated_samples_pct": round(100 * tr.mean(), 1),
            "problems_3of5_truncated": int((tr >= 0.6).sum()),
            "no_answer_samples_pct": round(100 * np.mean([r["no_answer_rate"] for r in recs]), 1),
            "mean_majority_share": round(np.mean([r["majority_share"] for r in recs]), 3),
            "mean_agree_on_correct": round(np.mean([r["agree_with_original"] for r in ok]), 3),
            "orig_answer_empty": sum(o == "" for o in origs),
            "orig_answer_string_only": sum(o != "" and parse_answer(o) is None for o in origs),
            "agree_auroc_no_truncation": round(agree_auc(tr == 0), 3), "n_no_truncation": int((tr == 0).sum()),
            "agree_auroc_some_truncation": round(agree_auc(tr > 0), 3), "n_some_truncation": int((tr > 0).sum())}


def cv_rows(name, recs, target, sets):
    """auroc = AUROC of the out-of-fold predictions averaged over the 10 repeats, the same statistic the
    bootstrap CI is computed on (stage7_evaluate.py reports the mean of the 10 per-repeat AUROCs instead,
    which for near-chance sets can fall outside that CI; it is kept here as auroc_mean_over_repeats)."""
    y = y_of(recs, target)
    boots = boot_idx(y)
    rows = []
    for fs, feats in sets:
        X = matrix(recs, feats)
        for m in MODELS:
            r = evaluate_set(X, y, m, boots)
            auc = roc_auc_score(y, r["p"])
            if not r["ci_low"] <= auc <= r["ci_high"]:
                print(f"  NOTE {name} / {target} / {fs} / {m}: AUROC {auc:.3f} outside its percentile CI "
                      f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
            rows.append({"group": name, "target": target, "feature_set": fs, "model": m,
                         "auroc": round(auc, 3), "ci_low": round(r["ci_low"], 3), "ci_high": round(r["ci_high"], 3),
                         "auroc_mean_over_repeats": round(r["auroc_mean"], 3), "sd_over_repeats": round(r["auroc_sd"], 3),
                         "ap_mean_over_repeats": round(r["ap_mean"], 3),
                         "base_rate": round(y.mean(), 3), "n": len(y), "n_pos": int(y.sum())})
    return rows


def transfer_rows(train_name, train, test_name, test, target):
    ytr, yte = y_of(train, target), y_of(test, target)
    boots = boot_idx(yte)
    rows = []
    for fs, feats in TRANSFER_SETS:
        for m in MODELS:
            clf = MODELS[m]().fit(matrix(train, feats), ytr)
            p = clf.predict_proba(matrix(test, feats))[:, 1]
            b = [roc_auc_score(yte[i], p[i]) for i in boots]
            rows.append({"train": train_name, "test": test_name, "target": target, "feature_set": fs, "model": m,
                         "auroc": round(roc_auc_score(yte, p), 3), "ci_low": round(np.percentile(b, 2.5), 3),
                         "ci_high": round(np.percentile(b, 97.5), 3), "base_rate_test": round(yte.mean(), 3),
                         "n_train": len(ytr), "n_test": len(yte)})
    return rows


def step_rows(name, parts):
    """Stage 3 (support_min <= 0.2) and Stage 4 first-step predictions on the sampled problems."""
    s3 = [r for d in parts for r in d["s3"]]
    s4 = {k: v for d in parts for k, v in d["s4"].items()}
    pb = {k: v for d in parts for k, v in d["pb"].items()}
    err = [r for r in s3 if r["label"] != -1]
    chance = round(100 * sum(1 / len(r["steps"]) for r in err) / len(err), 1)
    steps = [s for r in s3 for s in r["steps"]]
    no_key = round(100 * sum(s["n_key"] == 0 for s in steps) / len(steps), 1)
    preds = {"Stage 3 (support_min <= 0.2)": {r["id"]: None if r["s3_first_low_step"] == -1 else r["s3_first_low_step"] for r in s3},
             "Stage 4 (SymPy first fail)": {r["id"]: s4[r["id"]]["sympy"]["first_fail_step"] for r in s3}}
    ceil = {"Stage 3 (support_min <= 0.2)": lambda r: r["steps"][r["label"]]["support_min"] is not None,
            "Stage 4 (SymPy first fail)": lambda r: bool(pb[r["id"]]["steps"][r["label"]]["equations"])}
    rows = []
    for method, pred in preds.items():
        m = metrics(s3, pred, ceil[method])
        rows.append({"group": name, "method": method, "n_erroneous": m["n_erroneous"], "n_correct": m["n_correct"],
                     "flagged_pct": m["flagged_pct"], "exact_match_pct": m["exact_match_pct"], "chance_exact_pct": chance,
                     "false_alarm_pct": m["false_alarm_pct"], "precision_pct": m["precision_pct"],
                     "base_rate_pct": round(100 * len(err) / len(s3), 1), "ceiling_pct": m["ceiling_pct"],
                     "steps_no_key_number_pct": no_key, "n_steps": len(steps)})
    return rows


def write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")


def table(rows, cols):
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def compare(results_dir, suffix):
    """Side by side: default answers vs the variant (e.g. _truncnone), every AUROC row of both runs."""
    rows = []
    for kind, keys in [("auroc", ["group", "target", "feature_set", "model"]),
                       ("hard", ["group", "target", "feature_set", "model"]),
                       ("transfer", ["train", "test", "target", "feature_set", "model"])]:
        paths = [os.path.join(results_dir, f"all_splits_{kind}{s}.csv") for s in ("", suffix)]
        if not all(os.path.exists(p) for p in paths):
            continue
        base, var = [{tuple(r[k] for k in keys): r for r in csv.DictReader(open(p, encoding="utf-8"))} for p in paths]
        for k, r in base.items():
            v = var.get(k)
            if v is None:
                continue
            rows.append({"table": kind, "key": " / ".join(k),
                         "auroc_default": r["auroc"], "ci_default": f"[{r['ci_low']}, {r['ci_high']}]",
                         f"auroc{suffix}": v["auroc"], f"ci{suffix}": f"[{v['ci_low']}, {v['ci_high']}]",
                         "delta": round(float(v["auroc"]) - float(r["auroc"]), 3)})
    print(f"\n## Default answers vs {suffix.strip('_')}: every AUROC row (delta = {suffix.strip('_')} - default)\n")
    table(rows, list(rows[0]))
    write(rows, os.path.join(results_dir, f"all_splits_compare{suffix}.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--suffix", default="", help="s7 variant, e.g. _truncnone (from --truncated-no-answer)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--compare", action="store_true",
                    help="only compare the CSVs of the default run and the --suffix run (both must exist)")
    a = ap.parse_args()
    if a.compare:
        if not a.suffix:
            ap.error("--compare needs --suffix")
        compare(a.results_dir, a.suffix)
        raise SystemExit
    warnings.filterwarnings("ignore", category=UserWarning)
    check_no_leakage(FEATURE_SETS + TRANSFER_SETS + HARD_SETS)

    data = {}
    for sp in a.splits:
        d = load(sp, a.suffix)
        if d is None:
            print(f"skipping {sp}: s7_{sp}{a.suffix}.jsonl or s3_{sp}.jsonl not found (run src/run_split.py {sp})")
        else:
            data[sp] = d
    groups = [(sp, d["recs"]) for sp, d in data.items()]
    if len(data) > 1:
        groups.append(("pooled (" + " + ".join(data) + ")", [r for d in data.values() for r in d["recs"]]))
    os.makedirs(a.results_dir, exist_ok=True)
    out = lambda name: os.path.join(a.results_dir, f"all_splits_{name}{a.suffix}.csv")

    print(f"===== All-splits evaluation{' (' + a.suffix.strip('_') + ')' if a.suffix else ''}: {', '.join(data)}\n")
    ov = [overview(name, recs) for name, recs in groups]
    table(ov, ["group", "n", "n_erroneous", "base_rate", "n_wrong_answer", "base_rate_wrong_answer", "n_hard", "hard_reliable"])
    print()
    table(ov, ["group", "truncated_samples_pct", "problems_3of5_truncated", "no_answer_samples_pct", "mean_majority_share",
               "mean_agree_on_correct", "orig_answer_empty", "orig_answer_string_only",
               "agree_auroc_no_truncation", "n_no_truncation", "agree_auroc_some_truncation", "n_some_truncation"])
    for r in ov:
        if not r["hard_reliable"]:
            print(f"  {r['group']}: HARD subset has {r['n_hard']} correct-answer-wrong-step and {r['n_hard_correct_steps']} "
                  f"fully correct solutions (< {MIN_HARD} in a class): too small to trust")
    write(ov, out("overview"))

    auc = []
    for target in TARGETS:
        print(f"\n## AUROC [95% CI], target {target}, {N_SPLITS}-fold CV x 10 repeats\n")
        rows = [r for name, recs in groups for r in cv_rows(name, recs, target, FEATURE_SETS)]
        auc += rows
        for r in rows:
            r["auroc [95% CI]"] = f"{r['auroc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        table(rows, ["group", "feature_set", "model", "auroc [95% CI]", "ap_mean_over_repeats", "base_rate", "n"])
        for r in rows:
            del r["auroc [95% CI]"]
    write(auc, out("auroc"))

    print(f"\n## HARD subset (correct final answer): target process_error = a wrong step despite the right answer;"
          f" which feature groups carry the signal (only where both classes have >= {MIN_HARD})\n")
    hard_rows = []
    for (name, recs), o in zip(groups, ov):
        if o["hard_reliable"]:
            hard = [r for r in recs if r["final_answer_correct"] is True]
            hard_rows += cv_rows(name, hard, "process_error", HARD_SETS)
        else:
            print(f"  {name}: NOT EVALUATED: only {o['n_hard']} correct-answer-wrong-step cases "
                  f"(vs {o['n_hard_correct_steps']} fully correct), too few to train or score a classifier")
    if hard_rows:
        for r in hard_rows:
            r["auroc [95% CI]"] = f"{r['auroc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        table(hard_rows, ["group", "feature_set", "model", "auroc [95% CI]", "base_rate", "n", "n_pos"])
        for r in hard_rows:
            del r["auroc [95% CI]"]
        write(hard_rows, out("hard"))

    if not a.suffix:                                         # step level does not depend on the answer variant
        print("\n## Step level: predicted first wrong step vs the human label (sampled problems only)\n")
        st = [r for sp, d in data.items() for r in step_rows(sp, [d])]
        if len(data) > 1:
            st += step_rows(groups[-1][0], list(data.values()))
        table(st, ["group", "method", "n_erroneous", "flagged_pct", "exact_match_pct", "chance_exact_pct",
                   "false_alarm_pct", "precision_pct", "base_rate_pct", "ceiling_pct", "steps_no_key_number_pct"])
        write(st, out("steps"))

    tr = []
    for target in TARGETS:
        if "gsm8k" in data:
            for sp in data:
                if sp != "gsm8k":
                    tr += transfer_rows("gsm8k", data["gsm8k"]["recs"], sp, data[sp]["recs"], target)
        if len(data) > 2:
            for sp in data:
                others = [o for o in data if o != sp]
                tr += transfer_rows("+".join(others), [r for o in others for r in data[o]["recs"]],
                                    sp, data[sp]["recs"], target)
    if tr:
        print("\n## Generalisation: fit on the train splits, AUROC [95% CI] on the unseen test split\n")
        for r in tr:
            r["auroc [95% CI]"] = f"{r['auroc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        table(tr, ["target", "train", "test", "feature_set", "model", "auroc [95% CI]", "base_rate_test"])
        for r in tr:
            del r["auroc [95% CI]"]
        write(tr, out("transfer"))
