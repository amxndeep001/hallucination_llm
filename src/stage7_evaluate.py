"""Stage 7 evaluation: single-feature AUROCs, a first combined classifier, and an ablation.

Targets
  (a) process_error = label != -1                 (main target; the ablation uses it)
  (b) wrong_answer  = final_answer_correct is False
  HARD subset: only solutions with a correct final answer; target (a) there means
               "correct answer, wrong step", which answer agreement alone cannot see.
Models: logistic regression (standardized, C=1.0) and a random forest (200 trees, max_depth 4),
stratified 5-fold cross-validation repeated 10 times (seed 0). One row per problem, so folds never
split a problem. AUROC = mean over the 10 repeats; 95% CI = bootstrap over problems of the
out-of-fold predictions averaged over the repeats. Chance: AUROC 0.5, average precision = base rate.

Usage:
  python src/stage7_evaluate.py data/processed/s7_gsm8k.jsonl [--results-dir results]
Writes results/stage7_ablation_<split>.csv.
"""
import argparse, csv, json, os, re, warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

S4 = ["n_fail", "n_checked", "first_fail_step", "sympy_flag", "n_equations"]
S7 = ["majority_share", "agree_with_original", "n_clusters", "semantic_entropy", "no_answer_rate",
      "mean_logprob_mean", "mean_logprob_min", "mean_entropy_mean", "mean_entropy_max",
      "n_tokens_mean", "truncated_rate"]
ALL = S7 + S4
ABLATION = [("Stage 4 only", S4),
            ("Stage 2 agreement only", ["agree_with_original"]),
            ("Stage 7 only (all but Stage 4)", S7),
            ("Stage 4 + agreement", S4 + ["agree_with_original"]),
            ("ALL", ALL),
            ("ALL minus semantic_entropy", [f for f in ALL if f != "semantic_entropy"]),
            ("ALL minus Stage 4", S7)]                         # same set as 'Stage 7 only'
MODELS = {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
          "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0, n_jobs=-1)}
N_SPLITS, N_REPEATS, SEED, N_BOOT = 5, 10, 0, 2000


def matrix(recs, feats):
    """first_fail_step is None when nothing failed: encoded as -1 (sympy_flag carries the indicator)."""
    val = lambda r, f: -1 if r[f] is None else float(r[f])
    return np.array([[val(r, f) for f in feats] for r in recs], dtype=float)


def oof_predictions(X, y, model):
    """(N_REPEATS, n) out-of-fold probabilities."""
    oof = np.zeros((N_REPEATS, len(y)))
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m = MODELS[model]()
        m.fit(X[tr], y[tr])
        oof[i // N_SPLITS, te] = m.predict_proba(X[te])[:, 1]
    return oof


def boot_idx(y):
    """Bootstrap resamples of the problems (each must contain both classes)."""
    rng, out = np.random.default_rng(SEED), []
    while len(out) < N_BOOT:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            out.append(idx)
    return out


def evaluate_set(X, y, model, boots):
    oof = oof_predictions(X, y, model)
    aucs = [roc_auc_score(y, p) for p in oof]
    p = oof.mean(0)
    b = [roc_auc_score(y[i], p[i]) for i in boots]
    return {"auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs)),
            "ci_low": float(np.percentile(b, 2.5)), "ci_high": float(np.percentile(b, 97.5)),
            "ap_mean": float(np.mean([average_precision_score(y, q) for q in oof])), "p": p}


def paired_delta(y, p1, p2, boots):
    d = [roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]) for i in boots]
    return roc_auc_score(y, p1) - roc_auc_score(y, p2), np.percentile(d, 2.5), np.percentile(d, 97.5)


def fmt(r):
    return f"{r['auroc_mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def single_feature(recs, y, feat):
    x = matrix(recs, [feat])[:, 0]
    if len(set(x)) < 2:
        return float("nan"), "constant"
    auc = roc_auc_score(y, x)
    return max(auc, 1 - auc), "higher = error" if auc >= 0.5 else "lower = error"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("s7")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    split = re.sub(r"^s7_", "", os.path.splitext(os.path.basename(a.s7))[0])
    with open(a.s7, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    assert len({r["id"] for r in recs}) == len(recs), "expected one row per problem"

    y_a = np.array([r["label"] != -1 for r in recs], dtype=int)
    y_b = np.array([r["final_answer_correct"] is False for r in recs], dtype=int)
    hard = [r for r in recs if r["final_answer_correct"] is True]
    y_h = np.array([r["label"] != -1 for r in hard], dtype=int)
    base = {"a": y_a.mean(), "b": y_b.mean(), "hard": y_h.mean() if len(hard) else float("nan")}
    print(f"===== Stage 7 evaluation: {split}, {len(recs)} problems")
    print(f"(a) process_error: {y_a.sum()} of {len(y_a)}  base rate {base['a']:.3f}")
    print(f"(b) wrong_answer : {y_b.sum()} of {len(y_b)}  base rate {base['b']:.3f}")
    print(f"HARD subset (final answer correct): {len(hard)} problems, {y_h.sum()} with a wrong step "
          f"(base rate {base['hard']:.3f})")
    hard_ok = y_h.sum() >= 30 and len(hard) - y_h.sum() >= 30
    if not hard_ok:
        print(f"  WARNING: only {y_h.sum()} correct-answer-wrong-step cases (< 30): HARD-subset numbers are NOT reliable.")
    print("Chance level: AUROC 0.500 for every target; average precision (AP) = base rate.")

    # ------------------------------------------------ single features
    print(f"\n## Single-feature AUROC (sign-corrected: max(auc, 1-auc); chance 0.5)\n")
    print(f"| feature | (a) process_error, base {base['a']:.3f} | direction | (b) wrong_answer, base {base['b']:.3f} "
          f"| direction | HARD, base {base['hard']:.3f} (n_pos={y_h.sum()}) | direction |")
    print("|---|---|---|---|---|---|---|")
    for f in ALL:
        cells = [single_feature(recs, y_a, f), single_feature(recs, y_b, f),
                 single_feature(hard, y_h, f) if 0 < y_h.sum() < len(y_h) else (float("nan"), "n/a")]
        print(f"| {f} | " + " | ".join(f"{v:.3f} | {d}" for v, d in cells) + " |")
    print("(the direction is chosen on the same data, so single-feature AUROCs are slightly optimistic)")

    # ------------------------------------------------ ablation, target (a)
    boots_a = boot_idx(y_a)
    print(f"\n## Ablation, target (a) process_error: AUROC mean over {N_REPEATS}x{N_SPLITS}-fold CV "
          f"[bootstrap 95% CI]; chance 0.5, base rate {base['a']:.3f}\n")
    print("| feature set | n features | LR AUROC [95% CI] | LR AP | RF AUROC [95% CI] | RF AP | base rate |")
    print("|---|---|---|---|---|---|---|")
    res, rows = {}, []
    for name, feats in ABLATION:
        X = matrix(recs, feats)
        for m in MODELS:
            res[name, m] = evaluate_set(X, y_a, m, boots_a)
            r = res[name, m]
            rows.append({"split": split, "target": "process_error", "feature_set": name, "n_features": len(feats),
                         "model": m, "auroc_mean": round(r["auroc_mean"], 4), "auroc_sd_over_repeats": round(r["auroc_sd"], 4),
                         "ci_low": round(r["ci_low"], 4), "ci_high": round(r["ci_high"], 4),
                         "ap_mean": round(r["ap_mean"], 4), "base_rate": round(base["a"], 4), "n": len(y_a)})
        print(f"| {name} | {len(feats)} | {fmt(res[name, 'LR'])} | {res[name, 'LR']['ap_mean']:.3f} | "
              f"{fmt(res[name, 'RF'])} | {res[name, 'RF']['ap_mean']:.3f} | {base['a']:.3f} |")
    os.makedirs(a.results_dir, exist_ok=True)
    out_csv = os.path.join(a.results_dir, f"stage7_ablation_{split}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out_csv}")

    # ------------------------------------------------ does semantic entropy add anything?
    print(f"\n## Paired bootstrap: AUROC difference on the repeat-averaged out-of-fold predictions "
          f"(target a, base rate {base['a']:.3f})\n")
    X_ase = matrix(recs, ["agree_with_original", "semantic_entropy"])
    for m in MODELS:
        d, lo, hi = paired_delta(y_a, res["ALL", m]["p"], res["ALL minus semantic_entropy", m]["p"], boots_a)
        print(f"{m}: ALL - (ALL minus semantic_entropy)            = {d:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]")
        ase = evaluate_set(X_ase, y_a, m, boots_a)
        d, lo, hi = paired_delta(y_a, ase["p"], res["Stage 2 agreement only", m]["p"], boots_a)
        print(f"{m}: (agreement + semantic_entropy) - agreement    = {d:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]"
              f"   (agreement + SE: {fmt(ase)})")
        d, lo, hi = paired_delta(y_a, res["ALL", m]["p"], res["Stage 2 agreement only", m]["p"], boots_a)
        print(f"{m}: ALL - agreement only                          = {d:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]")

    # ------------------------------------------------ target (b) and the HARD subset
    print(f"\n## Target (b) wrong_answer, base rate {base['b']:.3f} (chance AUROC 0.5)")
    boots_b = boot_idx(y_b)
    for name, feats in [("Stage 2 agreement only", ["agree_with_original"]), ("Stage 4 only", S4), ("ALL", ALL)]:
        r = {m: evaluate_set(matrix(recs, feats), y_b, m, boots_b) for m in MODELS}
        print(f"  {name:24s} LR {fmt(r['LR'])}  AP {r['LR']['ap_mean']:.3f} | RF {fmt(r['RF'])}  AP {r['RF']['ap_mean']:.3f}"
              f" | base {base['b']:.3f}")

    print(f"\n## HARD subset: target (a) among {len(hard)} correct-answer solutions, {y_h.sum()} with a wrong step, "
          f"base rate {base['hard']:.3f}{'' if hard_ok else '  -- NOT RELIABLE (< 30 positives)'}")
    if y_h.sum() >= N_SPLITS and len(hard) - y_h.sum() >= N_SPLITS:
        boots_h = boot_idx(y_h)
        for name, feats in [("Stage 2 agreement only", ["agree_with_original"]), ("Stage 4 only", S4), ("ALL", ALL)]:
            r = {m: evaluate_set(matrix(hard, feats), y_h, m, boots_h) for m in MODELS}
            print(f"  {name:24s} LR {fmt(r['LR'])}  AP {r['LR']['ap_mean']:.3f} | RF {fmt(r['RF'])}  AP {r['RF']['ap_mean']:.3f}"
                  f" | base {base['hard']:.3f}")
    else:
        print(f"  fewer than {N_SPLITS} cases in a class: cross-validation not possible")

    # ------------------------------------------------ combined model: threshold, confusion matrix, worst errors
    p = res["ALL", "LR"]["p"]
    prec, rec, thr = precision_recall_curve(y_a, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    k = int(np.argmax(f1))
    pred = (p >= thr[k]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_a, pred, labels=[0, 1]).ravel()
    print(f"\n## Combined model (LR, ALL features), target (a), threshold {thr[k]:.3f} = max F1 on the same "
          f"out-of-fold predictions (OPTIMISTIC: the threshold was not chosen on held-out data)")
    print(f"  precision {prec[k]:.3f} (chance = base rate {base['a']:.3f})   recall {rec[k]:.3f}   F1 {f1[k]:.3f}")
    print(f"  confusion matrix          predicted ok   predicted error")
    print(f"    actual ok    (label -1)  {tn:12d}   {fp:15d}")
    print(f"    actual error            {fn:12d}   {tp:15d}")

    print(f"\n## 10 most confidently wrong problems (LR, ALL; repeat-averaged out-of-fold P(process error))\n")
    print("| id | label | final_answer_correct | P(error) | original answer | sample answers |")
    print("|---|---|---|---|---|---|")
    for i in np.argsort(-np.abs(y_a - p))[:10]:
        r = recs[i]
        print(f"| {r['id']} | {r['label']} | {r['final_answer_correct']} | {p[i]:.3f} | {r['original_answer']} | "
              f"{', '.join(x or '(none)' for x in r['sample_answers'])} |")
