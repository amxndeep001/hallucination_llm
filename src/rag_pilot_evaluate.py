"""RAG pilot evaluation of the Stage 6 evidence sources on the TEST half of results/rag_pilot_sample.csv.

Sources: A (problem NLI), A2 (problem + earlier steps NLI), B (samples NLI), C (numeric spot check),
D-bm25 / D-bge (ProofWiki retrieval + NLI, knowledge_claim steps only). Per solution, over all its claims:
  max_pc        max p_contradict (B: of the per-claim mean over samples)          higher = error
  1-min_pe      1 - min p_entail (B: of the per-claim max over samples)            higher = error
  C n_fail      number of claims whose numeric spot check fails
A solution without scored claims gets 0 (no signal).
1. Solution level: AUROC vs process error (label >= 0), bootstrap 95% CI (2000 resamples, seed 0, as
   stage7_evaluate.boot_idx); TEST pooled and per split (tiny n); HARD subset (correct final answer, wrong step) vs
   correct solutions if it has >= 15 cases.
2. Step level: erroneous TEST solutions; predicted step = the step with the highest step score (max p_contradict
   of its claims; C: earliest failing step), ties -> earliest; exact match and within one step of the human first
   wrong step, with chance levels for the same solutions (uniform random step).
3. False alarms: threshold chosen on DEV correct solutions (the most sensitive threshold with <= 10% of them
   flagged; score > t), applied to TEST: share of erroneous flagged, false alarm rate, precision.
4. NLI label distribution (argmax) on TEST claims per source and claim type; B per (claim, sample) pair.
5. Complementarity (exploratory, tiny n): the existing pipeline = ALL (Stage 4 + 7) logistic regression out-of-fold
   probability, recomputed with stage7_evaluate.evaluate_set exactly as evaluate_all.py does per split (same folds
   and seed, so the AUROC matches results/all_splits_auroc.csv), thresholded at 10% false alarms on the correct
   solutions outside TEST (all splits pooled). On erroneous TEST solutions: how many the pipeline misses and each
   source flags (at its DEV threshold), and the reverse. No fused AUROC is reported.
6. Runtime per 1000 claims from results/rag_pilot_runtime.csv (benchmark rows if present).
The sanity cases gsm8k-0, gsm8k-34, gsm8k-202 are printed with every claim and score.
Labels are used only here, as targets; the evidence files contain no label.

Usage:
  python src/rag_pilot_evaluate.py      -> results/rag_pilot_eval.csv, results/logs/rag_pilot_evaluate.txt
"""
import argparse, csv, json, os, warnings
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from stage5_claims import SANITY_IDS, SPLITS, read_jsonl
from stage7_evaluate import ALL, boot_idx, evaluate_set, matrix

P = "data/processed"
MIN_HARD = 15
FA_TARGET = 0.10
NLI_SOURCES = ["A", "A2", "B", "D-bm25", "D-bge"]
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def load_sample(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["n_steps"] = int(r["n_steps"])
        r["hard"] = r["stratum"] == "hard"
    return {r["id"]: r for r in rows}


def load_evidence():
    ev = {}
    for s in ("A", "A2", "B"):
        path = f"{P}/rag_pilot_nli_{s}.jsonl"
        if os.path.exists(path):
            ev[s] = {r["key"]: r for r in read_jsonl(path)}
    path = f"{P}/rag_pilot_nli_D.jsonl"
    if os.path.exists(path):
        rows = read_jsonl(path)
        for ret in ("bm25", "bge"):
            ev[f"D-{ret}"] = {r["key"]: r for r in rows if r["retriever"] == ret}
    path = f"{P}/rag_pilot_numeric.jsonl"
    if os.path.exists(path):
        ev["C"] = {r["key"]: r for r in read_jsonl(path)}
    return ev


def solution_scores(source, claims, ev):
    """-> {'max_pc', '1-min_pe'} (NLI) or {'n_fail'} (C), and step -> step score (higher = more suspicious)."""
    e = ev[source]
    step = defaultdict(float)
    if source == "C":
        fails = [c["step"] for c in claims if e.get(f"{c['id']}|{c['step']}.{c['claim_idx']}", {}).get("status") == "fail"]
        for st in fails:
            step[st] = max(step[st], 1.0)
        return {"n_fail": float(len(fails))}, dict(step)
    pcs, pes = [], []
    for c in claims:
        r = e.get(f"{c['id']}|{c['step']}.{c['claim_idx']}")
        if r is None or r.get("p_contradict") is None:
            continue
        pcs.append(r["p_contradict"])
        pes.append(r["p_entail"])
        step[c["step"]] = max(step.get(c["step"], 0.0), r["p_contradict"])
    return {"max_pc": max(pcs, default=0.0), "1-min_pe": 1 - min(pes, default=1.0), "n_scored": len(pcs)}, dict(step)


def auroc_ci(y, s, boots):
    if len(set(y)) < 2:
        return None
    b = [roc_auc_score(y[i], s[i]) for i in boots]
    return roc_auc_score(y, s), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt_auc(r):
    return "n/a" if r is None else f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"


def dev_threshold(dev_correct_scores):
    """Most sensitive t with share(score > t) <= 10% among DEV correct solutions."""
    s = np.asarray(dev_correct_scores, dtype=float)
    for t in [-np.inf] + sorted(set(s)):
        if np.mean(s > t) <= FA_TARGET + 1e-12:
            return t, float(np.mean(s > t))
    return float(s.max()), 0.0


def pipeline_oof(sample):
    """ALL (Stage 4 + 7) LR out-of-fold probability per problem, per split as in evaluate_all.py."""
    out, check = {}, {}
    for sp in SPLITS:
        recs = read_jsonl(f"{P}/s7_{sp}.jsonl")
        y = np.array([r["label"] != -1 for r in recs], dtype=int)
        res = evaluate_set(matrix(recs, ALL), y, "LR", boot_idx(y))
        out.update({r["id"]: (p, r["label"]) for r, p in zip(recs, res["p"])})
        check[sp] = roc_auc_score(y, res["p"])
    return out, check


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="results/rag_pilot_sample.csv")
    ap.add_argument("--claims", default=f"{P}/rag_pilot_claims.jsonl")
    ap.add_argument("--out", default="results/rag_pilot_eval.csv")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    sample = load_sample(a.sample)
    claims = defaultdict(list)
    for c in read_jsonl(a.claims):
        claims[c["id"]].append(c)
    ev = load_evidence()
    sources = [s for s in ["A", "A2", "B", "C", "D-bm25", "D-bge"] if s in ev]
    pb_types = {}
    for sp in SPLITS:
        for r in read_jsonl(f"{P}/pb_{sp}.jsonl"):
            if r["id"] in sample or r["id"] in SANITY_IDS:
                pb_types[r["id"]] = [s["type"] for s in r["steps"]]

    feats, steps = {}, {}
    for s in sources:
        for i in sample:
            feats[(s, i)], steps[(s, i)] = solution_scores(s, claims[i], ev)
    variants = {s: (["n_fail"] if s == "C" else ["max_pc", "1-min_pe"]) for s in sources}
    test = [i for i, r in sample.items() if r["half"] == "TEST"]
    dev = [i for i, r in sample.items() if r["half"] == "DEV"]
    rows_out = []
    n_err = sum(sample[i]["label"] >= 0 for i in test)
    log(f"===== RAG pilot evaluation on TEST: {len(test)} solutions ({n_err} erroneous, {len(test) - n_err} correct, "
        f"{sum(sample[i]['hard'] for i in test)} hard); DEV ({len(dev)}) is used only for thresholds")
    log(f"sources with evidence: {', '.join(sources)}")
    n_cl = sum(len(claims[i]) for i in test)
    log(f"TEST claims: {n_cl}; claims per solution {n_cl / len(test):.1f}")

    # ---------------------------------------------------------------- 1. solution level
    log("\n## 1. Solution level: AUROC [95% bootstrap CI] vs process error (label >= 0)\n")
    groups = [("TEST all", test)] + [(f"TEST {sp}", [i for i in test if sample[i]["split"] == sp]) for sp in SPLITS]
    hard_ids = [i for i in test if sample[i]["hard"] or sample[i]["label"] == -1]
    n_hard = sum(sample[i]["hard"] for i in test)
    log("| source | score | " + " | ".join(f"{g} (n={len(ids)})" for g, ids in groups) +
        f" | HARD vs correct ({n_hard} / {len(test) - n_err}) |")
    log("|---|---|" + "---|" * (len(groups) + 1))
    for s in sources:
        for v in variants[s]:
            cells = []
            for g, ids in groups + [("HARD", hard_ids)]:
                y = np.array([sample[i]["label"] >= 0 for i in ids], dtype=int)
                x = np.array([feats[(s, i)][v] for i in ids])
                if g == "HARD" and n_hard < MIN_HARD:
                    cells.append(f"too small ({n_hard} < {MIN_HARD})")
                    continue
                r = auroc_ci(y, x, boot_idx(y)) if len(set(x)) > 1 else None
                cells.append(fmt_auc(r) if r else "constant score")
                rows_out.append({"section": "auroc", "source": s, "score": v, "group": g, "n": len(ids),
                                 "n_pos": int(y.sum()), "value": round(r[0], 3) if r else "",
                                 "ci_low": round(r[1], 3) if r else "", "ci_high": round(r[2], 3) if r else ""})
            log(f"| {s} | {v} | " + " | ".join(cells) + " |")
    for s in sources:
        if s.startswith("D"):
            with_k = [i for i in test if any(t == "knowledge_claim" for t in pb_types[i])]
            log(f"\n{s}: only {len(with_k)} of {len(test)} TEST solutions have a knowledge_claim step "
                f"({sum(sample[i]['label'] >= 0 for i in with_k)} erroneous); all others score 0, so its AUROC above "
                f"mostly measures who has such a step, not evidence")

    # ---------------------------------------------------------------- 2. step level
    log("\n## 2. Step level (erroneous TEST solutions): highest-scoring step vs the human first wrong step\n")
    err = [i for i in test if sample[i]["label"] >= 0]
    ch_exact = 100 * np.mean([1 / sample[i]["n_steps"] for i in err])
    ch_within = 100 * np.mean([len({sample[i]["label"] + d for d in (-1, 0, 1)} & set(range(sample[i]["n_steps"])))
                               / sample[i]["n_steps"] for i in err])
    log(f"n = {len(err)}; chance: exact {ch_exact:.1f}%, within one step {ch_within:.1f}%\n")
    log("| source | solutions with a prediction | exact | within one step |")
    log("|---|---|---|---|")
    for s in sources:
        exact = within = has = 0
        for i in err:
            st = steps[(s, i)]
            if not st or max(st.values()) <= 0:
                continue
            has += 1
            best = max(st.values())
            pred = min(k for k, v in st.items() if v == best)
            exact += pred == sample[i]["label"]
            within += abs(pred - sample[i]["label"]) <= 1
        log(f"| {s} | {has} | {exact} ({100 * exact / len(err):.1f}%) | {within} ({100 * within / len(err):.1f}%) |")
        rows_out += [{"section": "step", "source": s, "score": "exact", "group": "TEST erroneous", "n": len(err),
                      "value": round(100 * exact / len(err), 1)},
                     {"section": "step", "source": s, "score": "within1", "group": "TEST erroneous", "n": len(err),
                      "value": round(100 * within / len(err), 1)}]
    rows_out += [{"section": "step", "source": "chance", "score": "exact", "group": "TEST erroneous", "n": len(err),
                  "value": round(ch_exact, 1)},
                 {"section": "step", "source": "chance", "score": "within1", "group": "TEST erroneous", "n": len(err),
                  "value": round(ch_within, 1)}]

    # ---------------------------------------------------------------- 3. false alarms at a DEV threshold
    log(f"\n## 3. Threshold chosen on DEV correct solutions (<= {100 * FA_TARGET:.0f}% flagged), applied to TEST\n")
    log("| source | score | threshold | DEV false alarms | TEST erroneous flagged | TEST false alarms | TEST precision |")
    log("|---|---|---|---|---|---|---|")
    flags = {}
    dev_ok = [i for i in dev if sample[i]["label"] == -1]
    test_ok = [i for i in test if sample[i]["label"] == -1]
    for s in sources:
        for v in variants[s]:
            t, dev_fa = dev_threshold([feats[(s, i)][v] for i in dev_ok])
            fl = {i: feats[(s, i)][v] > t for i in test}
            flags[(s, v)] = fl
            tp, fp = sum(fl[i] for i in err), sum(fl[i] for i in test_ok)
            prec = f"{100 * tp / (tp + fp):.1f}%" if tp + fp else "n/a"
            log(f"| {s} | {v} | {t:.4g} | {100 * dev_fa:.1f}% | {tp}/{len(err)} ({100 * tp / len(err):.1f}%) | "
                f"{fp}/{len(test_ok)} ({100 * fp / len(test_ok):.1f}%) | {prec} ({tp}/{tp + fp}) |")
            rows_out.append({"section": "threshold", "source": s, "score": v, "group": "TEST", "n": len(test),
                             "value": round(100 * tp / len(err), 1), "threshold": t, "dev_fa_pct": round(100 * dev_fa, 1),
                             "test_fa_pct": round(100 * fp / len(test_ok), 1),
                             "precision_pct": round(100 * tp / (tp + fp), 1) if tp + fp else ""})

    # ---------------------------------------------------------------- 4. NLI label distribution
    log("\n## 4. NLI argmax label on TEST claims (share entail / neutral / contradict), by claim type\n")
    log("| source | claim type | n | entail | neutral | contradict |")
    log("|---|---|---|---|---|---|")
    for s in [x for x in sources if x != "C"]:
        e = ev[s]
        by = defaultdict(Counter)
        for i in test:
            for c in claims[i]:
                r = e.get(f"{i}|{c['step']}.{c['claim_idx']}")
                if r is None:
                    continue
                probs = list(r["per_sample"].values()) if s == "B" else (r["per_passage"] if s.startswith("D") else
                                                                         [(r["p_entail"], r["p_neutral"], r["p_contradict"])])
                for p in probs:
                    lab = ("entail", "neutral", "contradict")[int(np.argmax(p))]
                    by[c["type"]][lab] += 1
                    by["all"][lab] += 1
        for t in ["computational", "reasoning", "knowledge_claim", "all"]:
            n = sum(by[t].values())
            if n:
                log(f"| {s} | {t} | {n} | " + " | ".join(f"{100 * by[t][k] / n:.1f}%" for k in ("entail", "neutral", "contradict")) + " |")
                rows_out.append({"section": "nli_labels", "source": s, "score": t, "group": "TEST claims", "n": n,
                                 "value": round(100 * by[t]["neutral"] / n, 1),
                                 "entail_pct": round(100 * by[t]["entail"] / n, 1),
                                 "contradict_pct": round(100 * by[t]["contradict"] / n, 1)})
    log("(B: one count per (claim, sample) pair; D: one per (claim, passage) pair)")
    if "C" in ev:
        cnt = Counter(ev["C"][f"{i}|{c['step']}.{c['claim_idx']}"]["status"] for i in test for c in claims[i])
        log(f"C claim status on TEST claims: {dict(cnt)}")

    # ---------------------------------------------------------------- 4b. A2: truncated vs not
    if "A2" in ev:
        e = ev["A2"]
        mode = lambda i, c: e.get(f"{i}|{c['step']}.{c['claim_idx']}", {}).get("premise_mode")
        log("\n## 4b. A2 premise length: problem + all earlier steps + claim over 512 tokens?\n")
        log("(shortening: keep the problem and the nearest earlier steps, drop the oldest steps first ('dropped_oldest');"
            " if even problem + nearest step is too long, the end of the premise is cut ('cut'); the claim is never cut)\n")
        log("| group | claims | full | dropped_oldest | cut | share over the limit |")
        log("|---|---|---|---|---|---|")
        for g, ids in [("all pilot", list(sample))] + [(f"pilot {sp}", [i for i in sample if sample[i]["split"] == sp])
                                                       for sp in SPLITS] + [("TEST", test)]:
            m = Counter(mode(i, c) for i in ids for c in claims[i])
            n = sum(m.values())
            log(f"| {g} | {n} | {m['full']} | {m['dropped_oldest']} | {m['cut']} | "
                f"{100 * (m['dropped_oldest'] + m['cut']) / n:.1f}% |")
        log("\nA2 solution scores from one kind of pair only (TEST; a solution without such pairs scores 0):\n")
        log("| pairs used | solutions with >= 1 such pair (erroneous / correct) | score | AUROC [95% CI] on those solutions |")
        log("|---|---|---|---|")
        for kind, keep in [("not truncated (full)", lambda m: m == "full"),
                           ("truncated (dropped_oldest or cut)", lambda m: m in ("dropped_oldest", "cut"))]:
            ids = [i for i in test if any(keep(mode(i, c)) for c in claims[i])]
            y = np.array([sample[i]["label"] >= 0 for i in ids], dtype=int)
            for v in ("max_pc", "1-min_pe"):
                x = np.array([solution_scores("A2", [c for c in claims[i] if keep(mode(i, c))], ev)[0][v] for i in ids])
                r = auroc_ci(y, x, boot_idx(y)) if 0 < y.sum() < len(y) and len(set(x)) > 1 else None
                log(f"| {kind} | {len(ids)} ({int(y.sum())} / {len(ids) - int(y.sum())}) | {v} | {fmt_auc(r)} |")
                rows_out.append({"section": "a2_truncation", "source": "A2", "score": v, "group": kind, "n": len(ids),
                                 "n_pos": int(y.sum()), "value": round(r[0], 3) if r else "",
                                 "ci_low": round(r[1], 3) if r else "", "ci_high": round(r[2], 3) if r else ""})
        log("(truncated pairs belong to late steps of long solutions, so the two groups differ in more than truncation)")
        by = defaultdict(Counter)
        for i in test:
            for c in claims[i]:
                r = e.get(f"{i}|{c['step']}.{c['claim_idx']}")
                if r:
                    k = "full" if r.get("premise_mode") == "full" else "truncated"
                    by[k][("entail", "neutral", "contradict")[int(np.argmax((r["p_entail"], r["p_neutral"], r["p_contradict"])))]] += 1
        for k, cnt in by.items():
            n = sum(cnt.values())
            log(f"A2 {k} pairs on TEST: n={n}, entail {100 * cnt['entail'] / n:.1f}%, neutral {100 * cnt['neutral'] / n:.1f}%, "
                f"contradict {100 * cnt['contradict'] / n:.1f}%")

    # ---------------------------------------------------------------- 5. complementarity with the existing pipeline
    log("\n## 5. Complementarity with the existing pipeline (EXPLORATORY: tiny n, no conclusions)\n")
    oof, check = pipeline_oof(sample)
    log("pipeline = ALL (Stage 4 + 7) LR out-of-fold probability; per-split AUROC (should match all_splits_auroc.csv): " +
        ", ".join(f"{sp} {v:.3f}" for sp, v in check.items()))
    ref = [p for i, (p, lab) in oof.items() if lab == -1 and sample.get(i, {}).get("half") != "TEST"]
    t_pipe, fa_pipe = dev_threshold(ref)
    pipe = {i: oof[i][0] > t_pipe for i in test}
    tp = sum(pipe[i] for i in err)
    fp = sum(pipe[i] for i in test_ok)
    log(f"pipeline threshold {t_pipe:.3f} (10% false alarms on {len(ref)} correct solutions outside TEST: {100 * fa_pipe:.1f}%); "
        f"on TEST it flags {tp}/{len(err)} erroneous and {fp}/{len(test_ok)} correct")
    missed = [i for i in err if not pipe[i]]
    caught = [i for i in err if pipe[i]]
    log(f"\nerroneous TEST solutions missed by the pipeline: {len(missed)} "
        f"({sum(sample[i]['hard'] for i in missed)} of them hard)\n")
    log("| source | score | flags among the pipeline's misses | misses among the pipeline's catches | TEST false alarms |")
    log("|---|---|---|---|---|")
    for (s, v), fl in flags.items():
        log(f"| {s} | {v} | {sum(fl[i] for i in missed)}/{len(missed)} | {sum(not fl[i] for i in caught)}/{len(caught)} | "
            f"{sum(fl[i] for i in test_ok)}/{len(test_ok)} |")
        rows_out.append({"section": "complementarity", "source": s, "score": v, "group": "TEST erroneous missed by pipeline",
                         "n": len(missed), "value": sum(fl[i] for i in missed)})

    # ---------------------------------------------------------------- 6. runtime
    log("\n## 6. Runtime per 1000 claims (CPU)\n")
    rt = "results/rag_pilot_runtime.csv"
    if os.path.exists(rt):
        with open(rt, encoding="utf-8", newline="") as f:
            rr = list(csv.DictReader(f))
        bench = [r for r in rr if r["note"].startswith("benchmark")]
        use = bench or rr
        log(f"({'benchmark rows: each source timed alone on a fixed claim subset' if bench else 'full-run rows; other jobs shared the CPU, so these are upper bounds'})\n")
        log("| source | model | claims | pairs | seconds | s per 1000 claims | threads | note |")
        log("|---|---|---|---|---|---|---|---|")
        for r in use:
            log(f"| {r['source']} | {r['model']} | {r['n_claims']} | {r['n_pairs']} | {r['seconds']} | "
                f"{r['sec_per_1000_claims']} | {r['cpu_threads']} | {r['note']} |")

    # ---------------------------------------------------------------- sanity cases
    log("\n## Sanity cases: every claim with its scores (A, A2, B: p_entail / p_contradict; C: status)\n")
    for i in SANITY_IDS:
        log(f"{i} ({'pilot ' + sample[i]['half'] if i in sample else 'not in the pilot sample'})")
        for c in claims[i]:
            key = f"{i}|{c['step']}.{c['claim_idx']}"
            parts = []
            for s in ("A", "A2", "B"):
                r = ev.get(s, {}).get(key)
                parts.append(f"{s} {r['p_entail']:.2f}/{r['p_contradict']:.2f}" if r and r["p_entail"] is not None else f"{s} -")
            parts.append(f"C {ev.get('C', {}).get(key, {}).get('status', '-')}")
            log(f"  {c['step']}.{c['claim_idx']} [{c['type'][:4]}] {' | '.join(parts)} | {c['text'][:110]}")
        log()

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows_out for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/rag_pilot_evaluate.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"\n-> {a.out}, results/logs/rag_pilot_evaluate.txt")
