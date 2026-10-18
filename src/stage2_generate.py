"""Stage 2: sample N solutions per problem (for the zero-resource signals in Stage 7).

For every problem we store N sampled solutions, each with
  - final_answer       (last \\boxed{...}, else last number)
  - mean_logprob       (average log-probability of the generated tokens)
  - mean_entropy       (average next-token entropy while generating)
  - n_tokens

Usage (GPU, e.g. Colab):
  python stage2_generate.py --input data/processed/pb_gsm8k.jsonl --output data/processed/samples_gsm8k.jsonl \
         --model Qwen/Qwen2.5-Math-1.5B-Instruct --n 5 --limit 400 --batch-size 2

Smoke test without any model:
  python stage2_generate.py --backend mock --input ... --output ... --limit 5

The script is resumable: problems already present in --output are skipped.
"""
import argparse, json, os, random, re, time
from fractions import Fraction

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------------------------------------------------------------- answers
def extract_answer(text: str) -> str:
    """Last \\boxed{...} (handles nested braces); otherwise the last number in the text."""
    text = str(text)
    idx = text.rfind("\\boxed{")
    if idx != -1:
        i, depth, start = idx + len("\\boxed{"), 1, idx + len("\\boxed{")
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            return text[start:i - 1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


_NUM = r"-?\\?\$?\s*\d[\d,]*(?:\.\d+)?"
_UNIT = r"[A-Za-z]{2,}[A-Za-z./ ]*"


def normalize_answer(a: str) -> str:
    """Light normalisation so '18', '18.0', '$18$', '\\$18.00', '18 hours' and '\\text{18}' compare equal."""
    a = str(a).strip()
    # trailing unit word, only when separated by a space or \text{}: '53 minutes', '5 \text{hours}' (not '2x')
    m = re.fullmatch(rf"({_NUM})\s*\\(?:text|mbox|mathrm)\{{\s*{_UNIT}\}}|({_NUM})\s+{_UNIT}", a)
    if m:
        a = m[1] or m[2]
    # mixed number '1 and 1/3', '1 \text{ and } 1/3' -> 4/3 (only if the fraction is proper)
    m = re.fullmatch(r"(\d+)\s*(?:\\text\{\s*and\s*\}|and)\s*(\d+)\s*/\s*(\d+)", a)
    if m and 0 < int(m[2]) < int(m[3]):
        return str(int(m[1]) + Fraction(int(m[2]), int(m[3])))
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = a.replace("\\$", "").replace("$", "").replace("\\!", "").replace(",", "")
    a = re.sub(r"\s+", "", a)                             # spaces and newlines ('\begin{pmatrix}\n...')
    a = re.sub(r"^(-?)\\(?=\d)", r"\1", a)               # stray leading backslash: '\15' (not '\\' or '\ ' inside)
    try:
        f = float(a)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):                   # '777...7' (80 digits) overflows to inf
        return a


# ---------------------------------------------------------------- backends
class MockBackend:
    """Fake model for testing the pipeline end to end (no GPU, no download)."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def generate(self, problems, n):
        out = []
        for p in problems:
            samples = []
            for _ in range(n):
                ans = self.rng.choice(["18", "18", "18", "24", "30"])
                samples.append({"text": f"Step 1 ... so the answer is \\boxed{{{ans}}}.",
                                "mean_logprob": -self.rng.random(), "mean_entropy": self.rng.random(),
                                "n_tokens": self.rng.randint(80, 300)})
            out.append(samples)
        return out


class HFBackend:
    def __init__(self, model_name, temperature, top_p, max_new_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: no GPU found. Sampling on CPU will be very slow; use a GPU (e.g. Colab).")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        eos = [e for e in eos if e is not None] + ([self.tok.eos_token_id] if self.tok.eos_token_id is not None else [])
        self.eos_ids = sorted(set(eos))
        self.temperature, self.top_p, self.max_new_tokens = temperature, top_p, max_new_tokens

    def _prompt(self, problem):
        if getattr(self.tok, "chat_template", None):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"{SYSTEM_PROMPT}\n\nProblem: {problem}\n\nSolution:"

    def generate(self, problems, n):
        torch = self.torch
        enc = self.tok([self._prompt(p) for p in problems], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, do_sample=True, temperature=self.temperature, top_p=self.top_p,
                max_new_tokens=self.max_new_tokens, num_return_sequences=n,
                pad_token_id=self.tok.pad_token_id, return_dict_in_generate=True, output_logits=True)
        prompt_len = enc["input_ids"].shape[1]
        gen = out.sequences[:, prompt_len:]                                   # (B*n, T)
        logits = out.logits                                                   # tuple of T tensors (B*n, V)
        eos = torch.tensor(self.eos_ids, device=gen.device)
        is_eos = torch.isin(gen, eos).long()
        valid = (torch.cumsum(is_eos, 1) - is_eos) == 0                       # tokens up to and incl. first EOS
        lp_sum = torch.zeros(gen.shape[0], device=gen.device)
        ent_sum = torch.zeros_like(lp_sum)
        for t, lg in enumerate(logits):
            logp = torch.log_softmax(lg.float(), dim=-1)
            tok_lp = logp.gather(1, gen[:, t:t + 1])[:, 0]
            ent = -(logp.exp() * logp).sum(-1)
            m = valid[:, t].float()
            lp_sum += tok_lp * m
            ent_sum += ent * m
        n_tok = valid.sum(1).clamp(min=1)
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        results = []
        for b in range(len(problems)):
            samples = []
            for j in range(n):
                k = b * n + j
                samples.append({"text": texts[k], "mean_logprob": float(lp_sum[k] / n_tok[k]),
                                "mean_entropy": float(ent_sum[k] / n_tok[k]), "n_tokens": int(n_tok[k])})
            results.append(samples)
        return results


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="problems per batch (each gives n sequences). Lower it if you hit CUDA out-of-memory.")
    ap.add_argument("--limit", type=int, default=None, help="only the first K problems")
    ap.add_argument("--sample", type=int, default=None,
                    help="random K problems (use this for MATH/Olympiad/Omni-MATH: the files are not shuffled)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    if a.backend == "hf":
        import torch
        torch.manual_seed(a.seed)
        backend = HFBackend(a.model, a.temperature, a.top_p, a.max_new_tokens)
    else:
        backend = MockBackend(a.seed)

    with open(a.input, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    if a.sample:
        records = random.Random(a.seed).sample(records, min(a.sample, len(records)))
    elif a.limit:
        records = records[:a.limit]

    done = set()
    if os.path.exists(a.output):
        with open(a.output, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(records)} problems, {len(done)} already done, {len(todo)} to do")

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    t0 = time.time()
    with open(a.output, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), a.batch_size):
            batch = todo[i:i + a.batch_size]
            res = backend.generate([r["problem"] for r in batch], a.n)
            for r, samples in zip(batch, res):
                for s in samples:
                    s["final_answer"] = extract_answer(s["text"])
                out.write(json.dumps({"id": r["id"], "problem": r["problem"], "samples": samples,
                                      "model": a.model if a.backend == "hf" else "mock",
                                      "temperature": a.temperature, "n": a.n}) + "\n")
            out.flush()
            d = min(i + a.batch_size, len(todo))
            el = time.time() - t0
            print(f"  {d}/{len(todo)} problems | {el/60:.1f} min elapsed | ETA {el/d*(len(todo)-d)/60:.1f} min")
    print("done ->", a.output)


if __name__ == "__main__":
    main()
