# ternary_qat_3b.py — PrismML-style ternary compression of a ~3B LLM | Colab Pro A100-80GB
#
# What it does (same idea as PrismML Bonsai, at Colab scale):
#   1. teacher = a real pretrained 3B model (bf16, frozen)
#   2. student = a copy of it where every Linear layer in the transformer blocks is a BitLinear:
#      weights are forced to {-s, 0, +s}, one FP16-style scale s per group of 128 weights (1.58-bit)
#   3. student is re-trained to copy the teacher (distillation) on Telugu text  -> quantization-aware training
#   4. export packs the ternary weights 4-per-byte + scales -> the real compressed file
#   5. eval reloads the packed file and compares it with the teacher (perplexity + sample answers)
#
# Stages (run one per Colab cell):
#   python ternary_qat_3b.py train  --out /content/drive/MyDrive/ternary-3b
#   python ternary_qat_3b.py export --out /content/drive/MyDrive/ternary-3b
#   python ternary_qat_3b.py eval   --out /content/drive/MyDrive/ternary-3b
# Training resumes automatically from the last checkpoint in --out (Colab disconnects are expected).

import argparse, json, math, os, time
import torch, torch.nn as nn
from torch.nn import functional as F

GROUP = 128


# ── BitLinear: the core of the compression ──────────────────────────────────────
class BitLinear(nn.Linear):
    """Linear layer that runs with ternary weights {-s, 0, +s}, one scale s per GROUP inputs.
    The full-precision weight is kept only for training (straight-through estimator)."""

    def __init__(self, in_features, out_features, bias=False, group=GROUP, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        assert in_features % group == 0, f"in_features={in_features} not divisible by group={group}"
        self.group = group
        # 0 → plain full-precision layer, 1 → fully ternary. Ramped up during training for stability.
        self.register_buffer("strength", torch.tensor(1.0), persistent=False)

    @staticmethod
    def ternarize(w, group=GROUP):
        """w [out, in] -> codes in {-1, 0, 1} shaped [n_groups, group] and scales [n_groups, 1]."""
        g = w.float().reshape(-1, group)
        s = g.abs().mean(1, keepdim=True).clamp(min=1e-5)
        return (g / s).round().clamp(-1, 1), s

    def forward(self, x):
        w = self.weight
        codes, s = self.ternarize(w, self.group)
        q = (codes * s).reshape_as(w).to(w.dtype)
        w_q = w + self.strength * (q - w).detach()   # forward uses ternary, gradient flows to w
        return F.linear(x, w_q, self.bias)


def to_bitlinear(model, skip=("lm_head",)):
    """Swap every nn.Linear (except lm_head) for a BitLinear that shares the same weights."""
    n = 0
    for parent_name, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            full = f"{parent_name}.{name}" if parent_name else name
            if type(child) is not nn.Linear or any(s in full for s in skip):
                continue
            new = BitLinear(child.in_features, child.out_features, bias=child.bias is not None, device="meta")
            new.weight = child.weight
            if child.bias is not None:
                new.bias = child.bias
            new.strength = torch.tensor(1.0, device=child.weight.device)
            setattr(parent, name, new)
            n += 1
    return n


def set_strength(model, value):
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.strength.fill_(value)


# ── data: same random-window batching as tiny_llm_*.py, on real tokens ─────────
def load_texts(args, tok):
    texts = []
    if args.dataset:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        texts += [r[args.text_field] for r in ds if r[args.text_field]]
    if args.qa_jsonl:   # lines like {"question": "...", "answer": "..."}
        for line in open(args.qa_jsonl, encoding="utf-8"):
            r = json.loads(line)
            msgs = [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False))
    assert texts, "no training text: set --dataset and/or --qa-jsonl"
    return texts


def tokenize_split(texts, tok, eval_frac=0.02):
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(tok.eos_token_id)
    data = torch.tensor(ids, dtype=torch.long)
    cut = int(len(data) * (1 - eval_frac))
    return data[:cut], data[cut:]


def get_batches(data, seq_len, batch_size, seed=1337):
    g = torch.Generator().manual_seed(seed)
    while True:
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=g)
        yield torch.stack([data[i:i + seq_len] for i in ix])


# ── training ────────────────────────────────────────────────────────────────────
def distill_loss(s_logits, t_logits, x, T=1.0, alpha=0.5):
    """KL(teacher || student) on every position + next-token cross-entropy on the real text."""
    V = s_logits.size(-1)
    s = s_logits.float() / T
    t = t_logits.float() / T
    kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True,
                  reduction="none").sum(-1).mean() * T * T
    ce = F.cross_entropy(s_logits[:, :-1].float().reshape(-1, V), x[:, 1:].reshape(-1))
    return (1 - alpha) * kl + alpha * ce, kl, ce


def make_optimizer(params, lr):
    try:
        import bitsandbytes as bnb   # 8-bit Adam: ~6 GB of optimizer state instead of ~24 GB for 3B
        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    except ImportError:
        print("bitsandbytes not installed — falling back to torch AdamW (needs ~24 GB more GPU memory for 3B)")
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)


def run_qat(teacher, student, batches, cfg, out_dir, device):
    """Distillation + quantization-aware training loop with checkpoint/resume. Returns final step."""
    opt = make_optimizer(student.parameters(), cfg["lr"])
    total, warmup = cfg["steps"], cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    state_path = os.path.join(out_dir, "train_state.pt")
    step = 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        student.load_state_dict({k: v.to(torch.float32) for k, v in st["weights"].items()})
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        print(f"resumed from step {step}")

    def save():
        tmp = state_path + ".tmp"
        torch.save({"weights": {k: v.detach().to("cpu", torch.bfloat16) for k, v in student.state_dict().items()},
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "cfg": cfg}, tmp)
        os.replace(tmp, state_path)   # never leave a half-written checkpoint behind
        print(f"  checkpoint saved at step {step} -> {state_path}")

    student.train()
    tokens_per_step = cfg["batch_size"] * cfg["seq_len"] * cfg["grad_accum"]
    t0, last_save = time.time(), time.time()
    while step < total:
        strength = min(1.0, step / max(1, cfg["ramp_steps"]))
        set_strength(student, strength)
        for _ in range(cfg["grad_accum"]):
            x = next(batches).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    t_logits = teacher(x).logits
                s_logits = student(x).logits
            loss, kl, ce = distill_loss(s_logits, t_logits, x, cfg["temperature"], cfg["alpha"])
            (loss / cfg["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg["log_every"] == 0 or step == 1:
            el = time.time() - t0
            tps = tokens_per_step * cfg["log_every"] / el if step > 1 else 0
            eta = (total - step) * tokens_per_step / tps / 3600 if tps else float("nan")
            print(f"step {step:6d}/{total}  loss {loss.item():.4f}  kl {kl.item():.4f}  ce {ce.item():.4f}  "
                  f"ternary {strength:.2f}  {tps:,.0f} tok/s  ~{eta:.1f} h left")
            t0 = time.time()
        if step % cfg["save_every"] == 0 or time.time() - last_save > cfg["save_minutes"] * 60 or step == total:
            save(); last_save = time.time()
    set_strength(student, 1.0)
    return step


# ── export: pack ternary codes 4-per-byte (2 bits each) + scales ────────────────
def pack_codes(codes):
    c = (codes.flatten().to(torch.int16) + 1).to(torch.uint8).reshape(-1, 4)   # {-1,0,1} -> {0,1,2}
    return c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)


def unpack_codes(packed):
    p = packed.to(torch.uint8)
    c = torch.stack([(p >> sh) & 3 for sh in (0, 2, 4, 6)], 1).flatten()
    return c.to(torch.int8) - 1


def export_packed(student, embed_bits=8):
    """Returns a dict holding the compressed model and prints the size breakdown."""
    packed, other, sizes = {}, {}, {"ternary": 0, "scales": 0, "other": 0}
    bitlinear_weights = {f"{n}.weight" for n, m in student.named_modules() if isinstance(m, BitLinear)}
    for name, p in student.named_parameters():   # named_parameters() lists tied weights once
        if name in bitlinear_weights:
            codes, s = BitLinear.ternarize(p.detach(), GROUP)
            packed[name] = {"codes": pack_codes(codes).cpu(), "scales": s.squeeze(1).half().cpu(),
                            "shape": tuple(p.shape)}
            sizes["ternary"] += packed[name]["codes"].numel()
            sizes["scales"] += packed[name]["scales"].numel() * 2
        elif p.dim() == 2 and embed_bits == 8:   # embeddings / untied lm_head: int8 with one scale per row
            w = p.detach().float()
            rs = (w.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8)
            other[name] = {"int8": (w / rs).round().clamp(-127, 127).to(torch.int8).cpu(),
                           "row_scale": rs.squeeze(1).half().cpu()}
            sizes["other"] += w.numel() + w.size(0) * 2
        else:
            other[name] = p.detach().to("cpu", torch.float16)
            sizes["other"] += p.numel() * 2
    return {"packed": packed, "other": other, "group": GROUP}, sizes


def unpacked_state_dict(blob):
    sd = {}
    for name, d in blob["packed"].items():
        n = math.prod(d["shape"])
        codes = unpack_codes(d["codes"])[:n].float().reshape(-1, blob["group"])
        sd[name] = (codes * d["scales"].float().unsqueeze(1)).reshape(d["shape"])
    for name, v in blob["other"].items():
        sd[name] = v["int8"].float() * v["row_scale"].float().unsqueeze(1) if isinstance(v, dict) else v.float()
    return sd


def load_compressed(blob, config, dtype=torch.bfloat16):
    """Build a normal Hugging Face model from the packed file (weights already ternary)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    missing, unexpected = model.load_state_dict(unpacked_state_dict(blob), strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]   # lm_head is tied to the embeddings
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    model.tie_weights()
    return model


# ── eval helpers ────────────────────────────────────────────────────────────────
@torch.no_grad()
def perplexity(model, data, seq_len, n_batches, batch_size, device):
    model.eval()
    gen, losses = get_batches(data, seq_len, batch_size, seed=7), []
    for _ in range(n_batches):
        x = next(gen).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x).logits
        losses.append(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1)).item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def answer(model, tok, question, device, max_new_tokens=120):
    msgs = [{"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = {k: v.to(device) for k, v in ids.items()}
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "export", "eval"])
    ap.add_argument("--teacher", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="any HF causal LM, e.g. your own Telugu fine-tuned 3B model folder")
    ap.add_argument("--out", default="/content/drive/MyDrive/ternary-3b")
    ap.add_argument("--dataset", default="ai4bharat/samanantar")
    ap.add_argument("--dataset-config", default="te")
    ap.add_argument("--split", default="train[:200000]")
    ap.add_argument("--text-field", default="tgt")
    ap.add_argument("--qa-jsonl", default=None, help="your Telugu Q&A pairs, one JSON per line")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ramp-steps", type=int, default=500, help="steps to go from full precision to fully ternary")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="0 = only copy teacher, 1 = only real text")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--embed-bits", type=int, default=8, choices=[8, 16])
    ap.add_argument("--prompts", nargs="*", default=[
        "తెలుగు భాష గురించి రెండు వాక్యాలు చెప్పండి.",
        "బండి ఇంజిన్ ఆయిల్ ఎప్పుడు మార్చాలి?",
        "భారతదేశ రాజధాని ఏది?"])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.teacher)
    packed_path = os.path.join(args.out, "ternary_model.pt")

    if args.stage == "train":
        train_data, _ = tokenize_split(load_texts(args, tok), tok)
        print(f"training tokens available: {len(train_data):,}")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device).eval()
        teacher.requires_grad_(False)
        student = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32).to(device)
        student.config.use_cache = False
        student.gradient_checkpointing_enable()
        n = to_bitlinear(student)
        print(f"params={sum(p.numel() for p in student.parameters()):,} | BitLinear layers={n}")
        cfg = {k: getattr(args, k) for k in ["steps", "warmup", "ramp_steps", "lr", "batch_size", "grad_accum",
                                             "seq_len", "temperature", "alpha", "log_every", "save_every",
                                             "save_minutes"]}
        run_qat(teacher, student, get_batches(train_data, args.seq_len, args.batch_size), cfg, args.out, device)

    elif args.stage == "export":
        student = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32)
        to_bitlinear(student)
        st = torch.load(os.path.join(args.out, "train_state.pt"), map_location="cpu", weights_only=False)
        student.load_state_dict({k: v.float() for k, v in st["weights"].items()})
        blob, sizes = export_packed(student, args.embed_bits)
        blob["teacher"] = args.teacher
        torch.save(blob, packed_path)
        n_params = sum(p.numel() for p in student.parameters())
        print(f"bf16 size      : {n_params * 2 / 1e9:.2f} GB")
        print(f"ternary weights: {sizes['ternary'] / 1e9:.2f} GB | scales {sizes['scales'] / 1e9:.3f} GB | "
              f"embeddings+norms {sizes['other'] / 1e9:.2f} GB")
        print(f"compressed file: {os.path.getsize(packed_path) / 1e9:.2f} GB -> {packed_path}")

    else:  # eval
        _, eval_data = tokenize_split(load_texts(args, tok), tok)
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device)
        blob = torch.load(packed_path, map_location="cpu", weights_only=False)
        compressed = load_compressed(blob, AutoConfig.from_pretrained(args.teacher)).to(device)
        for name, m in [("teacher (bf16)", teacher), ("ternary", compressed)]:
            print(f"{name:15s} Telugu perplexity: {perplexity(m, eval_data, args.seq_len, 20, 4, device):.2f}")
        for q in args.prompts:
            print(f"\nQ: {q}\n  teacher: {answer(teacher, tok, q, device)}\n  ternary: {answer(compressed, tok, q, device)}")


if __name__ == "__main__":
    main()
