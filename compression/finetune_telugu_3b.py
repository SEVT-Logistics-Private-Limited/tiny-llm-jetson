# finetune_telugu_3b.py — teach a real 3B model Telugu Q&A with LoRA | Colab Pro A100-80GB
#
# Step 1 of the 3B pipeline:  finetune_telugu_3b.py (this file)  ->  ternary_qat_3b.py (compression)
#
#   1. base   = a pretrained 3B chat model in bf16 (default meta-llama/Llama-3.2-3B-Instruct)
#   2. LoRA   = small trainable add-on matrices on every attention/MLP layer (~1-3% of the weights)
#   3. train  = on your Telugu Q&A pairs (loss only on the answer tokens), optionally mixed with raw Telugu text
#   4. merge  = fold LoRA into the base -> a normal Hugging Face model folder = --teacher for ternary_qat_3b.py
#   5. eval   = base vs fine-tuned on held-out questions: answer loss + how much of each answer is in Telugu script
#
# Stages (one per Colab cell):
#   python finetune_telugu_3b.py train --qa /content/drive/MyDrive/telugu_qa.jsonl --out /content/drive/MyDrive/telugu-3b
#   python finetune_telugu_3b.py merge --out /content/drive/MyDrive/telugu-3b
#   python finetune_telugu_3b.py eval  --qa /content/drive/MyDrive/telugu_qa.jsonl --out /content/drive/MyDrive/telugu-3b
# Training resumes automatically from the last checkpoint in --out.

import argparse, csv, json, math, os, random, time
import torch
from torch.nn import functional as F

# Same voice as voice-assistant/api.py build_system(), minus the live time/date line.
SYSTEM_PROMPT = (
    "నువ్వు యోధ — ఒక స్నేహితుడిలాంటి తెలుగు AI అసిస్టెంట్‌వి.\n"
    "— ప్రశ్నకు తగినట్టు జవాబు ఇవ్వు: చిన్న ప్రశ్నకు చిన్న జవాబు, వివరణ అవసరమైతే స్పష్టంగా చెప్పు.\n"
    "— ప్రధానంగా తెలుగులో మాట్లాడు. Technical terms లేదా common English పదాలు అవసరమైతే వాడవచ్చు.\n"
    "— *, #, _, / లాంటి symbols వాడకు — మాటల్లో సహజంగా చెప్పు.\n"
    "— తెలియని విషయాలు తెలియదని చెప్పు — imagine చేయకు.\n"
)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
KEY_PAIRS = [("question", "answer"), ("instruction", "output"), ("prompt", "response"), ("input", "output")]
IGNORE = -100


# ── data ────────────────────────────────────────────────────────────────────────
def read_rows(path):
    """.jsonl or .csv -> list of chat message lists. Accepts {"messages": [...]} or a question/answer pair
    under any of KEY_PAIRS."""
    if path.endswith(".csv"):
        raw = list(csv.DictReader(open(path, encoding="utf-8")))
    else:
        raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    convs = []
    for r in raw:
        if "messages" in r:
            convs.append(r["messages"]); continue
        for q, a in KEY_PAIRS:
            if r.get(q) and r.get(a):
                convs.append([{"role": "user", "content": r[q].strip()}, {"role": "assistant", "content": r[a].strip()}])
                break
        else:
            raise ValueError(f"row has none of {KEY_PAIRS} or 'messages': {list(r)}")
    return convs


def encode_conversation(msgs, tok, system, max_len):
    """Token ids + labels where only the assistant turns are trained (everything else = IGNORE)."""
    if system and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": system}] + msgs
    ids, labels, done = [], [], ""
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs[:i + 1], tokenize=False)
        assert prompt.startswith(done) and full.startswith(prompt), "chat template is not prefix-stable"
        p = tok(prompt[len(done):], add_special_tokens=False)["input_ids"]
        a = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids += p + a; labels += [IGNORE] * len(p) + a
        done = full
    return ids[:max_len], labels[:max_len]


def text_chunks(texts, tok, max_len):
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"]); ids.append(tok.eos_token_id)
    return [(ids[i:i + max_len], ids[i:i + max_len]) for i in range(0, len(ids) - max_len + 1, max_len)]


def collate(examples, pad_id):
    L = max(len(i) for i, _ in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), L), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(examples), L), dtype=torch.long)
    for b, (i, l) in enumerate(examples):
        ids[b, :len(i)] = torch.tensor(i); labels[b, :len(l)] = torch.tensor(l); mask[b, :len(i)] = 1
    return ids, labels, mask


def batches(qa, text, batch_size, text_mix, seed=1337):
    """Endless shuffled batches; each example is raw Telugu text with probability text_mix, else Q&A."""
    rng = random.Random(seed)
    order, pos = list(range(len(qa))), len(qa)
    while True:
        out = []
        for _ in range(batch_size):
            if text and rng.random() < text_mix:
                out.append(rng.choice(text)); continue
            if pos >= len(order):
                rng.shuffle(order); pos = 0
            out.append(qa[order[pos]]); pos += 1
        yield out


def answer_loss(model, ids, labels, mask):
    """Mean cross-entropy over trained (label != IGNORE) tokens; only those logits are upcast to fp32."""
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    tgt = labels[:, 1:]
    keep = tgt != IGNORE
    return F.cross_entropy(logits[keep].float(), tgt[keep])


# ── model ───────────────────────────────────────────────────────────────────────
def add_lora(model, r, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()   # LoRA weights in fp32, base stays bf16; compute runs in bf16 autocast
    return model


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def train_lora(model, tok, train_data, cfg, out_dir, device):
    """LoRA training loop with checkpoint/resume. train_data = endless iterator of example lists."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    total, warmup = cfg["steps"], cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    state_path, step = os.path.join(out_dir, "train_state.pt"), 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        missing = set(st["lora"]) - set(dict(model.named_parameters()))
        assert not missing, f"checkpoint does not match model: {sorted(missing)[:3]}"
        model.load_state_dict(st["lora"], strict=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        print(f"resumed from step {step}")

    def save():
        tmp = state_path + ".tmp"
        torch.save({"lora": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "cfg": cfg}, tmp)
        os.replace(tmp, state_path)
        model.save_pretrained(os.path.join(out_dir, "adapter"))
        print(f"  checkpoint saved at step {step} -> {out_dir}")

    model.train()
    t0 = t_run = last_save = time.time()
    tokens, start_step = 0, step
    while step < total:
        for _ in range(cfg["grad_accum"]):
            ids, labels, mask = (t.to(device) for t in collate(next(train_data), tok.pad_token_id))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = answer_loss(model, ids, labels, mask)
            (loss / cfg["grad_accum"]).backward()
            tokens += int(mask.sum())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg["log_every"] == 0 or step == 1:
            sec_per_step = (time.time() - t_run) / (step - start_step)
            print(f"step {step:5d}/{total}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{tokens / (time.time() - t0):,.0f} tok/s  ~{(total - step) * sec_per_step / 3600:.1f} h left")
            t0, tokens = time.time(), 0
        if step % cfg["save_every"] == 0 or time.time() - last_save > cfg["save_minutes"] * 60 or step == total:
            save(); last_save = time.time()
    return step


# ── eval helpers ────────────────────────────────────────────────────────────────
def telugu_ratio(text):
    """Share of letters that are Telugu script (U+0C00–U+0C7F). The app wants mostly-Telugu answers."""
    letters = [c for c in text if c.isalpha()]
    return sum("ఀ" <= c <= "౿" for c in letters) / len(letters) if letters else 0.0


@torch.no_grad()
def eval_loss(model, examples, tok, device, batch_size=8):
    model.eval()
    losses = []
    for i in range(0, len(examples), batch_size):
        ids, labels, mask = (t.to(device) for t in collate(examples[i:i + batch_size], tok.pad_token_id))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            losses.append(answer_loss(model, ids, labels, mask).item())
    return sum(losses) / len(losses)


@torch.no_grad()
def answer(model, tok, question, device, system=SYSTEM_PROMPT, max_new_tokens=200):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def first_turn(conv):
    """(first user message, first assistant reply) of a conversation."""
    return (next(m["content"] for m in conv if m["role"] == "user"),
            next(m["content"] for m in conv if m["role"] == "assistant"))


def split_qa(convs, eval_frac, seed=1337):
    convs = convs[:]
    random.Random(seed).shuffle(convs)
    n_eval = max(1, int(len(convs) * eval_frac))
    return convs[n_eval:], convs[:n_eval]


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "merge", "eval"])
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="/content/drive/MyDrive/telugu-3b")
    ap.add_argument("--qa", nargs="+", help="Telugu Q&A files (.jsonl or .csv)")
    ap.add_argument("--text-dataset", default=None, help="optional raw Telugu text, e.g. ai4bharat/samanantar")
    ap.add_argument("--text-config", default="te")
    ap.add_argument("--text-split", default="train[:100000]")
    ap.add_argument("--text-field", default="tgt")
    ap.add_argument("--text-mix", type=float, default=0.2, help="share of batch slots filled with raw text")
    ap.add_argument("--system", default=SYSTEM_PROMPT, help="'' to train without a system prompt")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=None, help="overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--n-samples", type=int, default=8, help="eval: questions to answer side by side")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    merged_dir = os.path.join(args.out, "merged")

    if args.stage in ("train", "eval"):
        assert args.qa, "--qa is required"
        convs = [c for path in args.qa for c in read_rows(path)]
        train_convs, eval_convs = split_qa(convs, args.eval_frac)
        print(f"Q&A conversations: {len(train_convs):,} train | {len(eval_convs):,} held out")

    if args.stage == "train":
        qa = [encode_conversation(c, tok, args.system, args.max_len) for c in train_convs]
        qa = [e for e in qa if any(l != IGNORE for l in e[1])]   # drop answers cut off by --max-len
        text = []
        if args.text_dataset:
            from datasets import load_dataset
            ds = load_dataset(args.text_dataset, args.text_config, split=args.text_split)
            text = text_chunks([r[args.text_field] for r in ds if r[args.text_field]], tok, args.max_len)
            print(f"raw Telugu text chunks: {len(text):,} x {args.max_len} tokens")
        steps = args.steps or math.ceil(args.epochs * len(qa) / ((1 - (args.text_mix if text else 0)) *
                                                                 args.batch_size * args.grad_accum))
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
        model.print_trainable_parameters()
        cfg = {k: getattr(args, k) for k in ["lr", "warmup", "grad_accum", "log_every", "save_every", "save_minutes"]}
        cfg["steps"] = steps
        print(f"training {steps} steps ({args.batch_size} x {args.grad_accum} examples per step)")
        train_lora(model, tok, batches(qa, text, args.batch_size, args.text_mix), cfg, args.out, device)

    elif args.stage == "merge":
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, os.path.join(args.out, "adapter")).merge_and_unload()
        model.save_pretrained(merged_dir); tok.save_pretrained(merged_dir)
        print(f"merged model saved -> {merged_dir}")
        print(f"next: python ternary_qat_3b.py train --teacher {merged_dir}")

    else:  # eval
        held = [encode_conversation(c, tok, args.system, args.max_len) for c in eval_convs]
        held = [e for e in held if any(l != IGNORE for l in e[1])]
        sample = eval_convs[:args.n_samples]
        results = {}
        for name, path in [("base", args.base), ("fine-tuned", merged_dir)]:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
            answers = [answer(model, tok, first_turn(c)[0], device, args.system) for c in sample]
            results[name] = answers
            print(f"{name:11s} held-out answer loss {eval_loss(model, held, tok, device):.3f} | "
                  f"Telugu script in answers {100 * sum(map(telugu_ratio, answers)) / len(answers):.0f}%")
            del model; torch.cuda.empty_cache()
        for i, c in enumerate(sample):
            q, expected = first_turn(c)
            print(f"\nQ: {q}\n  expected  : {expected}\n"
                  f"  base      : {results['base'][i]}\n  fine-tuned: {results['fine-tuned'][i]}")


if __name__ == "__main__":
    main()
