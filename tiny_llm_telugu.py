# tiny_llm_telugu.py
# Same GPT architecture as before
# BUT trained on real Telugu text corpus
# Vocab: 245 Telugu characters (not 13 English)

import torch, torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset

# ── CONFIG ──────────────────────────────────────────────────────
block_size, n_embd, n_head, n_layer = 128, 512, 8, 8
lr, steps, batch_size               = 3e-4, 10000, 16
device = "cuda"
dtype  = torch.bfloat16
torch.manual_seed(1337)

# ── LOAD TELUGU DATA ─────────────────────────────────────────────
print("Loading Telugu data...")
ds   = load_dataset("ai4bharat/samanantar", "te", split="train[:50000]")
text = " ".join([item["tgt"] for item in ds])
print(f"Corpus: {len(text):,} characters")

# ── TOKENIZER ────────────────────────────────────────────────────
chars      = sorted(set(text))
vocab_size = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for c, i in stoi.items()}
encode     = lambda s: [stoi[c] for c in s if c in stoi]
decode     = lambda l: "".join(itos[i] for i in l)
data       = torch.tensor(encode(text), dtype=torch.long)
print(f"Vocab size: {vocab_size} | Data tokens: {len(data):,}")

def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x  = torch.stack([data[i:i+block_size]     for i in ix]).to(device)
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# ── MODEL ─────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head,
                        batch_first=True, bias=False)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd, bias=False))
    def forward(self, x):
        T    = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device),
                          diagonal=1).bool()
        a    = self.ln1(x)
        x    = x + self.attn(a, a, a,
                    attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok    = nn.Embedding(vocab_size, n_embd)
        self.pos    = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.head   = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.use_ckpt = True
    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(
                torch.arange(idx.size(1), device=device))
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False) \
                if (self.use_ckpt and self.training) else blk(x)
        logits = self.head(self.ln_f(x))
        loss   = F.cross_entropy(
            logits.view(-1, vocab_size), targets.view(-1)
        ) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n):
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -block_size:])
            idx = torch.cat([idx,
                torch.multinomial(
                    F.softmax(logits[:, -1], dim=-1), 1)
            ], dim=1)
        return idx

# ── BUILD ─────────────────────────────────────────────────────────
model  = TinyGPT().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"device={device} | vocab={vocab_size} | parameters={params:,}")

# ── TRAIN ─────────────────────────────────────────────────────────
scaler = torch.cuda.amp.GradScaler()
opt    = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(steps):
    x, y = get_batch()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=dtype):
        _, loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    if step % 1000 == 0:
        print(f"step {step:5d}  loss {loss.item():.4f}")

# ── SAMPLE ────────────────────────────────────────────────────────
model.use_ckpt = False
seed_text = text[:block_size]
start = torch.tensor([encode(seed_text)[:block_size]],
    dtype=torch.long, device=device)
print("SAMPLE:", decode(model.generate(start, 200)[0].tolist()))

# ── SAVE ──────────────────────────────────────────────────────────
torch.save({
    "model":  model.state_dict(),
    "stoi":   stoi,
    "itos":   itos,
    "config": dict(block_size=block_size, n_embd=n_embd,
                   n_head=n_head, n_layer=n_layer,
                   vocab_size=vocab_size)
}, "/content/tiny_llm_telugu.pt")
size = __import__("os").path.getsize(
    "/content/tiny_llm_telugu.pt") / 1e6
print(f"Saved tiny_llm_telugu.pt ({size:.1f}MB)")
