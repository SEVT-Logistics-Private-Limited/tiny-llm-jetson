# tiny_llm.py — a ~1,000-parameter GPT, trained from scratch, on the Orin GPU.
import torch, torch.nn as nn
from torch.nn import functional as F

# --- config: tweak n_embd / block_size to change the parameter count ---
block_size, n_embd, n_head, n_layer = 128, 200, 8, 2
lr, steps, batch_size = 3e-3, 5000, 32
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# --- data: a tiny, highly-structured corpus so a 1k model can visibly learn ---
text = "to be or not to be that is the question " * 200
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]; decode = lambda l: "".join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size]   for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4*n_embd, bias=False), nn.GELU(),
                                 nn.Linear(4*n_embd, n_embd, bias=False))
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()  # causal
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying (no extra params)
    def forward(self, idx, targets=None):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n):
        for _ in range(n):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

model = TinyGPT().to(device)
print(f"device={device}  vocab={vocab_size}  parameters={sum(p.numel() for p in model.parameters())}")

opt = torch.optim.AdamW(model.parameters(), lr=lr)
for step in range(steps):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(f"step {step:5d}  loss {loss.item():.3f}")

start = torch.tensor([encode("to be")], dtype=torch.long, device=device)
print("SAMPLE:", decode(model.generate(start, 80)[0].tolist()))
torch.save({"model": model.state_dict(), "stoi": stoi,
            "config": dict(block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)},
           "tiny_llm_1m.pt")
print("saved tiny_llm_1m.pt")
