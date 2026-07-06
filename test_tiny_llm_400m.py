import torch, torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("tiny_llm_400m.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
block_size, n_embd, n_head, n_layer = cfg["block_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
stoi = ckpt["stoi"]; itos = {i: c for c, i in stoi.items()}; vocab_size = len(stoi)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd,4*n_embd,bias=False),nn.GELU(),nn.Linear(4*n_embd,n_embd,bias=False))
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        a = self.ln1(x)
        x = x + self.attn(a,a,a,attn_mask=mask,need_weights=False)[0]
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok=nn.Embedding(vocab_size,n_embd); self.pos=nn.Embedding(block_size,n_embd)
        self.blocks=nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,vocab_size,bias=False)
        self.head.weight=self.tok.weight
    def forward(self,idx):
        x=self.tok(idx)+self.pos(torch.arange(idx.size(1),device=idx.device))
        for b in self.blocks: x=b(x)
        return self.head(self.ln_f(x))
    @torch.no_grad()
    def generate(self,idx,n):
        for _ in range(n):
            probs=F.softmax(self(idx[:,-block_size:])[:,-1],dim=-1)
            idx=torch.cat([idx,torch.multinomial(probs,1)],dim=1)
        return idx

model = TinyGPT().to(device)
model.load_state_dict(ckpt["model"]); model.eval()
print(f"Loaded: parameters={sum(p.numel() for p in model.parameters()):,}")
for s in ["to be", "that i", "questi", "be or "]:
    start = torch.tensor([encode(s)], dtype=torch.long, device=device)
    print(f"SEED='{s}'  ->  {decode(model.generate(start, 60)[0].tolist())}")
