# test_ternary_qat.py — end-to-end check of ternary_qat_3b.py on a tiny random Llama (CPU, ~1 min)
#   python compression/test_ternary_qat.py
import os, sys, tempfile
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ternary_qat_3b as tq

torch.manual_seed(0)
device = torch.device("cpu")
cfg = LlamaConfig(vocab_size=320, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)

# 1. ternarize / pack round-trip
w = torch.randn(64, 256)
codes, s = tq.BitLinear.ternarize(w)
assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
assert torch.equal(tq.unpack_codes(tq.pack_codes(codes)).float(), codes.flatten())
print("ok  ternarize + 2-bit pack/unpack round-trip")

# 2. swap layers: 7 Linear per Llama block, lm_head untouched and still tied
teacher = LlamaForCausalLM(cfg).eval()
student = LlamaForCausalLM(cfg)
student.load_state_dict(teacher.state_dict())
n = tq.to_bitlinear(student)
assert n == 7 * cfg.num_hidden_layers, n
assert type(student.lm_head) is torch.nn.Linear
assert student.lm_head.weight is student.model.embed_tokens.weight
x = torch.randint(0, cfg.vocab_size, (2, 32))
tq.set_strength(student, 0.0)
with torch.no_grad():
    assert torch.allclose(student(x).logits, teacher(x).logits, atol=1e-5)
print(f"ok  {n} BitLinear layers; strength 0 == original model")

# 3. train a few steps, checkpoint, resume
data = torch.randint(0, cfg.vocab_size, (5000,))
run_cfg = dict(steps=6, warmup=2, ramp_steps=3, lr=1e-3, batch_size=2, grad_accum=2, seq_len=32,
               temperature=1.0, alpha=0.5, log_every=2, save_every=3, save_minutes=999)
out = tempfile.mkdtemp()
before = student.model.layers[0].mlp.up_proj.weight.detach().clone()
step = tq.run_qat(teacher, student, tq.get_batches(data, 32, 2), run_cfg, out, device)
assert step == 6 and os.path.exists(os.path.join(out, "train_state.pt"))
assert not torch.equal(before, student.model.layers[0].mlp.up_proj.weight)
run_cfg["steps"] = 8
resumed = LlamaForCausalLM(cfg); tq.to_bitlinear(resumed)
step = tq.run_qat(teacher, resumed, tq.get_batches(data, 32, 2), run_cfg, out, device)
assert step == 8
print("ok  training loop, checkpoint and resume")

# 4. export -> reload compressed model -> same outputs as the fully-ternary student
tq.set_strength(resumed, 1.0); resumed.eval()
blob, sizes = tq.export_packed(resumed, embed_bits=16)
compressed = tq.load_compressed(blob, cfg, dtype=torch.float32).eval()
assert compressed.lm_head.weight.data_ptr() == compressed.model.embed_tokens.weight.data_ptr()
with torch.no_grad():
    a, b = resumed(x).logits, compressed(x).logits
assert torch.allclose(a, b, atol=5e-2), (a - b).abs().max()
linear_params = sum(p.numel() for m in resumed.modules() if isinstance(m, tq.BitLinear) for p in [m.weight])
assert sizes["ternary"] == linear_params // 4
print(f"ok  export/reload matches student (max diff {(a - b).abs().max():.4f}); "
      f"ternary bytes {sizes['ternary']} = {linear_params} weights / 4")

blob8, sizes8 = tq.export_packed(resumed, embed_bits=8)
assert sizes8["other"] < sizes["other"]
tq.load_compressed(blob8, cfg)
print("ok  int8 embedding export")
print("ALL TESTS PASSED")
