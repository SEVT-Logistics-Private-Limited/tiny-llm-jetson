# Tiny LLM — Small-Model Foundations

From-scratch GPT-style transformers scaled from 984 params to 2B+
across Jetson Orin edge hardware and Google Colab Pro A100-80GB.

**Author:** Rishi Sasanala
**Hardware:** NVIDIA Jetson Orin Nano Super + Google Colab Pro A100-SXM4-80GB
**Stack:** PyTorch, CUDA, Git LFS

---

## Progression

| Rung | Params | Platform | Final Loss | Notes |
|------|--------|----------|------------|-------|
| 1 | 984 | Jetson Orin | 0.175 | fp32, batch=32 |
| 2 | ~1M | Jetson Orin | 0.016 | fp32, batch=32 — best result |
| 3 | ~151M | Jetson Orin | 1.954 | fp32, batch=8, memory-constrained |
| 3b | ~151M | Colab T4 | 1.114 | bf16, batch=16 |
| 4 | ~399M | Colab Pro A100 | TBD | bf16+GC+clipping, batch=16 |
| 5 | ~1B | Colab Pro A100-80GB | TBD | bf16+GC+clipping, batch=8 |
| 6 | ~2B | Colab Pro A100-80GB | TBD | bf16+GC+clipping, batch=4 |

---

## Rung 1 — 984 parameters (Jetson Orin)
Smallest viable GPT. Proves end-to-end pipeline on edge hardware.
Loss 6.6 → 0.175 in under 1 minute on-device.
Files: tiny_llm.py, tiny_llm.pt, test_tiny_llm.py, model-card.md

## Rung 2 — ~1M parameters (Jetson Orin)
1000x scale-up, same recipe. Flawless convergence to 0.016.
Best result across all rungs. Stable scaling when recipe is held constant.
Files: tiny_llm_1m.py, tiny_llm_1m.pt, test_tiny_llm_1m.py, model-card-1m.md

## Rung 3 — ~151M parameters (Jetson Orin + Colab T4)
Jetson: batch=8, loss plateaued at 1.954 (memory-constrained).
Colab T4: batch=16, bf16, loss 0.686 best — still oscillating.
Key finding: corpus saturated at this scale. Hardware not the bottleneck.
Files: tiny_llm_150m.py, tiny_llm_150m.pt, test_tiny_llm_150m.py, model-card-150m.md

## Rung 4 — ~399M parameters (Colab Pro A100)
First run with dedicated high-VRAM GPU. Introduced bf16 AMP,
gradient checkpointing, gradient clipping.
Files: tiny_llm_400m.py, tiny_llm_400m.pt, test_tiny_llm_400m.py, model-card-400m.md

## Rung 5 — ~1B parameters (Colab Pro A100-80GB)
36-layer, 1520-dim transformer. First billion-parameter run.
bf16 + gradient checkpointing + grad clipping. lr=2e-4, batch=8.
Files: tiny_llm_1b.py, tiny_llm_1b.pt, model-card-1b.md

## Rung 6 — ~2B parameters (Colab Pro A100-80GB)
16-layer, 3232-dim transformer. Checkpoint saved in bf16 (~3.8GB).
lr=1e-4, batch=4. First run exceeding 1B parameters.
Files: tiny_llm_2b.py, tiny_llm_2b.pt, model-card-2b.md

---

## Key Engineering Findings
- Parameter count alone does not determine convergence quality
- Training recipe (batch size, lr, precision) is the real constraint
- Jetson ceiling for clean training: ~1M params with fp32 batch=32
- Corpus exhausted above ~1M params; next rung needs real data
- GitHub LFS hard limit: 5GB/file — save checkpoints in bf16 directly

## Hardware Limits Reference
| Hardware | VRAM | Max trainable (bf16+GC) |
|----------|------|--------------------------|
| Jetson Orin Nano Super | 8GB shared | ~1M params (clean) |
| Colab T4 | 16GB | ~400M params |
| Colab A100-40GB | 40GB | ~2B params |
| Colab A100-80GB | 80GB | ~5B params |

## Next Steps
Rung 7: ~3B parameters on A100-80GB (same pipeline).
Rung 8: real Indic-language corpus — first run targeting genuinely
useful small regional-language model output.

## Rung 7 — ~3B parameters (Colab Pro A100-80GB)
16-layer, 3952-dim transformer. Best convergence of all rungs.
Loss 2222 → 0.1098. Near-perfect sample output.
Checkpoint (6GB) on HuggingFace: https://huggingface.co/RishikUttejSasanala/tiny-llm-3b
Files: tiny_llm_3b.py, model-card-3b.md

## Rung 8 — ~3B parameters MAX (Colab Pro A100-80GB)
Maximum parameter count achievable on A100-80GB with this architecture.
n_embd=4000, n_layer=16. Loss 2243 → 0.0977 best. Near-perfect sample.
Checkpoint (6.15GB) on HuggingFace: https://huggingface.co/RishikUttejSasanala/tiny-llm-5b
Files: tiny_llm_5b.py, model-card-5b.md
