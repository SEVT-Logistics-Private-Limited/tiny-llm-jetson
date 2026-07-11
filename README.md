# Tiny LLM — Small-Model Foundations

From-scratch GPT-style transformers scaled from 984 parameters to 3.07 billion,
across Jetson Orin edge hardware and Google Colab Pro A100-80GB.

**Author:** Rishi Sasanala
**Hardware:** NVIDIA Jetson Orin Nano Super + Google Colab Pro A100-SXM4-80GB
**Stack:** PyTorch, CUDA, Git LFS, HuggingFace Hub

---

## Complete Progression — Exact Numbers

| Rung | Exact Params | Platform | Final Loss | Notes |
|------|-------------|----------|------------|-------|
| 1 | 984 | Jetson Orin | 0.175 | fp32, batch=32 |
| 2 | 990,200 | Jetson Orin | 0.016 | fp32, batch=32 — best result |
| 3 | 151,204,504 | Jetson Orin | 1.954 | fp32, batch=8, memory-constrained |
| 3b | 151,204,504 | Colab T4 | 1.114 | bf16, batch=16 |
| 4 | ~399,038,848 | Colab Pro A100 | TBD | bf16+GC+clipping, batch=16 |
| 5 | ~998,529,040 | Colab Pro A100-80GB | TBD | bf16+GC+clipping, batch=8 |
| 6 | ~2,006,267,232 | Colab Pro A100-80GB | TBD | bf16+GC+clipping, batch=4 |
| 7 | 2,999,532,432 | Colab Pro A100-80GB | 0.1098 | bf16+GC+clipping, batch=8 |
| 8 MAX | 3,072,828,000 | Colab Pro A100-80GB | 0.0977 | HARDWARE MAXIMUM — see note below |

> **Rung 8 is the absolute maximum parameter count trainable on this hardware.**
> Config: n_embd=4000, n_layer=16, n_head=32. Uses all available VRAM on the
> A100-SXM4-80GB (85GB). Any larger config will OOM. Checkpoint (6.15GB)
> stored on HuggingFace Hub since it exceeds GitHub LFS 5GB limit.

---

## Rung 1 — 984 parameters (Jetson Orin)
Smallest viable GPT. Proves end-to-end pipeline on edge hardware.
Loss 6.6 → 0.175. Trained in under 1 minute on-device.
Files: tiny_llm.py, tiny_llm.pt, test_tiny_llm.py, model-card.md

## Rung 2 — 990,200 parameters (Jetson Orin)
1000x scale-up, same recipe. Flawless convergence to 0.016.
Best result across all rungs. Stable scaling when recipe is held constant.
Files: tiny_llm_1m.py, tiny_llm_1m.pt, test_tiny_llm_1m.py, model-card-1m.md

## Rung 3 — 151,204,504 parameters (Jetson Orin + Colab T4)
Jetson: batch=8, loss plateaued at 1.954 (memory-constrained).
Colab T4: batch=16, bf16, loss 0.686 best — still oscillating.
Key finding: corpus saturated at this scale, hardware not the bottleneck.
Files: tiny_llm_150m.py, tiny_llm_150m.pt, test_tiny_llm_150m.py, model-card-150m.md

## Rung 4 — ~399M parameters (Colab Pro A100)
First run with dedicated high-VRAM GPU. Introduced bf16 AMP,
gradient checkpointing, gradient clipping.
Files: tiny_llm_400m.py, tiny_llm_400m.pt, test_tiny_llm_400m.py, model-card-400m.md

## Rung 5 — ~1B parameters (Colab Pro A100-80GB)
First billion-parameter run. bf16 + gradient checkpointing + grad clipping.
Files: tiny_llm_1b.py, tiny_llm_1b.pt, model-card-1b.md

## Rung 6 — ~2B parameters (Colab Pro A100-80GB)
Checkpoint saved in bf16 (~3.8GB after compression).
Files: tiny_llm_2b.py, tiny_llm_2b.pt, model-card-2b.md

## Rung 7 — 2,999,532,432 parameters (Colab Pro A100-80GB)
16-layer, 3952-dim transformer. Loss 2222 → 0.1098. Near-perfect sample output.
Checkpoint (6GB) on HuggingFace: https://huggingface.co/RishikUttejSasanala/tiny-llm-3b
Files: tiny_llm_3b.py, model-card-3b.md

## Rung 8 — 3,072,828,000 parameters — HARDWARE MAXIMUM (Colab Pro A100-80GB)
**This is the maximum parameter count trainable on the A100-SXM4-80GB (85GB VRAM).**
Config: n_embd=4000, n_layer=16, n_head=32, block_size=128.
Loss 2243 → 0.0977 (best at step 2500). Near-perfect corpus reproduction.
Any larger config exceeds available VRAM and will OOM.
Checkpoint (6.15GB) on HuggingFace: https://huggingface.co/RishikUttejSasanala/tiny-llm-5b
Files: tiny_llm_5b.py, model-card-5b.md

---

## Key Engineering Findings
- Parameter count alone does not determine convergence quality
- Training recipe (batch size, lr, precision) is the real constraint
- Jetson ceiling for clean training: ~1M params with fp32 batch=32
- Corpus exhausted above ~1M params; real data needed for next rung
- GitHub LFS hard limit: 5GB per file — checkpoints above 3B params go to HuggingFace
- A100-80GB hardware ceiling: 3,072,828,000 parameters (this architecture, bf16+GC)

## Hardware Limits — Confirmed Empirically

| Hardware | VRAM | Max Params Trained | Notes |
|----------|------|-------------------|-------|
| Jetson Orin Nano Super | 8GB shared | ~1M (clean) | 150M attempted, unstable |
| Colab T4 | 16GB | ~400M | bf16 + gradient checkpointing |
| Colab A100-40GB | 40GB | ~2B | bf16 + gradient checkpointing |
| Colab A100-80GB | 85GB | **3,072,828,000** | CONFIRMED MAXIMUM |

## Checkpoints on HuggingFace
- Rung 7 (3B): https://huggingface.co/RishikUttejSasanala/tiny-llm-3b
- Rung 8 MAX (3.07B): https://huggingface.co/RishikUttejSasanala/tiny-llm-5b

## Next Steps
Rung 9: swap the placeholder corpus for a real Indic-language dataset.
Target: a genuinely useful small regional-language model using the
proven rung-2 training recipe as the convergence baseline.
