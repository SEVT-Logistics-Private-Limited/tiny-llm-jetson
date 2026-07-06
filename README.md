# Tiny LLM - Small-Model Foundations

From-scratch GPT-style transformers scaled from 984 params to 399M
across Jetson Orin edge hardware and Google Colab.

**Author:** Rishi Sasanala
**Hardware:** NVIDIA Jetson Orin Nano Super + Google Colab Pro A100
**Stack:** PyTorch, CUDA, Git LFS

---

## Progression

| Rung | Params | Platform | Final Loss | Notes |
|------|--------|----------|------------|-------|
| 1 | 984 | Jetson Orin | 0.175 | fp32, batch=32 |
| 2 | ~1M | Jetson Orin | 0.016 | fp32, batch=32, best result |
| 3 | ~151M | Jetson Orin | 1.954 | fp32, batch=8, memory-constrained |
| 3b | ~151M | Colab T4 | 1.114 | bf16, batch=16 |
| 4 | ~399M | Colab Pro A100 | TBD | bf16+GC+clipping, batch=16 |

---

## Rung 1 - 984 parameters (Jetson Orin)
Smallest viable GPT. Proves the full pipeline end-to-end on edge hardware.
Loss 6.6 to 0.175 in under 1 minute on-device.
Files: tiny_llm.py, tiny_llm.pt, test_tiny_llm.py, model-card.md

## Rung 2 - ~1M parameters (Jetson Orin)
1000x scale-up, same recipe. Flawless convergence to 0.016.
Best result across all rungs. Proves stable scaling when recipe is constant.
Files: tiny_llm_1m.py, tiny_llm_1m.pt, test_tiny_llm_1m.py, model-card-1m.md

## Rung 3 - ~151M parameters (Jetson + Colab)
Jetson: batch forced to 8, loss plateaued at 1.954 (memory-constrained).
Colab T4: batch=16, bf16, loss 0.686 best but still oscillating.
Key finding: corpus saturated at this scale, not a hardware limit.
Files: tiny_llm_150m.py, tiny_llm_150m.pt, test_tiny_llm_150m.py, model-card-150m.md

## Rung 4 - ~399M parameters (Colab Pro A100)
2.6x scale from rung 3. New techniques: bf16 AMP, gradient checkpointing,
gradient clipping. First rung with dedicated high-VRAM GPU.
Files: tiny_llm_400m.py, tiny_llm_400m.pt, test_tiny_llm_400m.py, model-card-400m.md

---

## Key Engineering Findings
- Parameter count alone does not determine convergence quality
- Training recipe (batch size, lr, precision) is the real constraint
- Jetson ceiling for clean training: ~1M params with fp32 batch=32
- Corpus exhausted above ~1M params; next rung needs real data

## Next Steps
Rung 5: real Indic-language corpus on A100, targeting a genuinely
useful small regional-language model.

## Rung 5 - ~1B parameters (Colab Pro A100-80GB)
36-layer, 1520-dim transformer. First billion-parameter run.
bf16 + gradient checkpointing + grad clipping.
Files: tiny_llm_1b.py, tiny_llm_1b.pt, model-card-1b.md
