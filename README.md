# Tiny LLM - Small-Model Foundations

From-scratch GPT transformers scaled from 984 params to 3.07B,
then applied to real Telugu language as the foundation for a Telugu voice assistant.

**Author:** Rishi Sasanala
**Hardware:** Jetson Orin Nano Super + Colab Pro A100-SXM4-80GB
**Stack:** PyTorch, CUDA, Git LFS, HuggingFace Hub

---

## What Is This?

Every model here is trained completely from scratch - no pretrained weights,
no fine-tuning. We start from random numbers and train until the model learns.
Rung 9 is the pivot: from English placeholder to real Telugu text.

---

## All 9 Rungs - Exact Numbers

| Rung | Exact Params | Platform | Loss | Language | Notes |
|------|-------------|----------|------|----------|-------|
| 1 | 984 | Jetson Orin | 0.175 | English | Pipeline proof |
| 2 | 990,200 | Jetson Orin | 0.016 | English | Best result ever |
| 3 | 151,204,504 | Jetson Orin | 1.954 | English | Memory-constrained |
| 3b | 151,204,504 | Colab T4 | 1.114 | English | bf16 rerun |
| 4 | ~399,038,848 | Colab A100 | TBD | English | First dedicated GPU |
| 5 | ~998,529,040 | Colab A100-80GB | TBD | English | First 1B+ |
| 6 | ~2,006,267,232 | Colab A100-80GB | TBD | English | 2B scale |
| 7 | 2,999,532,432 | Colab A100-80GB | 0.1098 | English | Near-perfect |
| 8 MAX | 3,072,828,000 | Colab A100-80GB | 0.0977 | English | HARDWARE MAXIMUM |
| 9 | 25,374,208 | Colab A100-80GB | 1.42 | Telugu | FIRST REAL LANGUAGE |

Rung 8 = absolute maximum on A100-80GB. Any larger config will OOM.
Rung 9 = pivot from English placeholder to real Telugu corpus.

---

## Rung Details

### Rung 1 - 984 params (Jetson Orin)
Proves full pipeline works on edge hardware. Under 1 minute.
Files: tiny_llm.py, tiny_llm.pt, test_tiny_llm.py, model-card.md

### Rung 2 - 990,200 params (Jetson Orin)
1000x scale, same recipe. Flawless loss 0.016. Best result.
Files: tiny_llm_1m.py, tiny_llm_1m.pt, test_tiny_llm_1m.py, model-card-1m.md

### Rung 3 - 151M params (Jetson + Colab T4)
Jetson memory-constrained (batch=8). Colab T4 rerun with bf16.
Files: tiny_llm_150m.py, tiny_llm_150m.pt, test_tiny_llm_150m.py, model-card-150m.md

### Rung 4 - ~399M params (Colab A100)
First dedicated GPU. Introduced bf16 + gradient checkpointing + grad clipping.
Files: tiny_llm_400m.py, tiny_llm_400m.pt, model-card-400m.md

### Rung 5 - ~1B params (Colab A100-80GB)
First billion-parameter run. lr=2e-4, batch=8.
Files: tiny_llm_1b.py, tiny_llm_1b.pt, model-card-1b.md

### Rung 6 - ~2B params (Colab A100-80GB)
bf16 checkpoint (3.8GB). Fits GitHub LFS limit.
Files: tiny_llm_2b.py, tiny_llm_2b.pt, model-card-2b.md

### Rung 7 - 2,999,532,432 params (Colab A100-80GB)
Loss 2222 to 0.1098. Near-perfect Telugu.
Checkpoint: https://huggingface.co/RishikUttejSasanala/tiny-llm-3b
Files: tiny_llm_3b.py, model-card-3b.md

### Rung 8 - 3,072,828,000 params - HARDWARE MAXIMUM
Maximum trainable on A100-80GB. n_embd=4000, n_layer=16.
Loss 2243 to 0.0977 best.
Checkpoint: https://huggingface.co/RishikUttejSasanala/tiny-llm-5b
Files: tiny_llm_5b.py, model-card-5b.md

### Rung 9 - 25,374,208 params - FIRST REAL LANGUAGE (Telugu)
Same GPT architecture trained on ai4bharat/samanantar Telugu corpus.
2.9M characters, 245 unique Telugu characters in vocab.
Loss 331 to 1.42. Generates real Telugu sentences.
This is the brain layer for the Telugu voice assistant.
Checkpoint: https://huggingface.co/RishikUttejSasanala/tiny-llm-telugu
Files: tiny_llm_telugu.py, tiny_llm_telugu.pt, model-card-telugu.md

---

## Key Findings

- Parameter count alone does not determine quality
- Training recipe (batch, lr, precision) is the real constraint
- Jetson ceiling: ~1M params clean with fp32 batch=32
- A100-80GB ceiling: 3,072,828,000 params (confirmed empirically)
- GitHub LFS limit: 5GB per file - above 3B goes to HuggingFace
- Telugu vocab = 245 chars vs 13 for English - model adapts automatically

---

## Hardware Limits (Empirically Confirmed)

| Hardware | VRAM | Max Params |
|----------|------|------------|
| Jetson Orin Nano Super | 8GB shared | ~1M clean |
| Colab T4 | 16GB | ~400M |
| Colab A100-40GB | 40GB | ~2B |
| Colab A100-80GB | 85GB | 3,072,828,000 |

---

## HuggingFace Checkpoints (files too large for GitHub LFS)

| Rung | Link |
|------|------|
| Rung 7 (3B) | https://huggingface.co/RishikUttejSasanala/tiny-llm-3b |
| Rung 8 MAX (3.07B) | https://huggingface.co/RishikUttejSasanala/tiny-llm-5b |
| Rung 9 Telugu | https://huggingface.co/RishikUttejSasanala/tiny-llm-telugu |

---

## Next Steps - Telugu Voice Assistant

1. Scale Telugu model to 1B+ params with more training data
2. Connect STT (Sarvam AI) + LLM + TTS (Bulbul) via FastAPI
3. Add WebSocket streaming for real-time conversation
4. Deploy as voice agent API
