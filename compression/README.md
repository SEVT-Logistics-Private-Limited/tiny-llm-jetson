# Compressing a 3B model (PrismML-style) — Colab only

Everything here runs in **Google Colab (A100-80GB)**. Nothing runs on the Jetson, and nothing changes in the live voice assistant (`voice-assistant/api.py` still uses Gemini).

## Why not compress `tiny_llm_3b.pt`?

`tiny_llm_3b.py` was trained only on `"to be or not to be that is the question " * 200` with 13 characters. It memorised one sentence, so a compressed copy would still only say that sentence. Compression keeps what a model already knows — it cannot add knowledge. So we compress a **real** pretrained 3B model instead (default: `meta-llama/Llama-3.2-3B-Instruct`, or your own Telugu fine-tuned 3B model).

## Two ways to compress

| | Path A — quick 4-bit | Path B — PrismML-style ternary |
|---|---|---|
| How | Round weights to 4 bits (no training) | Weights become {-s, 0, +s}, then **re-train** the model to recover quality |
| Size of a 3B model | 6.4 GB → ~2.0 GB | 6.4 GB → ~1.1 GB |
| Quality | ~97–99% of original | Depends on training length — must be measured |
| Time | ~15 minutes | ~4 hours (3,000 steps) up to 1–2 days (longer = better) |
| Script | llama.cpp (below) | `ternary_qat_3b.py` (this folder) |

Size estimate for Llama-3.2-3B, Path B: ternary layer weights 0.70 GB + scales 0.04 GB + int8 embeddings 0.39 GB ≈ **1.1 GB**. (PrismML also compresses the embeddings to 1 bit; we keep them at 8 bits because they are easy to damage.)

---

## Path A — quick 4-bit (Colab cells)

```bash
!git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && pip install -q -r requirements.txt
!cmake -S llama.cpp -B llama.cpp/build && cmake --build llama.cpp/build --target llama-quantize -j
!huggingface-cli login          # Llama 3.2 is gated: accept its license on Hugging Face first
!huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir /content/llama-3b
!python llama.cpp/convert_hf_to_gguf.py /content/llama-3b --outtype bf16 --outfile /content/llama-3b-bf16.gguf
!llama.cpp/build/bin/llama-quantize /content/llama-3b-bf16.gguf /content/llama-3b-Q4_K_M.gguf Q4_K_M
```

## Path B — PrismML-style ternary compression (Colab cells)

**Cell 1 — setup** (Runtime → A100 GPU, High-RAM)
```python
from google.colab import drive; drive.mount('/content/drive')   # checkpoints survive disconnects
!git clone -b claude/funny-thompson-o1yi7c https://github.com/SEVT-Logistics-Private-Limited/tiny-llm-jetson
!pip install -q transformers datasets accelerate bitsandbytes
# private repo? clone with a GitHub token, or upload compression/ternary_qat_3b.py to Colab directly
!huggingface-cli login
```

**Cell 2 — train** (re-run the same cell after a disconnect: it resumes from the last checkpoint)
```python
!python tiny-llm-jetson/compression/ternary_qat_3b.py train \
    --out /content/drive/MyDrive/ternary-3b --steps 3000
# optional: add your Telugu Q&A pairs:  --qa-jsonl /content/drive/MyDrive/telugu_qa.jsonl
#           (one line per pair: {"question": "...", "answer": "..."})
```

**Cell 3 — export the compressed file**
```python
!python tiny-llm-jetson/compression/ternary_qat_3b.py export --out /content/drive/MyDrive/ternary-3b
```
Prints the bf16 size vs the compressed size and writes `ternary_model.pt`.

**Cell 4 — check quality**
```python
!python tiny-llm-jetson/compression/ternary_qat_3b.py eval --out /content/drive/MyDrive/ternary-3b
```
Prints Telugu perplexity (lower = better) for the original and the ternary model, and their answers to the same Telugu questions side by side.

### What happens inside `train`

1. **Teacher**: the original 3B model in bf16, frozen.
2. **Student**: a copy where all 196 Linear layers (7 per block × 28 blocks) become `BitLinear`. For every group of 128 weights: `s = mean(|w|)`, weight → `round(w / s)` clipped to {-1, 0, 1}, times `s`.
3. For the first 500 steps the student slides from full precision to fully ternary (`--ramp-steps`) — jumping straight to ternary breaks a pretrained model.
4. Loss = 50% "copy the teacher's predictions" (KL distillation) + 50% "predict the real Telugu text" (`--alpha`).
5. Checkpoint to Google Drive every 200 steps and every 30 minutes.

### Memory and time on A100-80GB (estimates)

| Item | GPU memory |
|---|---|
| Student weights (fp32) | ~13 GB |
| Gradients | ~13 GB |
| 8-bit Adam state | ~6.5 GB |
| Teacher (bf16) | ~6.5 GB |
| Activations + logits (batch 4 × 512, gradient checkpointing) | ~10–15 GB |
| **Total** | **~50–55 GB** |

One step = 4 × 512 × 8 = 16,384 tokens. Expect roughly 3,000–5,000 tokens/s, so `--steps 3000` (~50M tokens) ≈ 3–5 hours. The log prints the real speed and hours left. For better quality, go to 10,000–30,000 steps over several Colab sessions, and use more Telugu text (`--split "train[:1000000]"`).

### Honest expectations

- PrismML trains on far more data than one A100 can process, so our ternary model will lose more quality than theirs. Always compare with `eval` before using it anywhere.
- The teacher's Telugu ability is the ceiling: if the 3B teacher answers Telugu poorly, fine-tune it on Telugu first, then pass that folder as `--teacher`.
- `ternary_model.pt` is our own packed format (2-bit codes + one FP16 scale per 128 weights). The `eval` stage loads it in PyTorch. Running it in llama.cpp needs a converter, because llama.cpp's ternary formats (`TQ1_0`/`TQ2_0`) use a different scale layout — a later step.

## Test

```bash
python compression/test_ternary_qat.py   # tiny random model on CPU, ~1 min: layers swap, training, resume, pack/unpack, reload
```
