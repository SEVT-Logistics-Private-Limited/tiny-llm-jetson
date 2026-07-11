# Model Card: tiny-llm-5b (Rung 8 MAX — Colab Pro A100-80GB)

## Architecture
- Parameters: 3,072,828,000
- n_embd: 4000 | n_layer: 16 | n_head: 32
- block_size: 128 | vocab: 13 | weight tying: yes

## Training
- Optimizer: AdamW lr=3e-5 | Steps: 5000 | Batch: 4
- Precision: bf16 AMP + gradient checkpointing + grad clipping
- Hardware: Google Colab Pro A100-SXM4-80GB (85GB VRAM)
- This is the maximum parameter count achievable on this hardware

## Results
- Initial loss: 2243.88
- Best loss (step 2500): 0.0977
- Final loss (step 4500): 0.2073
- Sample: "to be that is the question to be or not to be that is the question to be or not to be"

## Checkpoint
6.15GB checkpoint on HuggingFace:
https://huggingface.co/RishikUttejSasanala/tiny-llm-5b

## Files (this repo)
- tiny_llm_5b.py — training script
- model-card-5b.md — this file
