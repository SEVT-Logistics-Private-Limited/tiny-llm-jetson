# Model Card: tiny-llm-3b (Rung 7 — Colab Pro A100-80GB)

## Architecture
- Parameters: 2,999,532,432
- n_embd: 3952 | n_layer: 16 | n_head: 16
- block_size: 128 | vocab: 13 | weight tying: yes

## Training
- Optimizer: AdamW lr=5e-5 | Steps: 5000 | Batch: 8
- Precision: bf16 AMP + gradient checkpointing + grad clipping
- Hardware: Google Colab Pro A100-SXM4-80GB

## Results
- Initial loss: 2222.76
- Final loss (step 4500): 0.1098
- Sample: "to be or not to be that is the question to be or not to be that is the question to be"

## Checkpoint
6GB checkpoint on HuggingFace (exceeds GitHub LFS 5GB limit):
https://huggingface.co/RishikUttejSasanala/tiny-llm-3b

## Files (this repo)
- tiny_llm_3b.py — training script
- model-card-3b.md — this file
