# Model Card: tiny-llm-1b (Rung 5 — Colab Pro A100-80GB)

## Architecture
- Parameters: ~998,529,040
- n_embd: 1520 | n_layer: 36 | n_head: 16
- block_size: 128 | vocab: 13 | weight tying: yes

## Training
- Optimizer: AdamW lr=2e-4 | Steps: 5000 | Batch: 8
- Precision: bf16 AMP + gradient checkpointing + grad clipping (1.0)
- Hardware: Google Colab Pro A100-SXM4-80GB
- Corpus: "to be or not to be that is the question " x200

## Results
- Fill after training: initial loss / final loss / sample output
