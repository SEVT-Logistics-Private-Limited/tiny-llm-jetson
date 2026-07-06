# Model Card: tiny-llm-2b (Rung 6 — Colab Pro A100-80GB)

## Architecture
- Parameters: ~2,006,267,232
- n_embd: 3232 | n_layer: 16 | n_head: 32
- block_size: 128 | vocab: 13 | weight tying: yes

## Training
- Optimizer: AdamW lr=1e-4 | Steps: 5000 | Batch: 4
- Precision: bf16 AMP + gradient checkpointing + grad clipping (1.0)
- Hardware: Google Colab Pro A100-SXM4-80GB
- Corpus: "to be or not to be that is the question " x200

## Results
- Fill after training: initial loss / final loss / sample output

## Notes
- lr reduced to 1e-4 vs 1B (2e-4) for stability at 2B scale
- batch=4 to fit VRAM with 3232-dim embeddings
