# Model Card: tiny-llm-400m (Colab Pro A100 — Rung 4)

## Overview
~399M-parameter GPT-style transformer trained on Google Colab Pro A100.
Rung 4 of the Small-Model Foundations progression.

## Architecture
- Parameters: ~399,038,848
- Layers: 12 | Embedding dim: 1664 | Heads: 16
- Context length: 128 | Vocab: 13 (char-level)
- Weight tying: yes | Biases: none

## Memory Techniques
- bf16 AMP: halves weight memory vs fp32
- Gradient checkpointing: ~70% activation memory reduction
- Gradient clipping (norm=1.0): prevents loss spikes

## Training
- Corpus: "to be or not to be that is the question " x200
- Optimizer: AdamW lr=3e-4 | Steps: 5000 | Batch: 16
- Hardware: Google Colab Pro A100 High RAM
- Framework: PyTorch

## Files
- tiny_llm_400m.py, tiny_llm_400m.pt, test_tiny_llm_400m.py
