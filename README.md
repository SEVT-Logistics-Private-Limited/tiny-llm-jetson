# Tiny LLM on Jetson Orin

A 984-parameter GPT-style transformer, built and trained entirely from
scratch on an NVIDIA Jetson Orin Nano Super edge device.

This is the rung-1 artifact in the Small-Model Foundations progression --
proving the complete from-scratch training pipeline (data, tokenizer,
transformer, optimizer, checkpoint) works end-to-end on accessible edge
hardware, before scaling toward larger and more useful models.

See model-card.md for full architecture, training details, and results.

## Author
Rishi

## Files
- tiny_llm.py -- training script
- tiny_llm.pt -- trained checkpoint (984 params, 8.7KB)
- test_tiny_llm.py -- independent load-and-generate verification script
- model-card.md -- full model card
