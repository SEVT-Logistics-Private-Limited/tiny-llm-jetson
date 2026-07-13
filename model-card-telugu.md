# Model Card: tiny-llm-telugu (Rung 9)

## Overview
First real-language model in the Small-Model Foundations progression.
Trained on actual Telugu text (ai4bharat/samanantar) — not a placeholder corpus.

## Architecture
- Parameters: 25,374,208
- n_embd: 512 | n_layer: 8 | n_head: 8 | block_size: 128
- Vocab: 245 Telugu characters

## Training
- Corpus: ai4bharat/samanantar (50,000 sentences, 2.9M characters)
- Optimizer: AdamW lr=3e-4 | Steps: 10,000 | Batch: 16
- Hardware: Google Colab Pro A100-SXM4-80GB | bf16 + gradient checkpointing

## Results
- Initial loss: 331.16
- Final loss (step 9000): 1.42
- Output: real Telugu sentences generated correctly

## Checkpoint
- GitHub: this repo (97MB)
- HuggingFace: https://huggingface.co/RishikUttejSasanala/tiny-llm-telugu

## Purpose
Foundation layer for Telugu voice assistant pipeline.
Next steps: connect Sarvam AI STT + this model + Bulbul TTS via FastAPI.
