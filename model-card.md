# Model Card: tiny-llm-984

## Overview
A 984-parameter GPT-style transformer, trained entirely from scratch on an
NVIDIA Jetson Orin Nano Super. This is the rung-1 artifact in the
Small-Model Foundations progression -- a teaching-scale model that proves
the full from-scratch training pipeline on edge hardware.

## Architecture
- Type: Decoder-only transformer (GPT architecture)
- Parameters: 984
- Layers: 1
- Embedding dim: 8
- Attention heads: 2
- Context length (block_size): 8
- Vocabulary: 13 (character-level)
- Weight tying: yes (output head shares weights with token embedding)
- Biases: none (linear layers and attention are bias-free)

## Training
- Corpus: "to be or not to be that is the question " x 200 (char-level)
- Optimizer: AdamW, lr=3e-3
- Steps: 5000, batch size 32
- Hardware: Jetson Orin Nano Super, JetPack 6.2.1, CUDA 12.6, MAXN_SUPER power mode
- Framework: PyTorch (dustynv/l4t-pytorch:r36.4.0 container)
- Training time: under 1 minute on-device

## Results
- Initial loss: 6.604
- Final loss (step 4500): 0.175
- Best loss (step 3500): 0.165
- Sample output: "to be or not to be that is the ques thes n t ionns
  ttthe que tionne o or not to be th"

## Independent Verification
Reloaded from the saved checkpoint in a separate script (test_tiny_llm.py)
and used to generate text from 4 unseen seed strings. All 4 produced
coherent, corpus-consistent continuations, confirming the checkpoint is
self-contained and usable independent of the training script.

## Interpretation
The model learns the surface structure and recurring tokens of the
training corpus (correctly reproduces "to be or not to be that is the
qu..." as an opening) but does not produce fluent or grammatical text --
expected and intended at this parameter scale. This validates the
from-scratch training pipeline (data -> tokenizer -> transformer ->
optimizer -> checkpoint) end-to-end on Jetson edge hardware.

## Files
- tiny_llm.py -- training script
- tiny_llm.pt -- checkpoint (model weights + tokenizer + config), 8.7KB
- test_tiny_llm.py -- independent load-and-generate verification script

## Next Steps (Scale-Up Ladder)
Increase n_embd (8->16->32) and block_size (8->32) to grow capacity.
Target direction: swap in a real Indic-language corpus for the next rung,
producing a small Indic-language model rather than scaling parameter
count on the English placeholder corpus alone.
