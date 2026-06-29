# Tiny LLM on Jetson Orin

A series of GPT-style transformers, built and trained entirely from
scratch on an NVIDIA Jetson Orin Nano Super edge device -- no pretrained
weights, no fine-tuning, starting from random initialization every time.

This is the Small-Model Foundations progression: prove the complete
from-scratch training pipeline (data, tokenizer, transformer, optimizer,
checkpoint) works end-to-end on accessible edge hardware, then scale it
up step by step, rung by rung, and measure exactly what changes.

## Author
Rishi

## Hardware
NVIDIA Jetson Orin Nano Super, JetPack 6.2.1, CUDA 12.6, MAXN_SUPER
power mode, PyTorch via the dustynv/l4t-pytorch:r36.4.0 container.

## What's in this repo

### Rung 1: tiny_llm.py / tiny_llm.pt
A 984-parameter, 1-layer transformer. The smallest possible "real" GPT
architecture: token + position embeddings, single attention block,
weight-tied output head, no biases. Trained on a short repeated English
sentence for 5,000 steps in under a minute.

Result: loss fell from 6.604 to 0.175. The model learns the corpus's
structure -- correctly reproduces long stretches of the training text --
but does not fully memorize it, with some garbled segments. This is
expected and intentional at this scale: the goal is to prove the
pipeline works, not to produce fluent or perfect text.

Full details: model-card.md
Verification: test_tiny_llm.py independently reloads the checkpoint and
generates from 4 unseen seed strings.

### Rung 2: tiny_llm_1m.py / tiny_llm_1m.pt
A 990,200-parameter, 2-layer transformer -- roughly 1000x the parameter
count of rung 1, on the same architecture family and the same training
corpus. Built to test how training behavior changes purely from scale.

Result: loss fell to 0.016 -- a flawless, fully memorized reproduction
of the training corpus with zero garbling. At this capacity, the model
has more than enough room to perfectly learn the short repeated sentence,
so the corpus itself becomes the limiting factor rather than the model.

Full details: model-card-1m.md
Verification: test_tiny_llm_1m.py, same independent reload-and-generate
test as rung 1.

## Why two rungs, same corpus
The point of holding the training text constant across rungs is to
isolate exactly one variable: parameter count. Rung 1 proves the
pipeline runs at all. Rung 2 proves it scales correctly by 1000x without
any change beyond the architecture config (embedding dimension, context
length, number of layers). Both checkpoints were independently verified
by reloading them in a separate script and generating text from seed
strings the training run never explicitly returned to -- not just
trusting the training log.

## Where this goes next
The English placeholder corpus is now fully saturated at rung-2 scale --
further scaling parameter count on the same short sentence won't teach
us anything new. The planned next step is to introduce a real, larger
training corpus, with the longer-term goal of producing a genuinely
useful small Indic-language model, rather than continuing to scale
parameters purely for their own sake on throwaway text.

## Files
- tiny_llm.py, tiny_llm.pt, test_tiny_llm.py, model-card.md -- rung 1
- tiny_llm_1m.py, tiny_llm_1m.pt, test_tiny_llm_1m.py, model-card-1m.md -- rung 2
