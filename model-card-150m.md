# Model Card: tiny-llm-150m

## Overview
A 151,204,504-parameter GPT-style transformer, trained entirely from
scratch on an NVIDIA Jetson Orin Nano Super. This is rung 3 in the
Small-Model Foundations progression -- roughly 150x rung 2 and 153,000x
rung 1. Unlike the earlier rungs, this run did not converge cleanly, and
the reasons why are documented here as a deliberate part of the result.

## Architecture
- Type: Decoder-only transformer (GPT architecture)
- Parameters: 151,204,504
- Layers: 6
- Embedding dim: 1448
- Attention heads: 8
- Context length (block_size): 128
- Vocabulary: 13 (character-level)
- Weight tying: yes
- Biases: none

## Training
- Corpus: "to be or not to be that is the question " x 200 (char-level) -- unchanged from rungs 1 and 2
- Optimizer: AdamW, lr=3e-3 (unchanged from rungs 1 and 2)
- Steps: 5000 (unchanged from rungs 1 and 2)
- Batch size: 8 (reduced from 32, deliberately, to fit available memory)
- Hardware: Jetson Orin Nano Super, JetPack 6.2.1, CUDA 12.6, MAXN_SUPER power mode
- Framework: PyTorch (dustynv/l4t-pytorch:r36.4.0 container)
- Training time: several minutes on-device

## Results
- Initial loss: 966.684
- Final loss (step 4500): 1.954
- Best loss (step 3500): 0.940
- Loss did not converge cleanly -- dropped quickly in the first 500
  steps, then plateaued and oscillated between roughly 1.0 and 2.0 for
  the remainder of training, never reaching the near-zero loss achieved
  by the rung-2 (990,200 param) model.
- Sample output: "to ber i nor nonor t besthathathe or the be ist ist n
  be ot that" -- visibly more garbled than rung 2, comparable to or
  worse than rung 1, despite 150x more parameters than rung 2.

## Independent Verification
Reloaded from the saved checkpoint in a separate script
(test_tiny_llm_150m.py) and tested against 4 seed strings. All 4
confirm the architecture loads and runs correctly (parameter count
matches exactly), but generation quality is poor across all seeds --
consistent with the training loss never settling.

## Memory and Reliability Notes
This run required real troubleshooting before succeeding:
- A first attempt at this exact config crashed mid-training with a
  CUDA allocator failure (CUDACachingAllocator.cpp assertion), after
  successfully completing step 0.
- A "system throttled" thermal warning appeared during that first
  attempt. Direct tegrastats monitoring during a subsequent idle period
  showed junction temperature stable at ~53C, well below throttling
  thresholds -- ruling out heat as the actual cause.
- The fix that worked: clearing page cache (sync && echo 3 >
  /proc/sys/vm/drop_caches) on the host before re-entering the
  container. After this, training completed all 5000 steps without
  error. This points to memory allocator pressure (not heat) as the
  real cause of the first failure, consistent with earlier findings
  at much smaller scale (the original 984-param run).

## Interpretation
This result demonstrates that parameter count alone does not determine
output quality. Three changes were made simultaneously to fit this
model into available memory and get it running at all (much larger
architecture, smaller batch size, same unchanged learning rate and step
count) -- and the combination produced a model that is technically
larger and harder to train, but qualitatively worse than the much
smaller rung-2 model. This is a genuine, useful finding: it shows the
practical ceiling for *clean, well-converged* training on this specific
hardware sits closer to rung 2's scale than to this one, at least
without further hyperparameter tuning (lower learning rate, more steps,
or a larger batch size achieved through other memory optimizations
such as mixed-precision training).

## Files
- tiny_llm_150m.py -- training script
- tiny_llm_150m.pt -- checkpoint (model weights + tokenizer + config)
- test_tiny_llm_150m.py -- independent load-and-generate verification script

## Honest Conclusion on Scale-Up
Across three rungs (984 -> 990,200 -> 151,204,504 parameters), the
clear winner on output quality is rung 2, not rung 3. This is the
correct, evidence-based stopping point for parameter-count scaling on
this corpus and this hardware, absent further training-recipe tuning.
The original question -- "what's the maximum parameters we can train
on this Jetson" -- now has a real, tested answer: the device can build
and run a forward/backward pass at 150M+ parameters, but reliable,
well-converged training at that scale was not achieved with this
training recipe. 990,200 parameters represents the best verified
result obtained so far.
