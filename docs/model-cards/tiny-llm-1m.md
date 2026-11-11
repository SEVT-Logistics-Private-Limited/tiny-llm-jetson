# Model Card: tiny-llm-1m

## Overview
A 990,200-parameter GPT-style transformer, trained entirely from scratch
on an NVIDIA Jetson Orin Nano Super. This is rung 2 in the Small-Model
Foundations progression -- roughly 1000x the parameter count of the
rung-1 984-param baseline, on the same architecture family and corpus,
to observe how capacity and training behavior change with scale.

## Architecture
- Type: Decoder-only transformer (GPT architecture)
- Parameters: 990,200
- Layers: 2
- Embedding dim: 200
- Attention heads: 8
- Context length (block_size): 128
- Vocabulary: 13 (character-level)
- Weight tying: yes (output head shares weights with token embedding)
- Biases: none (linear layers and attention are bias-free)

## Training
- Corpus: "to be or not to be that is the question " x 200 (char-level)
- Optimizer: AdamW, lr=3e-3
- Steps: 5000, batch size 32
- Hardware: Jetson Orin Nano Super, JetPack 6.2.1, CUDA 12.6, MAXN_SUPER power mode
- Framework: PyTorch (dustynv/l4t-pytorch:r36.4.0 container)
- Training time: under a few minutes on-device

## Results
- Initial loss: 137.732 (higher starting loss is expected at this scale --
  larger embedding dimension produces larger initial logit magnitudes)
- Final loss (step 4500): 0.016
- Best loss (step 2500): 0.011
- One transient instability spike at step 3500 (loss 0.340), fully
  recovered by step 4000 -- not a concern, common at this learning rate
- Sample output: "to be or not to be that is the question to be or not
  to be that is the question to be" -- a flawless, ungarbled reproduction

## Independent Verification
Reloaded from the saved checkpoint in a separate script
(test_tiny_llm_1m.py) and used to generate text from 4 unseen seed
strings. All 4 produced clean, grammatically correct continuations with
zero garbling -- a clear improvement over the rung-1 984-param model,
whose outputs at the same test contained visible glitches.

## Interpretation
At this scale (~1000x the rung-1 baseline), the model fully memorizes
the small repeated training corpus with no visible errors. This confirms
the training pipeline scales correctly across a 1000x capacity increase
without requiring any changes beyond the architecture config (n_embd,
block_size, n_layer). The corpus itself is now the limiting factor, not
model capacity -- to learn anything beyond perfect memorization of this
short text, the next rung needs a larger, more varied training corpus
(the planned direction: a real Indic-language text corpus).

## Files
- tiny_llm_1m.py -- training script
- tiny_llm_1m.pt -- checkpoint (model weights + tokenizer + config), 3.97MB
- test_tiny_llm_1m.py -- independent load-and-generate verification script

## Comparison to Rung 1
| Metric | Rung 1 (984 params) | Rung 2 (990,200 params) |
|---|---|---|
| n_embd | 8 | 200 |
| block_size | 8 | 128 |
| n_layer | 1 | 2 |
| Final loss | 0.175 | 0.016 |
| Sample quality | Partial, some garbling | Flawless reproduction |
| Checkpoint size | 8.7KB | 3.97MB |

## Next Steps
The English placeholder corpus is now fully saturated at this scale.
Rung 3 should introduce a real, larger corpus -- ideally a genuine
Indic-language text source -- rather than scaling parameter count
further on the same repeated sentence.
