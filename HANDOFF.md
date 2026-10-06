# Session handoff — PrismML-style 3B Telugu model (6 Oct 2026)

Read this first in a new session. It records what was decided, what is done, and what to do next.

## Goal

Copy the idea behind **PrismML** for Yodha: take a real model that is already smart, teach it Telugu, then compress it heavily so it can run cheaply and offline. PrismML's Bonsai models are 1-bit and ternary (1.58-bit) LLMs. They were not trained from scratch: Qwen models were compressed to ±1 or {-1, 0, +1} weights with one FP16 scale per 128 weights, then **re-trained** (quantization-aware training) so they keep their quality. Bonsai 8B is 1.15 GB; Bonsai 2 27B is 5.9 GB.

## Facts established

- **The live voice assistant (`voice-assistant/api.py`) uses neither our 25M nor our 3B model.** Everything runs on Gemini: speech-to-text, Telugu→English translation for the manual search, and the answers (`LLM_MODELS`, `api.py:19`). Text-to-speech is gTTS. The manual search uses `multilingual-e5-small` + ChromaDB.
- `tiny_llm_telugu.py` (25M parameters) was a training experiment and was never connected to the app.
- `tiny_llm_3b.py` was trained only on `"to be or not to be…" * 200` with 13 characters. It memorised one sentence, so it **can't be compressed into anything useful**. We therefore compress a real pretrained 3B model instead (default: `meta-llama/Llama-3.2-3B-Instruct`).
- **Constraints from the owner:** don't disturb the working app architecture; don't run anything on the Jetson. All training is done in Google Colab (A100-80GB).
- The Azure VM (D2s v3, 2 CPUs, no GPU) would run a 3B model only slowly (estimated 30–60 s or more per answer), so any later live use would need a GPU VM, or an optional switch with Gemini as the fallback.

## Done (merged to main in PR #43)

Everything is in `compression/`; read `compression/README.md` for the copy-paste Colab cells.

| File | What it does |
|---|---|
| `finetune_telugu_3b.py` | **Step 1.** LoRA fine-tune of a 3B model on Telugu Q&A (.jsonl/.csv). Uses Yodha's system prompt from `api.py`. Trains only on the answer tokens. Can optionally mix in raw Telugu text (`--text-dataset ai4bharat/samanantar`). Resumes from Drive after a disconnect. Stages: `train` → `merge` → `eval`. |
| `ternary_qat_3b.py` | **Step 2.** PrismML-style compression: `BitLinear` layers ({-s, 0, +s}, one scale per 128 weights), distillation from the frozen teacher, ramp from full precision to ternary over 500 steps, pack 2-bit weights + int8 embeddings. Stages: `train` → `export` → `eval`. Expected size for a 3B model: 6.4 GB → ~1.1 GB. |
| `test_finetune_telugu.py`, `test_ternary_qat.py` | CPU end-to-end tests on tiny random models; both pass. **The scripts have not yet been run on a real 3B model.** |
| `README.md` | Full guide: Colab cells, data formats, memory and time estimates, limitations. |

## Next steps (in order)

1. **Prepare data.** Put the 8,439 Telugu Q&A pairs, plus Q&A written from the vehicle manuals, into a `.jsonl` file. Use one line per pair: `{"question": "...", "answer": "..."}`. Upload it to Google Drive as `telugu_qa.jsonl`. These files are **not** in the repo.
2. **Accounts.** On Hugging Face, accept the Llama 3.2 license, then run `huggingface-cli login` in Colab. The repo is private, so Colab needs a GitHub token to clone it, or upload the `compression/` folder instead.
3. **Run Step 1 in Colab:** `finetune_telugu_3b.py train`, then `merge`, then `eval`. Estimated time is ~1–3 h. Check the eval output: held-out answer loss should go down, the share of Telugu script in answers should be high, and the side-by-side answers should look right.
4. **Run Step 2 in Colab:** `ternary_qat_3b.py train --teacher /content/drive/MyDrive/telugu-3b/merged`, then `export`, then `eval`. Start with 3,000 steps (~3–5 h); use 10k–30k steps for better quality. Compare ternary against the teacher in eval.
5. **Optional quick baseline:** 4-bit GGUF with llama.cpp (Path A in the README, ~15 min, 6.4 GB → ~2 GB). Use it to compare against the ternary model.
6. **Later, only if quality is good:**
   - Write a converter from `ternary_model.pt` to llama.cpp. Its ternary formats use a different scale layout, so this is not written yet.
   - Add an optional `LOCAL_LLM_URL` switch in `_llm_call_with_fallback` (`api.py:71`), with Gemini as automatic fallback. Off by default; run in shadow mode first. The app must not be disturbed.

## Other notes

- `tiny_llm.pt` and `tiny_llm_1m.pt` show as "modified" right after cloning, because `.gitattributes` routes `*.pt` through Git LFS but these two were committed as plain files. Don't commit them as they are, or the checkpoints get replaced by LFS pointer files. Ignore them locally with `git update-index --assume-unchanged tiny_llm.pt tiny_llm_1m.pt`.
- GitHub Dependabot reports 5 vulnerabilities on `main` (1 critical). These come from existing dependencies, not this work, and haven't been looked at yet.
