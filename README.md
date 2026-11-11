# tiny-llm-jetson

From-scratch GPT transformers scaled from 984 params to 3B+. They are the foundation of **Yodha**, a live Telugu voice AI assistant.

**Author:** Rishi Sasanala · Yodha AI (SEVT Logistics Private Limited)
**Hardware:** Jetson Orin Nano Super + Colab Pro A100-SXM4-80GB
**Stack:** PyTorch · FastAPI · Google Gemini · ChromaDB + multilingual-e5 (RAG) · Docker · Azure Container Apps · GitHub Actions

---

## Overview

Three tracks:

1. **Scale experiments:** a GPT trained on a toy English corpus, scaled from 984 params to 3B+ to learn what changes with scale.
2. **Telugu language models:**
   - a 25M param GPT trained from scratch on real Telugu text
   - a conversational Telugu fine-tune
   - a LoRA → ternary compression pipeline for a real 3B model ([`compression/`](compression/README.md))
3. **Yodha voice assistant:** a live Telugu voice assistant (Gemini LLM + RAG over vehicle manuals), deployed on Azure Container Apps.

---

## Model Rungs

| Rung | Params | Platform | Loss | Notes |
|------|--------|----------|------|-------|
| 1 | 984 | Jetson Orin | 0.175 | Pipeline proof on edge hardware |
| 2 | 990K | Jetson Orin | 0.016 | 1000x scale, flawless |
| 3 | 151M | Jetson + Colab T4 | 1.114 | First large model, bf16 |
| 4 | 399M | Colab A100 | — | Gradient checkpointing introduced |
| 5 | 1B | Colab A100-80GB | — | First billion-parameter run |
| 6 | 2B | Colab A100-80GB | — | 3.8GB checkpoint |
| 7 | 3B | Colab A100-80GB | 0.1098 | Near-perfect loss |
| 8 MAX | 3.07B | Colab A100-80GB | 0.0977 | Hardware maximum on A100-80GB (`tiny_llm_5b.py`) |
| 9 | 25M | Colab A100-80GB | 1.42 | Telugu: first real language |

Rungs 1–8 train on one repeated sentence, so low loss there means memorization, not language ability. Rung 9 is the first real-language model.

Checkpoints: [HuggingFace/RishikUttejSasanala](https://huggingface.co/RishikUttejSasanala)

---

## Telugu Voice AI Assistant (Yodha), v10

**Live:** https://telugu-voice-app.gentletree-f4002f64.southindia.azurecontainerapps.io/assistant
**API docs:** https://telugu-voice-app.gentletree-f4002f64.southindia.azurecontainerapps.io/docs
**Health:** https://telugu-voice-app.gentletree-f4002f64.southindia.azurecontainerapps.io/healthz

### Pipeline

```
Mic ──► Browser Web Speech API (te-IN) ──────────────► text ─┐
   └─► (fallback) MediaRecorder WebM ──► Gemini STT ──► text ─┤
                                                             ▼
              instant reply? (time / date / greetings, <100ms) ──► yes ──► answer
                                                             │ no
                                                             ▼
        RAG: translate → multilingual-e5 → ChromaDB (vehicle manuals), top-1, distance ≤ 0.35
                                                             ▼
        Gemini LLM (fallback chain: gemini-3.6-flash → gemini-2.0-flash → gemini-1.5-flash)
                                                             ▼
        clean_for_tts (strip symbols / English-only lines) ──► gTTS (Telugu) ──► audio
                                                             └─► (fallback) browser speechSynthesis
```

| Stage | Primary | Fallback |
|---|---|---|
| Speech-to-text | Browser Web Speech API (`te-IN`), so no audio upload | Gemini (inline `audio/webm`) |
| LLM | Gemini (`gemini-3.6-flash`) | `gemini-2.0-flash` → `gemini-1.5-flash` → Telugu apology message |
| Knowledge | RAG over vehicle manuals (ChromaDB + `intfloat/multilingual-e5-small`) | Answers without manual context |
| Text-to-speech | gTTS Telugu (`co.in` → `com`, 15s timeout) | Browser `speechSynthesis` |

### API Endpoints (`voice-assistant/api.py`)

| Endpoint | Function |
|----------|----------|
| GET `/healthz` | Liveness probe (no I/O) |
| GET `/` | Status: version, IST time, knowledge-base state (`kb: loaded / disabled`) |
| GET `/assistant` | Web voice assistant UI |
| GET `/docs` | OpenAPI docs |
| GET `/debug` | Diagnostics: tests each Gemini model, TTS backends, KB |
| GET `/speak` | TTS test (greeting) |
| POST `/tts` | Text → Telugu audio |
| POST `/converse` | Audio + history → transcript, answer, audio, history |
| POST `/converse_stream` | Same as `/converse`, as Server-Sent Events (`transcript` → `response` → `done`) |
| POST `/converse_text_stream` | Pre-transcribed text + history → SSE (skips STT; used by the UI) |
| POST `/voice` | Single-turn audio in → audio out |

### Features
- Continuous conversation loop with memory (last 8 turns)
- Answers grounded in vehicle manuals via RAG, with the source attached to the prompt
- Real-time context: current IST time, date and weekday in every prompt
- Instant replies (time, date, greetings) in under 100ms without calling the LLM
- Mostly-Telugu answers (technical English allowed), with markdown and symbols stripped before TTS
- SSE streaming. TTS is fetched in parallel as soon as the answer text arrives.
- Barge-in: speak over Yodha to interrupt
- Web Audio VAD for the recording fallback, plus a shared audio element for mobile autoplay

> **Knowledge base status:** RAG loads in the background from a ChromaDB volume at `/mnt/vehicledata`. If the volume isn't mounted, the API keeps working without manual context and `GET /` reports `"kb": "disabled"`. Check that field after every deploy.

### Infrastructure
- **Hosting:** Azure Container Apps (South India), min 1 replica
- **Image:** `voice-assistant/Dockerfile` (python:3.11-slim, pinned deps; `multilingual-e5-small` baked in so RAG runs offline)
- **Registry:** Azure Container Registry (`telugu-voice-ai:<git-sha>` and `:latest`)
- **CI/CD:** `.github/workflows/deploy.yml`. Every push to `main` runs `az acr build`, then `az containerapp update`.
- **Secrets:** `GEMINI_API_KEY` (GitHub Actions secret → container env)
- **TLS:** managed by Container Apps ingress

### Key Files
| File | Purpose |
|---|---|
| `voice-assistant/api.py` | **Current** FastAPI server (v10) |
| `voice-assistant/telugu_assistant.html` | **Current** voice assistant UI |
| `voice-assistant/Dockerfile`, `requirements.txt` | Container build |
| `.github/workflows/deploy.yml` | Build & deploy to Azure Container Apps |
| `tiny_llm_telugu.py` / `tiny_llm_telugu.pt` | Telugu GPT architecture / checkpoint |
| `telugu_voice_api.py`, `telugu_assistant.html` (repo root) | Legacy v8 (all-Gemini incl. Gemini TTS), kept for reference |

---

## Compression pipeline (Colab)

[`compression/`](compression/README.md) takes a real pretrained 3B model through two stages:
1. **LoRA Telugu fine-tune** (`finetune_telugu_3b.py`).
2. **PrismML-style ternary QAT** (`ternary_qat_3b.py`): 6.4 GB → ~1.1 GB. This is the path to replacing the hosted LLM with our own model on the Jetson.

---

## Roadmap
- Replace Gemini with our own Telugu model (LoRA fine-tuned 3B → 4-bit / ternary)
- Stream LLM tokens with sentence-level TTS for lower end-to-end latency
- Mount and version the vehicle-manual knowledge base in IaC; add a retrieval eval set
- Auth + rate limiting on the public API
- Integrate into Paperclip / YODA platform
- Offline edge deployment on Jetson Orin Nano

---

## Changelog

### v10 (current): Gemini + RAG on Azure Container Apps
- Moved from Azure VM (systemd + Nginx + Let's Encrypt, `52.140.52.237.nip.io`) to **Azure Container Apps** with GitHub Actions CI/CD
- LLM: Sarvam-105b → **Gemini** with a 3-model fallback chain and retry
- STT: browser Web Speech API primary, Gemini fallback
- TTS: gTTS with browser fallback
- **RAG over vehicle manuals** (ChromaDB + multilingual-e5-small)
- SSE streaming endpoints, barge-in, `/healthz`, `/debug`
- Prompt now allows technical English (natural code-switching) and sizes answers to the question

### v8: all-Gemini
- Gemini 2.0 Flash for STT, LLM and TTS (`gemini-2.0-flash-preview-tts`, voice "Aoede"); see `telugu_voice_api.py`

### v4.2 (20 July 2026): Sarvam on Azure VM
- Pipeline: Sarvam Saarika STT → Sarvam-105b LLM → Sarvam Bulbul TTS
- Instant response cache: time, date and greetings answered in <100ms without hitting the LLM
- Real-time context injection: current IST time, date and day of week in every response
- Strict Telugu-only output: English words and `*`, `/`, `#`, `_` stripped before TTS
- `max_tokens` reduced to 300 for faster responses
- Web Audio API VAD. A shared audio element fixed mobile autoplay.
- Infrastructure: Azure VM Standard D2s v3 (Ubuntu 24.04, South India), Nginx + Let's Encrypt, systemd

### Fine-tuned conversational model
- Trained on maya-research/IndicVault Telugu dataset (93,028 samples, 2M+ chunks)
- 3 epochs on Colab A100-80GB, loss 2.17 → 0.76
- Saved to HuggingFace: RishikUttejSasanala/tiny-llm-telugu-conv
