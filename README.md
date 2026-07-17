# tiny-llm-jetson

From-scratch GPT transformers scaled from 984 params to 3B+, applied as the foundation of a live Telugu Voice AI Assistant.

**Author:** Rishi Sasanala · Yodha AI (SEVT Logistics Private Limited)
**Hardware:** Jetson Orin Nano Super + Colab Pro A100-SXM4-80GB
**Stack:** PyTorch · FastAPI · Sarvam AI · Azure · Nginx

---

## Overview

Two parts:

1. **Scale experiments** — GPT trained on English text, scaled 984 params to 3B+ to understand what changes with scale
2. **Telugu Language** — 25M param GPT trained on real Telugu corpus, fine-tuned on 8,439 Q&A pairs, deployed as a live voice assistant

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
| 8 MAX | 3.07B | Colab A100-80GB | 0.0977 | Hardware maximum on A100-80GB |
| 9 | 25M | Colab A100-80GB | 1.42 | Telugu — first real language |

Checkpoints: [HuggingFace/RishikUttejSasanala](https://huggingface.co/RishikUttejSasanala)

---

## Telugu Voice AI Assistant

**Live:** https://52.140.52.237.nip.io/assistant
**API:** https://52.140.52.237.nip.io/docs

### Pipeline
Audio In → Sarvam Saarika STT → Sarvam-105b LLM → Sarvam Bulbul TTS → Audio Out

### API Endpoints
| Endpoint | Function |
|----------|----------|
| GET / | Health check |
| POST /tts | Text to Telugu audio |
| POST /stt | Telugu audio to text |
| POST /speak | Text to LLM to Telugu audio |
| POST /voice | Full pipeline audio in to audio out |
| POST /converse | Conversational with history |
| GET /assistant | Web voice assistant UI |

### Features
- Continuous conversation loop
- Conversation memory across turns
- Pure Telugu responses, no markdown artifacts
- 1.2s LLM response time
- Premium dark UI with typing indicators

### Infrastructure
- Azure VM Standard D2s v3, Ubuntu 24.04, South India
- Nginx + Let's Encrypt SSL, auto-renews
- systemd service, auto-restarts on crash or reboot
- Domain: 52.140.52.237.nip.io

### Key Files
- telugu_voice_api.py — FastAPI server
- telugu_assistant.html — Voice assistant UI
- tiny_llm_telugu.py — Telugu GPT architecture
- tiny_llm_telugu.pt — Trained checkpoint

---

## Roadmap
- Build conversational Telugu dataset, replace Sarvam-105b with own model
- Streaming responses for under 4 second latency
- Integrate into Paperclip / YODA platform
- Offline edge deployment on Jetson Orin Nano
