import base64, requests, os, re, io, traceback, json, time
import concurrent.futures
from google import genai as google_genai
from google.genai import types as genai_types
from gtts import gTTS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pytz

GEMINI_KEY = os.environ["GEMINI_API_KEY"]

gemini_client = google_genai.Client(api_key=GEMINI_KEY)

# Ordered fallback chain — first model that returns non-null text wins
LLM_MODELS = ["gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

import threading

_embed_model = None
_chroma_client = None
_manuals_collection = None

def _load_kb():
    global _embed_model, _chroma_client, _manuals_collection
    print("Loading manual knowledge base in background...")
    try:
        import os as _os
        # Use cached model only — never download from HuggingFace at runtime
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        import chromadb
        em = SentenceTransformer("intfloat/multilingual-e5-small")
        cc = chromadb.PersistentClient(path="/mnt/vehicledata")
        mc = cc.get_collection("vehicle_manuals")
        _embed_model = em
        _chroma_client = cc
        _manuals_collection = mc
        print("Vehicle manuals DB loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load vehicle manuals DB: {e}. RAG disabled.")

threading.Thread(target=_load_kb, daemon=True).start()


def _extract_text_from_response(resp):
    """Extract text from a GenerateContentResponse, bypassing response.text which
    raises ValueError when finish_reason != STOP (e.g. MAX_TOKENS in google-genai>=2.x)."""
    # Fast path: response.text works when finish_reason == STOP
    try:
        txt = resp.text
        if txt:
            return txt
    except Exception:
        pass
    # Slow path: extract from candidates directly (handles MAX_TOKENS, etc.)
    try:
        candidate = resp.candidates[0]
        if candidate.content and candidate.content.parts:
            txt = "".join(p.text for p in candidate.content.parts if getattr(p, "text", None)) or None
            return txt
    except Exception:
        pass
    return None


def _llm_call_with_fallback(contents, config, context_label="llm"):
    """Call generate_content with model fallback + 1 retry on transient ServerErrors.
    Returns (text, model_used) or (None, None)."""
    for model in LLM_MODELS:
        for attempt in range(2):
            try:
                resp = gemini_client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                txt = _extract_text_from_response(resp)
                if txt:
                    return txt, model
                # Diagnose why text is still empty
                try:
                    n = len(resp.candidates) if resp.candidates else 0
                    reason = str(resp.candidates[0].finish_reason) if n else "no_candidates"
                    print(f"{context_label}: empty text (model={model}, attempt={attempt}, candidates={n}, finish_reason={reason})")
                except Exception:
                    print(f"{context_label}: empty text (model={model}, attempt={attempt}, could not inspect candidates)")
                break  # empty text — don't retry, try next model
            except Exception as e:
                err_type = type(e).__name__
                if "ServerError" in err_type and attempt == 0:
                    print(f"{context_label}: ServerError (model={model}), retrying after 1s...")
                    time.sleep(1)
                    continue
                print(f"{context_label}: error (model={model}, attempt={attempt}, {err_type}): {e}")
                break
    return None, None


def gemini_translate(text, source_lang, target_lang):
    prompt = f"Translate the following text from {source_lang} to {target_lang}. Return only the translated text, no explanations.\n\nText: {text}"
    txt, _ = _llm_call_with_fallback(
        contents=prompt,
        config=genai_types.GenerateContentConfig(max_output_tokens=500),
        context_label="gemini_translate"
    )
    return txt.strip() if txt else text

def retrieve_manual_context(question_te, distance_threshold=0.35):
    if _manuals_collection is None:
        return None
    try:
        question_en = gemini_translate(question_te, "Telugu", "English")
        query_emb = _embed_model.encode(["query: " + question_en]).tolist()
        results = _manuals_collection.query(query_embeddings=query_emb, n_results=1)
        if not results["documents"][0]:
            return None
        doc = results["documents"][0][0]
        dist = results["distances"][0][0]
        source = results["metadatas"][0][0]["source"]
        if dist > distance_threshold:
            return None
        return {"passage": doc[:1200], "source": source}
    except Exception:
        return None

def get_context():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    days = ['సోమవారం','మంగళవారం','బుధవారం','గురువారం','శుక్రవారం','శనివారం','ఆదివారం']
    months = ['జనవరి','ఫిబ్రవరి','మార్చి','ఏప్రిల్','మే','జూన్','జులై','ఆగస్టు','సెప్టెంబర్','అక్టోబర్','నవంబర్','డిసెంబర్']
    return {
        "time": now.strftime("%I:%M %p"),
        "date": f"{now.day} {months[now.month-1]} {now.year}",
        "day": days[now.weekday()],
        "hour": now.hour
    }

INSTANT_RESPONSES = {
    "సమయం": lambda ctx: f"ఇప్పుడు సమయం {ctx['time']} IST.",
    "టైమ్": lambda ctx: f"ఇప్పుడు సమయం {ctx['time']} IST.",
    "time": lambda ctx: f"ఇప్పుడు సమయం {ctx['time']} IST.",
    "తేదీ": lambda ctx: f"ఈరోజు {ctx['date']}, {ctx['day']}.",
    "ఈరోజు": lambda ctx: f"ఈరోజు {ctx['date']}, {ctx['day']}.",
    "నమస్కారం": lambda ctx: "నమస్కారం! నేను యోధను. మీకు ఎలా సహాయం చేయగలను?",
    "హలో": lambda ctx: "హలో! నేను యోధను. చెప్పండి.",
    "హాయ్": lambda ctx: "హాయ్! నేను యోధను. మీకు ఎలా సహాయం చేయగలను?",
}

def check_instant(text, ctx):
    t = text.strip().lower().rstrip('?.,!')
    for kw, fn in INSTANT_RESPONSES.items():
        if t == kw.lower():
            return fn(ctx)
    return None

def gemini_stt(audio_bytes):
    """Transcribe Telugu audio using Gemini with model fallback."""
    contents = [
        genai_types.Content(parts=[
            genai_types.Part(
                inline_data=genai_types.Blob(mime_type="audio/webm", data=audio_bytes)
            ),
            genai_types.Part(
                text="Transcribe exactly what is spoken in Telugu in this audio. Return only the transcribed Telugu text, no explanations or translations."
            )
        ])
    ]
    txt, _ = _llm_call_with_fallback(
        contents=contents,
        config=genai_types.GenerateContentConfig(max_output_tokens=200),
        context_label="gemini_stt"
    )
    return txt.strip() if txt else ""

def clean_for_tts(text):
    text = re.sub(r'[*/#_+|\[\]{}<>^~`\\]', '', text)
    text = re.sub(r'\([^)]*[a-zA-Z]{4,}[^)]*\)', '', text)  # remove (parenthetical English explanations)
    text = re.sub(r'\b\d+\.\s*', '', text)  # remove numbered list markers
    # Remove lines that are entirely English (no Telugu chars) — preserve code-switching
    lines = text.split('\n')
    kept = [l for l in lines if sum(1 for c in l if 'ఀ' <= c <= '౿') > 0 or not re.search(r'[a-zA-Z]{4,}', l)]
    text = ' '.join(kept)
    text = re.sub(r'[.,;:!?]{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()
    telugu_count = sum(1 for c in text if 'ఀ' <= c <= '౿')
    if telugu_count < 3:
        return "మళ్ళీ చెప్పగలరా?"
    return text

def build_system(ctx, manual_context=None):
    base = (
        "నువ్వు యోధ — ఒక స్నేహితుడిలాంటి తెలుగు AI అసిస్టెంట్‌వి.\n"
        f"ఇప్పుడు సమయం {ctx['time']} IST, {ctx['date']}, {ctx['day']}.\n"
        "భారత రాజధాని న్యూఢిల్లీ, తెలంగాణ రాజధాని హైదరాబాద్, ఆంధ్రప్రదేశ్ రాజధాని అమరావతి.\n\n"
        "నువ్వు మాట్లాడే తీరు:\n"
        "— ప్రశ్నకు తగినట్టు జవాబు ఇవ్వు: చిన్న ప్రశ్నకు చిన్న జవాబు, వివరణ అవసరమైతే స్పష్టంగా చెప్పు.\n"
        "— ప్రధానంగా తెలుగులో మాట్లాడు. Technical terms లేదా common English పదాలు అవసరమైతే వాడవచ్చు.\n"
        "— *, #, _, / లాంటి symbols వాడకు — మాటల్లో సహజంగా చెప్పు.\n"
        "— వేడుకగా, నిజాయితీగా ఉండు. వినేవారికి comfortable గా అనిపించాలి.\n"
        "— Real-time డేటా (cricket scores, weather, breaking news) తెలియకపోతే చెప్పు: 'అది నాకు live గా తెలియదు మిత్రమా'\n"
        "— తెలియని విషయాలు తెలియదని చెప్పు — imagine చేయకు.\n"
    )
    if manual_context:
        base += (
            f"\nవాహన మాన్యువల్ నుండి సమాచారం ({manual_context['source']}):\n"
            f"{manual_context['passage']}\n"
            "పై సమాచారం ఆధారంగా వినియోగదారు ప్రశ్నకు సమాధానం ఇవ్వు.\n"
        )
    return base

def telugu_llm(messages):
    system_msg = messages[0]["content"]
    # Prepend system instruction as user+model turn — more universally supported
    # than system_instruction in GenerateContentConfig (some models return ClientError
    # when system_instruction is used, silently failing to generate any response).
    contents = [
        genai_types.Content(role="user", parts=[genai_types.Part(text=system_msg)]),
        genai_types.Content(role="model", parts=[genai_types.Part(text="సరే.")]),
    ]
    for m in messages[1:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(genai_types.Content(
            role=role,
            parts=[genai_types.Part(text=m["content"])]
        ))
    raw, model_used = _llm_call_with_fallback(
        contents=contents,
        config=genai_types.GenerateContentConfig(max_output_tokens=500),
        context_label="telugu_llm"
    )
    if raw:
        print(f"telugu_llm: response from model={model_used}")
        return clean_for_tts(raw)
    print("telugu_llm: all models returned None — returning fallback")
    return "క్షమించాలి, మళ్ళీ అడగగలరా?"

def _gtts_generate(text, tld):
    tts = gTTS(text=text, lang='te', slow=False, tld=tld)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

def do_tts(text):
    """TTS via gTTS with 15s server-side timeout and TLD fallback."""
    text = text[:500]
    for tld in ('co.in', 'com'):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_gtts_generate, text, tld)
                data = future.result(timeout=15)
            if not data:
                raise ValueError("Empty gTTS output")
            return data, "audio/mpeg"
        except concurrent.futures.TimeoutError:
            print(f"gTTS timed out after 15s (tld={tld})")
        except Exception as e:
            print(f"gTTS failed (tld={tld}, {type(e).__name__}): {e}")
    raise RuntimeError("All gTTS attempts failed")

app = FastAPI(title="Telugu Voice AI", version="10.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Msg(BaseModel):
    role: str
    content: str

class ConvReq(BaseModel):
    audio_base64: str
    history: Optional[List[Msg]] = []

class TextReq(BaseModel):
    text: str

class AudioReq(BaseModel):
    audio_base64: str

class TextConvReq(BaseModel):
    text: str
    history: Optional[List[Msg]] = []

@app.get("/healthz")
def healthz():
    """Lightweight liveness probe — returns immediately without any I/O."""
    return {"status": "ok"}

@app.get("/")
def health():
    ctx = get_context()
    kb = "loaded" if _manuals_collection is not None else "disabled"
    return {"status": "running", "version": "10.0", "time": ctx['time'], "date": ctx['date'], "kb": kb}

@app.get("/assistant")
def assistant():
    return FileResponse(
        "/home/azureuser/telugu_assistant.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}
    )

@app.get("/debug")
def debug():
    """Diagnostic endpoint — tests LLM, Gemini TTS, and gTTS independently."""
    result = {}

    # Test each LLM model: once without system_instruction, once with (to detect support)
    llm_results = {}
    for model in LLM_MODELS:
        entry = {}
        # Plain test (no system_instruction)
        try:
            resp = gemini_client.models.generate_content(
                model=model,
                contents="Say 'ok' in Telugu (one word only).",
                config=genai_types.GenerateContentConfig(max_output_tokens=50)
            )
            txt = _extract_text_from_response(resp)
            entry["plain"] = {"ok": bool(txt), "response": txt}
            if not txt:
                try:
                    n = len(resp.candidates) if resp.candidates else 0
                    entry["plain"]["candidates"] = n
                    if n:
                        entry["plain"]["finish_reason"] = str(resp.candidates[0].finish_reason)
                except Exception:
                    pass
        except Exception as e:
            entry["plain"] = {"ok": False, "error": type(e).__name__}
        # Test with system_instruction (same as old telugu_llm code path)
        try:
            resp2 = gemini_client.models.generate_content(
                model=model,
                contents=[genai_types.Content(role="user", parts=[genai_types.Part(text="హైదరాబాద్ ఏ రాష్ట్రానికి రాజధాని?")])],
                config=genai_types.GenerateContentConfig(
                    system_instruction="నువ్వు తెలుగు AI అసిస్టెంట్‌వి. తెలుగులో మాత్రమే జవాబు ఇవ్వు.",
                    max_output_tokens=200
                )
            )
            txt2 = _extract_text_from_response(resp2)
            entry["with_sys_instruction"] = {"ok": bool(txt2), "response": txt2}
            if not txt2:
                try:
                    n = len(resp2.candidates) if resp2.candidates else 0
                    entry["with_sys_instruction"]["candidates"] = n
                    if n:
                        entry["with_sys_instruction"]["finish_reason"] = str(resp2.candidates[0].finish_reason)
                except Exception:
                    pass
        except Exception as e:
            entry["with_sys_instruction"] = {"ok": False, "error": type(e).__name__}
        # Test with system as first user turn (new approach)
        try:
            resp3 = gemini_client.models.generate_content(
                model=model,
                contents=[
                    genai_types.Content(role="user", parts=[genai_types.Part(text="నువ్వు తెలుగు AI అసిస్టెంట్‌వి. తెలుగులో మాత్రమే జవాబు ఇవ్వు.")]),
                    genai_types.Content(role="model", parts=[genai_types.Part(text="సరే.")]),
                    genai_types.Content(role="user", parts=[genai_types.Part(text="హైదరాబాద్ ఏ రాష్ట్రానికి రాజధాని?")]),
                ],
                config=genai_types.GenerateContentConfig(max_output_tokens=200)
            )
            txt3 = _extract_text_from_response(resp3)
            entry["with_sys_as_turn"] = {"ok": bool(txt3), "response": txt3}
            if not txt3:
                try:
                    n = len(resp3.candidates) if resp3.candidates else 0
                    entry["with_sys_as_turn"]["candidates"] = n
                    if n:
                        entry["with_sys_as_turn"]["finish_reason"] = str(resp3.candidates[0].finish_reason)
                except Exception:
                    pass
        except Exception as e:
            entry["with_sys_as_turn"] = {"ok": False, "error": type(e).__name__}
        llm_results[model] = entry
    result["llm_models"] = llm_results
    # Summary: first model where plain test works
    working = next((m for m, v in llm_results.items() if v.get("plain", {}).get("ok")), None)
    result["llm"] = {"ok": working is not None, "active_model": working,
                     "response": llm_results[working]["plain"]["response"] if working else None}

    # Test Gemini TTS models (informational only — not in active production path)
    for model in ["gemini-2.5-flash-preview-tts", "gemini-2.0-flash-preview-tts"]:
        key = f"gemini_tts_{model}"
        try:
            resp = gemini_client.models.generate_content(
                model=model,
                contents="నమస్కారం",
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name="Kore"
                            )
                        )
                    )
                )
            )
            pcm = resp.candidates[0].content.parts[0].inline_data.data
            if isinstance(pcm, str):
                pcm = base64.b64decode(pcm)
            result[key] = {"ok": bool(pcm), "bytes": len(pcm) if pcm else 0,
                           "mime": resp.candidates[0].content.parts[0].inline_data.mime_type}
        except Exception as e:
            result[key] = {"ok": False, "error": type(e).__name__}

    # Test gTTS
    try:
        tts = gTTS(text="నమస్కారం", lang='te', slow=False, tld='co.in')
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        data = buf.read()
        result["gtts"] = {"ok": bool(data), "bytes": len(data)}
    except Exception as e:
        result["gtts"] = {"ok": False, "error": type(e).__name__}

    # Version info
    result["google_genai_version"] = getattr(google_genai, "__version__", "unknown")
    result["kb"] = "loaded" if _manuals_collection is not None else "disabled"

    return result


@app.get("/speak")
def speak():
    """Test TTS endpoint — plays a greeting without needing STT."""
    try:
        audio, content_type = do_tts("నమస్కారం! నేను యోధను. మీకు ఎలా సహాయం చేయగలను?")
        return {"audio_base64": base64.b64encode(audio).decode(), "content_type": content_type}
    except Exception as e:
        print(f"ERROR in /speak: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")

@app.post("/tts")
def tts(req: TextReq):
    try:
        audio, content_type = do_tts(req.text)
        return {"audio_base64": base64.b64encode(audio).decode(), "content_type": content_type}
    except Exception as e:
        print(f"ERROR in /tts: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")

@app.post("/converse")
def converse(req: ConvReq):
    try:
        ctx = get_context()
        user_text = gemini_stt(base64.b64decode(req.audio_base64))
        if not user_text.strip():
            raise HTTPException(400, "Empty transcript")
        instant = check_instant(user_text, ctx)
        if instant:
            audio, ct = do_tts(instant)
            history = list(req.history or [])
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": instant})
            return {"transcript": user_text, "response": instant,
                    "audio_base64": base64.b64encode(audio).decode(), "content_type": ct, "history": history}
        manual_ctx = retrieve_manual_context(user_text)
        msgs = [{"role": "system", "content": build_system(ctx, manual_ctx)}]
        for m in (req.history or [])[-8:]:
            msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": user_text})
        response = telugu_llm(msgs)
        audio, ct = do_tts(response)
        history = list(req.history or [])
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": response})
        return {"transcript": user_text, "response": response,
                "audio_base64": base64.b64encode(audio).decode(), "content_type": ct, "history": history}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in /converse: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")

@app.post("/converse_stream")
def converse_stream(req: ConvReq):
    """SSE streaming endpoint: emits transcript → response text → audio in order."""
    def _sse(obj):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def generate():
        try:
            ctx = get_context()
            user_text = gemini_stt(base64.b64decode(req.audio_base64))
            if not user_text.strip():
                yield _sse({"type": "error", "msg": "Empty transcript"})
                return

            yield _sse({"type": "transcript", "text": user_text})

            instant = check_instant(user_text, ctx)
            response_text = instant
            if not instant:
                manual_ctx = retrieve_manual_context(user_text)
                msgs = [{"role": "system", "content": build_system(ctx, manual_ctx)}]
                for m in (req.history or [])[-8:]:
                    msgs.append({"role": m.role, "content": m.content})
                msgs.append({"role": "user", "content": user_text})
                try:
                    response_text = telugu_llm(msgs)
                except Exception as llm_err:
                    print(f"LLM error in /converse_stream ({type(llm_err).__name__}): {traceback.format_exc()}")
                    response_text = "క్షమించాలి, నాకు ఇప్పుడు జవాబు ఇవ్వడం కష్టంగా ఉంది. మళ్ళీ అడగగలరా?"

            yield _sse({"type": "response", "text": response_text})

            # Convert Pydantic Msg objects to plain dicts so json.dumps can serialize them
            hist = [{"role": m.role, "content": m.content} for m in (req.history or [])]
            hist.append({"role": "user", "content": user_text})
            hist.append({"role": "assistant", "content": response_text})
            yield _sse({"type": "done", "history": hist})
        except Exception as e:
            print(f"ERROR in /converse_stream ({type(e).__name__}: {e}):\n{traceback.format_exc()}")
            yield _sse({"type": "error", "msg": "Request failed. Please try again."})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )

@app.post("/converse_text_stream")
def converse_text_stream(req: TextConvReq):
    """SSE streaming for pre-transcribed text — skips STT for lower latency."""
    def _sse(obj):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def generate():
        try:
            ctx = get_context()
            user_text = req.text.strip()
            if not user_text:
                yield _sse({"type": "error", "msg": "Empty text"})
                return

            yield _sse({"type": "transcript", "text": user_text})

            instant = check_instant(user_text, ctx)
            response_text = instant
            if not instant:
                manual_ctx = retrieve_manual_context(user_text)
                msgs = [{"role": "system", "content": build_system(ctx, manual_ctx)}]
                for m in (req.history or [])[-8:]:
                    msgs.append({"role": m.role, "content": m.content})
                msgs.append({"role": "user", "content": user_text})
                try:
                    response_text = telugu_llm(msgs)
                except Exception as llm_err:
                    print(f"LLM error in /converse_text_stream ({type(llm_err).__name__}): {traceback.format_exc()}")
                    response_text = "క్షమించాలి, నాకు ఇప్పుడు జవాబు ఇవ్వడం కష్టంగా ఉంది. మళ్ళీ అడగగలరా?"

            yield _sse({"type": "response", "text": response_text})

            # Convert Pydantic Msg objects to plain dicts so json.dumps can serialize them
            hist = [{"role": m.role, "content": m.content} for m in (req.history or [])]
            hist.append({"role": "user", "content": user_text})
            hist.append({"role": "assistant", "content": response_text})
            yield _sse({"type": "done", "history": hist})
        except Exception as e:
            print(f"ERROR in /converse_text_stream ({type(e).__name__}: {e}):\n{traceback.format_exc()}")
            yield _sse({"type": "error", "msg": "Request failed. Please try again."})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )

@app.post("/voice")
def voice(req: AudioReq):
    try:
        ctx = get_context()
        text = gemini_stt(base64.b64decode(req.audio_base64))
        if not text.strip():
            raise HTTPException(400, "Empty transcript")
        instant = check_instant(text, ctx)
        if instant:
            audio, ct = do_tts(instant)
            return {"transcript": text, "response": instant,
                    "audio_base64": base64.b64encode(audio).decode(), "content_type": ct}
        manual_ctx = retrieve_manual_context(text)
        msgs = [{"role": "system", "content": build_system(ctx, manual_ctx)},
                {"role": "user", "content": text}]
        response = telugu_llm(msgs)
        audio, ct = do_tts(response)
        return {"transcript": text, "response": response,
                "audio_base64": base64.b64encode(audio).decode(), "content_type": ct}
    except Exception as e:
        print(f"ERROR in /voice: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")
