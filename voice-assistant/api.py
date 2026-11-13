import base64, requests, os, re, io, traceback, json, time
from google import genai as google_genai
from google.genai import types as genai_types
from gtts import gTTS
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional
from datetime import datetime
import pytz
import auth
from auth import require_user

GEMINI_KEY = os.environ["GEMINI_API_KEY"]

gemini_client = google_genai.Client(api_key=GEMINI_KEY)

# Ordered fallback chain — first model that returns non-null text wins.
# gemini-2.0-flash and gemini-1.5-flash have been shut down by Google, so they are not used.
# Override without a code change: LLM_MODELS="model-a,model-b"
LLM_MODELS = [m.strip() for m in os.environ.get(
    "LLM_MODELS", "gemini-3.6-flash,gemini-3.8-flash,gemini-3.5-flash-lite").split(",") if m.strip()]

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


def _llm_call_with_fallback(contents, config, context_label="llm", models=None):
    """Call generate_content with model fallback + 1 retry on transient ServerErrors.
    Returns (text, model_used) or (None, None)."""
    for model in (models or LLM_MODELS):
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


# Translation is a small task: try the fastest model first.
TRANSLATE_MODELS = [m.strip() for m in os.environ.get(
    "TRANSLATE_MODELS", "gemini-3.5-flash-lite").split(",") if m.strip()] + LLM_MODELS

def gemini_translate(text, source_lang, target_lang):
    prompt = f"Translate the following text from {source_lang} to {target_lang}. Return only the translated text, no explanations.\n\nText: {text}"
    txt, _ = _llm_call_with_fallback(
        contents=prompt,
        config=genai_types.GenerateContentConfig(max_output_tokens=500),
        context_label="gemini_translate",
        models=TRANSLATE_MODELS,
    )
    return txt.strip() if txt else text

RAG_DISTANCE_THRESHOLD = float(os.environ.get("RAG_DISTANCE_THRESHOLD", "0.35"))
# If the direct Telugu search is further than this, the question is treated as unrelated to
# the manuals and the translation retry is skipped. Unset = always retry (same answers as
# before). Pick a value from the "rag:" log lines once there is real traffic.
_skip = os.environ.get("RAG_SKIP_TRANSLATE_ABOVE", "").strip()
RAG_SKIP_TRANSLATE_ABOVE = float(_skip) if _skip else None

def _search_manuals(query_text):
    """Best match as (distance, passage, source), or None when the collection is empty."""
    query_emb = _embed_model.encode(["query: " + query_text]).tolist()
    results = _manuals_collection.query(query_embeddings=query_emb, n_results=1)
    if not results["documents"][0]:
        return None
    return (results["distances"][0][0], results["documents"][0][0],
            results["metadatas"][0][0]["source"])

def retrieve_manual_context(question_te):
    """Search with the Telugu question directly (the embedding model is multilingual, ~20 ms).
    Only when that misses, translate to English and retry (an extra LLM call, ~1 s)."""
    if _manuals_collection is None:
        return None
    try:
        t0 = time.monotonic()
        hit = _search_manuals(question_te)
        if hit is None:
            return None
        direct_dist, path, translated_dist = hit[0], "direct", None
        if direct_dist > RAG_DISTANCE_THRESHOLD:
            if RAG_SKIP_TRANSLATE_ABOVE is not None and direct_dist > RAG_SKIP_TRANSLATE_ABOVE:
                path = "skipped"
            else:
                question_en = gemini_translate(question_te, "Telugu", "English")
                path = "translated"
                if question_en != question_te:
                    hit = _search_manuals(question_en) or hit
                    translated_dist = hit[0]
        matched = hit[0] <= RAG_DISTANCE_THRESHOLD
        print("rag: " + json.dumps({"path": path, "direct": round(direct_dist, 3),
              "translated": None if translated_dist is None else round(translated_dist, 3),
              "hit": matched, "ms": int((time.monotonic() - t0) * 1000)}), flush=True)
        return {"passage": hit[1][:1200], "source": hit[2]} if matched else None
    except Exception as e:
        print(f"rag: error ({type(e).__name__}): {e}")
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

def gemini_stt(audio_bytes, mime_type="audio/webm"):
    """Transcribe Telugu audio using Gemini with model fallback.
    mime_type must match the recording (Safari records audio/mp4, Chrome/Firefox audio/webm)."""
    contents = [
        genai_types.Content(parts=[
            genai_types.Part(
                inline_data=genai_types.Blob(mime_type=mime_type, data=audio_bytes)
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

TELUGU_RE = re.compile(r'[\u0C00-\u0C7F]')

def _telugu_count(text):
    return len(TELUGU_RE.findall(text))

def clean_for_display(text):
    """Strip markdown formatting only. Keeps English, numbers, part names and line breaks,
    because this is what the user reads and what goes into conversation history."""
    text = text.replace('`', '')
    text = re.sub(r'\*\*|__', '', text)                    # bold markers
    text = re.sub(r'(?m)^\s*#+\s*', '', text)              # headings
    text = re.sub(r'(?m)^\s*[*\-•]\s+', '', text)          # bullet markers
    text = re.sub(r'(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)', '', text)  # stray single asterisks
    lines = [re.sub(r'[ \t]+', ' ', l).strip() for l in text.split('\n')]
    return '\n'.join(l for l in lines if l)

def clean_for_tts(text):
    """Make text speakable by a Telugu voice. Applied only to the audio, never to what is shown."""
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', text)        # numbered-list markers at line start only,
                                                           # so decimals like 3.5 survive
    text = re.sub(r'[*#_+|\[\]{}<>^~`\\]', '', text)
    text = text.replace('/', ' ')
    spoken = re.sub(r'\([^)]*[a-zA-Z]{4,}[^)]*\)', '', text)  # (English explanations in brackets)
    # Lines with no Telugu that are mostly English words are usually translations the model
    # appended; drop them from the audio only. Short specs like "SAE 5W-30" are kept.
    kept = [l for l in spoken.split('\n')
            if _telugu_count(l) or len(re.findall(r'[a-zA-Z]{4,}', l)) < 4]
    spoken = ' '.join(kept)
    if _telugu_count(spoken) < 3:   # nothing Telugu left: speak the original rather than drop it
        spoken = text.replace('\n', ' ')
    spoken = re.sub(r'[,;:!?]{2,}|\.{2,}', '.', spoken)
    return re.sub(r'\s+', ' ', spoken).strip()

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

LLM_FALLBACK_TEXT = "క్షమించాలి, మళ్ళీ అడగగలరా?"

def _build_contents(messages):
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
    return contents

def telugu_llm(messages):
    raw, model_used = _llm_call_with_fallback(
        contents=_build_contents(messages),
        config=genai_types.GenerateContentConfig(max_output_tokens=500),
        context_label="telugu_llm"
    )
    if raw:
        print(f"telugu_llm: response from model={model_used}")
        return clean_for_display(raw)
    print("telugu_llm: all models returned None — returning fallback")
    return LLM_FALLBACK_TEXT

def telugu_llm_stream(messages):
    """Yield raw text pieces as the model produces them. Falls back to the next model only
    while nothing has been sent yet; a failure mid-answer keeps what was already produced.
    Yields nothing if every model fails."""
    contents = _build_contents(messages)
    config = genai_types.GenerateContentConfig(max_output_tokens=500)
    for model in LLM_MODELS:
        produced = False
        try:
            for chunk in gemini_client.models.generate_content_stream(
                    model=model, contents=contents, config=config):
                piece = _extract_text_from_response(chunk)
                if piece:
                    produced = True
                    yield piece
        except Exception as e:
            print(f"telugu_llm_stream: error (model={model}, mid_answer={produced}, {type(e).__name__}): {e}")
        if produced:
            print(f"telugu_llm_stream: response from model={model}")
            return
    print("telugu_llm_stream: all models returned nothing")

# A piece is sent for speech once it ends a sentence (or line) and is long enough to sound
# natural. Requiring whitespace after the full stop keeps decimals like 3.5 together.
_SENTENCE_END = re.compile(r'[.!?।]["\')\]]*\s+|\n+')
MIN_SPEECH_CHARS = 40

def split_sentences(pieces, min_chars=MIN_SPEECH_CHARS):
    """Group streamed text pieces into speakable segments."""
    buf = ""
    for piece in pieces:
        buf += piece
        while True:
            cut = next((m.end() for m in _SENTENCE_END.finditer(buf) if m.end() >= min_chars), None)
            if cut is None:
                break
            yield buf[:cut]
            buf = buf[cut:]
    if buf.strip():
        yield buf

MAX_TTS_CHARS = 4_000
GTTS_TIMEOUT = (5, 10)   # (connect, read) seconds per HTTP request to Google

def _truncate_at_sentence(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(p) for p in ('. ', '? ', '! ', '।'))
    return cut[:end + 1] if end > limit // 2 else cut

def do_tts(text):
    """TTS via gTTS with a real per-request timeout and TLD fallback.
    gTTS's own timeout aborts a stuck request; the old ThreadPoolExecutor approach waited
    for the hung thread anyway when the `with` block exited, so its timeout never applied."""
    text = _truncate_at_sentence(clean_for_tts(text), MAX_TTS_CHARS)
    if not text:
        raise ValueError("Nothing to speak")
    for tld in ('co.in', 'com'):
        try:
            buf = io.BytesIO()
            gTTS(text=text, lang='te', slow=False, tld=tld, timeout=GTTS_TIMEOUT).write_to_fp(buf)
            data = buf.getvalue()
            if not data:
                raise ValueError("Empty gTTS output")
            return data, "audio/mpeg"
        except Exception as e:
            print(f"gTTS failed (tld={tld}, {type(e).__name__}): {e}")
    raise RuntimeError("All gTTS attempts failed")

app = FastAPI(title="Telugu Voice AI", version="10.1")

# The web UI is served from this same origin, so no CORS is needed by default.
# Set ALLOWED_ORIGINS (comma-separated) only if another site must call this API.
_allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins,
                       allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

DEBUG_ENABLED = os.environ.get("ENABLE_DEBUG") == "1"

# Request size limits — bound cost and memory per request.
HISTORY_TURNS = 8                 # messages sent to the LLM and echoed back to the client
MAX_TEXT_CHARS = 2_000
MAX_AUDIO_B64_CHARS = 8_000_000   # ≈ 6 MB of audio
MAX_HISTORY_ITEMS = 20

class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4_000)

AUDIO_MIME_TYPES = {"audio/webm", "audio/ogg", "audio/mp4", "audio/aac", "audio/mpeg", "audio/wav"}

class AudioReq(BaseModel):
    audio_base64: str = Field(min_length=1, max_length=MAX_AUDIO_B64_CHARS)
    mime_type: str = Field(default="audio/webm", max_length=64)

    @field_validator("mime_type")
    @classmethod
    def _check_mime(cls, v):
        base = v.split(";")[0].strip().lower()   # "audio/webm;codecs=opus" -> "audio/webm"
        if base not in AUDIO_MIME_TYPES:
            raise ValueError(f"unsupported audio type {base!r}")
        return base

class ConvReq(AudioReq):
    history: Optional[List[Msg]] = Field(default=[], max_length=MAX_HISTORY_ITEMS)

class TextReq(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)

class FeedbackReq(BaseModel):
    rating: int = Field(ge=1, le=5)
    call_seconds: int = Field(default=0, ge=0, le=86_400)
    turns: int = Field(default=0, ge=0, le=10_000)

class TextConvReq(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    history: Optional[List[Msg]] = Field(default=[], max_length=MAX_HISTORY_ITEMS)

class LoginReq(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)

@app.post("/auth/login")
def login(req: LoginReq, request: Request):
    if not auth.login_limiter.allow(auth.client_ip(request)):
        raise HTTPException(429, "Too many login attempts. Try again in a minute.")
    if not auth.config.authenticate(req.email, req.password):
        raise HTTPException(401, "Incorrect email or password.")
    token, exp = auth.issue_token(req.email.strip().lower(), auth.config.secret, auth.config.ttl)
    return {"token": token, "expires_at": exp}

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
def debug(user: str = Depends(require_user)):
    """Diagnostic endpoint — tests LLM, Gemini TTS, and gTTS independently.
    Costs ~11 Gemini calls per hit: disabled unless ENABLE_DEBUG=1, and login required."""
    if not DEBUG_ENABLED:
        raise HTTPException(404, "Not found")
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
def speak(user: str = Depends(require_user)):
    """Test TTS endpoint — plays a greeting without needing STT."""
    try:
        audio, content_type = do_tts("నమస్కారం! నేను యోధను. మీకు ఎలా సహాయం చేయగలను?")
        return {"audio_base64": base64.b64encode(audio).decode(), "content_type": content_type}
    except Exception as e:
        print(f"ERROR in /speak: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")

@app.post("/tts")
def tts(req: TextReq, user: str = Depends(require_user)):
    try:
        audio, content_type = do_tts(req.text)
        return {"audio_base64": base64.b64encode(audio).decode(), "content_type": content_type}
    except Exception as e:
        print(f"ERROR in /tts: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")

@app.post("/converse")
def converse(req: ConvReq, user: str = Depends(require_user)):
    try:
        ctx = get_context()
        user_text = gemini_stt(base64.b64decode(req.audio_base64), req.mime_type)
        if not user_text.strip():
            raise HTTPException(400, "Empty transcript")
        instant = check_instant(user_text, ctx)
        if instant:
            audio, ct = do_tts(instant)
            history = list(req.history or [])
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": instant})
            return {"transcript": user_text, "response": instant,
                    "audio_base64": base64.b64encode(audio).decode(), "content_type": ct, "history": history[-HISTORY_TURNS:]}
        manual_ctx = retrieve_manual_context(user_text)
        msgs = [{"role": "system", "content": build_system(ctx, manual_ctx)}]
        for m in (req.history or [])[-HISTORY_TURNS:]:
            msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": user_text})
        response = telugu_llm(msgs)
        audio, ct = do_tts(response)
        history = list(req.history or [])
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": response})
        return {"transcript": user_text, "response": response,
                "audio_base64": base64.b64encode(audio).decode(), "content_type": ct, "history": history[-HISTORY_TURNS:]}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in /converse: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")

def _sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}

def _conversation_events(get_user_text, history, label):
    """Shared SSE pipeline: transcript → chunk (one per sentence, as generated) → response
    (full text) → done. The client starts speech on the first chunk instead of waiting for
    the whole answer."""
    try:
        t0 = time.monotonic()
        ctx = get_context()
        user_text = (get_user_text() or "").strip()
        if not user_text:
            yield _sse({"type": "error", "msg": "Empty transcript"})
            return
        yield _sse({"type": "transcript", "text": user_text})

        first_ms, raw = None, ""
        instant = check_instant(user_text, ctx)
        if instant:
            response_text = instant
            yield _sse({"type": "chunk", "text": instant})
        else:
            try:
                manual_ctx = retrieve_manual_context(user_text)
                msgs = [{"role": "system", "content": build_system(ctx, manual_ctx)}]
                for m in (history or [])[-HISTORY_TURNS:]:
                    msgs.append({"role": m.role, "content": m.content})
                msgs.append({"role": "user", "content": user_text})
                for segment in split_sentences(telugu_llm_stream(msgs)):
                    raw += segment
                    shown = clean_for_display(segment)
                    if shown:
                        if first_ms is None:
                            first_ms = int((time.monotonic() - t0) * 1000)
                        yield _sse({"type": "chunk", "text": shown})
            except Exception:
                print(f"LLM error in {label}: {traceback.format_exc()}")
            response_text = clean_for_display(raw)
            if not response_text:
                response_text = LLM_FALLBACK_TEXT
                yield _sse({"type": "chunk", "text": response_text})

        yield _sse({"type": "response", "text": response_text})
        hist = [{"role": m.role, "content": m.content} for m in (history or [])]
        hist.append({"role": "user", "content": user_text})
        hist.append({"role": "assistant", "content": response_text})
        yield _sse({"type": "done", "history": hist[-HISTORY_TURNS:]})
        print("timing: " + json.dumps({"endpoint": label, "instant": bool(instant), "first_chunk_ms": first_ms,
              "total_ms": int((time.monotonic() - t0) * 1000)}), flush=True)
    except Exception as e:
        print(f"ERROR in {label} ({type(e).__name__}: {e}):\n{traceback.format_exc()}")
        yield _sse({"type": "error", "msg": "Request failed. Please try again."})

@app.post("/converse_stream")
def converse_stream(req: ConvReq, user: str = Depends(require_user)):
    """SSE streaming endpoint: audio in → transcript → answer sentence by sentence."""
    return StreamingResponse(
        _conversation_events(lambda: gemini_stt(base64.b64decode(req.audio_base64), req.mime_type),
                             req.history, "/converse_stream"),
        media_type="text/event-stream", headers=_SSE_HEADERS)

@app.post("/converse_text_stream")
def converse_text_stream(req: TextConvReq, user: str = Depends(require_user)):
    """SSE streaming for pre-transcribed text — skips STT for lower latency."""
    return StreamingResponse(
        _conversation_events(lambda: req.text, req.history, "/converse_text_stream"),
        media_type="text/event-stream", headers=_SSE_HEADERS)

@app.post("/feedback")
def feedback(req: FeedbackReq, user: str = Depends(require_user)):
    """Post-call star rating. Written as one JSON log line, queryable in Log Analytics:
    ContainerAppConsoleLogs_CL | where Log_s startswith "FEEDBACK " """
    print("FEEDBACK " + json.dumps({"user": user, "rating": req.rating, "call_seconds": req.call_seconds,
                                    "turns": req.turns, "ts": int(time.time())}), flush=True)
    return {"ok": True}

@app.post("/voice")
def voice(req: AudioReq, user: str = Depends(require_user)):
    try:
        ctx = get_context()
        text = gemini_stt(base64.b64decode(req.audio_base64), req.mime_type)
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in /voice: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")
