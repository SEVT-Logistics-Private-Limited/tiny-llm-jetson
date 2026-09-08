import base64, requests, os, re, io, wave, traceback
from google import genai as google_genai
from google.genai import types as genai_types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pytz

GEMINI_KEY = os.environ["GEMINI_API_KEY"]

gemini_client = google_genai.Client(api_key=GEMINI_KEY)

from sentence_transformers import SentenceTransformer
import chromadb
import threading

_embed_model = None
_chroma_client = None
_manuals_collection = None

def _load_kb():
    global _embed_model, _chroma_client, _manuals_collection
    print("Loading manual knowledge base in background...")
    try:
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


def gemini_translate(text, source_lang, target_lang):
    prompt = f"Translate the following text from {source_lang} to {target_lang}. Return only the translated text, no explanations.\n\nText: {text}"
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(max_output_tokens=500)
    )
    return response.text.strip()

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
    t = text.lower().strip()
    for kw, fn in INSTANT_RESPONSES.items():
        if kw in t:
            return fn(ctx)
    return None

def gemini_stt(audio_bytes):
    """Transcribe Telugu audio using Gemini."""
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            genai_types.Content(parts=[
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        mime_type="audio/webm",
                        data=audio_bytes
                    )
                ),
                genai_types.Part(
                    text="Transcribe exactly what is spoken in Telugu in this audio. Return only the transcribed Telugu text, no explanations or translations."
                )
            ])
        ]
    )
    return response.text.strip()

def clean_for_tts(text):
    text = re.sub(r'[*/#_+|\[\]{}<>^~`\\]', '', text)
    text = re.sub(r'\([^)]*[a-zA-Z]{2,}[^)]*\)', '', text)
    text = re.sub(r'\b\d+\.\s*', '', text)
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)
    text = re.sub(r'[.,;:!?]{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()
    telugu_count = sum(1 for c in text if 'ఀ' <= c <= '౿')
    if telugu_count < 3:
        return "మళ్ళీ చెప్పగలరా?"
    return text

def build_system(ctx, manual_context=None):
    base = (
        "నువ్వు యోధ అనే తెలుగు AI అసిస్టెంట్‌వి. నువ్వు చాలా స్నేహంగా, వేడుకగా మాట్లాడతావు - ఒక మంచి స్నేహితుడిలా.\n"
        f"సమయం: {ctx['time']} IST, తేదీ: {ctx['date']}, వారం: {ctx['day']}\n"
        "భారత రాజధాని న్యూఢిల్లీ, తెలంగాణ రాజధాని హైదరాబాద్, ఆంధ్రప్రదేశ్ రాజధాని అమరావతి.\n"
    )
    if manual_context:
        base += (
            f"\nవాహన మాన్యువల్ నుండి సమాచారం ({manual_context['source']}):\n"
            f"{manual_context['passage']}\n"
            "పై సమాచారం ఆధారంగా వినియోగదారు ప్రశ్నకు సమాధానం ఇవ్వు.\n"
        )
    base += (
        "\nనియమాలు:\n"
        "1. ఎల్లప్పుడూ శుద్ధ తెలుగులో మాత్రమే మాట్లాడు\n"
        "2. ఇంగ్లీష్ పదాలు వాడకు\n"
        "3. *, /, #, _ లాంటి గుర్తులు వాడకు\n"
        "4. జవాబు చాలా చిన్నగా ఉండాలి - 1-2 వాక్యాలు మాత్రమే\n"
        "5. స్నేహంగా, వేడుకగా, సహజంగా మాట్లాడు - ఒక స్నేహితుడిలా\n"
        "6. సంక్షిప్తంగా, స్పష్టంగా చెప్పు"
    )
    return base

def telugu_llm(messages):
    system_msg = messages[0]["content"]
    contents = []
    for m in messages[1:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(genai_types.Content(
            role=role,
            parts=[genai_types.Part(text=m["content"])]
        ))
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_msg,
            max_output_tokens=200
        )
    )
    return clean_for_tts(response.text)

def _pcm_to_wav(pcm_data, rate=24000):
    """Convert raw PCM (16-bit signed LE, mono) to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm_data)
    return buf.getvalue()

def do_tts(text):
    """Gemini TTS — returns WAV bytes."""
    text = text[:500]
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-preview-tts",
        contents=text,
        config=genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name="Aoede"
                    )
                )
            )
        )
    )
    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    return _pcm_to_wav(pcm_data)

app = FastAPI(title="Telugu Voice AI", version="8.0")
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

@app.get("/")
def health():
    ctx = get_context()
    return {"status": "running", "version": "8.0", "time": ctx['time'], "date": ctx['date']}

@app.get("/assistant")
def assistant():
    return FileResponse("/home/azureuser/telugu_assistant.html")

@app.get("/speak")
def speak():
    """Test TTS endpoint — plays a greeting without needing STT."""
    try:
        audio = do_tts("నమస్కారం! నేను యోధను. మీకు ఎలా సహాయం చేయగలను?")
        return {"audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        print(f"ERROR in /speak: {traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.post("/tts")
def tts(req: TextReq):
    try:
        audio = do_tts(req.text)
        return {"audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        print(f"ERROR in /tts: {traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.post("/converse")
def converse(req: ConvReq):
    try:
        ctx = get_context()
        user_text = gemini_stt(base64.b64decode(req.audio_base64))
        if not user_text.strip():
            raise HTTPException(400, "Empty transcript")
        instant = check_instant(user_text, ctx)
        if instant:
            audio = do_tts(instant)
            history = list(req.history or [])
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": instant})
            return {"transcript": user_text, "response": instant,
                    "audio_base64": base64.b64encode(audio).decode(), "history": history}
        manual_ctx = retrieve_manual_context(user_text)
        msgs = [{"role": "system", "content": build_system(ctx, manual_ctx)}]
        for m in (req.history or [])[-8:]:
            msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": user_text})
        response = telugu_llm(msgs)
        audio = do_tts(response)
        history = list(req.history or [])
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": response})
        return {"transcript": user_text, "response": response,
                "audio_base64": base64.b64encode(audio).decode(), "history": history}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in /converse: {traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.post("/voice")
def voice(req: AudioReq):
    try:
        ctx = get_context()
        text = gemini_stt(base64.b64decode(req.audio_base64))
        if not text.strip():
            raise HTTPException(400, "Empty transcript")
        instant = check_instant(text, ctx)
        if instant:
            audio = do_tts(instant)
            return {"transcript": text, "response": instant,
                    "audio_base64": base64.b64encode(audio).decode()}
        manual_ctx = retrieve_manual_context(text)
        msgs = [{"role": "system", "content": build_system(ctx, manual_ctx)},
                {"role": "user", "content": text}]
        response = telugu_llm(msgs)
        audio = do_tts(response)
        return {"transcript": text, "response": response,
                "audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        print(f"ERROR in /voice: {traceback.format_exc()}")
        raise HTTPException(500, str(e))
