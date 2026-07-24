import base64, sarvamai, requests, tempfile, subprocess, os, re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pytz

SARVAM_KEY = "sk_javz0il8_d4GhUv9PhBOiH1jlHEG3ryOT"
client = sarvamai.SarvamAI(api_subscription_key=SARVAM_KEY)
from sentence_transformers import SentenceTransformer
import chromadb

print("Loading manual knowledge base...")
_embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
_chroma_client = chromadb.PersistentClient(path="/home/azureuser/vehicle_manuals_db")
_manuals_collection = _chroma_client.get_collection("vehicle_manuals")

def sarvam_translate(text, source_lang, target_lang):
    headers = {"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"}
    payload = {"input": text, "source_language_code": source_lang, "target_language_code": target_lang}
    r = requests.post("https://api.sarvam.ai/translate", json=payload, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()["translated_text"]

def retrieve_manual_context(question_te, distance_threshold=0.35):
    try:
        question_en = sarvam_translate(question_te, "te-IN", "en-IN")
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

def convert_to_wav(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
        f.write(audio_bytes)
        webm_path = f.name
    wav_path = webm_path.replace('.webm', '.wav')
    subprocess.run(['/usr/bin/ffmpeg', '-y', '-i', webm_path,
        '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path], capture_output=True)
    with open(wav_path, 'rb') as f:
        wav_bytes = f.read()
    os.unlink(webm_path)
    os.unlink(wav_path)
    return wav_bytes

def clean_for_tts(text):
    text = re.sub(r'[*/#_+|\[\]{}<>^~`\\]', '', text)
    text = re.sub(r'\([^)]*[a-zA-Z]{2,}[^)]*\)', '', text)
    text = re.sub(r'\b\d+\.\s*', '', text)
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)
    text = re.sub(r'[.,;:!?]{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()
    telugu_count = sum(1 for c in text if '\u0c00' <= c <= '\u0c7f')
    if telugu_count < 3:
        return "మళ్ళీ చెప్పగలరా?"
    return text

def build_system(ctx, manual_context=None):
    base = (
        "నువ్వు యోధ అనే తెలుగు AI అసిస్టెంట్‌వి.\n"
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
        "నియమాలు:\n"
        "1. ఎల్లప్పుడూ తెలుగులో మాత్రమే మాట్లాడు\n"
        "2. ఇంగ్లీష్ పదాలు వాడకు\n"
        "3. *, /, #, _ లాంటి గుర్తులు వాడకు\n"
        "4. జవాబు 1-2 వాక్యాలలో ఇవ్వు\n"
        "5. స్నేహంగా మాట్లాడు"
    )
    return base

def telugu_llm(messages):
    r = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"api-subscription-key": SARVAM_KEY},
        json={"model": "sarvam-105b", "messages": messages, "max_tokens": 1200},
        timeout=30
    )
    data = r.json()
    if "choices" not in data:
        raise Exception(f"LLM error: {data}")
    content = data["choices"][0]["message"].get("content")
    if not content:
        reasoning = data["choices"][0]["message"].get("reasoning_content", "")
        telugu_lines = [l.strip() for l in reasoning.split('\n')
                        if l.strip() and sum(1 for c in l if '\u0c00' <= c <= '\u0c7f') > 3]
        content = ' '.join(telugu_lines[-2:]) if telugu_lines else "మళ్ళీ చెప్పగలరా?"
    return clean_for_tts(content)

def do_tts(text):
    r = client.text_to_speech.convert(
        text=text, target_language_code="te-IN",
        speaker="anushka", model="bulbul:v2")
    return base64.b64decode(r.audios[0])

app = FastAPI(title="Telugu Voice AI", version="4.2")
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
    return {"status": "running", "version": "4.2", "time": ctx['time'], "date": ctx['date']}

@app.get("/assistant")
def assistant():
    return FileResponse("/home/azureuser/telugu_assistant.html")

@app.post("/tts")
def tts(req: TextReq):
    try:
        audio = do_tts(req.text)
        return {"audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/converse")
def converse(req: ConvReq):
    try:
        ctx = get_context()
        wav = convert_to_wav(base64.b64decode(req.audio_base64))
        stt = client.speech_to_text.transcribe(
            file=("audio.wav", wav), model="saarika:v2.5", language_code="te-IN")
        user_text = stt.transcript
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
        raise HTTPException(500, str(e))

@app.post("/voice")
def voice(req: AudioReq):
    try:
        ctx = get_context()
        wav = convert_to_wav(base64.b64decode(req.audio_base64))
        stt = client.speech_to_text.transcribe(
            file=("audio.wav", wav), model="saarika:v2.5", language_code="te-IN")
        text = stt.transcript
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
        raise HTTPException(500, str(e))
