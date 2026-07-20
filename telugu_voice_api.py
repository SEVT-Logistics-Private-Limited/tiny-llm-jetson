import base64, sarvamai, requests, tempfile, subprocess, os, re, concurrent.futures
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pytz

SARVAM_KEY = "sk_javz0il8_d4GhUv9PhBOiH1jlHEG3ryOT"
client = sarvamai.SarvamAI(api_subscription_key=SARVAM_KEY)

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

# Cache instant responses for common questions
INSTANT_RESPONSES = {
    "సమయం": lambda ctx: f"ఇప్పుడు సమయం {ctx['time']} IST.",
    "time": lambda ctx: f"ఇప్పుడు సమయం {ctx['time']} IST.",
    "టైమ్": lambda ctx: f"ఇప్పుడు సమయం {ctx['time']} IST.",
    "తేదీ": lambda ctx: f"ఈరోజు {ctx['date']}, {ctx['day']}.",
    "date": lambda ctx: f"ఈరోజు {ctx['date']}, {ctx['day']}.",
    "ఈరోజు": lambda ctx: f"ఈరోజు {ctx['date']}, {ctx['day']}.",
    "నమస్కారం": lambda ctx: "నమస్కారం! నేను యోధను. మీకు ఎలా సహాయం చేయగలను?",
    "హలో": lambda ctx: "హలో! నేను యోధను. మీకు ఎలా సహాయం చేయగలను?",
    "హాయ్": lambda ctx: "హాయ్! నేను యోధను. చెప్పండి.",
}

def check_instant(text, ctx):
    """Return instant response if question matches known patterns"""
    text_lower = text.lower().strip()
    for keyword, fn in INSTANT_RESPONSES.items():
        if keyword in text_lower:
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
    text = re.sub(r'\([^)]*[a-zA-Z]{3,}[^)]*\)', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)
    text = re.sub(r'"([^"]*)"', r'\1', text)
    lines = text.split('\n')
    telugu_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        telugu_count = sum(1 for c in line if '\u0c00' <= c <= '\u0c7f')
        english_count = sum(1 for c in line if 'a' <= c.lower() <= 'z')
        if telugu_count > english_count:
            telugu_lines.append(line)
    result = ' '.join(telugu_lines) if telugu_lines else text
    return re.sub(r'\s+', ' ', result).strip()

def telugu_llm(messages):
    r = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"api-subscription-key": SARVAM_KEY},
        json={"model": "sarvam-105b", "messages": messages, "max_tokens": 300},
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

def build_system(ctx):
    return f"""యోధ — తెలుగు AI అసిస్టెంట్.
సమయం: {ctx['time']} IST | తేదీ: {ctx['date']} | వారం: {ctx['day']}
జ్ఞానం: భారత రాజధాని న్యూఢిల్లీ, తెలంగాణ రాజధాని హైదరాబాద్, ఆంధ్రప్రదేశ్ రాజధాని అమరావతి.
నియమాలు: తెలుగులో మాట్లాడు, * వాడకు, 1-2 వాక్యాలలో జవాబు ఇవ్వు, స్నేహంగా ఉండు."""

app = FastAPI(title="Telugu Voice AI", version="4.1")
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
    return {"status": "running", "version": "4.1", "time": ctx['time'], "date": ctx['date']}

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
    import time
    t0 = time.time()
    try:
        ctx = get_context()

        # STT
        wav = convert_to_wav(base64.b64decode(req.audio_base64))
        stt = client.speech_to_text.transcribe(
            file=("audio.wav", wav), model="saarika:v2.5", language_code="te-IN")
        user_text = stt.transcript
        if not user_text.strip():
            raise HTTPException(400, "Empty transcript")

        t1 = time.time()
        print(f"STT: {t1-t0:.2f}s | text: {user_text}")

        # Check instant response first
        instant = check_instant(user_text, ctx)
        if instant:
            print(f"INSTANT response: {instant}")
            audio = do_tts(instant)
            t2 = time.time()
            print(f"TTS: {t2-t1:.2f}s | Total: {t2-t0:.2f}s")
            history = list(req.history or [])
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": instant})
            return {"transcript": user_text, "response": instant,
                    "audio_base64": base64.b64encode(audio).decode(), "history": history}

        # LLM + TTS in parallel where possible
        msgs = [{"role": "system", "content": build_system(ctx)}]
        for m in (req.history or [])[-8:]:
            msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": user_text})

        response = telugu_llm(msgs)
        t2 = time.time()
        print(f"LLM: {t2-t1:.2f}s | response: {response}")

        audio = do_tts(response)
        t3 = time.time()
        print(f"TTS: {t3-t2:.2f}s | Total: {t3-t0:.2f}s")

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
        msgs = [{"role": "system", "content": build_system(ctx)},
                {"role": "user", "content": text}]
        response = telugu_llm(msgs)
        audio = do_tts(response)
        return {"transcript": text, "response": response,
                "audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))
