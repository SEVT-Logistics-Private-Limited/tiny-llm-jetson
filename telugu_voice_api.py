import base64, sarvamai, requests, tempfile, subprocess, os, re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

SARVAM_KEY = "sk_javz0il8_d4GhUv9PhBOiH1jlHEG3ryOT"
client = sarvamai.SarvamAI(api_subscription_key=SARVAM_KEY)

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
    text = re.sub(r'\s*-\s*(This|It|That|Good|Also|Very|Here|I |The )[^.]*\.?', '', text)
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
        json={"model": "sarvam-105b", "messages": messages, "max_tokens": 500},
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
        content = ' '.join(telugu_lines[-3:]) if telugu_lines else "మళ్ళీ చెప్పగలరా?"
    return clean_for_tts(content)

SYSTEM = """మీరు యోధ అనే తెలుగు AI అసిస్టెంట్. నియమాలు:
1. కేవలం తెలుగులో మాట్లాడండి - ఇంగ్లీష్ వాడకండి
2. * లేదా _ లేదా markdown వాడకండి
3. బ్రాకెట్లలో అనువాదాలు ఇవ్వకండి
4. 1-2 వాక్యాలలో జవాబు ఇవ్వండి
5. స్నేహంగా మాట్లాడండి"""

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

app = FastAPI(title="Telugu Voice AI", version="3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"status": "running", "version": "3.1"}

@app.get("/assistant")
def assistant():
    return FileResponse("/home/azureuser/telugu_assistant.html")

@app.post("/tts")
def tts(req: TextReq):
    try:
        r = client.text_to_speech.convert(text=req.text, target_language_code="te-IN", speaker="anushka", model="bulbul:v2")
        audio = base64.b64decode(r.audios[0])
        return {"audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/converse")
def converse(req: ConvReq):
    try:
        wav = convert_to_wav(base64.b64decode(req.audio_base64))
        stt = client.speech_to_text.transcribe(file=("audio.wav", wav), model="saarika:v2.5", language_code="te-IN")
        user_text = stt.transcript
        if not user_text.strip():
            raise HTTPException(400, "Empty transcript")
        msgs = [{"role": "system", "content": SYSTEM}]
        for m in (req.history or [])[-10:]:
            msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": user_text})
        response = telugu_llm(msgs)
        r = client.text_to_speech.convert(text=response, target_language_code="te-IN", speaker="anushka", model="bulbul:v2")
        audio = base64.b64decode(r.audios[0])
        history = list(req.history or [])
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": response})
        return {"transcript": user_text, "response": response, "audio_base64": base64.b64encode(audio).decode(), "history": history}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/voice")
def voice(req: AudioReq):
    try:
        wav = convert_to_wav(base64.b64decode(req.audio_base64))
        stt = client.speech_to_text.transcribe(file=("audio.wav", wav), model="saarika:v2.5", language_code="te-IN")
        text = stt.transcript
        if not text.strip():
            raise HTTPException(400, "Empty transcript")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        response = telugu_llm(msgs)
        r = client.text_to_speech.convert(text=response, target_language_code="te-IN", speaker="anushka", model="bulbul:v2")
        audio = base64.b64decode(r.audios[0])
        return {"transcript": text, "response": response, "audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))
