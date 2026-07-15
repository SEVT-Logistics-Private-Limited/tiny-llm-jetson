import base64, sarvamai, requests, threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SARVAM_KEY = "sk_javz0il8_d4GhUv9PhBOiH1jlHEG3ryOT"
client = sarvamai.SarvamAI(api_subscription_key=SARVAM_KEY)

def telugu_llm(user_text: str) -> str:
    r = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"api-subscription-key": SARVAM_KEY},
        json={
            "model": "sarvam-105b",
            "messages": [
                {"role": "system", "content": "మీరు ఒక సహాయకారి తెలుగు AI అసిస్టెంట్. తెలుగులో సంక్షిప్తంగా జవాబు ఇవ్వండి."},
                {"role": "user", "content": user_text}
            ],
            "max_tokens": 4096
        },
        timeout=180
    )
    data = r.json()
    if "choices" not in data:
        raise Exception(f"LLM error: {data}")
    return data["choices"][0]["message"]["content"].strip()

app = FastAPI(title="Telugu Voice AI", version="2.0")

class TextRequest(BaseModel):
    text: str

class AudioRequest(BaseModel):
    audio_base64: str

@app.get("/")
def health():
    return {"status": "running", "llm": "sarvam-105b", "language": "Telugu"}

@app.post("/tts")
def tts(req: TextRequest):
    try:
        r     = client.text_to_speech.convert(
            text=req.text, target_language_code="te-IN",
            speaker="anushka", model="bulbul:v2")
        audio = base64.b64decode(r.audios[0])
        return {"text": req.text, "audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/stt")
def stt(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        r = client.speech_to_text.transcribe(
            file=("audio.wav", audio_bytes),
            model="saarika:v2.5", language_code="te-IN")
        return {"transcript": r.transcript}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/speak")
def speak(req: TextRequest):
    try:
        response = telugu_llm(req.text)
        r = client.text_to_speech.convert(
            text=response, target_language_code="te-IN",
            speaker="anushka", model="bulbul:v2")
        audio = base64.b64decode(r.audios[0])
        return {"input": req.text, "response": response,
                "audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/voice")
def voice(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        stt_r = client.speech_to_text.transcribe(
            file=("audio.wav", audio_bytes),
            model="saarika:v2.5", language_code="te-IN")
        text = stt_r.transcript
        if not text.strip():
            raise HTTPException(400, "Empty transcript")
        response = telugu_llm(text)
        r = client.text_to_speech.convert(
            text=response, target_language_code="te-IN",
            speaker="anushka", model="bulbul:v2")
        audio = base64.b64decode(r.audios[0])
        return {"transcript": text, "response": response,
                "audio_base64": base64.b64encode(audio).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))
