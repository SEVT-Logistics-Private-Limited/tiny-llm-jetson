FROM python:3.11-slim

# Unprivileged account the service runs as, and the path api.py expects for the HTML file.
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin yodha \
 && mkdir -p /home/azureuser

WORKDIR /app

# Model cache in a location the unprivileged user can read (the default is /root/.cache).
ENV HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY voice-assistant/requirements.txt .
# CPU-only PyTorch first: the default wheel pulls ~2 GB of CUDA libraries this service never
# uses (no GPU on Container Apps). Smaller image = faster builds, pulls and cold starts.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image.
# TRANSFORMERS_OFFLINE=1 is set at runtime — without this the model
# can't be fetched and the RAG pipeline stays disabled.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
 && chmod -R a+rX /opt/hf-cache

# App code
COPY voice-assistant/api.py voice-assistant/auth.py ./

# HTML served at /assistant — path hardcoded in api.py
COPY voice-assistant/telugu_assistant.html /home/azureuser/telugu_assistant.html

# Runs as root for now: after the switch to the unprivileged user (uid 10001) the knowledge
# base stopped loading from the /mnt/vehicledata share in production. Restore "USER yodha"
# once the share is mounted writable for that uid.

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
