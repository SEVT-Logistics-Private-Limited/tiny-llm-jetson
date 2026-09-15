FROM python:3.11-slim

# Create the path api.py expects for the HTML file
RUN mkdir -p /home/azureuser

WORKDIR /app

COPY voice-assistant/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image.
# TRANSFORMERS_OFFLINE=1 is set at runtime — without this the model
# can't be fetched and the RAG pipeline stays disabled.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# App code
COPY voice-assistant/api.py .

# HTML served at /assistant — path hardcoded in api.py
COPY voice-assistant/telugu_assistant.html /home/azureuser/telugu_assistant.html

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
