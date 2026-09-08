FROM python:3.11-slim

# Create the path api.py expects for the HTML file
RUN mkdir -p /home/azureuser

WORKDIR /app

COPY voice-assistant/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY voice-assistant/api.py .

# HTML served at /assistant — path hardcoded in api.py
COPY voice-assistant/telugu_assistant.html /home/azureuser/telugu_assistant.html

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
