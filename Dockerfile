FROM python:3.11-slim

# Install ffmpeg (and basic deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

# (Optional) If you plan to use GPU or external libs, add them here.

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Env placeholders (set real ones in the platform dashboard)
ENV SUPABASE_URL=""
ENV SUPABASE_KEY=""
ENV RAW_BUCKET=raw
ENV PROCESSED_BUCKET=processed
ENV TRANSCRIPTS_BUCKET=transcripts
ENV WHISPER_MODEL=base

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

