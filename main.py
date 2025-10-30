import os
import tempfile
import shutil
import subprocess
import requests
import datetime
import json

from fastapi import FastAPI, HTTPException, Header
from faster_whisper import WhisperModel
from supabase import create_client, Client

# ---- ENV VARIABLES ----
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # service role key
RAW_BUCKET = os.environ.get("RAW_BUCKET", "raw")
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET", "processed")
TRANSCRIPTS_BUCKET = os.environ.get("TRANSCRIPTS_BUCKET", "transcripts")
API_KEY = os.environ.get("TRANSCRIBE_API_KEY", "changeme")  # protect endpoint

# ---- INIT ----
app = FastAPI()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load Whisper model once (CPU, int8 for Render free tier; switch to cuda/float16 for GPU)
model = WhisperModel("medium", device="cpu", compute_type="int8")


# ---- HELPERS ----
def download_file_from_supabase(raw_path: str, temp_dir: str) -> str:
    """Download file from private Supabase bucket into temp dir, return local path."""
    # Strip bucket prefix if present (avoid raw/raw bug)
    if raw_path.startswith(f"{RAW_BUCKET}/"):
        raw_path = raw_path[len(RAW_BUCKET) + 1 :]

    url = f"{SUPABASE_URL}/storage/v1/object/raw/{RAW_BUCKET}/{raw_path}"

    headers = {"Authorization": f"Bearer {SUPABASE_KEY}"}
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to download {url}: {response.text}")

    local_path = os.path.join(temp_dir, os.path.basename(raw_path))
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path


def convert_to_wav(input_path: str, output_path: str):
    """Convert webm/mp4/etc to WAV using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        output_path,
        "-y",
    ]
    subprocess.run(cmd, check=True)


def run_transcription(audio_path: str):
    """Run Whisper transcription, return both JSON + TXT."""
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)

    transcript_data = {
        "duration": info.duration,
        "language": info.language,
        "segments": [],
    }

    transcript_txt = []

    for seg in segments:
        segment_dict = {
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        }
        if seg.words:
            segment_dict["words"] = [
                {"word": w.word, "start": w.start, "end": w.end} for w in seg.words
            ]
        transcript_data["segments"].append(segment_dict)
        transcript_txt.append(seg.text.strip())

    return transcript_data, "\n".join(transcript_txt)


def upload_to_supabase(bucket: str, path: str, content: bytes, content_type: str):
    """Upload content to Supabase storage."""
    res = supabase.storage.from_(bucket).upload(path, content, {"content-type": content_type, "upsert": True})
    if "error" in str(res).lower():
        raise HTTPException(status_code=502, detail=f"Upload failed: {res}")


# ---- ROUTES ----
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process_file(data: dict, x_api_key: str = Header(None)):
    # Security check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw_path = data.get("rawPath")
    processed_prefix = data.get("processedPrefix")
    if not raw_path or not processed_prefix:
        raise HTTPException(status_code=400, detail="Missing rawPath or processedPrefix")

    temp_dir = tempfile.mkdtemp()

    try:
        # ---- Download raw file ----
        local_raw = download_file_from_supabase(raw_path, temp_dir)

        # ---- Convert to wav ----
        temp_wav = os.path.join(temp_dir, "audio.wav")
        convert_to_wav(local_raw, temp_wav)

        # ---- Transcribe ----
        transcript_json, transcript_txt = run_transcription(temp_wav)

        # ---- Upload HLS processed version (for streaming) ----
        processed_path = f"{processed_prefix}/master.m3u8"
        # Example ffmpeg HLS export
        hls_dir = os.path.join(temp_dir, "hls")
        os.makedirs(hls_dir, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                local_raw,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-f",
                "hls",
                "-hls_time",
                "4",
                "-hls_list_size",
                "0",
                os.path.join(hls_dir, "master.m3u8"),
            ],
            check=True,
        )
        # Upload HLS files (m3u8 + segments)
        for fname in os.listdir(hls_dir):
            path = f"{processed_prefix}/{fname}"
            with open(os.path.join(hls_dir, fname), "rb") as f:
                upload_to_supabase(PROCESSED_BUCKET, path, f.read(), "application/vnd.apple.mpegurl" if fname.endswith(".m3u8") else "video/mp2t")

        # ---- Upload transcripts ----
        transcript_prefix = f"{processed_prefix}/transcript"
        transcript_json_path = f"{transcript_prefix}.json"
        transcript_txt_path = f"{transcript_prefix}.txt"

        upload_to_supabase(TRANSCRIPTS_BUCKET, transcript_json_path, json.dumps(transcript_json).encode("utf-8"), "application/json")
        upload_to_supabase(TRANSCRIPTS_BUCKET, transcript_txt_path, transcript_txt.encode("utf-8"), "text/plain; charset=utf-8")

        # ---- Response ----
        return {
            "processed_path": processed_path,
            "transcript_json": transcript_json_path,
            "transcript_txt": transcript_txt_path,
            "duration": transcript_json["duration"],
            "language": transcript_json["language"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
