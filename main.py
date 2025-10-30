import os
import tempfile
import shutil
import subprocess
import requests
import json
from fastapi import FastAPI, HTTPException, Header
from faster_whisper import WhisperModel

# ---- ENV ----
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE")
RAW_BUCKET = os.environ.get("RAW_BUCKET", "raw")
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET", "processed")
TRANSCRIPTS_BUCKET = os.environ.get("TRANSCRIPTS_BUCKET", "transcripts")
API_KEY = os.environ.get("TRANSCRIBE_API_KEY", "changeme")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE) must be set")

app = FastAPI()

# Load once (CPU; switch to cuda/float16 on GPU)
model_name = os.environ.get("WHISPER_MODEL", "medium")
model = WhisperModel(model_name, device="cpu", compute_type="int8")

# ---- Supabase Storage helpers (REST) ----
def sb_headers(extra=None):
    h = {"Authorization": f"Bearer {SUPABASE_KEY}"}
    if extra:
        h.update(extra)
    return h

def sb_download_raw(bucket: str, path_in_bucket: str, dest_path: str):
    # Ensure path does NOT include bucket prefix
    if path_in_bucket.startswith(f"{bucket}/"):
        path_in_bucket = path_in_bucket[len(bucket) + 1:]
    url = f"{SUPABASE_URL}/storage/v1/object/raw/{bucket}/{path_in_bucket}"
    r = requests.get(url, headers=sb_headers(), stream=True, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Download failed {r.status_code}: {r.text}")
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

def sb_upload(bucket: str, path_in_bucket: str, content_bytes: bytes, content_type: str):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path_in_bucket}"
    headers = sb_headers({"Content-Type": content_type, "x-upsert": "true"})
    r = requests.put(url, headers=headers, data=content_bytes, timeout=120)
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"Upload failed {r.status_code}: {r.text}")

# ---- Media / Transcription helpers ----
def convert_to_wav(input_path: str, output_path: str):
    cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path, "-y"]
    subprocess.run(cmd, check=True)

def run_transcription(audio_path: str):
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    transcript_data = {"duration": info.duration, "language": info.language, "segments": []}
    txt_lines = []
    for seg in segments:
        seg_dict = {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
        if seg.words:
            seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
        transcript_data["segments"].append(seg_dict)
        txt_lines.append(seg.text.strip())
    return transcript_data, "\n".join(txt_lines)

def export_hls(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    m3u8_path = os.path.join(output_dir, "master.m3u8")
    # single bitrate HLS
    subprocess.run(
        [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-f", "hls", "-hls_time", "4", "-hls_list_size", "0",
            m3u8_path
        ],
        check=True
    )
    return m3u8_path

# ---- API ----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
def process(data: dict, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw_path = data.get("rawPath")
    processed_prefix = data.get("processedPrefix")
    if not raw_path or not processed_prefix:
        raise HTTPException(status_code=400, detail="Missing rawPath or processedPrefix")

    temp_dir = tempfile.mkdtemp()
    try:
        # 1) Download raw (strip bucket prefix if caller included it)
        local_raw = os.path.join(temp_dir, os.path.basename(raw_path))
        path_in_bucket = raw_path
        if path_in_bucket.startswith(f"{RAW_BUCKET}/"):
            path_in_bucket = path_in_bucket[len(RAW_BUCKET) + 1:]
        sb_download_raw(RAW_BUCKET, path_in_bucket, local_raw)

        # 2) Convert to wav
        local_wav = os.path.join(temp_dir, "audio.wav")
        convert_to_wav(local_raw, local_wav)

        # 3) Transcribe (JSON with per-word + TXT)
        transcript_json, transcript_txt = run_transcription(local_wav)

        # 4) HLS export & upload
        hls_dir = os.path.join(temp_dir, "hls")
        master_local = export_hls(local_raw, hls_dir)
        processed_path = f"{processed_prefix}/master.m3u8"
        # upload m3u8 + segments
        for fname in os.listdir(hls_dir):
            local_fp = os.path.join(hls_dir, fname)
            remote_fp = f"{processed_prefix}/{fname}"
            ctype = "application/vnd.apple.mpegurl" if fname.endswith(".m3u8") else "video/mp2t"
            with open(local_fp, "rb") as f:
                sb_upload(PROCESSED_BUCKET, remote_fp, f.read(), ctype)

        # 5) Upload transcripts next to processed (so path is predictable for playback)
        transcript_base = f"{processed_prefix}/transcript"
        transcript_json_path = f"{transcript_base}.json"
        transcript_txt_path = f"{transcript_base}.txt"
        sb_upload(TRANSCRIPTS_BUCKET, transcript_json_path, json.dumps(transcript_json).encode("utf-8"), "application/json")
        sb_upload(TRANSCRIPTS_BUCKET, transcript_txt_path, transcript_txt.encode("utf-8"), "text/plain; charset=utf-8")

        return {
            "processed_path": processed_path,           # e.g. processed/<...>/master.m3u8
            "transcript_json": transcript_json_path,    # e.g. transcripts/<...>/transcript.json
            "transcript_txt": transcript_txt_path,      # e.g. transcripts/<...>/transcript.txt
            "duration": transcript_json["duration"],
            "language": transcript_json["language"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
