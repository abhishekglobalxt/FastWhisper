import os, tempfile, shutil, json, time, subprocess, uuid
from pathlib import Path
from typing import Optional
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------- ENV VARS (set these in Railway/Render) ----------
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE"]
RAW_BUCKET = os.environ.get("RAW_BUCKET", "raw")
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET", "processed")
TRANSCRIPTS_BUCKET = os.environ.get("TRANSCRIPTS_BUCKET", "transcripts")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # "small" for better quality

#----Health
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

# --------- FASTAPI ----------
app = FastAPI(title="Video Processing Worker", version="1.0.0")

# --------- MODELS ----------
class ProcessIn(BaseModel):
    rawPath: str                      # e.g. "intv123/TOK123/q1.webm"
    processedPrefix: str              # e.g. "intv123/TOK123/q1"
    transcriptPrefix: Optional[str] = None  # defaults to processedPrefix

# --------- UTILS ----------
def supabase_headers(extra=None):
    h = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "apikey": SUPABASE_SERVICE_ROLE,
    }
    if extra:
        h.update(extra)
    return h

def sb_download_object(bucket: str, path: str, dest_file: Path):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    with requests.get(url, headers=supabase_headers(), stream=True) as r:
        if r.status_code >= 300:
            raise HTTPException(status_code=502, detail=f"Supabase download failed {r.status_code}: {r.text}")
        with open(dest_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def sb_upload_file(bucket: str, dst_path: str, src_file: Path, content_type: Optional[str] = None, upsert=True):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{dst_path}"
    headers = supabase_headers({"x-upsert": "true" if upsert else "false"})
    if content_type:
        headers["Content-Type"] = content_type
    with open(src_file, "rb") as f:
        r = requests.post(url, headers=headers, data=f)
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"Supabase upload failed {r.status_code}: {r.text}")

def run_ffmpeg_to_hls(input_file: Path, out_dir: Path) -> float:
    """
    Convert input_file -> HLS in out_dir.
    Returns duration (seconds) via ffprobe.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(input_file)],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip()) if probe.returncode == 0 and probe.stdout.strip() else 0.0

    # HLS encode (single bitrate; you can add renditions later)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-c:v", "h264", "-profile:v", "main", "-level", "3.1",
        "-preset", "veryfast", "-b:v", "2500k", "-maxrate", "2800k", "-bufsize", "5000k",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000",
        "-hls_time", "4", "-hls_playlist_type", "vod",
        "-hls_segment_filename", str(out_dir / "segment_%03d.ts"),
        str(out_dir / "master.m3u8")
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"FFmpeg failed: {proc.stderr[:5000]}")
    return duration

def run_whisper(input_file: Path, out_json: Path, out_txt: Path):
    # Import here so app boots even if model isn’t downloaded yet.
    from faster_whisper import WhisperModel
    model_size = WHISPER_MODEL
    model = WhisperModel(model_size, compute_type="int8")  # memory friendly
    segments, info = model.transcribe(str(input_file), vad_filter=True)

    # Write plain text + segments
    texts = []
    seg_json = []
    for s in segments:
        seg_json.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip()
        })
        texts.append(s.text.strip())

    out_txt.write_text("\n".join(texts), encoding="utf-8")
    out_json.write_text(json.dumps({
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "segments": seg_json
    }, ensure_ascii=False), encoding="utf-8")

# --------- API ----------
@app.post("/process")
def process_video(payload: ProcessIn):
    t0 = time.time()
    transcript_prefix = payload.transcriptPrefix or payload.processedPrefix

    # Temp workspace
    workdir = Path(tempfile.mkdtemp(prefix="proc-"))
    try:
        raw_local = workdir / "input.raw"
        hls_dir = workdir / "hls"
        transcript_json = workdir / "transcript.json"
        transcript_txt = workdir / "transcript.txt"

        # 1) Download raw from Supabase
        sb_download_object(RAW_BUCKET, payload.rawPath, raw_local)

        # 2) FFmpeg -> HLS
        duration = run_ffmpeg_to_hls(raw_local, hls_dir)

        # 3) Whisper -> transcripts
        run_whisper(raw_local, transcript_json, transcript_txt)

        # 4) Upload HLS files to processed bucket
        #    processed path layout: processed/<prefix>/master.m3u8  + segments
        processed_base = f"{payload.processedPrefix}"
        sb_upload_file(PROCESSED_BUCKET, f"{processed_base}/master.m3u8", hls_dir / "master.m3u8", "application/vnd.apple.mpegurl")
        for seg in sorted(hls_dir.glob("segment_*.ts")):
            sb_upload_file(PROCESSED_BUCKET, f"{processed_base}/{seg.name}", seg, "video/mp2t")

        # 5) Upload transcripts to transcripts bucket
        transcript_base = f"{transcript_prefix}"
        sb_upload_file(TRANSCRIPTS_BUCKET, f"{transcript_base}.json", transcript_json, "application/json")
        sb_upload_file(TRANSCRIPTS_BUCKET, f"{transcript_base}.txt", transcript_txt, "text/plain; charset=utf-8")

        elapsed = round(time.time() - t0, 2)
        return {
            "ok": True,
            "processed_path": f"{processed_base}/master.m3u8",
            "transcript_path": f"{transcript_base}.json",
            "duration_seconds": int(duration),
            "took_seconds": elapsed
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

