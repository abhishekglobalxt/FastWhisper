import os
import tempfile
import shutil
import subprocess
import requests
import json
import traceback
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from faster_whisper import WhisperModel

# ========= ENV =========
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE")
RAW_BUCKET = os.environ.get("RAW_BUCKET", "raw")
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET", "processed")
TRANSCRIPTS_BUCKET = os.environ.get("TRANSCRIPTS_BUCKET", "transcripts")
API_KEY = os.environ.get("TRANSCRIBE_API_KEY", "changeme")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
DEBUG = os.environ.get("DEBUG", "0") == "1"
LOG_BODY = os.environ.get("LOG_BODY", "0") == "1"  # print response snippets

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE) must be set")

app = FastAPI()

# ========= LOG HELPERS =========
def now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def log(msg: str, rid: str = ""):
    prefix = f"[{now()}]"
    if rid:
        prefix += f"[rid:{rid}]"
    print(prefix, msg, flush=True)

def log_exc(rid: str = ""):
    tb = traceback.format_exc()
    log("EXCEPTION:\n" + tb, rid)

def safe_snip(txt: str, n: int = 500) -> str:
    try:
        s = txt if isinstance(txt, str) else str(txt)
        return s[:n]
    except Exception:
        return "<unprintable>"

# ========= INIT HEAVY MODEL ONCE =========
log(f"Loading Whisper model '{WHISPER_MODEL}' ...")
model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
log("Whisper model loaded.")

# ========= SUPABASE STORAGE (REST) =========
def sb_headers(extra=None):
    # Some Supabase deployments require both Authorization and apikey
    h = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
    }
    if extra:
        h.update(extra)
    return h

def normalize_path_in_bucket(raw_path: str, rid: str) -> str:
    """
    Accepts:
      - signed URL (…/storage/v1/object/sign/<bucket>/<path>?token=…)
      - raw URL (…/storage/v1/object/raw/<bucket>/<path>)
      - path with bucket prefix (raw/abc/def.webm)
      - path inside bucket (abc/def.webm)
    Returns: path inside the bucket only (abc/def.webm)
    """
    p = raw_path or ""
    if p.startswith("http"):
        # extract bucket & path from URL
        # matches both .../object/sign/<bucket>/<path>?... and .../object/raw/<bucket>/<path>
        import re
        m = re.search(r"/storage/v1/object/(?:sign|raw)/([^/]+)/(.+?)(?:\?|$)", p)
        if m:
            bucket = m.group(1)
            path_in_bucket = m.group(2)
            log(f"normalize: detected URL; bucket={bucket} path={path_in_bucket}", rid)
            p = path_in_bucket
        else:
            log("normalize: URL pattern not recognized; passing as-is (may fail).", rid)
    # strip leading 'raw/' if present
    if p.startswith(f"{RAW_BUCKET}/"):
        p = p[len(RAW_BUCKET) + 1 :]
        log(f"normalize: stripped leading '{RAW_BUCKET}/' -> {p}", rid)
    return p

def sb_download(bucket: str, path_in_bucket: str, dest_path: str, rid: str):
    """
    Downloads either:
     - from a path inside the bucket (authorized call)
     - or from a signed URL (if caller passed a full URL in 'path_in_bucket')
    """
    if path_in_bucket.startswith("http"):
        url = path_in_bucket
        log(f"DOWNLOAD (signed URL) -> {url}", rid)
        r = requests.get(url, stream=True, timeout=120)
    else:
        url = f"{SUPABASE_URL}/storage/v1/object/raw/{bucket}/{path_in_bucket}"
        log(f"DOWNLOAD (bucket path) -> {url}", rid)
        r = requests.get(url, headers=sb_headers(), stream=True, timeout=120)

    if r.status_code != 200:
        body_snip = safe_snip(r.text, 500) if LOG_BODY else f"<{len(r.text)} bytes>"
        log(f"DOWNLOAD_FAIL status={r.status_code} url={url} body={body_snip}", rid)
        raise HTTPException(status_code=502, detail=f"Download failed {r.status_code}: {body_snip}")

    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    log(f"DOWNLOAD OK -> {dest_path}", rid)

def sb_upload(bucket: str, path_in_bucket: str, content_bytes: bytes, content_type: str, rid: str):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path_in_bucket}"
    headers = sb_headers({"Content-Type": content_type, "x-upsert": "true"})
    log(f"UPLOAD -> {url}  ({len(content_bytes)} bytes, {content_type})", rid)
    r = requests.put(url, headers=headers, data=content_bytes, timeout=120)
    if r.status_code not in (200, 201):
        body_snip = safe_snip(r.text, 500) if LOG_BODY else f"<{len(r.text)} bytes>"
        log(f"UPLOAD_FAIL status={r.status_code} url={url} body={body_snip}", rid)
        raise HTTPException(status_code=502, detail=f"Upload failed {r.status_code}: {body_snip}")
    log("UPLOAD OK", rid)

# ========= MEDIA / TRANSCRIPTION =========
def convert_to_wav(input_path: str, output_path: str, rid: str):
    cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path, "-y"]
    log(f"FFMPEG to WAV: {' '.join(cmd)}", rid)
    subprocess.run(cmd, check=True)
    log("FFMPEG WAV OK", rid)

def export_hls(input_path: str, output_dir: str, rid: str):
    os.makedirs(output_dir, exist_ok=True)
    m3u8_path = os.path.join(output_dir, "master.m3u8")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-f", "hls", "-hls_time", "4", "-hls_list_size", "0",
        m3u8_path
    ]
    log(f"FFMPEG HLS: {' '.join(cmd)}", rid)
    subprocess.run(cmd, check=True)
    log("FFMPEG HLS OK", rid)
    return m3u8_path

def run_transcription(audio_path: str, rid: str):
    log("WHISPER: transcribe start", rid)
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    log(f"WHISPER: language={info.language} duration={info.duration}", rid)

    transcript_json = {"duration": info.duration, "language": info.language, "segments": []}
    txt_lines = []

    for seg in segments:
        seg_dict = {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
        if seg.words:
            seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
        transcript_json["segments"].append(seg_dict)
        txt_lines.append(seg.text.strip())

    transcript_txt = "\n".join(txt_lines)
    log("WHISPER: transcribe done", rid)
    return transcript_json, transcript_txt

# ========= ROUTES =========
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
def process(data: dict, x_api_key: str = Header(None)):
    rid = str(uuid.uuid4())[:8]  # short request id for log correlation
    try:
        log(f"REQ: /process body={safe_snip(json.dumps(data))}", rid)

        if x_api_key != API_KEY:
            log("AUTH FAIL (X-API-KEY mismatch)", rid)
            raise HTTPException(status_code=401, detail="Unauthorized")

        raw_path_in = data.get("rawPath")
        processed_prefix = data.get("processedPrefix")
        if not raw_path_in or not processed_prefix:
            log("BAD REQUEST: missing rawPath or processedPrefix", rid)
            raise HTTPException(status_code=400, detail="Missing rawPath or processedPrefix")

        # Normalize raw path
        path_in_bucket = normalize_path_in_bucket(raw_path_in, rid)
        log(f"NORMALIZED rawPath -> {path_in_bucket}", rid)

        temp_dir = tempfile.mkdtemp()
        log(f"TEMP DIR -> {temp_dir}", rid)

        # 1) Download RAW
        local_raw = os.path.join(temp_dir, os.path.basename(path_in_bucket) or "input.webm")
        sb_download(RAW_BUCKET, path_in_bucket, local_raw, rid)

        # 2) Convert to WAV
        local_wav = os.path.join(temp_dir, "audio.wav")
        convert_to_wav(local_raw, local_wav, rid)

        # 3) Transcribe
        transcript_json, transcript_txt = run_transcription(local_wav, rid)

        # 4) HLS export & upload
        hls_dir = os.path.join(temp_dir, "hls")
        master_local = export_hls(local_raw, hls_dir, rid)
        processed_path = f"{processed_prefix}/master.m3u8"
        # upload m3u8 + segments
        for fname in os.listdir(hls_dir):
            local_fp = os.path.join(hls_dir, fname)
            remote_fp = f"{processed_prefix}/{fname}"
            ctype = "application/vnd.apple.mpegurl" if fname.endswith(".m3u8") else "video/mp2t"
            with open(local_fp, "rb") as f:
                sb_upload(PROCESSED_BUCKET, remote_fp, f.read(), ctype, rid)

        # 5) Upload transcripts next to processed prefix
        transcript_base = f"{processed_prefix}/transcript"
        transcript_json_path = f"{transcript_base}.json"
        transcript_txt_path = f"{transcript_base}.txt"
        sb_upload(TRANSCRIPTS_BUCKET, transcript_json_path, json.dumps(transcript_json).encode("utf-8"), "application/json", rid)
        sb_upload(TRANSCRIPTS_BUCKET, transcript_txt_path, transcript_txt.encode("utf-8"), "text/plain; charset=utf-8", rid)

        resp = {
            "processed_path": processed_path,
            "transcript_json": transcript_json_path,
            "transcript_txt": transcript_txt_path,
            "duration": transcript_json["duration"],
            "language": transcript_json["language"],
            "request_id": rid,
        }
        log("RESP: " + safe_snip(json.dumps(resp)), rid)
        return resp

    except HTTPException as he:
        log(f"HTTPException {he.status_code}: {safe_snip(str(he.detail))}", rid)
        raise
    except subprocess.CalledProcessError as cpe:
        log(f"FFMPEG ERROR: returncode={cpe.returncode}", rid)
        log_exc(rid)
        raise HTTPException(status_code=500, detail=f"ffmpeg_failed:{cpe.returncode}")
    except Exception:
        log_exc(rid)
        raise HTTPException(status_code=500, detail="internal_error")
    finally:
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
                log("CLEANUP temp dir", rid)
        except Exception:
            log("CLEANUP error (ignored)", rid)
