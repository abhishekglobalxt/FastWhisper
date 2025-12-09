import os
import tempfile
import shutil
import subprocess
import requests
import json
import traceback
import uuid
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from faster_whisper import WhisperModel

# ========= ENV =========
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE")
RAW_BUCKET = os.environ.get("RAW_BUCKET", "raw")
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET", "processed")  # unused now, kept for compatibility
TRANSCRIPTS_BUCKET = os.environ.get("TRANSCRIPTS_BUCKET", "transcripts")
API_KEY = os.environ.get("TRANSCRIBE_API_KEY", "changeme")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")  # override via env for speed (tiny/small/etc.)

DEBUG = os.environ.get("DEBUG", "0") == "1"
LOG_BODY = os.environ.get("LOG_BODY", "0") == "1"

# Supabase I/O tuning
SB_TIMEOUT = int(os.environ.get("SUPABASE_TIMEOUT", "60"))       # per-request timeout (seconds)
SB_RETRIES = int(os.environ.get("SUPABASE_RETRIES", "3"))        # retries for download/upload

# Transcription tuning
MAX_AUDIO_SECONDS = float(os.environ.get("MAX_AUDIO_SECONDS", "900.0"))  # hard guard (15 mins default)
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "3"))
WHISPER_WORD_TS = os.environ.get("WHISPER_WORD_TS", "0") == "1"
WHISPER_VAD = os.environ.get("WHISPER_VAD", "1") == "1"

APP_VERSION = os.environ.get("APP_VERSION", "2025-12-09-01")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE) must be set")

app = FastAPI()


# ========= LOG HELPERS =========
def now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log(msg: str, rid: str = "", level: str = "INFO"):
    """
    Simple structured-ish logger:
    [time][level][rid:xxxx] message
    """
    prefix = f"[{now()}][{level}]"
    if rid:
        prefix += f"[rid:{rid}]"
    print(prefix, msg, flush=True)


def log_exc(rid: str = ""):
    tb = traceback.format_exc()
    log("EXCEPTION:\n" + tb, rid, level="ERROR")


def safe_snip(txt: str, n: int = 500) -> str:
    try:
        s = txt if isinstance(txt, str) else str(txt)
        return s[:n]
    except Exception:
        return "<unprintable>"


# ========= INIT WHISPER ONCE =========
log(f"Loading Whisper model '{WHISPER_MODEL}' with beam_size={WHISPER_BEAM_SIZE} word_ts={WHISPER_WORD_TS} vad={WHISPER_VAD} ...")
model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
log("Whisper model loaded.")


# ========= SUPABASE HELPERS =========
def sb_headers(extra=None):
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
      - signed/raw URL
      - 'raw/<path>'
      - '<path>'
    Returns the path inside the RAW bucket, without leading 'raw/'.
    """
    p = raw_path or ""
    if p.startswith("http"):
        import re
        m = re.search(r"/storage/v1/object/(?:sign|raw)/([^/]+)/(.+?)(?:\?|$)", p)
        if m:
            bucket = m.group(1)
            path_in_bucket = m.group(2)
            log(f"[normalize] detected URL; bucket={bucket} path={path_in_bucket}", rid)
            p = path_in_bucket
        else:
            log("[normalize] URL pattern not recognized; passing as-is (may fail).", rid)
    if p.startswith(f"{RAW_BUCKET}/"):
        p = p[len(RAW_BUCKET) + 1 :]
        log(f"[normalize] stripped leading '{RAW_BUCKET}/' -> {p}", rid)
    return p


def sb_download(bucket: str, path_in_bucket: str, dest_path: str, rid: str):
    """
    Download from Supabase (or signed URL) with retries + detailed logging.
    """
    last_status = None
    last_body_snip = None
    last_exc = None

    for attempt in range(1, SB_RETRIES + 1):
        try:
            if path_in_bucket.startswith("http"):
                url = path_in_bucket
                log(f"[download] attempt={attempt} (signed URL) -> {url}", rid)
                r = requests.get(url, stream=True, timeout=SB_TIMEOUT)
            else:
                url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path_in_bucket}"
                log(f"[download] attempt={attempt} (bucket path) -> {url}", rid)
                r = requests.get(url, headers=sb_headers(), stream=True, timeout=SB_TIMEOUT)

            last_status = r.status_code
            if r.status_code != 200:
                body_snip = safe_snip(r.text, 300) if LOG_BODY else f"<{len(r.text)} bytes>"
                last_body_snip = body_snip
                log(f"[download] FAIL status={r.status_code} url={url} body={body_snip}", rid, level="WARN")
            else:
                # Stream to file
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                log(f"[download] OK -> {dest_path}", rid)
                return
        except requests.RequestException as e:
            last_exc = e
            log(f"[download] EXCEPTION attempt={attempt}: {e}", rid, level="WARN")

        # Backoff before next attempt (if any)
        if attempt < SB_RETRIES:
            sleep_for = 2 * attempt
            log(f"[download] retrying in {sleep_for}s ...", rid)
            time.sleep(sleep_for)

    # If we fall through, all attempts failed
    detail = f"download_failed_after_retries status={last_status} body={safe_snip(last_body_snip, 200)} exc={last_exc}"
    log(f"[download] GIVING UP: {detail}", rid, level="ERROR")
    raise HTTPException(status_code=502, detail=f"download_failed:{detail}")


def sb_upload(bucket: str, path_in_bucket: str, content_bytes: bytes, content_type: str, rid: str):
    """
    Upload to Supabase with retries + detailed logging.
    """
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path_in_bucket}"
    last_status = None
    last_body_snip = None
    last_exc = None

    for attempt in range(1, SB_RETRIES + 1):
        try:
            headers = sb_headers({"Content-Type": content_type, "x-upsert": "true"})
            log(f"[upload] attempt={attempt} -> {url}  ({len(content_bytes)} bytes, {content_type})", rid)
            r = requests.put(url, headers=headers, data=content_bytes, timeout=SB_TIMEOUT)
            last_status = r.status_code

            if r.status_code not in (200, 201):
                body_snip = safe_snip(r.text, 300) if LOG_BODY else f"<{len(r.text)} bytes>"
                last_body_snip = body_snip
                log(f"[upload] FAIL status={r.status_code} url={url} body={body_snip}", rid, level="WARN")
            else:
                log("[upload] OK", rid)
                return
        except requests.RequestException as e:
            last_exc = e
            log(f"[upload] EXCEPTION attempt={attempt}: {e}", rid, level="WARN")

        if attempt < SB_RETRIES:
            sleep_for = 2 * attempt
            log(f"[upload] retrying in {sleep_for}s ...", rid)
            time.sleep(sleep_for)

    detail = f"upload_failed_after_retries status={last_status} body={safe_snip(last_body_snip, 200)} exc={last_exc}"
    log(f"[upload] GIVING UP: {detail}", rid, level="ERROR")
    raise HTTPException(status_code=502, detail=f"upload_failed:{detail}")


# ========= MEDIA / TRANSCRIPTION =========
def convert_to_wav(input_path: str, output_path: str, rid: str):
    cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path, "-y"]
    log(f"[ffmpeg] to WAV: {' '.join(cmd)}", rid)
    subprocess.run(cmd, check=True)
    log("[ffmpeg] WAV OK", rid)


def run_transcription(audio_path: str, rid: str):
    """
    Run Faster-Whisper with env-driven options, and include detailed logs.
    """
    log(
        f"[whisper] transcribe start model={WHISPER_MODEL} beam_size={WHISPER_BEAM_SIZE} "
        f"word_ts={WHISPER_WORD_TS} vad={WHISPER_VAD}",
        rid,
    )

    # Main transcription call
    segments, info = model.transcribe(
        audio_path,
        beam_size=WHISPER_BEAM_SIZE,
        word_timestamps=WHISPER_WORD_TS,
        vad_filter=WHISPER_VAD,
    )
    log(f"[whisper] language={info.language} duration={info.duration:.3f}s", rid)

    # Duration guard (after fact; still useful for logs + future decisions)
    if info.duration and info.duration > MAX_AUDIO_SECONDS:
        msg = f"audio_too_long:{info.duration:.2f}s > max={MAX_AUDIO_SECONDS:.2f}s"
        log(f"[whisper] {msg}", rid, level="WARN")
        # 413 = Payload Too Large (semantically appropriate)
        raise HTTPException(status_code=413, detail=msg)

    transcript_json = {"duration": info.duration, "language": info.language, "segments": []}
    txt_lines = []

    include_words = WHISPER_WORD_TS  # only embed per-word timings when explicitly requested

    for seg in segments:
        seg_dict = {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
        if include_words and getattr(seg, "words", None):
            seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
        transcript_json["segments"].append(seg_dict)
        txt_lines.append(seg.text.strip())

    transcript_txt = "\n".join(txt_lines)
    log("[whisper] transcribe done", rid)
    return transcript_json, transcript_txt


# ========= ROUTES =========
@app.get("/health")
def health():
    """
    Simple healthcheck with version + model info.
    """
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": WHISPER_MODEL,
        "beam_size": WHISPER_BEAM_SIZE,
        "word_ts": WHISPER_WORD_TS,
        "vad": WHISPER_VAD,
    }


@app.post("/process")
def process(data: dict, x_api_key: str = Header(None)):
    """
    Main entrypoint.

    Steps (all logged with rid):
    1) Auth check
    2) Input validation (rawPath, processedPrefix)
    3) Path normalization
    4) Download from Supabase
    5) FFmpeg to WAV
    6) Whisper transcription
    7) Upload transcripts to TRANSCRIPTS_BUCKET
    8) Respond with paths + metadata
    """
    rid = str(uuid.uuid4())[:8]
    temp_dir = None
    try:
        body_log = safe_snip(json.dumps(data), 500) if (DEBUG or LOG_BODY) else "<hidden>"
        log(f"REQ /process body={body_log}", rid)

        # 1) Auth
        if x_api_key != API_KEY:
            log("[auth] FAIL (X-API-KEY mismatch)", rid, level="WARN")
            raise HTTPException(status_code=401, detail="unauthorized:bad_api_key")

        # 2) Basic validation
        raw_path_in = data.get("rawPath") or data.get("raw_path")
        processed_prefix = data.get("processedPrefix") or data.get("processed_prefix")

        if not raw_path_in:
            log("[input] missing rawPath/raw_path", rid, level="WARN")
            raise HTTPException(status_code=400, detail="missing_rawPath")

        if not processed_prefix:
            log("[input] missing processedPrefix/processed_prefix", rid, level="WARN")
            raise HTTPException(status_code=400, detail="missing_processedPrefix")

        # Optional meta for extra debugging context
        meta = data.get("meta") or {}
        log(f"[input] rawPath={safe_snip(raw_path_in, 200)} processedPrefix={processed_prefix} meta={safe_snip(meta, 200)}", rid)

        # 3) Normalize the input path (we won't upload the video again)
        path_in_bucket = normalize_path_in_bucket(raw_path_in, rid)
        log(f"[stage] NORMALIZED rawPath -> {path_in_bucket}", rid)

        # 4) Prepare temp dir
        temp_dir = tempfile.mkdtemp()
        log(f"[stage] TEMP DIR -> {temp_dir}", rid)

        # 5) Download existing WEBM (or other container) to transcribe
        local_raw = os.path.join(temp_dir, os.path.basename(path_in_bucket) or "input.webm")
        log(f"[stage] DOWNLOAD -> local_raw={local_raw}", rid)
        sb_download(RAW_BUCKET, path_in_bucket, local_raw, rid)

        # 6) Convert to WAV
        local_wav = os.path.join(temp_dir, "audio.wav")
        log(f"[stage] FFMPEG convert -> {local_wav}", rid)
        convert_to_wav(local_raw, local_wav, rid)

        # 7) Transcribe
        log("[stage] WHISPER transcribe", rid)
        transcript_json, transcript_txt = run_transcription(local_wav, rid)

        # 8) Upload ONLY transcripts (NO video upload here)
        transcript_base = f"{processed_prefix}/transcript"
        transcript_json_path = f"{transcript_base}.json"
        transcript_txt_path = f"{transcript_base}.txt"

        log(f"[stage] UPLOAD transcripts -> bucket={TRANSCRIPTS_BUCKET} base={transcript_base}", rid)
        sb_upload(
            TRANSCRIPTS_BUCKET,
            transcript_json_path,
            json.dumps(transcript_json).encode("utf-8"),
            "application/json",
            rid,
        )
        sb_upload(
            TRANSCRIPTS_BUCKET,
            transcript_txt_path,
            transcript_txt.encode("utf-8"),
            "text/plain; charset=utf-8",
            rid,
        )

        # 9) Respond with paths expected by your n8n/Supabase schema
        resp = {
            # existing video location; we NEVER re-upload it here
            "processed_path": f"{RAW_BUCKET}/{path_in_bucket}",
            "transcript_json": transcript_json_path,
            "transcript_txt": transcript_txt_path,
            "duration": transcript_json["duration"],
            "language": transcript_json["language"],
            "request_id": rid,
        }
        log("RESP " + safe_snip(json.dumps(resp), 500), rid)
        return resp

    except HTTPException as he:
        # HTTPExceptions already have a status + detail; just log and re-raise
        log(f"[error] HTTPException {he.status_code}: {safe_snip(str(he.detail), 300)}", rid, level="ERROR")
        raise
    except subprocess.CalledProcessError as cpe:
        # Specifically ffmpeg failure
        log(f"[error] FFMPEG ERROR: returncode={cpe.returncode}", rid, level="ERROR")
        log_exc(rid)
        raise HTTPException(status_code=500, detail=f"ffmpeg_failed:{cpe.returncode}")
    except Exception:
        # Catch-all internal error
        log_exc(rid)
        raise HTTPException(status_code=500, detail="internal_error")
    finally:
        # Best-effort cleanup
        try:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
                log("[cleanup] temp dir removed", rid)
        except Exception:
            log("[cleanup] temp dir removal error (ignored)", rid, level="WARN")
