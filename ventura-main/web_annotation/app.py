#!/usr/bin/env python3
"""
Flask video annotation tool (timeline blocks, start/end, arrow-key scrubbing).

• Serve videos from static/videos
• Annotate scenario intervals with start/end + label
• Editor-like timeline at bottom with draggable/resizable blocks
• Saves per-video annotations as JSON to --out_dir
"""
from __future__ import annotations

import os, re, hashlib
import argparse, json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask import send_file, abort, url_for

from spinflow.dataset.frodo_helpers import (
    get_available_sequences
)

# ────────── CLI ───────────────────────────────────────────────────
pa = argparse.ArgumentParser()
pa.add_argument("--in_dir", default="./data/fai_spinflow_raw", help="Directory containing .mp4 files")
pa.add_argument("--out_dir",   default="./data/fai_spinflow_raw/language_labels",   help="Where to save JSON annotations")
pa.add_argument("--host",      default="0.0.0.0")
pa.add_argument("--port",      type=int, default=5000)
ARGS = pa.parse_args()

IN_DIR = Path(ARGS.in_dir)
OUT_DIR   = Path(ARGS.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_INDEX: dict[str, Path] = {}

# ────────── Flask init ────────────────────────────────────────────
app = Flask(__name__)

# ────────── helpers ───────────────────────────────────────────────
# def list_videos():
#     """Return list of (display_name, static_url, basename) for allowed video types."""
#     exts = {".mp4", ".mov", ".webm", ".m4v"}
#     vids = []
#     if IN_DIR.exists():
#         for fp in sorted(IN_DIR.iterdir()):
#             if fp.suffix.lower() in exts and fp.is_file():
#                 rel = f"videos/{fp.name}"
#                 # cache-bust with mtime so we avoid 304 issues during dev
#                 mtime = int(fp.stat().st_mtime)
#                 url = url_for("static", filename=rel) + f"?v={mtime}"
#                 vids.append((fp.name, url, fp.stem))
#     return vids

def _safe_id(text: str) -> str:
    """Sanitize into a mostly filesystem-safe ID."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)

def _short_hash(p: Path) -> str:
    return hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]

def format_sequences_for_ui(paths: list[str | Path]) -> list[tuple[str, str, str]]:
    """
    Turn absolute video paths into (display, url, id) triples for the template:
      display: relative path from common root (nice label)
      url:     /media_id/<id>?v=<mtime>   (cache-busted)
      id:      stable id used in /save and /load
    Also populates VIDEO_INDEX[id] = absolute Path.
    """
    ps = [Path(p).expanduser().resolve() for p in paths]
    ps = [p for p in ps if p.is_file()]  # ignore non-files

    if not ps:
        VIDEO_INDEX.clear()
        return []

    # Find a common root for pretty display labels
    try:
        common = Path(os.path.commonpath([str(p) for p in ps]))
    except Exception:
        common = ps[0].parent

    VIDEO_INDEX.clear()
    out: list[tuple[str, str, str]] = []

    seen_ids: set[str] = set()
    for p in ps:
        rel = p
        try:
            rel = p.relative_to(common)
        except Exception:
            pass
        display = rel.as_posix()

        # Base id from relative path; ensure uniqueness with short hash on collisions
        base_id = _safe_id(display)
        vid_id = base_id if base_id not in seen_ids else f"{base_id}-{_short_hash(p)}"
        vid_id = vid_id.replace("_lossy", "")  # remove "lossy" suffix if present
        seen_ids.add(vid_id)
        VIDEO_INDEX[vid_id] = p

        mtime = int(p.stat().st_mtime)
        url = url_for("media_by_id", vid=vid_id) + f"?v={mtime}"
        out.append((display, url, vid_id))
    
    # Sort by display name
    out.sort(key=lambda t: t[0].lower())
    return out

def ann_path(video_id: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{video_id}.json"

# ────────── routes ────────────────────────────────────────────────
@app.route("/")
def index():
    # If your function needs an argument (e.g., a root directory), pass it here:
    # seqs = get_available_sequences(ARGS.video_dir)
    seqs = get_available_sequences(IN_DIR, "front_camera_lossy", "mp4")
    
    videos = format_sequences_for_ui(seqs)
    if not videos:
        return render_template("index.html", videos=[], default_url="", default_basename="")

    disp, url, vid_id = videos[0]
    return render_template(
        "index.html",
        videos=videos,                 # iterable of (display, url, id)
        default_url=url,               # src for <video>
        default_basename=vid_id,       # used by your JS for cache-busting
    )

@app.route("/media_id/<vid>")
def media_by_id(vid: str):
    p = VIDEO_INDEX.get(vid)
    if not p or not p.exists() or not p.is_file():
        abort(404)
    # conditional=True enables ETag/If-Modified-Since + Range support
    return send_file(p, conditional=True)

@app.after_request
def add_no_cache_headers(resp):
    # Dev-friendly: ensure latest JS/CSS/video always load.
    # You can narrow this to specific paths if you prefer.
    if request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

@app.route("/load", methods=["GET"])
def load():
    video_id = (request.args.get("video") or "").strip()
    fp = ann_path(video_id)
    if not fp.exists():
        return jsonify({"video": video_id, "annotations": [], "meta": {}})
    try:
        data = json.loads(fp.read_text())
    except Exception:
        data = {"video": video_id, "annotations": [], "meta": {}}
    return jsonify(data)

@app.route("/save", methods=["POST"])
def save():
    data = request.get_json(force=True)
    video_id = (data.get("video") or "").strip()
    if not video_id:
        return jsonify({"error": "missing 'video' id"}), 400
    
    data["video"] = video_id
    fp = ann_path(video_id)
    fp.write_text(json.dumps(data, indent=2))
    return jsonify({"saved": "ok", "file": str(fp)})

if __name__ == "__main__":
    app.run(host=ARGS.host, port=ARGS.port, debug=True, threaded=False)
