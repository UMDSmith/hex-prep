"""
hex-prep web UI — drag-and-drop front end for the extraction chain.

Serves a single page: drop (or pick) an audio file, click Process, get
download links for the clean vocals (.wav) and the clean music (.mp3).
Wraps process() from hex_prep.cli, always with mix_music=True.

Data flow: uploaded file -> HEXPREP_OUTPUT_DIR/_uploads/<name> ->
process() -> HEXPREP_OUTPUT_DIR/<name>/ stems -> served via /download.
One GPU job at a time (global lock); extra requests wait their turn.

Run:  hex-prep-web            (default http://127.0.0.1:7870)
      hex-prep-web --port 8123 --host 0.0.0.0
"""
import argparse
import re
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from hex_prep import cli

UPLOAD_DIR = cli.OUTPUT_DIR / "_uploads"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

# job_id -> {"vocals": Path, "music": Path}; populated after each run
JOBS: dict[str, dict[str, Path]] = {}
GPU_LOCK = threading.Lock()

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hex-prep — Vocal Stripper</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; background: #14141c; color: #e8e8f0;
         display: flex; flex-direction: column; align-items: center;
         min-height: 100vh; margin: 0; padding: 2rem 1rem; box-sizing: border-box; }
  h1 { font-size: 1.4rem; margin: 0 0 1.5rem; }
  #drop { width: 100%; max-width: 560px; border: 2px dashed #555; border-radius: 12px;
          padding: 3rem 1rem; text-align: center; cursor: pointer;
          transition: border-color .15s, background .15s; }
  #drop.hover { border-color: #e05555; background: #1e1e2a; }
  #drop p { margin: .3rem 0; color: #aaa; }
  #fname { color: #e8e8f0; font-weight: 600; }
  button { margin-top: 1.2rem; font-size: 1.05rem; padding: .7rem 2.4rem;
           border: 0; border-radius: 8px; background: #e05555; color: #fff;
           cursor: pointer; }
  button:disabled { background: #444; color: #888; cursor: default; }
  #status { margin-top: 1.2rem; min-height: 1.4em; color: #aaa; }
  #links { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;
           justify-content: center; }
  #links a { background: #2a2a3a; color: #fff; text-decoration: none;
             padding: .7rem 1.4rem; border-radius: 8px; }
  #links a:hover { background: #3a3a50; }
  input[type=file] { display: none; }
</style>
</head>
<body>
<h1>hex-prep — drop a song, get clean vocals + clean music</h1>
<div id="drop">
  <p id="fname">Drag &amp; drop a song here</p>
  <p>or click to browse</p>
</div>
<input type="file" id="file" accept="audio/*,.mp3,.flac,.wav,.m4a,.ogg,.opus,.aac,.wma">
<button id="go" disabled>Process</button>
<div id="status"></div>
<div id="links"></div>
<script>
const drop = document.getElementById('drop');
const input = document.getElementById('file');
const go = document.getElementById('go');
const status = document.getElementById('status');
const links = document.getElementById('links');
let file = null;

function setFile(f) {
  file = f;
  document.getElementById('fname').textContent = f ? f.name : 'Drag & drop a song here';
  go.disabled = !f;
  links.innerHTML = '';
  status.textContent = '';
}
drop.addEventListener('click', () => input.click());
input.addEventListener('change', () => setFile(input.files[0] || null));
['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add('hover');
}));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove('hover');
}));
drop.addEventListener('drop', e => setFile(e.dataTransfer.files[0] || null));

go.addEventListener('click', async () => {
  if (!file) return;
  go.disabled = true;
  links.innerHTML = '';
  const t0 = Date.now();
  const tick = setInterval(() => {
    status.textContent = `Processing… ${Math.round((Date.now() - t0) / 1000)}s`;
  }, 500);
  try {
    const fd = new FormData();
    fd.append('song', file);
    const resp = await fetch('/process', { method: 'POST', body: fd });
    const data = await resp.json();
    clearInterval(tick);
    if (!resp.ok) { status.textContent = 'ERROR: ' + (data.error || resp.statusText); return; }
    status.textContent = `Done in ${data.seconds}s`;
    links.innerHTML =
      `<a href="${data.vocals}" download>⬇ Clean vocals (.wav)</a>` +
      `<a href="${data.music}" download>⬇ Clean music (.mp3)</a>`;
  } catch (err) {
    clearInterval(tick);
    status.textContent = 'ERROR: ' + err;
  } finally {
    go.disabled = !file;
  }
});
</script>
</body>
</html>"""


@app.get("/")
def index():
    return PAGE


@app.post("/process")
def process_song():
    upload = request.files.get("song")
    if upload is None or not upload.filename:
        return jsonify({"error": "no file uploaded"}), 400

    # Sanitize the filename but keep it readable — it names the output folder.
    name = Path(upload.filename).name
    name = re.sub(r"[^\w .&'()\[\]-]", "_", name).strip() or "upload"
    if Path(name).suffix.lower() not in cli.AUDIO_EXTS:
        return jsonify({"error": f"unsupported file type: {name}"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    song_path = UPLOAD_DIR / name
    upload.save(song_path)

    t0 = time.time()
    with GPU_LOCK:
        results = cli.process(song_path, mix_music=True)
    seconds = round(time.time() - t0, 1)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"vocals": results["lead"], "music": results["music"]}
    return jsonify({
        "seconds": seconds,
        "vocals": f"/download/{job_id}/vocals",
        "music": f"/download/{job_id}/music",
    })


@app.get("/download/<job_id>/<stem>")
def download(job_id: str, stem: str):
    job = JOBS.get(job_id)
    if job is None or stem not in job:
        return jsonify({"error": "unknown download"}), 404
    return send_file(job[stem], as_attachment=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--host", type=str, default="127.0.0.1",
                    help="Bind address (0.0.0.0 to allow other devices on "
                         "your network)")
    args = ap.parse_args()
    print(f"hex-prep web UI -> http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
