import threading
import uuid
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, abort

from pipeline import run_pipeline

app = Flask(__name__)

# In-memory job store: job_id -> dict(status, stage, message, result, error)
JOBS = {}
JOBS_LOCK = threading.Lock()

DOWNLOAD_DIR = Path("/home/ubuntu/portal_downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _update_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _run_job(job_id, prompt):
    def progress_cb(stage, message):
        _update_job(job_id, stage=stage, message=message)

    try:
        _update_job(job_id, status="running", stage=0, message="Starting pipeline...")
        result = run_pipeline(prompt, progress_cb=progress_cb)

        # Copy CIF to a stable download location for this job
        cif_src = Path(result["cif_path"])
        cif_dest = DOWNLOAD_DIR / f"{job_id}.cif"
        shutil.copy(cif_src, cif_dest)

        _update_job(
            job_id,
            status="done",
            stage=4,
            message="Pipeline completed successfully.",
            result={
                "summary": result["summary"],
                "hypothesis": result["hypothesis"],
                "cif_filename": cif_dest.name,
                "log_file": result["log_file"],
            },
        )
    except Exception as e:
        _update_job(job_id, status="error", message=str(e))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True)
    prompt = (data or {}).get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "stage": 0, "message": "Queued...", "result": None, "error": None}

    thread = threading.Thread(target=_run_job, args=(job_id, prompt), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify(job)


@app.route("/api/download/<job_id>")
def api_download(job_id):
    cif_path = DOWNLOAD_DIR / f"{job_id}.cif"
    if not cif_path.exists():
        abort(404)
    return send_file(cif_path, as_attachment=True, download_name=f"structure_{job_id}.cif")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
