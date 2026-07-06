"""
Script Doctor — Flask Web Server
---------------------------------
GET  /          → serves the web UI (api/static/index.html)
POST /run       → starts a pipeline job, returns { job_id }
GET  /stream/<job_id> → SSE stream: progress events then the final report HTML
"""

import sys
import os
import json
import uuid
import queue
import logging
import threading
import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from main import run_pipeline, RUNS_DIR
from report.generator import generate_report

logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB max request body

# ── Concurrency limits ────────────────────────────────────────────────────────
# Each pipeline job fires ~15+ Gemini calls; running many at once will exhaust
# the API quota and thrash the shared vector store. Cap how many pipelines run
# concurrently, and how many jobs may be outstanding (running + queued + awaiting
# their SSE stream) before new requests are refused with 429.
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
MAX_ACTIVE_JOBS = int(os.getenv("MAX_ACTIVE_JOBS", "8"))
_pipeline_slots = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)

# In-memory job store: job_id → {"queue": queue.Queue, "created_at": float}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_JOBS_TTL_SECONDS = 3600  # 1 hour — abandon jobs older than this


def _cleanup_stale_jobs():
    """Background thread: evict orphaned job entries older than TTL."""
    import time
    while True:
        time.sleep(300)  # run every 5 minutes
        now = datetime.datetime.now().timestamp()
        with _jobs_lock:
            stale = [jid for jid, meta in list(_jobs.items())
                     if now - meta["created_at"] > _JOBS_TTL_SECONDS]
            for jid in stale:
                _jobs.pop(jid, None)
                logger.info("Evicted stale job %s", jid)


_cleanup_thread = threading.Thread(target=_cleanup_stale_jobs, daemon=True)
_cleanup_thread.start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/run", methods=["POST"])
def start_run():
    """Accept a prompt, spawn a background worker, return the job_id."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    # Refuse new work when the server is already saturated, so clients get a
    # clear signal instead of piling up threads that starve the API quota.
    with _jobs_lock:
        if len(_jobs) >= MAX_ACTIVE_JOBS:
            logger.warning("Rejecting job — %d active jobs (cap %d).", len(_jobs), MAX_ACTIVE_JOBS)
            return jsonify({
                "error": "Server is at capacity. Please retry in a moment."
            }), 429
        job_id = str(uuid.uuid4())
        q: queue.Queue = queue.Queue()
        _jobs[job_id] = {"queue": q, "created_at": datetime.datetime.now().timestamp()}

    logger.info("Starting pipeline job %s", job_id)

    def _worker():
        def _progress(node_name: str, message: str):
            q.put({"event": "progress", "node": node_name, "message": message})

        # Bound concurrent pipeline execution. If no slot is free, tell the
        # client it's queued and block until one opens up.
        slot_acquired = _pipeline_slots.acquire(blocking=False)
        if not slot_acquired:
            _progress("queued", "⏳ Waiting for an open pipeline slot...")
            _pipeline_slots.acquire()
            slot_acquired = True

        try:
            final_state, iteration_history, timestamp = run_pipeline(
                prompt=prompt,
                on_node=_progress,
                save_files=True,
                output_dir=str(RUNS_DIR),
            )

            # Generate and save HTML report once; read HTML back from disk
            saved_path = generate_report(
                prompt=final_state["prompt"],
                final_script=final_state["draft"],
                iteration_history=iteration_history,
                final_score=final_state["score"],
                timestamp=timestamp,
                output_dir=str(RUNS_DIR),
                save=True,
            )
            html_str = open(saved_path, encoding="utf-8").read()

            q.put({
                "event": "done",
                "score": final_state["score"],
                "iterations": final_state["iteration_count"],
                "saved_as": Path(saved_path).name,
                "html": html_str,
            })
            logger.info("Job %s completed — score %d/10", job_id, final_state["score"])

        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
            q.put({"event": "error", "message": str(exc)})
        finally:
            if slot_acquired:
                _pipeline_slots.release()
            q.put(None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream_job(job_id: str):
    """SSE endpoint: stream progress events for the given job."""
    with _jobs_lock:
        job_meta = _jobs.get(job_id)
    if job_meta is None:
        return jsonify({"error": "Job not found."}), 404
    q = job_meta["queue"]

    def _generate():
        try:
            while True:
                item = q.get(timeout=300)  # 5 min max
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            with _jobs_lock:
                _jobs.pop(job_id, None)

    headers = {
        "Cache-Control":   "no-cache",
        "X-Accel-Buffering": "no",
        "Connection":      "keep-alive",
    }
    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers=headers,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Script Doctor Web UI")
    logger.info("  Open http://localhost:5000 in your browser")
    logger.info("=" * 60)
    # threaded=True so SSE and POST can run concurrently
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
