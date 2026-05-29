from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# ── PATHS ─────────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).parent.resolve()
JOBS_DIR      = DASHBOARD_DIR.parent
BUILDER_DIR   = JOBS_DIR / "resume_builder"
PYTHON_EXE    = JOBS_DIR / "venv" / "Scripts" / "python.exe"

JOBSPY_SCRIPT = JOBS_DIR / "run_jobspy.py"
TAILOR_SCRIPT = BUILDER_DIR / "tailor.py"
JOBS_CSV      = JOBS_DIR / "jobs_results.csv"
PROFILES_DIR  = BUILDER_DIR / "profiles"
OUTPUT_DIR    = BUILDER_DIR / "output"
SETTINGS_FILE = DASHBOARD_DIR / "settings.json"

TEMPLATES = ["modern", "classic", "minimal", "ats_safe", "tech_bold"]

# ── APP ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
_lock = threading.Lock()

scrape_state: dict = {"status": "idle", "log": [], "last_run": None}
tailor_state: dict = {"status": "idle", "log": [], "last_output": None}

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DEFAULTS = {"notify_threshold": 40, "hours_old": 2,
            "default_template": "modern", "default_profile": "default"}

def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return {**DEFAULTS, **json.loads(SETTINGS_FILE.read_text("utf-8"))}
        except Exception:
            pass
    return DEFAULTS.copy()

def save_settings(data: dict) -> None:
    merged = {**load_settings(), **{k: v for k, v in data.items() if k in DEFAULTS}}
    SETTINGS_FILE.write_text(json.dumps(merged, indent=2), "utf-8")

# ── JOBS ──────────────────────────────────────────────────────────────────────
@app.route("/api/jobs")
def api_jobs():
    if not JOBS_CSV.exists():
        return jsonify({"jobs": [], "total": 0})
    try:
        jobs = []
        with open(JOBS_CSV, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                try:
                    score = float(row.get("relevance_score") or 0)
                except (ValueError, TypeError):
                    score = 0
                jobs.append({
                    "site":            row.get("site", ""),
                    "title":           row.get("title", ""),
                    "company":         row.get("company", ""),
                    "location":        row.get("location", ""),
                    "date_posted":     row.get("date_posted", ""),
                    "job_type":        row.get("job_type", ""),
                    "is_remote":       row.get("is_remote", ""),
                    "min_amount":      row.get("min_amount", ""),
                    "max_amount":      row.get("max_amount", ""),
                    "currency":        row.get("currency", "GBP"),
                    "relevance_score": score,
                    "job_url":         row.get("job_url", ""),
                })
        jobs.sort(key=lambda j: j["relevance_score"], reverse=True)
        return jsonify({"jobs": jobs, "total": len(jobs)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ── SCRAPER ───────────────────────────────────────────────────────────────────
def _run_scrape():
    with _lock:
        scrape_state.update({"status": "running", "log": [], "last_run": None})
    try:
        proc = subprocess.Popen(
            [str(PYTHON_EXE), str(JOBSPY_SCRIPT)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(JOBS_DIR),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        for line in proc.stdout:
            with _lock:
                scrape_state["log"].append(line.rstrip())
        proc.wait()
        with _lock:
            scrape_state["status"] = "done" if proc.returncode == 0 else "error"
            scrape_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        with _lock:
            scrape_state.update({"status": "error", "log": scrape_state["log"] + [f"[ERROR] {exc}"]})

@app.route("/api/scrape", methods=["POST"])
def api_scrape_start():
    with _lock:
        if scrape_state["status"] == "running":
            return jsonify({"error": "Already running"}), 409
    threading.Thread(target=_run_scrape, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/scrape/status")
def api_scrape_status():
    with _lock:
        return jsonify(scrape_state.copy())

# ── RESUME PROFILES ───────────────────────────────────────────────────────────
@app.route("/api/profiles")
def api_profiles():
    if not PROFILES_DIR.exists():
        return jsonify({"profiles": []})
    profiles = []
    for p in sorted(PROFILES_DIR.iterdir()):
        if not p.is_dir():
            continue
        rp, hist = p / "resume.json", p / "history"
        pname = p.name
        if rp.exists():
            try:
                pname = json.loads(rp.read_text("utf-8")).get("name", p.name) or p.name
            except Exception:
                pass
        pdfs = sorted(hist.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True) if hist.exists() else []
        profiles.append({
            "id": p.name, "name": pname,
            "has_resume": rp.exists(),
            "history_count": len(pdfs),
            "recent_pdf": pdfs[0].name if pdfs else None,
        })
    return jsonify({"profiles": profiles})

# ── TAILOR ────────────────────────────────────────────────────────────────────
def _run_tailor(profile, jd, template, company, model):
    with _lock:
        tailor_state.update({"status": "running", "log": [], "last_output": None})
    jd_file = BUILDER_DIR / "_temp_jd.txt"
    jd_file.write_text(jd, encoding="utf-8")
    try:
        cmd = [str(PYTHON_EXE), str(TAILOR_SCRIPT), "tailor",
               "--file", str(jd_file),
               "--profile", profile,
               "--template", template,
               "--company", company or "job",
               "--model", model]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(BUILDER_DIR),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        for line in proc.stdout:
            with _lock:
                tailor_state["log"].append(line.rstrip())
        proc.wait()
        last_pdf = None
        if OUTPUT_DIR.exists():
            pdfs = sorted(OUTPUT_DIR.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)
            last_pdf = pdfs[0].name if pdfs else None
        with _lock:
            tailor_state["status"] = "done" if proc.returncode == 0 else "error"
            tailor_state["last_output"] = last_pdf
    except Exception as exc:
        with _lock:
            tailor_state.update({"status": "error",
                                  "log": tailor_state["log"] + [f"[ERROR] {exc}"]})
    finally:
        if jd_file.exists():
            jd_file.unlink(missing_ok=True)

@app.route("/api/tailor", methods=["POST"])
def api_tailor_start():
    with _lock:
        if tailor_state["status"] == "running":
            return jsonify({"error": "Already running"}), 409
    body = request.json or {}
    if not body.get("jd", "").strip():
        return jsonify({"error": "Job description required"}), 400
    threading.Thread(
        target=_run_tailor,
        args=(body.get("profile", "default"), body["jd"],
              body.get("template", "modern"), body.get("company", ""),
              body.get("model", "qwen2.5-coder:32b").replace("auto", "qwen2.5-coder:32b")),
        daemon=True,
    ).start()
    return jsonify({"status": "started"})

@app.route("/api/tailor/status")
def api_tailor_status():
    with _lock:
        return jsonify(tailor_state.copy())

@app.route("/api/output/<path:filename>")
def api_download(filename):
    target = OUTPUT_DIR / Path(filename).name  # prevent path traversal
    if not target.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(target), as_attachment=True)

@app.route("/api/output/list")
def api_output_list():
    if not OUTPUT_DIR.exists():
        return jsonify({"files": []})
    files = [
        {"name": f.name,
         "size_kb": round(f.stat().st_size / 1024, 1),
         "created": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
        for f in sorted(OUTPUT_DIR.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)[:30]
    ]
    return jsonify({"files": files})

# ── SETTINGS ──────────────────────────────────────────────────────────────────
@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        save_settings(request.json or {})
        return jsonify({"ok": True})
    return jsonify(load_settings())

# ── MAIN ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    print("\n  JobSpy Dashboard  →  http://localhost:5000\n")
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
