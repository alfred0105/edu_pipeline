"""
edu_pipeline Web UI
브라우저에서 파이프라인을 실행하고 실시간 로그를 확인할 수 있는 웹 인터페이스.

실행: python web_ui.py
접속: http://localhost:5000
"""

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

app = Flask(__name__)

# 실행 중인 작업 관리
_jobs: dict[str, dict] = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/input-files")
def list_input_files():
    """data/input 폴더의 영상 파일 목록"""
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
    files = []
    if INPUT_DIR.exists():
        for f in sorted(INPUT_DIR.iterdir()):
            if f.suffix.lower() in exts:
                files.append({"name": f.name, "path": str(f)})
    return jsonify(files)


@app.route("/api/output-files")
def list_output_files():
    """data/output 폴더의 결과물 목록"""
    results = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir():
                files = [f.name for f in sorted(d.iterdir()) if f.is_file() and not f.name.startswith("_")]
                results.append({"name": d.name, "files": files})
    return jsonify(results)


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """파이프라인 실행 (비동기)"""
    data = request.json
    filepath = data.get("file", "")
    if not filepath or not Path(filepath).is_file():
        return jsonify({"error": "파일을 찾을 수 없습니다"}), 400

    job_id = str(uuid.uuid4())[:8]
    cmd = [str(VENV_PYTHON), "main.py", "-i", filepath]

    # 옵션 조립
    lang = data.get("lang")
    if lang:
        cmd += ["--lang", lang]
    cmd += ["--target-lang", data.get("targetLang", "ko")]

    step = data.get("startFrom", 1)
    if step > 1:
        cmd += ["--start-from", str(step)]

    if data.get("forceTranslate"):
        cmd.append("--force-translate")
    if data.get("skipTts"):
        cmd.append("--skip-tts")
    if data.get("chat"):
        cmd.append("--chat")

    _jobs[job_id] = {"status": "running", "logs": [], "process": None}

    thread = threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True)
    thread.start()

    return jsonify({"jobId": job_id})


def _run_job(job_id: str, cmd: list[str]):
    job = _jobs[job_id]
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(BASE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        job["process"] = proc
        for line in proc.stdout:
            job["logs"].append(line)
        proc.wait()
        job["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as e:
        job["logs"].append(f"\n[오류] {e}\n")
        job["status"] = "error"
    finally:
        job["process"] = None


@app.route("/api/logs/<job_id>")
def stream_logs(job_id):
    """SSE 방식으로 실시간 로그 스트리밍"""
    def generate():
        idx = 0
        while True:
            job = _jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'type': 'error', 'msg': 'Job not found'})}\n\n"
                break

            while idx < len(job["logs"]):
                line = job["logs"][idx]
                yield f"data: {json.dumps({'type': 'log', 'msg': line})}\n\n"
                idx += 1

            if job["status"] != "running":
                yield f"data: {json.dumps({'type': 'done', 'status': job['status']})}\n\n"
                break

            time.sleep(0.3)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/stop/<job_id>", methods=["POST"])
def stop_job(job_id):
    job = _jobs.get(job_id)
    if job and job.get("process"):
        job["process"].terminate()
        job["status"] = "stopped"
        job["logs"].append("\n[중지됨] 사용자에 의해 중지되었습니다.\n")
        return jsonify({"ok": True})
    return jsonify({"error": "실행 중인 작업이 없습니다"}), 404


if __name__ == "__main__":
    print("=" * 50)
    print("  edu_pipeline Web UI")
    print("  http://localhost:5000")
    print("=" * 50)
    import webbrowser
    webbrowser.open("http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
