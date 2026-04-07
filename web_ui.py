"""
edu_pipeline Web UI (FastAPI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NotebookLM 스타일 웹 인터페이스

- 영상 또는 문서 업로드 → STT (또는 텍스트 추출)
- 번역 + 자막 생성
- TTS 더빙 (선택)
- RAG 챗봇으로 질의응답

실행:
  python web_ui.py
  http://localhost:8000
"""

import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from config import BASE_DIR, INPUT_DIR, OUTPUT_DIR

TEMPLATES_DIR    = BASE_DIR / "templates"
STATIC_DIR       = BASE_DIR / "static"
VENV_PYTHON      = Path(sys.executable)
MAX_UPLOAD_BYTES = 2 * 1024 ** 3   # 2 GB
BOOKMARKS_PATH   = BASE_DIR / "data" / "bookmarks.json"
_bookmarks_lock  = threading.Lock()

# 허용 업로드 확장자
ALLOWED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
ALLOWED_DOC_EXTS   = {".pdf", ".txt", ".md"}
ALLOWED_DATA_EXTS  = {".csv", ".xlsx", ".xls", ".json"}
ALLOWED_EXTS       = ALLOWED_VIDEO_EXTS | ALLOWED_DOC_EXTS | ALLOWED_DATA_EXTS


def _resolve_output_dir(stem: str) -> Path:
    """컬렉션 stem → 실제 출력 디렉토리.

    1) _collection_map.json 우선 조회
    2) OUTPUT_DIR/stem 직접 매칭
    3) source_info.json output_stem 필드 스캔
    """
    # 1) 매핑 파일 조회
    map_path = OUTPUT_DIR / "_collection_map.json"
    if map_path.exists():
        try:
            cmap = json.loads(map_path.read_text(encoding="utf-8"))
            if stem in cmap:
                resolved = OUTPUT_DIR / cmap[stem]
                if resolved.exists():
                    return resolved
        except Exception:
            pass

    # 2) 직접 매칭
    direct = OUTPUT_DIR / stem
    if direct.exists() and any(f for f in direct.iterdir() if f.is_file()):
        return direct

    # 3) source_info.json 스캔
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            info_path = d / "source_info.json"
            if not info_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                if info.get("output_stem") == stem or info.get("title") == stem:
                    return d
            except Exception:
                continue

    return direct  # fallback


def _safe_stem(stem: str) -> str:
    """stem 파라미터에서 경로 탐색 문자만 제거합니다 (파일명 자체는 보존)."""
    safe = Path(stem).name          # 디렉토리 구분자 제거
    safe = safe.replace("\x00", "").strip()
    if not safe or safe in (".", ".."):
        raise ValueError("유효하지 않은 stem입니다.")
    return safe


def _safe_output_path(stem: str, filename: str) -> Path:
    """OUTPUT_DIR/{stem}/{filename} 경로를 OUTPUT_DIR 내로 제한합니다."""
    base = OUTPUT_DIR.resolve()
    path = (base / _safe_stem(stem) / Path(filename).name).resolve()
    if not str(path).startswith(str(base)):
        raise ValueError(f"경로 접근 거부: {path}")
    return path

# 로깅
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI 앱
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan: 시작 시 워밍업, 종료 시 리소스 해제."""
    threading.Thread(target=_warm_up, daemon=True).start()
    yield
    # 종료 시 RAG 봇 정리
    with _rag_bots_lock:
        for bot in list(_rag_bots.values()):
            try:
                bot.unload()
            except Exception:
                pass
        _rag_bots.clear()

app = FastAPI(title="edu_pipeline Web UI", lifespan=_lifespan)

# 정적 파일 서빙
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_warmup_state: dict = {"ready": False, "message": "AI 초기화 중..."}


def _cleanup_empty_collections():
    """ChromaDB에 데이터(count==0) 없는 잔여 컬렉션 삭제."""
    try:
        from modules.rag import _get_chroma_client
        client = _get_chroma_client()
        removed = 0
        for c in client.list_collections():
            name = c if isinstance(c, str) else c.name
            try:
                if client.get_collection(name).count() == 0:
                    client.delete_collection(name)
                    removed += 1
            except Exception:
                pass
        if removed:
            logger.info(f"[Cleanup] 빈 컬렉션 {removed}개 삭제")
    except Exception as e:
        logger.warning(f"[Cleanup] 실패 (무시): {e}")


def _auto_reindex_missing():
    """서버 시작 시: ChromaDB에 없거나 비어있는 컬렉션을 segments.json으로부터 자동 재색인."""
    try:
        from modules.rag import list_collections
        existing = set(list_collections())
        if not OUTPUT_DIR.exists():
            return
        for d in OUTPUT_DIR.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if not (d / "segments.json").exists():
                continue
            if d.name in existing:
                continue
            logger.info(f"[AutoReindex] 누락된 컬렉션 복구: {d.name}")
            _reindex_bg(d.name)
    except Exception as e:
        logger.warning(f"[AutoReindex] 실패 (무시): {e}")


def _warm_up():
    """
    서버 시작 시 백그라운드에서 실행:
    1. 기존 컬렉션 목록 확인 (데이터 유효성 검증)
    2. 공유 임베더 사전 로드 (첫 질문 응답 속도 향상)
    """
    try:
        _warmup_state["message"] = "빈 컬렉션 정리 중..."
        _cleanup_empty_collections()
        _warmup_state["message"] = "누락 컬렉션 자동 재색인 중..."
        _auto_reindex_missing()
        _warmup_state["message"] = "컬렉션 목록 확인 중..."
        from modules.rag import list_collections, _get_shared_embedder
        cols = list_collections()
        logger.info(f"[WarmUp] 저장된 컬렉션 {len(cols)}개: {cols or '(없음)'}")
        if cols:
            _warmup_state["message"] = "임베딩 모델 로드 중..."
            _get_shared_embedder()
            logger.info("[WarmUp] 임베더 로드 완료 — 첫 질문 즉시 처리 가능")
    except Exception as e:
        logger.warning(f"[WarmUp] 초기화 중 오류 (무시): {e}")
    finally:
        _warmup_state["ready"] = True
        _warmup_state["message"] = "준비 완료"


# ── 작업 / RAG 캐시 ──────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_JOBS_MAX_COMPLETED = 20          # 완료된 작업 최대 보관 수
_JOBS_MAX_LOG_LINES = 2_000       # 작업당 최대 로그 줄 수

# LRU 캐시 (OrderedDict): 최대 5개 MultiRAGChatbot 인스턴스 유지
_rag_bots: OrderedDict[str, object] = OrderedDict()
_rag_bots_lock = threading.Lock()
_RAG_BOTS_MAX = 5


def _get_or_create_rag(cols: list[str]):
    """MultiRAGChatbot LRU 캐시. 초과 시 가장 오래된 항목 해제."""
    from modules.rag import MultiRAGChatbot
    key = ",".join(sorted(cols))
    with _rag_bots_lock:
        if key in _rag_bots:
            _rag_bots.move_to_end(key)
            return _rag_bots[key]

        if len(_rag_bots) >= _RAG_BOTS_MAX:
            _, oldest = _rag_bots.popitem(last=False)
            try:
                oldest.unload()
            except Exception:
                pass

        bot = MultiRAGChatbot(collection_names=cols)
        _rag_bots[key] = bot
        return bot


def _evict_rag_key(stem: str):
    """컬렉션 삭제 시 해당 키를 포함하는 캐시 항목 무효화."""
    with _rag_bots_lock:
        stale = [k for k in list(_rag_bots) if stem in k.split(",")]
        for k in stale:
            try:
                _rag_bots[k].unload()
            except Exception:
                pass
            del _rag_bots[k]


def _jobs_gc():
    """완료된 오래된 작업 정리."""
    with _jobs_lock:
        done = [jid for jid, j in _jobs.items() if j["status"] != "running"]
        for jid in done[:-_JOBS_MAX_COMPLETED]:
            del _jobs[jid]


# ────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ────────────────────────────────────────────────────────────────────────────
class YoutubeRequest(BaseModel):
    url: str
    target_lang: str = "ko"
    skip_tts: bool = False


class RunPipelineRequest(BaseModel):
    file: str          # 파일 경로 (video) 또는 문서 텍스트 (document)
    file_type: str = "video"   # "video" | "document"
    lang: Optional[str] = None
    target_lang: str = "ko"
    force_translate: bool = False
    skip_tts: bool = False


class ChatRequest(BaseModel):
    question: str
    collections: list[str] = []
    threshold: float = 0.6
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]


class BookmarkRequest(BaseModel):
    question: str
    answer: str
    chunks: list[dict] = []
    collections: list[str] = []


# ────────────────────────────────────────────────────────────────────────────
# HTML Pages
# ────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 페이지"""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>edu_pipeline</h1><p>templates/index.html을 찾을 수 없습니다.</p>"


@app.get("/chat/{collection}", response_class=HTMLResponse)
async def chat_page(collection: str):
    """RAG 챗봇 페이지"""
    html_path = TEMPLATES_DIR / "chat.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return f"<h1>RAG Chatbot: {collection}</h1><p>templates/chat.html을 찾을 수 없습니다.</p>"


# ────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ────────────────────────────────────────────────────────────────────────────
@app.get("/api/input-files")
async def list_input_files():
    """data/input 폴더의 영상 파일 목록"""
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
    files = []
    if INPUT_DIR.exists():
        for f in sorted(INPUT_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in video_exts:
                files.append({"name": f.name, "path": str(f)})
    return files


@app.get("/api/collections")
async def get_collections():
    """RAG에 인덱싱된 컬렉션 + 출력 폴더에 존재하는 소스 병합 목록 반환"""
    # 1) ChromaDB 인덱싱된 컬렉션
    try:
        from modules.rag import list_collections
        indexed = set(list_collections())
    except Exception:
        indexed = set()

    # 2) data/output 폴더 스캔 (segments.json 또는 source_info.json 이 있는 디렉터리)
    disk = set()
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if (d / "segments.json").exists() or (d / "source_info.json").exists():
                disk.add(d.name)

    # 인덱싱된 것 우선, 그 다음 디스크 전용 항목 추가
    merged = list(indexed) + sorted(disk - indexed)
    return merged


@app.get("/api/ready")
async def get_ready_state():
    """AI 초기화 상태 반환"""
    return _warmup_state


@app.get("/api/output-files")
async def list_output_files():
    """data/output 폴더의 결과물 목록"""
    results = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir():
                files = [f.name for f in sorted(d.iterdir()) if f.is_file() and not f.name.startswith("_")]
                results.append({"name": d.name, "files": files})
    return results


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """파일 업로드 (영상 또는 문서 또는 실험 데이터)"""
    from fastapi import HTTPException
    # 확장자 검증
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"허용되지 않는 파일 형식: {suffix}")
    # 파일명 정제
    safe_name = Path(file.filename).name
    import re
    safe_name = re.sub(r"[^\w\-. ]", "_", safe_name).strip(". ") or "upload"
    safe_name = safe_name if safe_name.endswith(suffix) else (safe_name + suffix)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = INPUT_DIR / safe_name

    # 청크 단위 스트리밍 쓰기 (2GB 파일의 RAM 폭발 방지)
    CHUNK = 1024 * 1024  # 1 MiB
    total_written = 0
    try:
        with open(file_path, "wb") as out_f:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > MAX_UPLOAD_BYTES:
                    out_f.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413, detail="파일 크기가 2GB를 초과합니다."
                    )
                out_f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")
    return {"filename": safe_name, "path": str(file_path)}


@app.post("/api/youtube")
async def youtube_pipeline(req: YoutubeRequest, background_tasks: BackgroundTasks):
    """유튜브 URL에서 영상 다운로드 후 파이프라인 실행"""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "logs": [], "process": None}
    def _wrap():
        try:
            _run_youtube_job(job_id, req)
        except Exception as e:
            j = _jobs.get(job_id)
            if j is not None:
                j["logs"].append(f"\n[오류] {e}\n")
                j["status"] = "error"
    threading.Thread(target=_wrap, daemon=True).start()
    _jobs_gc()
    return {"jobId": job_id}


def _run_youtube_job(job_id: str, req: YoutubeRequest):
    """유튜브 다운로드 + 파이프라인 실행"""
    job = _jobs[job_id]

    def _log(msg: str):
        msg = msg.rstrip("\n")
        if not msg:
            return
        job["logs"].append(msg + "\n")
        if len(job["logs"]) > _JOBS_MAX_LOG_LINES:
            job["logs"] = job["logs"][-_JOBS_MAX_LOG_LINES:]

    # yt-dlp 로그를 job logs로 리디렉션
    class _YdlLogger:
        def debug(self, msg):
            if msg.startswith("[debug]"):
                return  # 상세 디버그는 생략
            _log(f"[yt-dlp] {msg}")
        def info(self, msg):
            _log(f"[INFO] {msg}")
        def warning(self, msg):
            _log(f"[경고] {msg}")
        def error(self, msg):
            _log(f"[오류] {msg}")

    def _progress_hook(d):
        if d["status"] == "downloading":
            pct   = d.get("_percent_str", "").strip()
            speed = d.get("_speed_str", "").strip()
            eta   = d.get("_eta_str", "").strip()
            if pct:
                _log(f"[INFO] 다운로드 {pct}  {speed}  남은시간 {eta}")
        elif d["status"] == "finished":
            _log(f"[INFO] 다운로드 완료: {d.get('filename', '')}")

    try:
        import yt_dlp, re, shutil as _shutil
        _log(f"[INFO] 유튜브 영상 정보 가져오는 중: {req.url}")

        # Node.js 설치 여부 확인 → js_runtimes 설정 (yt-dlp 경고 제거)
        _node_path = _shutil.which("node") or _shutil.which("nodejs")
        _js_runtime_opt: dict = {}
        if _node_path:
            _js_runtime_opt["js_runtimes"] = {"nodejs": {}}
            _log(f"[INFO] Node.js 감지됨: {_node_path}")

        info_opts: dict = {
            "logger": _YdlLogger(),
            "quiet": False,
            **_js_runtime_opt,
        }

        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(req.url, download=False)

        if not info:
            raise RuntimeError("영상 정보를 가져올 수 없습니다. URL을 확인하거나 잠시 후 다시 시도해 주세요.")

        title = info.get("title", "youtube_video")
        safe  = re.sub(r'[\\/*?:"<>|]', "", title).strip()[:60] or "youtube_video"
        _log(f"[INFO] 영상 제목: {title}")

        out_path = INPUT_DIR / f"{safe}.mp4"

        ydl_opts: dict = {
            "format":              "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl":             str(out_path),
            "merge_output_format": "mp4",
            "logger":              _YdlLogger(),
            "quiet":               False,
            "progress_hooks":      [_progress_hook],
            **_js_runtime_opt,
        }

        _log(f"[INFO] 다운로드 시작...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([req.url])

        if not out_path.exists():
            raise RuntimeError(f"다운로드된 파일을 찾을 수 없습니다: {out_path}")

        _log(f"[INFO] 다운로드 완료 → {out_path.name}")

        cmd = [
            str(VENV_PYTHON), "main.py",
            "-i", str(out_path),
            "--target-lang", req.target_lang,
            "--skip-tts",
        ]
        _run_subprocess(job, cmd)

    except Exception as e:
        job["logs"].append(f"\n[오류] {e}\n")
        job["status"] = "error"
    finally:
        job["process"] = None


@app.post("/api/run")
async def run_pipeline(req: RunPipelineRequest, background_tasks: BackgroundTasks):
    """파이프라인 실행"""
    job_id = str(uuid.uuid4())[:8]
    cmd = [str(VENV_PYTHON), "main.py"]

    if req.file_type == "video":
        if not Path(req.file).is_file():
            return {"error": "파일을 찾을 수 없습니다"}
        cmd += ["-i", req.file]
        tmp_doc_path = None
    else:
        # 문서 텍스트를 임시 파일에 저장 후 --document-file 로 전달
        # (CLI arg 길이 제한 및 인코딩 문제 방지)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8",
            delete=False, dir=str(BASE_DIR / "data")
        )
        tmp.write(req.file)
        tmp.close()
        tmp_doc_path = tmp.name
        cmd += ["--document-file", tmp_doc_path]

    if req.lang:
        cmd += ["--lang", req.lang]
    cmd += ["--target-lang", req.target_lang]
    if req.force_translate:
        cmd.append("--force-translate")
    cmd.append("--skip-tts")

    _jobs[job_id] = {"status": "running", "logs": [], "process": None,
                     "_tmp": tmp_doc_path}
    background_tasks.add_task(_run_job, job_id, cmd)
    _jobs_gc()
    return {"jobId": job_id}


def _run_subprocess(job: dict, cmd: list[str]):
    """공통 subprocess 실행 로직."""
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(BASE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        job["process"] = proc
        for line in proc.stdout:
            job["logs"].append(line)
            if len(job["logs"]) > _JOBS_MAX_LOG_LINES:
                job["logs"] = job["logs"][-_JOBS_MAX_LOG_LINES:]
        proc.wait()
        job["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as e:
        job["logs"].append(f"\n[오류] {e}\n")
        job["status"] = "error"
    finally:
        job["process"] = None


def _run_job(job_id: str, cmd: list[str]):
    """파이프라인 작업 실행 (스레드)"""
    job = _jobs[job_id]
    _run_subprocess(job, cmd)

    # 임시 문서 파일 정리
    tmp = job.pop("_tmp", None)
    if tmp:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass



@app.get("/api/logs/{job_id}")
async def stream_logs(job_id: str):
    """SSE로 실시간 로그 스트리밍"""
    async def generate():
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

            await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/stop/{job_id}")
async def stop_job(job_id: str):
    """작업 중지"""
    job = _jobs.get(job_id)
    if job and job.get("process"):
        job["process"].terminate()
        job["status"] = "stopped"
        job["logs"].append("\n[중지됨] 사용자에 의해 중지되었습니다.\n")
        return {"ok": True}
    return {"error": "실행 중인 작업이 없습니다"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """멀티 컬렉션 RAG 질의응답"""
    from modules.rag import list_collections

    cols = req.collections if req.collections else list_collections()
    if not cols:
        return {"answer": "아직 처리된 파일이 없습니다. 파일을 먼저 업로드해 주세요.", "chunks": []}

    rag = _get_or_create_rag(cols)
    history = req.history or None
    answer, chunks = rag.query(req.question, threshold=req.threshold, history=history)
    return {"answer": answer, "chunks": chunks}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """스트리밍 RAG 응답 (SSE)"""
    import queue
    from modules.rag import list_collections

    cols = req.collections if req.collections else list_collections()
    if not cols:
        async def no_cols():
            yield f"data: {json.dumps({'token': '아직 처리된 파일이 없습니다.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(no_cols(), media_type="text/event-stream")

    rag = _get_or_create_rag(cols)
    question  = req.question
    threshold = req.threshold
    history   = req.history or None
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            for kind, val in rag.query_stream(question, threshold=threshold, history=history):
                q.put((kind, val))
            q.put(("done", None))
        except Exception as e:
            q.put(("error", str(e)))

    threading.Thread(target=_run, daemon=True).start()

    async def generate():
        loop = asyncio.get_event_loop()
        while True:
            try:
                kind, val = await loop.run_in_executor(None, q.get, True, 600.0)
            except Exception:
                yield f"data: {json.dumps({'error': 'timeout'})}\n\n"
                break
            if kind == "status":
                yield f"data: {json.dumps({'status': val})}\n\n"
            elif kind == "chunks":
                yield f"data: {json.dumps({'chunks': val})}\n\n"
            elif kind == "token":
                yield f"data: {json.dumps({'token': val})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            else:
                yield f"data: {json.dumps({'error': val})}\n\n"
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/collection/{stem}")
async def delete_collection(stem: str):
    """ChromaDB 컬렉션 삭제"""
    from fastapi import HTTPException
    from modules.rag import _get_chroma_client, _sanitize_name
    from config import cfg
    try:
        safe = _safe_stem(stem)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        client = _get_chroma_client()
        col_name = _sanitize_name(f"{cfg.rag.collection_prefix}{safe}")
        client.delete_collection(col_name)
        _evict_rag_key(safe)
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/source-info/{stem}")
async def get_source_info(stem: str):
    """AI가 생성한 소스 제목·요약 반환. summary가 비어있으면 segments.json 앞부분으로 보완."""
    from fastapi import HTTPException
    try:
        safe = _safe_stem(stem)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out_dir = _resolve_output_dir(safe)
    info_path = out_dir / "source_info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    except Exception:
        info = {}
    if not info.get("title"):
        info["title"] = safe
    # summary가 비어있으면 segments.json 앞 6개 문장을 미리보기로 대체
    if not info.get("summary"):
        seg_path = out_dir / "segments.json"
        if seg_path.exists():
            try:
                segs = json.loads(seg_path.read_text(encoding="utf-8"))
                preview = " ".join(
                    s.get("translated", s.get("text", ""))
                    for s in segs[:6]
                ).strip()
                info["summary"] = preview[:400] if preview else ""
            except Exception:
                pass
    return info


@app.get("/api/output-info/{stem}")
async def get_output_info(stem: str):
    """특정 컬렉션의 결과 파일 목록"""
    from fastapi import HTTPException
    try:
        safe = _safe_stem(stem)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out_dir = _resolve_output_dir(safe)
    if not out_dir.exists():
        return []
    return [
        f.name for f in sorted(out_dir.iterdir())
        if f.is_file() and not f.name.startswith("_")
    ]


@app.get("/api/check-duplicate/{stem}")
async def check_duplicate(stem: str):
    """새 컬렉션과 기존 컬렉션 간 중복 콘텐츠 감지"""
    from fastapi import HTTPException
    from modules.rag import _get_chroma_client, _sanitize_name
    from config import cfg
    try:
        stem = _safe_stem(stem)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        client   = _get_chroma_client()
        prefix   = cfg.rag.collection_prefix
        new_name = _sanitize_name(f"{prefix}{stem}")
        # ChromaDB 1.x: list_collections() → str list
        # ChromaDB 0.5.x: list_collections() → Collection object list
        _raw = client.list_collections()
        col_names = [c if isinstance(c, str) else c.name for c in _raw]
        if new_name not in col_names:
            return {"duplicates": []}

        new_col = client.get_collection(new_name)
        count   = new_col.count()
        if count == 0:
            return {"duplicates": []}

        # 최대 12개 샘플 벡터 추출
        sample_size = min(12, count)
        got = new_col.get(limit=sample_size, include=["embeddings"])
        sample_vecs = got.get("embeddings") or []
        if not sample_vecs:
            return {"duplicates": []}

        duplicates = []
        for col_name_item in col_names:
            if col_name_item == new_name:
                continue
            try:
                other = client.get_collection(col_name_item)
                if other.count() == 0:
                    continue
                hits  = other.query(
                    query_embeddings=sample_vecs,
                    n_results=1,
                    include=["distances"],
                )
                dists = [d[0] for d in hits["distances"] if d]
                if not dists:
                    continue
                avg_dist   = sum(dists) / len(dists)
                similarity = round((1 - avg_dist) * 100)
                if similarity >= 60:   # 60% 이상 유사 → 경고
                    other_stem = col_name_item[len(prefix):] if col_name_item.startswith(prefix) else col_name_item
                    duplicates.append({"stem": other_stem, "similarity": similarity})
            except Exception:
                continue

        duplicates.sort(key=lambda x: -x["similarity"])
        return {"duplicates": duplicates}
    except Exception as e:
        return {"duplicates": [], "error": str(e)}


@app.get("/api/content-preview/{stem}")
async def content_preview(stem: str):
    """segments.json 앞부분으로 빠른 콘텐츠 미리보기 반환"""
    from fastapi import HTTPException
    try:
        stem = _safe_stem(stem)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out_dir = _resolve_output_dir(stem)
    seg_path = out_dir / "segments.json"
    if not seg_path.exists():
        return {"preview": "", "count": 0, "duration": 0}
    try:
        segs     = json.loads(seg_path.read_text(encoding="utf-8"))
        count    = len(segs)
        duration = segs[-1]["end"] if segs else 0
        preview  = " ".join(
            s.get("translated", s.get("text", ""))
            for s in segs[:6]
        )[:600]
        return {"preview": preview, "count": count, "duration": round(duration)}
    except Exception:
        return {"preview": "", "count": 0, "duration": 0}


@app.post("/api/chat/{collection}")
async def chat_single(collection: str, req: ChatRequest):
    """단일 컬렉션 RAG (하위 호환)"""
    rag = _get_or_create_rag([collection])
    history = req.history or None
    answer, chunks = rag.query(req.question, threshold=req.threshold, history=history)
    return {"answer": answer, "chunks": chunks}


# ────────────────────────────────────────────────────────────────────────────
# 북마크 API
# ────────────────────────────────────────────────────────────────────────────
def _load_bookmarks() -> list[dict]:
    if not BOOKMARKS_PATH.exists():
        return []
    try:
        return json.loads(BOOKMARKS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_bookmarks(bms: list[dict]):
    BOOKMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOKMARKS_PATH.write_text(json.dumps(bms, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/bookmarks")
async def get_bookmarks():
    with _bookmarks_lock:
        return _load_bookmarks()


@app.post("/api/bookmarks")
async def add_bookmark(req: BookmarkRequest):
    with _bookmarks_lock:
        bms = _load_bookmarks()
        bm = {
            "id":          str(uuid.uuid4())[:8],
            "question":    req.question,
            "answer":      req.answer,
            "chunks":      req.chunks,
            "collections": req.collections,
            "created_at":  time.strftime("%Y-%m-%d %H:%M"),
        }
        bms.insert(0, bm)
        _save_bookmarks(bms)
    return bm


@app.delete("/api/bookmarks/{bookmark_id}")
async def delete_bookmark(bookmark_id: str):
    with _bookmarks_lock:
        bms = _load_bookmarks()
        bms = [b for b in bms if b.get("id") != bookmark_id]
        _save_bookmarks(bms)
    return {"ok": True}


# ────────────────────────────────────────────────────────────────────────────
# 재인덱싱 API (segments.json → ChromaDB)
# ────────────────────────────────────────────────────────────────────────────
@app.post("/api/reindex/{stem}")
async def reindex(stem: str, background_tasks: BackgroundTasks):
    """디스크의 segments.json을 ChromaDB에 재인덱싱합니다 (파이프라인 재실행 불필요)."""
    from fastapi import HTTPException
    try:
        safe = _safe_stem(stem)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    background_tasks.add_task(_reindex_bg, safe)
    return {"ok": True, "message": f"재인덱싱 시작: {safe}"}


@app.post("/api/reindex-all")
async def reindex_all(background_tasks: BackgroundTasks):
    """data/output 폴더의 모든 segments.json을 재인덱싱합니다."""
    stems = []
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("_"):
                if (d / "segments.json").exists():
                    stems.append(d.name)
    for s in stems:
        background_tasks.add_task(_reindex_bg, s)
    return {"ok": True, "message": f"{len(stems)}개 재인덱싱 시작", "stems": stems}


def _reindex_bg(stem: str):
    """segments.json → ChromaDB 재인덱싱 (백그라운드)."""
    try:
        out_dir = _resolve_output_dir(stem)
        seg_path = out_dir / "segments.json"
        if not seg_path.exists():
            logger.warning(f"[Reindex] segments.json 없음: {out_dir}")
            return
        segs = json.loads(seg_path.read_text(encoding="utf-8"))
        if not segs:
            logger.warning(f"[Reindex] 세그먼트 0개: {stem}")
            return
        logger.info(f"[Reindex] {stem}: {len(segs)}개 세그먼트 인덱싱 시작")
        from modules.rag import RAGChatbot
        bot = RAGChatbot(collection_name=stem)
        bot.index_segments(segs)
        logger.info(f"[Reindex] {stem}: 완료  총 {bot._collection.count()}개")
        _evict_rag_key(stem)
    except Exception as e:
        logger.error(f"[Reindex] {stem} 실패: {e}", exc_info=True)


# ────────────────────────────────────────────────────────────────────────────
# 요약 인덱싱 API
# ────────────────────────────────────────────────────────────────────────────
@app.post("/api/categorize-all")
async def categorize_all_sources(background_tasks: BackgroundTasks):
    """topic이 없는 모든 소스를 LLM으로 분류 (백그라운드 실행)"""
    background_tasks.add_task(_run_categorize_all)
    return {"ok": True, "message": "분류 작업이 시작되었습니다."}


_categorize_progress: dict = {"running": False, "done": 0, "total": 0, "current": ""}


@app.post("/api/reclassify-all")
async def reclassify_all(background_tasks: BackgroundTasks):
    """모든 소스를 강제 재분류 (topic/topics 덮어쓰기)."""
    if _categorize_progress["running"]:
        return {"ok": False, "message": "이미 진행 중"}
    background_tasks.add_task(_run_categorize_all, True)
    return {"ok": True, "message": "전체 재분류 시작됨"}


@app.get("/api/categorize-progress")
async def categorize_progress():
    p = _categorize_progress
    pct = int(p["done"] / p["total"] * 100) if p["total"] else 0
    return {**p, "percent": pct}


@app.get("/api/topics")
async def list_topics():
    """모든 소스의 topics 태그 집계 → {tag: [stem, ...]}"""
    import json as _json
    result: dict[str, list[str]] = {}
    if not OUTPUT_DIR.exists():
        return result
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        info_path = d / "source_info.json"
        if not info_path.exists():
            continue
        try:
            info = _json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tags = info.get("topics") or ([info["topic"]] if info.get("topic") else ["미분류"])
        for t in tags:
            result.setdefault(t, []).append(d.name)
    return result


def _run_categorize_all(force: bool = False):
    """모든 소스의 source_info.json을 읽어 topic/topics 생성.

    force=False: topic 없는 항목만
    force=True : 모든 항목 강제 재분류 (topics 덮어쓰기)
    """
    import json as _json
    if not OUTPUT_DIR.exists():
        return

    targets: list[tuple[Path, dict]] = []
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        info_path = d / "source_info.json"
        seg_path  = d / "segments.json"
        if not seg_path.exists():
            continue
        info: dict = {}
        if info_path.exists():
            try:
                info = _json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not force and info.get("topics"):
            continue  # 이미 다중 태그 분류됨
        targets.append((d, info))

    if not targets:
        return

    _categorize_progress.update(running=True, done=0, total=len(targets), current="")
    logger.info(f"[Categorize] {len(targets)}개 분류 시작 (force={force})")
    from modules.llm import LLMProcessor
    llm = LLMProcessor()

    try:
        for out_dir, info in targets:
            _categorize_progress["current"] = out_dir.name
            seg_path  = out_dir / "segments.json"
            info_path = out_dir / "source_info.json"
            try:
                segs = _json.loads(seg_path.read_text(encoding="utf-8"))
                text = " ".join(
                    s.get("translated", s.get("text", ""))
                    for s in segs[:8]
                ).strip()
                if not text:
                    continue
                topics = llm.generate_topics(text)
                # 제목·요약도 없으면 함께 생성
                if not info.get("title") or info["title"] == out_dir.name:
                    info["title"] = llm.generate_title(text)
                if not info.get("summary"):
                    info["summary"] = llm.generate_summary(text)
                info["topic"]  = topics[0] if topics else "기타"
                info["topics"] = topics
                info.setdefault("output_stem", out_dir.name)
                with open(str(info_path), "w", encoding="utf-8") as f:
                    _json.dump(info, f, ensure_ascii=False, indent=2)
                logger.info(f"[Categorize] {out_dir.name} → {topics}")
            except Exception as e:
                logger.warning(f"[Categorize] {out_dir.name} 실패: {e}")
            finally:
                _categorize_progress["done"] += 1
    finally:
        llm.unload()
        _categorize_progress.update(running=False, current="")
    logger.info("[Categorize] 전체 분류 완료")


@app.get("/api/output/{stem}/{filename}")
async def get_output_file(stem: str, filename: str):
    """결과 파일 다운로드"""
    from fastapi import HTTPException
    try:
        safe = _safe_stem(stem)
        out_dir = _resolve_output_dir(safe)
        filename_safe = Path(filename).name
        file_path = (out_dir / filename_safe).resolve()
        # 경로 탈출 방지: OUTPUT_DIR 내부인지 확인
        if not str(file_path).startswith(str(OUTPUT_DIR.resolve())):
            raise ValueError(f"경로 접근 거부: {file_path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    return FileResponse(file_path)


# ────────────────────────────────────────────────────────────────────────────
# Startup
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    import webbrowser

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    port = 8000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                break
            port += 1

    logger.info("=" * 60)
    logger.info("  edu_pipeline Web UI (FastAPI)")
    logger.info(f"  http://localhost:{port}")
    logger.info("=" * 60)

    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
