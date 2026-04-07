#!/usr/bin/env python3
"""
edu_pipeline/main.py
────────────────────────────────────────────────────────────────────
교육용 영상 자동 처리 파이프라인 진입점.

파이프라인 순서:
  1. 오디오 추출 (ffmpeg)
  2. STT — Whisper large-v3-turbo  ▶ VRAM 해제
  3. LLM — 번역 + 연령별 요약      ▶ VRAM 해제
  4. TTS — AI 더빙 (F5-TTS)       ▶ VRAM 해제
  5. 임베딩 — Qwen3-Embedding-0.6B ▶ VRAM 해제
  6. 인덱싱 — ChromaDB 저장
  7. (선택) RAG 챗봇 실행

사용 예시:
  # 기본 실행 (한국어 → 한국어, 성인 요약)
  python main.py -i lecture.mp4

  # 영어 강의 → 한국어 자막 + 더빙 + 챗봇
  python main.py -i lecture.mp4 --lang en --target-lang ko --chat

  # TTS 스킵하고 자막만 생성
  python main.py -i lecture.mp4 --skip-tts

  # 화자 클로닝 적용
  python main.py -i lecture.mp4 --ref-audio speaker.wav --ref-text "안녕하세요, 반갑습니다."

  # 이미 인덱싱된 영상으로 챗봇만 실행
  python main.py --chat-only --collection lecture
"""

import argparse
import faulthandler
import json
import logging
import os
import sys
from pathlib import Path

# Windows cp949 환경에서 UTF-8 로그 출력 보장
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 모델 캐시가 있으면 오프라인으로 실행, 없으면 자동 다운로드
import huggingface_hub as _hf_hub
from pathlib import Path as _Path

def _all_models_cached() -> bool:
    """주요 모델 세 가지(LLM GGUF, Whisper, Embedder)가 모두 캐시에 있는지 확인"""
    import os
    from config import cfg
    # huggingface_hub 버전에 따라 constants 모듈이 없을 수 있으므로 env var 직접 사용
    hf_cache_env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_cache_env:
        cache = _Path(hf_cache_env)
    else:
        try:
            cache = _Path(_hf_hub.constants.HF_HUB_CACHE)
        except AttributeError:
            cache = _Path.home() / ".cache" / "huggingface" / "hub"
    if not cache.exists():
        return False
    # LLM: GGUF 파일 직접 검색
    gguf_found = any(cache.rglob(cfg.llm.gguf_file))
    # Whisper: faster-whisper는 HF Hub 네이밍 사용 (turbo → Systran/faster-whisper-turbo)
    whisper_dir = cache / f"models--Systran--faster-whisper-{cfg.stt.model_size}"
    whisper_found = whisper_dir.exists()
    # Embedder: Qwen/Qwen3-Embedding-0.6B
    embed_repo = cfg.embed.model_id.replace("/", "--")
    embedder_dir = cache / f"models--{embed_repo}"
    embedder_found = embedder_dir.exists()
    return gguf_found and whisper_found and embedder_found

if _all_models_cached():
    os.environ["HF_HUB_OFFLINE"]        = "1"
    os.environ["TRANSFORMERS_OFFLINE"]  = "1"
else:
    # 최초 실행 또는 캐시 없음 → 다운로드 허용
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
# segfault 시 traceback 출력
faulthandler.enable(file=sys.stderr, all_threads=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── SRT 파서 (--start-from 재개용) ────────────────────────────────────────────
def _parse_srt(srt_path: str) -> list[dict]:
    """SRT 파일을 세그먼트 리스트로 변환합니다."""
    import re
    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        ts_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            lines[1],
        )
        if not ts_match:
            continue
        g = [int(x) for x in ts_match.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end   = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text  = " ".join(lines[2:]).strip()
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return segments


# ── 문서 파서 (텍스트 → 세그먼트) ────────────────────────────────────────────────
def _parse_document(text: str, max_length: int = 300) -> list[dict]:
    """문서 텍스트를 문장 단위로 세그먼트로 변환합니다."""
    import re

    # 문장 단위로 분할 (마침표, 느낌표, 물음표 기준)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    segments = []
    current_time = 0.0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 단어 경계 기준으로 max_length 글자 이하 청크로 쪼개기
        words = sentence.split()
        chunk_words: list[str] = []
        chunk_len = 0
        for word in words:
            # 단어 추가 시 max_length 초과하면 현재 청크 확정
            if chunk_words and chunk_len + 1 + len(word) > max_length:
                chunk = " ".join(chunk_words)
                duration = len(chunk) / 35.0
                segments.append({
                    "start": round(current_time, 3),
                    "end":   round(current_time + duration, 3),
                    "text":  chunk,
                })
                current_time += duration
                chunk_words = []
                chunk_len   = 0
            chunk_words.append(word)
            chunk_len += (1 if chunk_len else 0) + len(word)

        # 마지막 남은 단어들 처리
        if chunk_words:
            chunk = " ".join(chunk_words)
            duration = len(chunk) / 35.0
            segments.append({
                "start": round(current_time, 3),
                "end":   round(current_time + duration, 3),
                "text":  chunk,
            })
            current_time += duration

    return segments


# ── 파이프라인 ────────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace) -> None:
    from config import cfg, MODE
    from utils.audio import extract_audio, get_video_duration, mix_dubbed_into_video, export_srt, check_ffmpeg
    from utils.timestamp_sync import merge_dubbed_audio
    from utils.memory import clear_vram

    check_ffmpeg()

    start_from = args.start_from

    logger.info(f"{'=' * 55}")
    logger.info(f"  edu_pipeline 시작  (mode={MODE})")

    # --document-file 우선, 없으면 --document 인라인 텍스트 사용
    doc_text: str | None = None
    if getattr(args, "document_file", None):
        doc_path = Path(args.document_file)
        if not doc_path.is_file():
            logger.error(f"문서 파일이 존재하지 않습니다: {args.document_file}")
            sys.exit(1)
        doc_text = doc_path.read_text(encoding="utf-8")
    elif args.document:
        doc_text = args.document

    # 실험 데이터 모드 (CSV / Excel / JSON)
    DATA_EXTS = {".csv", ".xlsx", ".xls", ".json"}
    is_data = (
        not doc_text
        and args.input
        and Path(args.input).suffix.lower() in DATA_EXTS
    )

    # 문서 모드 vs 실험 데이터 모드 vs 영상 모드
    is_document = doc_text is not None
    if is_document:
        logger.info(f"  입력: 문서 텍스트 ({len(doc_text)} 글자)")
        stem = "document"
    elif is_data:
        logger.info(f"  입력: 실험 데이터 파일 {args.input}")
        data_path = args.input
        if not Path(data_path).is_file():
            logger.error(f"입력 파일이 존재하지 않습니다: {data_path}")
            sys.exit(1)
        stem = Path(data_path).stem
    else:
        logger.info(f"  입력 파일: {args.input}")
        video_path = args.input
        if not Path(video_path).is_file():
            logger.error(f"입력 파일이 존재하지 않습니다: {video_path}")
            sys.exit(1)
        stem = Path(video_path).stem

    if start_from > 1:
        logger.info(f"  STEP {start_from}부터 재개")
    logger.info(f"{'=' * 55}")

    out_dir = Path(cfg.output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: 오디오 추출 / 문서 파싱 / 실험 데이터 파싱
    # ──────────────────────────────────────────────────────────────────────────
    if is_document:
        logger.info("━━━ STEP 1: 문서 파싱 ━━━")
        segments = _parse_document(doc_text)
        logger.info(f"문서: {len(segments)}개 세그먼트 생성")
    elif is_data:
        logger.info("━━━ STEP 1: 실험 데이터 파싱 ━━━")
        from utils.data_parser import parse_data_file
        segments, data_info = parse_data_file(data_path)
        logger.info(f"실험 데이터: {len(segments)}개 세그먼트 생성  "
                    f"({data_info['shape']['rows']}행 × {data_info['shape']['cols']}열)")
        # source_info 저장 (통계 + output_stem 포함)
        import json as _json
        data_info["output_stem"] = stem
        with open(str(out_dir / "source_info.json"), "w", encoding="utf-8") as _f:
            _json.dump(data_info, _f, ensure_ascii=False, indent=2)
        # 실험 데이터는 번역/TTS 불필요 → STEP 3(LLM번역) 이후로 바로 임베딩
        segments = [{**s, "translated": s["text"]} for s in segments]
        logger.info("━━━ STEP 2~4 건너뜀 (실험 데이터 모드) ━━━")
        # STEP 5~6 진행
        logger.info("━━━ STEP 5: 임베딩 ━━━")
        from modules.embedder import EmbeddingProcessor
        from utils.memory import clear_vram
        embedder = EmbeddingProcessor()
        segments = embedder.embed_segments(segments)
        embedder.unload(); clear_vram()
        logger.info("━━━ STEP 6: 벡터 DB 인덱싱 ━━━")
        from modules.rag import RAGChatbot
        rag = RAGChatbot(collection_name=stem)
        rag.index_segments(segments)
        manifest = str(out_dir / "segments.json")
        with open(manifest, "w", encoding="utf-8") as f:
            import json as _j
            _j.dump([{k: v for k, v in s.items() if k != "embedding"} for s in segments],
                    f, ensure_ascii=False, indent=2)
        logger.info(f"{'=' * 55}")
        logger.info("  파이프라인 완료!")
        logger.info(f"  출력 디렉토리: {out_dir}")
        logger.info(f"{'=' * 55}")
        return
    else:
        audio_path = str(out_dir / "extracted_audio.wav")
        if start_from <= 1:
            logger.info("━━━ STEP 1: 오디오 추출 ━━━")
            extract_audio(video_path, audio_path, sample_rate=16_000)

        duration = get_video_duration(video_path)
        logger.info(f"영상 길이: {duration:.1f}초")

        # ──────────────────────────────────────────────────────────────────────────
        # STEP 2: STT
        # ──────────────────────────────────────────────────────────────────────────
        srt_orig = str(out_dir / "subtitles_original.srt")
        if start_from <= 2:
            logger.info("━━━ STEP 2: STT (자막 추출) ━━━")
            from modules.stt import transcribe
            try:
                segments = transcribe(audio_path, language=args.lang)
                export_srt(segments, srt_orig, use_translated=False)
                logger.info(f"원본 자막 저장: {srt_orig}")
            except FileNotFoundError:
                logger.error(f"[STEP 2] 오디오 파일을 찾을 수 없습니다: {audio_path}")
                sys.exit(1)
            except RuntimeError as e:
                logger.error(f"[STEP 2] STT 실행 실패 (CUDA/모델 문제): {e}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"[STEP 2] STT 알 수 없는 오류: {e}", exc_info=True)
                sys.exit(1)
        else:
            logger.info(f"━━━ STEP 2: STT 건너뜀 — SRT 파일에서 세그먼트 복원 ━━━")
            try:
                segments = _parse_srt(srt_orig)
            except Exception as e:
                logger.error(f"[STEP 2] SRT 파일 복원 실패 ({srt_orig}): {e}")
                sys.exit(1)
            logger.info(f"SRT에서 {len(segments)}개 세그먼트 복원")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: LLM — 번역 + 요약
    # ──────────────────────────────────────────────────────────────────────────
    if start_from <= 3:
        logger.info("━━━ STEP 3: LLM (번역 + 요약) ━━━")
        from modules.llm import LLMProcessor

        try:
            llm = LLMProcessor()
        except Exception as e:
            logger.error(f"[STEP 3] LLM 로드 실패: {e}")
            sys.exit(1)

        need_translate = args.force_translate or (args.lang is not None and args.lang != args.target_lang)
        if need_translate:
            segments = llm.translate_segments(segments, target_lang=args.target_lang)
            srt_trans = str(out_dir / f"subtitles_{args.target_lang}.srt")
            export_srt(segments, srt_trans, use_translated=True)
            logger.info(f"번역 자막 저장: {srt_trans}")
        else:
            segments = [{**s, "translated": s["text"]} for s in segments]

        # ── AI 제목 + 콘텐츠 설명 생성 (항상 실행)
        import json as _json
        source_info_path = str(out_dir / "source_info.json")
        first_text = " ".join(s.get("translated", s["text"]) for s in segments[:8])
        try:
            ai_title = llm.generate_title(first_text)
            if not ai_title or len(ai_title) < 2:
                ai_title = stem
        except Exception:
            ai_title = stem

        try:
            ai_description = llm.generate_summary(first_text)
        except Exception:
            ai_description = ""

        try:
            ai_topic = llm.generate_topic(first_text)
        except Exception:
            ai_topic = "기타"

        try:
            ai_topics = llm.generate_topics(first_text)
        except Exception:
            ai_topics = [ai_topic] if ai_topic else ["기타"]

        source_info: dict = {
            "title": ai_title,
            "summary": ai_description,
            "topic": ai_topic,
            "topics": ai_topics,
            "output_stem": stem,
        }
        try:
            with open(source_info_path, "w", encoding="utf-8") as _f:
                _json.dump(source_info, _f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        llm.unload()
        clear_vram()
    else:
        logger.info("━━━ STEP 3: LLM 건너뜀 ━━━")
        segments = [{**s, "translated": s["text"]} for s in segments]

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: TTS — AI 더빙 (영상 모드에서만 실행)
    # STT/LLM의 CUDA 컨텍스트 오염으로 Qwen3-TTS가 크래시하므로
    # 별도 프로세스에서 실행하여 깨끗한 CUDA 컨텍스트를 보장합니다.
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_tts and not is_document:
        logger.info("━━━ STEP 4: TTS (AI 더빙) ━━━")
        from utils.audio import extract_ref_clip

        ref_audio = args.ref_audio
        ref_text  = args.ref_text

        if not ref_audio:
            ref_clip_path = str(out_dir / "ref_clip.wav")
            result = extract_ref_clip(video_path, ref_clip_path, segments)
            if result:
                ref_audio, ref_text = result
                logger.info(f"[TTS] 자동 화자 추출 성공: {ref_audio}")
            else:
                logger.warning("[TTS] 화자 추출 실패 — 기본 화자 사용")

        # segments를 JSON으로 저장 → subprocess에서 로드
        seg_json = str(out_dir / "_segments_for_tts.json")
        with open(seg_json, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)

        tts_dir = str(out_dir / "tts_segments")

        # 별도 프로세스에서 TTS 실행 (깨끗한 CUDA 컨텍스트)
        logger.info("[TTS] 별도 프로세스로 TTS 실행 중...")
        import subprocess as sp
        tts_cmd = [
            sys.executable, "-m", "utils.tts_worker",
            "--segments-json", seg_json,
            "--tts-dir", tts_dir,
        ]
        if ref_audio:
            tts_cmd += ["--ref-audio", ref_audio]
        if ref_text:
            tts_cmd += ["--ref-text", ref_text]
        tts_cmd += ["--language", args.target_lang]

        # 세그먼트 수 기반 타임아웃: 세그먼트당 최대 10초 + 여유 120초
        tts_timeout = len(segments) * 10 + 120
        try:
            proc = sp.run(tts_cmd, cwd=str(Path(__file__).parent), timeout=tts_timeout)
        except sp.TimeoutExpired:
            logger.error(f"[TTS] 서브프로세스 타임아웃 ({tts_timeout}초)")
            sys.exit(1)
        if proc.returncode != 0:
            logger.error(f"[TTS] 서브프로세스 실패 (exit {proc.returncode})")
            sys.exit(1)

        # subprocess가 저장한 결과 로드
        tts_result_json = str(out_dir / "_segments_tts_result.json")
        with open(tts_result_json, "r", encoding="utf-8") as f:
            segments = json.load(f)
        logger.info(f"[TTS] 서브프로세스 완료 — {sum(1 for s in segments if s.get('tts_path'))}개 성공")

        # 세그먼트를 단일 트랙으로 합치기
        dubbed_audio = str(out_dir / "dubbed_audio.wav")
        merge_dubbed_audio(segments, duration, dubbed_audio)

        # 원본 영상에 더빙 합성
        dubbed_video = str(out_dir / f"{stem}_dubbed.mp4")
        mix_dubbed_into_video(video_path, dubbed_audio, dubbed_video)
        logger.info(f"더빙 영상: {dubbed_video}")
    else:
        logger.info("━━━ STEP 4: TTS 건너뜀 (--skip-tts) ━━━")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5: 임베딩
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("━━━ STEP 5: 임베딩 ━━━")
    from modules.embedder import EmbeddingProcessor

    embedder = EmbeddingProcessor()
    segments = embedder.embed_segments(segments)
    embedder.unload()
    clear_vram()

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6: ChromaDB 인덱싱
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("━━━ STEP 6: 벡터 DB 인덱싱 ━━━")
    from modules.rag import RAGChatbot

    rag = RAGChatbot(collection_name=stem)
    rag.index_segments(segments)

    # 컬렉션명 → output 디렉토리 매핑 저장 (web UI 역조회용)
    _map_path = Path(str(out_dir.parent)) / "_collection_map.json"
    try:
        from modules.rag import _sanitize_name
        from config import cfg as _cfg
        col_stem = _sanitize_name(f"{_cfg.rag.collection_prefix}{stem}")
        try:
            _cmap = json.loads(_map_path.read_text(encoding="utf-8")) if _map_path.exists() else {}
        except Exception:
            _cmap = {}
        _cmap[col_stem] = stem   # sanitized_collection_name → original_stem
        _cmap[stem] = stem       # 직접 매핑도 저장
        _map_path.write_text(json.dumps(_cmap, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # 세그먼트 매니페스트 저장 (임베딩 벡터 제외)
    manifest = str(out_dir / "segments.json")
    with open(manifest, "w", encoding="utf-8") as f:
        safe_segs = [{k: v for k, v in s.items() if k != "embedding"} for s in segments]
        json.dump(safe_segs, f, ensure_ascii=False, indent=2)
    logger.info(f"매니페스트 저장: {manifest}")

    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"{'=' * 55}")
    logger.info("  파이프라인 완료!")
    logger.info(f"  출력 디렉토리: {out_dir}")
    logger.info(f"{'=' * 55}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7 (선택): RAG 챗봇
    # ──────────────────────────────────────────────────────────────────────────
    if args.chat:
        rag.chat()


def run_chat_only(args: argparse.Namespace) -> None:
    """이미 인덱싱된 컬렉션으로 챗봇을 시작합니다."""
    from modules.rag import RAGChatbot
    rag = RAGChatbot(collection_name=args.collection)
    rag.chat()


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edu_pipeline",
        description="교육용 영상 자동 처리 파이프라인 (STT → 번역 → TTS → RAG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 입력
    p.add_argument("-i", "--input",       metavar="VIDEO",  help="입력 영상 파일 경로")
    p.add_argument("--document",         metavar="TEXT",   help="문서 텍스트 직접 입력 (짧은 텍스트)")
    p.add_argument("--document-file",    metavar="PATH",   help="문서 텍스트 파일 경로 (긴 텍스트 권장)")
    p.add_argument("--lang",             default=None,      help="원본 언어 코드 (ko, en, ja 등). 생략 시 자동 감지")
    p.add_argument("--target-lang",      default="ko",     help="번역 목표 언어 (기본: ko)")
    p.add_argument("--force-translate",  action="store_true", help="원본과 목표 언어가 같아도 번역 실행")

    # TTS 옵션
    p.add_argument("--skip-tts",         action="store_true", help="TTS 더빙 단계 건너뜀")
    p.add_argument("--ref-audio",        metavar="WAV",    help="화자 클로닝용 참조 음성 파일")
    p.add_argument("--ref-text",         metavar="TEXT",   help="참조 음성의 텍스트")

    # 요약 옵션

    # 재개 / 디버그
    p.add_argument("--start-from",       type=int, default=1, metavar="STEP",
                   help="지정한 STEP부터 시작 (1=오디오, 2=STT, 3=LLM, 4=TTS, 5=임베딩)")

    # 챗봇
    p.add_argument("--chat",             action="store_true", help="파이프라인 완료 후 RAG 챗봇 실행")
    p.add_argument("--chat-only",        action="store_true", help="파이프라인 없이 챗봇만 실행")
    p.add_argument("--collection",       default="edu_video", help="챗봇 전용 실행 시 ChromaDB 컬렉션 이름")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.chat_only:
        run_chat_only(args)
    elif args.input or args.document or getattr(args, "document_file", None):
        run_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)
