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

# 모델이 이미 캐시에 있으면 HuggingFace 네트워크 체크 건너뜀 (로드 3~5초 단축)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
    logger.info(f"  입력 파일: {args.input}")
    if start_from > 1:
        logger.info(f"  STEP {start_from}부터 재개")
    logger.info(f"{'=' * 55}")

    video_path = args.input
    if not Path(video_path).is_file():
        logger.error(f"입력 파일이 존재하지 않습니다: {video_path}")
        sys.exit(1)

    stem       = Path(video_path).stem
    out_dir    = Path(cfg.output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: 오디오 추출
    # ──────────────────────────────────────────────────────────────────────────
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
        segments = transcribe(audio_path, language=args.lang)
        export_srt(segments, srt_orig, use_translated=False)
        logger.info(f"원본 자막 저장: {srt_orig}")
    else:
        logger.info(f"━━━ STEP 2: STT 건너뜀 — SRT 파일에서 세그먼트 복원 ━━━")
        segments = _parse_srt(srt_orig)
        logger.info(f"SRT에서 {len(segments)}개 세그먼트 복원")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: LLM — 번역 + 요약
    # ──────────────────────────────────────────────────────────────────────────
    if start_from <= 3:
        logger.info("━━━ STEP 3: LLM (번역 + 요약) ━━━")
        from modules.llm import LLMProcessor

        llm = LLMProcessor()

        need_translate = args.force_translate or (args.lang is not None and args.lang != args.target_lang)
        if need_translate:
            segments = llm.translate_segments(segments, target_lang=args.target_lang)
            srt_trans = str(out_dir / f"subtitles_{args.target_lang}.srt")
            export_srt(segments, srt_trans, use_translated=True)
            logger.info(f"번역 자막 저장: {srt_trans}")
        else:
            segments = [{**s, "translated": s["text"]} for s in segments]

        full_text = " ".join(s["text"] for s in segments)
        MAX_SUMMARY_CHARS = 5000  # Qwen3-8B: 컨텍스트 여유분 확보
        if len(full_text) > MAX_SUMMARY_CHARS:
            logger.info(f"[LLM] 요약 텍스트 {len(full_text)}자 → {MAX_SUMMARY_CHARS}자로 축소")
            half = MAX_SUMMARY_CHARS // 2
            full_text = full_text[:half] + "\n...\n" + full_text[-half:]

        summaries = llm.summarize_all_ages(full_text)
        for age, summary in summaries.items():
            path = str(out_dir / f"summary_{age}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(summary)
            logger.info(f"요약 ({age}): {path}")

        llm.unload()
        clear_vram()
    else:
        logger.info("━━━ STEP 3: LLM 건너뜀 ━━━")
        segments = [{**s, "translated": s["text"]} for s in segments]

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: TTS — AI 더빙
    # STT/LLM의 CUDA 컨텍스트 오염으로 F5-TTS가 크래시하므로
    # 별도 프로세스에서 실행하여 깨끗한 CUDA 컨텍스트를 보장합니다.
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_tts:
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
    p.add_argument("--lang",             default=None,      help="원본 언어 코드 (ko, en, ja 등). 생략 시 자동 감지")
    p.add_argument("--target-lang",      default="ko",     help="번역 목표 언어 (기본: ko)")
    p.add_argument("--force-translate",  action="store_true", help="원본과 목표 언어가 같아도 번역 실행")

    # TTS 옵션
    p.add_argument("--skip-tts",         action="store_true", help="TTS 더빙 단계 건너뜀")
    p.add_argument("--ref-audio",        metavar="WAV",    help="화자 클로닝용 참조 음성 파일")
    p.add_argument("--ref-text",         metavar="TEXT",   help="참조 음성의 텍스트")

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
    elif args.input:
        run_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)
