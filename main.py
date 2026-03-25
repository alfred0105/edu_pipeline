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
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 파이프라인 ────────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace) -> None:
    from config import cfg, MODE
    from utils.audio import extract_audio, get_video_duration, mix_dubbed_into_video, export_srt, check_ffmpeg
    from utils.timestamp_sync import merge_dubbed_audio
    from utils.memory import clear_vram

    check_ffmpeg()

    logger.info(f"{'=' * 55}")
    logger.info(f"  edu_pipeline 시작  (mode={MODE})")
    logger.info(f"  입력 파일: {args.input}")
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
    logger.info("━━━ STEP 1: 오디오 추출 ━━━")
    audio_path = str(out_dir / "extracted_audio.wav")
    extract_audio(video_path, audio_path, sample_rate=16_000)
    duration = get_video_duration(video_path)
    logger.info(f"영상 길이: {duration:.1f}초")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: STT
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("━━━ STEP 2: STT (자막 추출) ━━━")
    from modules.stt import transcribe

    segments = transcribe(audio_path, language=args.lang)

    srt_orig = str(out_dir / "subtitles_original.srt")
    export_srt(segments, srt_orig, use_translated=False)
    logger.info(f"원본 자막 저장: {srt_orig}")

    # VRAM은 transcribe() 내부에서 자동 해제됨

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: LLM — 번역 + 요약
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("━━━ STEP 3: LLM (번역 + 요약) ━━━")
    from modules.llm import LLMProcessor

    llm = LLMProcessor()

    # 번역 (언어가 다르거나, 자동감지 모드이거나, 강제 번역 시)
    need_translate = args.force_translate or (args.lang is not None and args.lang != args.target_lang)
    if need_translate:
        segments = llm.translate_segments(segments, target_lang=args.target_lang)
        srt_trans = str(out_dir / f"subtitles_{args.target_lang}.srt")
        export_srt(segments, srt_trans, use_translated=True)
        logger.info(f"번역 자막 저장: {srt_trans}")
    else:
        # 번역 없이 원본을 translated로 복사 (이후 단계 통일을 위해)
        segments = [{**s, "translated": s["text"]} for s in segments]

    # 연령별 요약 (3가지 통합 1회 호출) — context window 초과 방지 위해 최대 12000자
    full_text = " ".join(s["text"] for s in segments)
    MAX_SUMMARY_CHARS = 12000
    if len(full_text) > MAX_SUMMARY_CHARS:
        logger.warning(f"[LLM] 요약 텍스트가 너무 깁니다 ({len(full_text)}자) → {MAX_SUMMARY_CHARS}자로 자릅니다.")
        full_text = full_text[:MAX_SUMMARY_CHARS]

    summaries = llm.summarize_all_ages(full_text)
    for age, summary in summaries.items():
        path = str(out_dir / f"summary_{age}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info(f"요약 ({age}): {path}")

    llm.unload()
    clear_vram()

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: TTS — AI 더빙
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_tts:
        logger.info("━━━ STEP 4: TTS (AI 더빙) ━━━")
        from modules.tts import TTSProcessor
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

        tts = TTSProcessor(
            ref_audio_path=ref_audio,
            ref_text=ref_text,
        )
        tts_dir  = str(out_dir / "tts_segments")
        segments = tts.synthesize_all(segments, tts_dir, use_translated=True)
        tts.unload()
        clear_vram()

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
