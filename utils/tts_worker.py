"""
utils/tts_worker.py
────────────────────────────────────────────────────────────────────
TTS 서브프로세스 워커.

STT/LLM이 사용한 CUDA 컨텍스트(CTranslate2, bitsandbytes)가
F5-TTS와 충돌하여 silent crash를 유발하므로,
TTS를 별도 프로세스에서 실행하여 깨끗한 CUDA 컨텍스트를 보장합니다.

main.py에서 subprocess로 호출됩니다:
    python -m utils.tts_worker --segments-json ... --tts-dir ... [--ref-audio ...] [--ref-text ...]
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
logger = logging.getLogger("tts_worker")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segments-json", required=True, help="입력 세그먼트 JSON 경로")
    p.add_argument("--tts-dir", required=True, help="TTS WAV 출력 디렉토리")
    p.add_argument("--ref-audio", default=None, help="참조 음성 파일")
    p.add_argument("--ref-text", default=None, help="참조 음성 텍스트")
    p.add_argument("--language", default="ko", help="생성 언어 코드 (ko/en/vi/lo 등)")
    args = p.parse_args()

    # 세그먼트 로드
    with open(args.segments_json, "r", encoding="utf-8") as f:
        segments = json.load(f)
    logger.info(f"[TTS Worker] {len(segments)}개 세그먼트 로드")

    # TTS 실행
    from modules.tts import TTSProcessor
    from utils.memory import clear_vram

    tts = TTSProcessor(
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
    )
    segments = tts.synthesize_all(segments, args.tts_dir, use_translated=True)
    tts.unload()
    clear_vram()

    # 결과 저장 (main.py가 로드)
    result_path = str(Path(args.segments_json).parent / "_segments_tts_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)
    logger.info(f"[TTS Worker] 결과 저장: {result_path}")


if __name__ == "__main__":
    main()
