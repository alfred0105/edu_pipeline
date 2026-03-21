"""
modules/stt.py
────────────────────────────────────────────────────────────────────
STT 모듈 — faster-whisper를 이용한 Whisper large-v3-turbo 추론.

PC (RTX 4060 Ti 8GB):
  - device="cuda", compute_type="float16"
  - large-v3-turbo는 기존 v3 대비 VRAM 약 40% 절감, 속도 2-3배 향상

Mac (Apple Silicon):
  - device="auto" (faster-whisper가 MPS 자동 감지)
  - compute_type="float32"

출력 형식:
  [
    {
      "start": 0.0,
      "end":   3.5,
      "text":  "안녕하세요.",
      "words": [{"word": "안녕하세요.", "start": 0.1, "end": 0.8}]
    }, ...
  ]
"""

import logging
from config import cfg
from utils.memory import clear_vram, log_vram

logger = logging.getLogger(__name__)


def transcribe(
    audio_path: str,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> list[dict]:
    """
    음성 파일을 텍스트로 변환합니다.

    Args:
        audio_path:     WAV/MP3/MP4 등 오디오 파일 경로
        language:       언어 코드 ("ko", "en", "ja" 등). None이면 자동 감지
        initial_prompt: Whisper 초기 프롬프트 (전문 용어 힌트 등)

    Returns:
        세그먼트 리스트
    """
    from faster_whisper import WhisperModel

    logger.info(
        f"[STT] 모델 로드: whisper-{cfg.stt.model_size}  "
        f"device={cfg.stt.device}  compute={cfg.stt.compute_type}"
    )
    log_vram("STT 시작")

    model = WhisperModel(
        cfg.stt.model_size,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
        download_root=None,           # 기본 캐시 사용 (~/.cache/huggingface)
        local_files_only=False,
        num_workers=2,
    )

    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        initial_prompt=initial_prompt,
        beam_size=3,
        best_of=1,
        word_timestamps=False,
        vad_filter=cfg.stt.vad_filter,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 100,
        },
        condition_on_previous_text=True,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    segments: list[dict] = []
    for seg in segments_gen:
        words = []
        if seg.words:
            words = [
                {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in seg.words
            ]
        segments.append({
            "start": round(seg.start, 3),
            "end":   round(seg.end,   3),
            "text":  seg.text.strip(),
            "words": words,
        })

    logger.info(
        f"[STT] 완료: {len(segments)}개 세그먼트 | "
        f"감지 언어={info.language} ({info.language_probability:.1%})"
    )

    # VRAM 해제 — 다음 단계(LLM)가 메모리를 사용할 수 있도록
    del model
    clear_vram()
    log_vram("STT 종료")

    return segments
