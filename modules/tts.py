"""
modules/tts.py
────────────────────────────────────────────────────────────────────
TTS 모듈 — F5-TTS를 이용한 AI 더빙 생성.

F5-TTS 특징:
  - Flow Matching 기반 비자기회귀 모델 → 빠른 추론 속도
  - 참조 음성(ref_audio) 제공 시 화자 목소리 복제(클로닝) 가능
  - 8GB VRAM 내에서 안정적으로 동작

타임스탬프 동기화:
  - Whisper 세그먼트의 (end - start) 시간에 맞게
    stretch_audio_to_duration()으로 음성 길이를 조정합니다.
  - 조정 가능 범위: 0.5배(느리게) ~ 2.5배(빠르게)
  - 범위 초과 시 클램프 + 무음 패딩으로 보완

참조 음성 없이도 기본 화자로 생성 가능합니다.
"""

import logging
import os
import tempfile
from pathlib import Path

from config import cfg
from utils.memory import clear_vram, log_vram
from utils.timestamp_sync import stretch_audio_to_duration

logger = logging.getLogger(__name__)

# F5-TTS 내부 샘플레이트
F5_TTS_SR = 24_000


class TTSProcessor:
    """
    F5-TTS 래퍼.

    Args:
        ref_audio_path: 화자 클로닝용 참조 WAV 파일 경로 (선택)
        ref_text:       참조 음성에 대한 정확한 텍스트 (클로닝 품질 향상)
    """

    def __init__(
        self,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
    ):
        logger.info(f"[TTS] F5-TTS 로드 중  device={cfg.tts.device}")
        log_vram("TTS 로드 전")

        from f5_tts.api import F5TTS

        self._tts      = F5TTS(device=cfg.tts.device)
        self._ref_audio = ref_audio_path
        self._ref_text  = ref_text or ""

        if ref_audio_path:
            logger.info(f"[TTS] 참조 음성 설정: {ref_audio_path}")
        else:
            logger.info("[TTS] 참조 음성 없음 — 기본 화자 사용")

        log_vram("TTS 로드 후")

    # ── 단일 세그먼트 합성 ────────────────────────────────────────────────
    def synthesize_segment(
        self,
        text: str,
        target_duration: float,
        output_path: str,
    ) -> str:
        """
        텍스트를 합성하고 target_duration에 맞게 시간 축 스트레칭합니다.

        Returns:
            output_path
        """
        import soundfile as sf

        # 임시 파일에 먼저 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if self._ref_audio:
                wav, sr, _ = self._tts.infer(
                    ref_file=self._ref_audio,
                    ref_text=self._ref_text,
                    gen_text=text,
                    target_rms=0.1,
                )
            else:
                wav, sr, _ = self._tts.infer(gen_text=text)

            sf.write(tmp_path, wav, sr)

            # 타임스탬프에 맞게 스트레칭
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            stretch_audio_to_duration(
                tmp_path,
                target_duration,
                output_path,
                sr=sr,
                min_ratio=cfg.tts.min_stretch_ratio,
                max_ratio=cfg.tts.max_stretch_ratio,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return output_path

    # ── 전체 세그먼트 배치 합성 ───────────────────────────────────────────
    def synthesize_all(
        self,
        segments: list[dict],
        output_dir: str,
        use_translated: bool = True,
    ) -> list[dict]:
        """
        모든 세그먼트를 합성합니다.

        Args:
            segments:       [{start, end, text, translated?, ...}, ...]
            output_dir:     TTS WAV 파일 저장 디렉토리
            use_translated: True이면 'translated' 필드 사용, 없으면 'text' 폴백

        Returns:
            segments에 'tts_path' 필드가 추가된 리스트
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result = []
        total = len(segments)

        for i, seg in enumerate(segments):
            text = seg.get("translated", seg["text"]) if use_translated else seg["text"]
            duration = seg["end"] - seg["start"]
            output_path = str(Path(output_dir) / f"seg_{i:04d}.wav")

            logger.info(
                f"[TTS] {i + 1}/{total}  "
                f"duration={duration:.1f}s  "
                f"text={text[:45]}{'...' if len(text) > 45 else ''}"
            )

            try:
                self.synthesize_segment(text, duration, output_path)
                result.append({**seg, "tts_path": output_path})
            except Exception as e:
                logger.error(f"[TTS] 세그먼트 {i} 합성 실패: {e}")
                result.append({**seg, "tts_path": None})

        successful = sum(1 for r in result if r.get("tts_path"))
        logger.info(f"[TTS] 합성 완료: {successful}/{total}개 성공")
        return result

    # ── VRAM 해제 ────────────────────────────────────────────────────────
    def unload(self) -> None:
        del self._tts
        clear_vram()
        log_vram("TTS 해제 후")
        logger.info("[TTS] 모델 해제 완료")
