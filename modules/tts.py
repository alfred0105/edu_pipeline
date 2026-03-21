"""
modules/tts.py
────────────────────────────────────────────────────────────────────
TTS 모듈 — F5-TTS를 이용한 AI 더빙 생성.

F5-TTS 특징:
  - Flow Matching 기반 비자기회귀 모델 → 빠른 추론 속도
  - 참조 음성(ref_audio) 제공 시 화자 목소리 복제(클로닝) 가능
  - 8GB VRAM 내에서 안정적으로 동작

속도 최적화:
  - nfe_step=8 (기본 32 → 4배 빠름, 품질 소폭 저하)
  - GPU 추론과 CPU 후처리(스트레칭)를 ThreadPoolExecutor로 오버랩

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
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from config import cfg
from utils.memory import clear_vram, log_vram
from utils.timestamp_sync import stretch_audio_to_duration

logger = logging.getLogger(__name__)

# F5-TTS 내부 샘플레이트
F5_TTS_SR = 24_000


def _get_default_ref():
    from importlib.resources import files
    return (
        str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        "Some call me nature, others call me mother nature.",
    )


class TTSProcessor:
    """
    F5-TTS 래퍼.

    Args:
        ref_audio_path: 화자 클로닝용 참조 WAV 파일 경로 (선택)
        ref_text:       참조 음성에 대한 정확한 텍스트 (클로닝 품질 향상)
        nfe_step:       추론 스텝 수 (기본 8, 낮을수록 빠름/품질 저하)
    """

    def __init__(
        self,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
        nfe_step: int = 8,
    ):
        logger.info(f"[TTS] F5-TTS 로드 중  device={cfg.tts.device}  nfe_step={nfe_step}")
        log_vram("TTS 로드 전")

        from f5_tts.api import F5TTS

        self._tts       = F5TTS(device=cfg.tts.device)
        self._ref_audio = ref_audio_path
        self._ref_text  = ref_text or ""
        self._nfe_step  = nfe_step

        if not ref_audio_path:
            self._ref_audio, self._ref_text = _get_default_ref()
            logger.info("[TTS] 참조 음성 없음 — 기본 화자 사용")
        else:
            logger.info(f"[TTS] 참조 음성 설정: {ref_audio_path}")

        log_vram("TTS 로드 후")

    # ── 단일 세그먼트 GPU 추론 (wav 반환) ────────────────────────────────
    def _infer(self, text: str):
        """GPU 추론만 수행하고 (wav, sr)을 반환합니다."""
        wav, sr, _ = self._tts.infer(
            ref_file=self._ref_audio,
            ref_text=self._ref_text,
            gen_text=text,
            target_rms=0.1,
            nfe_step=self._nfe_step,
        )
        return wav, sr

    # ── CPU 후처리 (스트레칭 + 파일 저장) ────────────────────────────────
    @staticmethod
    def _postprocess(wav, sr: int, target_duration: float, output_path: str) -> str:
        """CPU에서 오디오 스트레칭 후 저장합니다."""
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, wav, sr)
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

    # ── 전체 세그먼트 합성 (GPU 추론 + CPU 후처리 오버랩) ────────────────
    def synthesize_all(
        self,
        segments: list[dict],
        output_dir: str,
        use_translated: bool = True,
    ) -> list[dict]:
        """
        모든 세그먼트를 합성합니다.
        GPU 추론과 CPU 후처리를 오버랩해 속도를 높입니다.

        Args:
            segments:       [{start, end, text, translated?, ...}, ...]
            output_dir:     TTS WAV 파일 저장 디렉토리
            use_translated: True이면 'translated' 필드 사용, 없으면 'text' 폴백

        Returns:
            segments에 'tts_path' 필드가 추가된 리스트
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        total = len(segments)
        result = [None] * total

        # CPU 후처리 스레드풀 (GPU 추론과 오버랩)
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending: dict[int, Future] = {}

            for i, seg in enumerate(segments):
                text     = seg.get("translated", seg["text"]) if use_translated else seg["text"]
                duration = seg["end"] - seg["start"]
                out_path = str(Path(output_dir) / f"seg_{i:04d}.wav")

                logger.info(
                    f"[TTS] {i + 1}/{total}  "
                    f"duration={duration:.1f}s  "
                    f"text={text[:45]}{'...' if len(text) > 45 else ''}"
                )

                # 이전 세그먼트 CPU 후처리 완료 확인 (2개 앞서 대기)
                if i - 2 in pending:
                    fut = pending.pop(i - 2)
                    try:
                        fut.result()
                    except Exception as e:
                        logger.error(f"[TTS] 세그먼트 {i - 2} 후처리 실패: {e}")

                # GPU 추론 (동기 실행)
                try:
                    wav, sr = self._infer(text)
                    # CPU 후처리는 비동기로 제출 → 다음 GPU 추론과 오버랩
                    pending[i] = executor.submit(
                        self._postprocess, wav, sr, duration, out_path
                    )
                    result[i] = {**seg, "tts_path": out_path}
                except Exception as e:
                    logger.error(f"[TTS] 세그먼트 {i} 합성 실패: {e}")
                    result[i] = {**seg, "tts_path": None}

            # 남은 CPU 후처리 완료 대기
            for idx, fut in pending.items():
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"[TTS] 세그먼트 {idx} 후처리 실패: {e}")
                    result[idx] = {**result[idx], "tts_path": None}

        successful = sum(1 for r in result if r and r.get("tts_path"))
        logger.info(f"[TTS] 합성 완료: {successful}/{total}개 성공")
        return result

    # ── VRAM 해제 ────────────────────────────────────────────────────────
    def unload(self) -> None:
        del self._tts
        clear_vram()
        log_vram("TTS 해제 후")
        logger.info("[TTS] 모델 해제 완료")
