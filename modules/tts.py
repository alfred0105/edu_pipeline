"""
modules/tts.py
────────────────────────────────────────────────────────────────────
TTS 모듈 — Qwen3-TTS를 이용한 AI 더빙 생성.

Qwen3-TTS 특징:
  - 0.6B / 1.7B 모델 선택 가능
  - 3초 참조 음성으로 화자 클로닝 (Voice Cloning)
  - 한국어·영어·베트남어 등 10개 언어 지원
  - create_voice_clone_prompt()로 참조 특성을 미리 추출해
    세그먼트마다 재추출 없이 빠르게 생성

타임스탬프 동기화:
  - Whisper 세그먼트의 (end - start) 시간에 맞게
    stretch_audio_to_duration()으로 음성 길이를 조정합니다.
  - 조정 가능 범위: 0.5배(느리게) ~ 2.5배(빠르게)
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from config import cfg
from utils.memory import clear_vram, log_vram
from utils.timestamp_sync import stretch_audio_array

logger = logging.getLogger(__name__)

# Qwen3-TTS 내부 샘플레이트
QWEN3_TTS_SR = 24_000

# 언어 코드 → Qwen3-TTS 언어 이름 매핑 (소문자)
LANG_MAP = {
    "ko": "korean",
    "en": "english",
    "vi": "vietnamese",
    "lo": "lao",
    "zh": "chinese",
    "ja": "japanese",
    "fr": "french",
    "es": "spanish",
    "de": "german",
    "ar": "arabic",
}


class TTSProcessor:
    """
    Qwen3-TTS 래퍼.

    Args:
        ref_audio_path: 화자 클로닝용 참조 WAV 파일 경로 (선택)
        ref_text:       참조 음성에 대한 텍스트 (클로닝 품질 향상, 생략 가능)
        language:       생성 언어 코드 (기본 "ko")
    """

    def __init__(
        self,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
        language: str = "ko",
    ):
        logger.info(f"[TTS] Qwen3-TTS 로드 중  model={cfg.tts.model_id}  device={cfg.tts.device}")
        log_vram("TTS 로드 전")

        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = torch.bfloat16
        self._model = Qwen3TTSModel.from_pretrained(
            cfg.tts.model_id,
            device_map=cfg.tts.device,
            dtype=dtype,
        )
        self._language = LANG_MAP.get(language, "korean")
        self._voice_clone_prompt = None

        if ref_audio_path and Path(ref_audio_path).is_file():
            logger.info(f"[TTS] 화자 클로닝 프롬프트 생성 중: {ref_audio_path}")
            prompts = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=ref_text or None,
            )
            self._voice_clone_prompt = prompts[0] if prompts else None
            logger.info("[TTS] 화자 클로닝 프롬프트 생성 완료")
        else:
            logger.info("[TTS] 참조 음성 없음 — 무음 더미로 기본 화자 프롬프트 생성 중")
            self._voice_clone_prompt = self._make_default_prompt()

        log_vram("TTS 로드 후")

    # ── 기본 화자 프롬프트 생성 (참조 음성 없을 때) ───────────────────────
    def _make_default_prompt(self):
        """무음 3초 WAV를 임시 파일로 만들어 voice_clone_prompt를 생성합니다."""
        import tempfile
        import numpy as np
        import soundfile as sf

        silence = np.zeros(QWEN3_TTS_SR * 3, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, silence, QWEN3_TTS_SR)
        try:
            prompts = self._model.create_voice_clone_prompt(ref_audio=tmp_path, ref_text=None)
            prompt = prompts[0] if prompts else None
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        logger.info("[TTS] 기본 화자 프롬프트 생성 완료")
        return prompt

    # ── 단일 세그먼트 추론 ────────────────────────────────────────────────
    def _infer(self, text: str):
        """텍스트를 음성으로 변환하고 (wav_array, sr)을 반환합니다."""
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=self._language,
            voice_clone_prompt=self._voice_clone_prompt,
            non_streaming_mode=True,
        )
        return wavs[0], sr

    # ── CPU 후처리 (스트레칭 + 저장) ─────────────────────────────────────
    @staticmethod
    def _postprocess(wav, sr: int, target_duration: float, output_path: str) -> str:
        import numpy as np
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        wav_np = np.asarray(wav, dtype=np.float32).flatten()
        stretch_audio_array(
            wav_np,
            target_duration,
            output_path,
            sr=sr,
            min_ratio=cfg.tts.min_stretch_ratio,
            max_ratio=cfg.tts.max_stretch_ratio,
        )
        return output_path

    # ── 전체 세그먼트 합성 ────────────────────────────────────────────────
    def synthesize_all(
        self,
        segments: list[dict],
        output_dir: str,
        use_translated: bool = True,
    ) -> list[dict]:
        """
        모든 세그먼트를 합성합니다.
        GPU 추론과 CPU 후처리를 오버랩해 속도를 높입니다.
        """
        import time

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        total = len(segments)
        result = [None] * total
        success_count = 0
        t_start = time.time()

        log_interval = max(1, min(10, total // 10))

        with ThreadPoolExecutor(max_workers=2) as executor:
            pending: dict[int, Future] = {}

            for i, seg in enumerate(segments):
                text = seg.get("translated", seg["text"]) if use_translated else seg["text"]
                duration = seg["end"] - seg["start"]
                out_path = str(Path(output_dir) / f"seg_{i:04d}.wav")

                if not text.strip() or duration < 0.5:
                    result[i] = {**seg, "tts_path": None}
                    continue

                if i == 0 or (i + 1) % log_interval == 0 or i == total - 1:
                    elapsed = time.time() - t_start
                    eta_str = ""
                    if success_count > 0:
                        eta = elapsed / success_count * (total - i)
                        eta_str = f"  ETA {eta:.0f}s"
                    logger.info(
                        f"[TTS] {i + 1}/{total} ({(i + 1) * 100 // total}%)  "
                        f"성공={success_count}  경과={elapsed:.0f}s{eta_str}  "
                        f"text={text[:35]}{'...' if len(text) > 35 else ''}"
                    )

                # 이전 CPU 후처리 완료 확인
                if i - 2 in pending:
                    fut = pending.pop(i - 2)
                    try:
                        fut.result()
                    except Exception as e:
                        logger.error(f"[TTS] 세그먼트 {i - 2} 후처리 실패: {e}")

                # GPU 추론
                try:
                    wav, sr = self._infer(text)
                    pending[i] = executor.submit(
                        self._postprocess, wav, sr, duration, out_path
                    )
                    result[i] = {**seg, "tts_path": out_path}
                    success_count += 1
                except Exception as e:
                    logger.error(f"[TTS] 세그먼트 {i} 합성 실패: {e}")
                    result[i] = {**seg, "tts_path": None}

            for idx, fut in pending.items():
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"[TTS] 세그먼트 {idx} 후처리 실패: {e}")
                    if result[idx]:
                        result[idx] = {**result[idx], "tts_path": None}

        successful = sum(1 for r in result if r and r.get("tts_path"))
        elapsed_total = time.time() - t_start
        logger.info(
            f"[TTS] 합성 완료: {successful}/{total}개 성공  "
            f"총 {elapsed_total:.1f}초 ({elapsed_total / 60:.1f}분)"
        )
        return result

    # ── VRAM 해제 ────────────────────────────────────────────────────────
    def unload(self) -> None:
        del self._model
        self._model = None
        clear_vram()
        log_vram("TTS 해제 후")
        logger.info("[TTS] 모델 해제 완료")
