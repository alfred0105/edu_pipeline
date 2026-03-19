"""
utils/timestamp_sync.py
────────────────────────────────────────────────────────────────────
타임스탬프 동기화 — TTS 음성의 길이를 Whisper 세그먼트 구간에 맞춥니다.

전략:
  1. TTS 생성 음성의 실제 길이를 측정
  2. 목표 구간 길이(end - start)와 비율 계산
  3. librosa STFT phase vocoder로 시간 축 스트레칭
  4. 스트레칭 비율이 [0.5, 2.5] 범위를 벗어나면 끝에 무음 패딩
  5. merge_dubbed_audio()로 전체 트랙을 단일 WAV로 합칩니다.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def stretch_audio_to_duration(
    input_path: str,
    target_duration: float,
    output_path: str,
    sr: int = 24_000,
    min_ratio: float = 0.5,
    max_ratio: float = 2.5,
) -> None:
    """
    WAV 파일을 target_duration(초)에 맞게 시간 축 스트레칭합니다.

    Args:
        input_path:       원본 TTS WAV 파일
        target_duration:  목표 길이 (Whisper 세그먼트 end - start)
        output_path:      결과 WAV 파일
        sr:               샘플레이트 (F5-TTS 기본값 24 kHz)
        min_ratio:        최소 스트레칭 비율 (이보다 느려지면 패딩으로 보완)
        max_ratio:        최대 스트레칭 비율 (이보다 빨라지면 클램프 후 패딩)
    """
    import librosa
    import soundfile as sf

    y, file_sr = librosa.load(input_path, sr=sr)
    actual_duration = len(y) / sr

    # 무음 세그먼트 처리
    if actual_duration < 0.05 or np.abs(y).max() < 1e-5:
        silence = np.zeros(int(sr * target_duration), dtype=np.float32)
        sf.write(output_path, silence, sr)
        logger.debug(f"[Sync] 무음 구간 → {target_duration:.2f}s 무음 패딩")
        return

    ratio = target_duration / actual_duration  # >1: 느리게, <1: 빠르게

    if abs(ratio - 1.0) < 0.02:
        # 거의 동일 — 스트레칭 불필요
        sf.write(output_path, y, sr)
        logger.debug(f"[Sync] 스트레칭 불필요 ({actual_duration:.2f}s ≈ {target_duration:.2f}s)")
        return

    # 비율 클램프
    clamped = max(min_ratio, min(ratio, max_ratio))
    if clamped != ratio:
        logger.warning(
            f"[Sync] 비율 클램프: {ratio:.2f} → {clamped:.2f} "
            f"(target={target_duration:.2f}s, actual={actual_duration:.2f}s)"
        )

    # librosa phase vocoder 스트레칭 (rate = 1/ratio: >1이면 빠르게, <1이면 느리게)
    stretched = librosa.effects.time_stretch(y, rate=1.0 / clamped)

    # 샘플 수 정확히 맞추기 (무음 패딩 or 트리밍)
    target_samples = int(sr * target_duration)
    if len(stretched) < target_samples:
        stretched = np.pad(stretched, (0, target_samples - len(stretched)))
    else:
        stretched = stretched[:target_samples]

    # 클리핑 방지
    peak = np.abs(stretched).max()
    if peak > 0.99:
        stretched = stretched * (0.95 / peak)

    sf.write(output_path, stretched.astype(np.float32), sr)
    logger.debug(
        f"[Sync] {actual_duration:.2f}s → {target_duration:.2f}s "
        f"(ratio={ratio:.2f}, clamped={clamped:.2f})"
    )


def merge_dubbed_audio(
    segments: list[dict],
    total_duration: float,
    output_path: str,
    sr: int = 24_000,
) -> None:
    """
    TTS 세그먼트들을 단일 오디오 트랙으로 합칩니다.
    세그먼트 사이의 간격(갭)은 무음으로 채웁니다.

    Args:
        segments:       [{start, end, tts_path, ...}, ...]  (tts_path가 None이면 건너뜀)
        total_duration: 영상 전체 길이 (초)
        output_path:    최종 WAV 출력 경로
        sr:             샘플레이트
    """
    import librosa
    import soundfile as sf

    total_samples = int(sr * total_duration)
    merged = np.zeros(total_samples, dtype=np.float32)

    placed = 0
    for seg in segments:
        tts_path = seg.get("tts_path")
        if not tts_path:
            continue
        try:
            y, file_sr = librosa.load(tts_path, sr=sr)
        except Exception as e:
            logger.warning(f"[Sync] 세그먼트 로드 실패 ({tts_path}): {e}")
            continue

        start_sample = int(seg["start"] * sr)
        end_sample   = min(start_sample + len(y), total_samples)
        copy_len     = end_sample - start_sample

        if copy_len <= 0:
            continue

        merged[start_sample:end_sample] += y[:copy_len]
        placed += 1

    # 최종 정규화
    peak = np.abs(merged).max()
    if peak > 0.98:
        merged = merged * (0.95 / peak)

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, merged, sr)
    logger.info(
        f"[Sync] 더빙 트랙 합성 완료: {output_path}  "
        f"({placed}/{len(segments)} 세그먼트, {total_duration:.1f}s)"
    )
