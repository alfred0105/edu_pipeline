"""
utils/audio.py
────────────────────────────────────────────────────────────────────
ffmpeg 래퍼 — 영상에서 오디오 추출, 더빙 음성 합성, SRT 자막 내보내기.
ffmpeg 바이너리가 PATH에 설치되어 있어야 합니다.
"""

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


# ── ffmpeg 헬퍼 ────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str = "ffmpeg") -> None:
    """subprocess 실행 후 실패 시 상세 오류를 출력합니다."""
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"[{label}] 실행 실패 (exit {result.returncode})\n"
            f"CMD : {' '.join(cmd)}\n"
            f"STDERR: {result.stderr[-800:]}"
        )


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """
    영상에서 오디오를 추출합니다.
    STT 입력용으로 16 kHz 모노 PCM WAV로 변환합니다.

    Returns:
        output_path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ], label="extract_audio")
    logger.info(f"[Audio] 오디오 추출 완료: {output_path}")
    return output_path


def get_video_duration(video_path: str) -> float:
    """ffprobe로 영상 총 길이(초)를 반환합니다."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 실패: {result.stderr}")
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def mix_dubbed_into_video(
    video_path: str,
    dubbed_audio_path: str,
    output_path: str,
    original_volume: float = 0.08,
) -> str:
    """
    더빙 음성을 영상에 합성합니다.
    원본 음성은 original_volume 비율로 낮춰 배경음으로 유지합니다.

    Args:
        original_volume: 0.0 → 원본 음소거, 1.0 → 원본 유지. 기본값 0.08 (거의 음소거)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", dubbed_audio_path,
        "-filter_complex",
        (
            f"[0:a]volume={original_volume}[orig];"
            "[orig][1:a]amix=inputs=2:duration=first:normalize=0[aout]"
        ),
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], label="mix_dubbed")
    logger.info(f"[Audio] 더빙 영상 저장: {output_path}")
    return output_path


def extract_ref_clip(
    video_path: str,
    output_path: str,
    segments: list[dict],
    target_duration: float = 10.0,
) -> tuple[str, str] | None:
    """
    STT 세그먼트 중 화자 클로닝에 적합한 구간을 찾아 WAV로 추출합니다.

    조건: 5~15초 길이, 텍스트 길이 15자 이상인 세그먼트 중 가장 긴 것 선택

    Returns:
        (output_path, ref_text) 또는 실패 시 None
    """
    candidates = [
        s for s in segments
        if 5.0 <= (s["end"] - s["start"]) <= 15.0 and len(s["text"]) >= 15
    ]
    if not candidates:
        return None

    best = max(candidates, key=lambda s: s["end"] - s["start"])
    start, end = best["start"], best["end"]
    ref_text = best["text"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start), "-to", str(end),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "24000", "-ac", "1",
        output_path,
    ], label="extract_ref_clip")
    logger.info(f"[Audio] 화자 참조 클립 추출: {start:.1f}s~{end:.1f}s  '{ref_text[:40]}'")
    return output_path, ref_text


# ── SRT 자막 내보내기 ───────────────────────────────────────────────────────

def _fmt_srt_time(secs: float) -> str:
    """초 → SRT 타임스탬프 형식 (HH:MM:SS,mmm)"""
    h  = int(secs // 3600)
    m  = int((secs % 3600) // 60)
    s  = int(secs % 60)
    ms = int(round((secs % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(
    segments: list[dict],
    output_path: str,
    use_translated: bool = False,
) -> str:
    """
    세그먼트 리스트를 SRT 자막 파일로 내보냅니다.

    Args:
        use_translated: True면 'translated' 필드 사용, False면 원본 'text' 사용
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            text = seg.get("translated", seg["text"]) if use_translated else seg["text"]
            f.write(f"{i}\n")
            f.write(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}\n")
            f.write(f"{text.strip()}\n\n")
    logger.info(f"[SRT] 자막 저장: {output_path}  ({len(segments)}개 구간)")
    return output_path
