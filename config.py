"""
edu_pipeline/config.py
────────────────────────────────────────────────────────────────────
환경 설정 파일. Mac(MLX) ↔ PC(CUDA) 자동 감지 및 수동 오버라이드 지원.

오버라이드 방법:
  Windows: set EDU_PIPELINE_MODE=pc
  Mac/Linux: export EDU_PIPELINE_MODE=mac
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

# ── 환경 자동 감지 ─────────────────────────────────────────────────────────
_DEFAULT_MODE = "mac" if platform.system() == "Darwin" else "pc"
MODE: str = os.environ.get("EDU_PIPELINE_MODE", _DEFAULT_MODE).lower()  # "mac" | "pc"

assert MODE in ("mac", "pc"), f"EDU_PIPELINE_MODE must be 'mac' or 'pc', got: {MODE!r}"


# ── 데이터 경로 ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INPUT_DIR  = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
VECTOR_DIR = DATA_DIR / "vector_db"


# ── 모델 설정 ──────────────────────────────────────────────────────────────
@dataclass
class _STTConfig:
    model_size: str      = "large-v3-turbo"      # faster-whisper 모델 크기
    device: str          = field(default_factory=lambda: "cuda"    if MODE == "pc" else "auto")
    compute_type: str    = field(default_factory=lambda: "float16" if MODE == "pc" else "float32")
    beam_size: int       = 3
    vad_filter: bool     = True                  # 무음 구간 자동 제거

@dataclass
class _LLMConfig:
    model_id: str        = "Qwen/Qwen3-4B"
    # PC: llama.cpp GGUF 양자화 (transformers 대비 2~3배 빠름, GPU 가속)
    gguf_repo: str       = "Qwen/Qwen3-4B-GGUF"
    gguf_file: str       = "Qwen3-4B-Q4_K_M.gguf"
    n_gpu_layers: int    = -1                   # -1 = 전체 GPU 오프로드
    n_ctx: int           = 4096                 # 컨텍스트 길이
    # Mac: mlx-lm 자체 양자화
    load_in_4bit: bool   = field(default_factory=lambda: MODE == "pc")
    max_new_tokens: int  = 1024
    temperature: float   = 0.1                  # 번역/요약은 낮게 유지

@dataclass
class _TTSConfig:
    # F5-TTS 모델 (https://github.com/SWivid/F5-TTS)
    model_name: str      = "F5TTS_v1_Base"
    device: str          = field(default_factory=lambda: "cuda" if MODE == "pc" else "mps")
    # 타임스탬프 동기화 허용 스트레칭 범위
    min_stretch_ratio: float = 0.5               # 원본 속도의 0.5배까지만 느리게
    max_stretch_ratio: float = 2.5               # 원본 속도의 2.5배까지만 빠르게

@dataclass
class _EmbedConfig:
    model_id: str        = "Qwen/Qwen3-Embedding-0.6B"
    device: str          = field(default_factory=lambda: "cuda" if MODE == "pc" else "mps")
    batch_size: int      = 32
    max_length: int      = 512

@dataclass
class _RAGConfig:
    top_k: int           = 5                     # 검색 청크 수
    distance_threshold: float = 0.60            # 코사인 거리 임계값 (낮을수록 엄격)
    collection_prefix: str = "edu_"

@dataclass
class Config:
    mode:   str          = MODE
    stt:    _STTConfig   = field(default_factory=_STTConfig)
    llm:    _LLMConfig   = field(default_factory=_LLMConfig)
    tts:    _TTSConfig   = field(default_factory=_TTSConfig)
    embed:  _EmbedConfig = field(default_factory=_EmbedConfig)
    rag:    _RAGConfig   = field(default_factory=_RAGConfig)
    input_dir:  str      = str(INPUT_DIR)
    output_dir: str      = str(OUTPUT_DIR)
    vector_dir: str      = str(VECTOR_DIR)


cfg = Config()

# 디렉토리 자동 생성
for d in (INPUT_DIR, OUTPUT_DIR, VECTOR_DIR):
    d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Mode     : {cfg.mode}")
    print(f"STT      : {cfg.stt.model_size}  device={cfg.stt.device}  compute={cfg.stt.compute_type}")
    print(f"LLM      : {cfg.llm.model_id}  4bit={cfg.llm.load_in_4bit}")
    print(f"TTS      : {cfg.tts.model_name}  device={cfg.tts.device}")
    print(f"Embed    : {cfg.embed.model_id}  device={cfg.embed.device}")
    print(f"VectorDB : {cfg.vector_dir}")
