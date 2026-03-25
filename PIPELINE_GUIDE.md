# edu_pipeline 동작 메커니즘 완전 정리

---

## 전체 흐름 요약

```
영상 파일 (.mp4)
     │
     ▼
[STEP 1] 오디오 추출 (ffmpeg)
     │  → extracted_audio.wav (16kHz 모노)
     ▼
[STEP 2] 음성인식 STT (faster-whisper)
     │  → segments: [{start, end, text}, ...]
     │  → subtitles_original.srt
     ▼
[STEP 3] LLM 번역 + 요약 (Qwen3-4B)
     │  → segments에 "translated" 필드 추가
     │  → subtitles_ko.srt (번역 시)
     │  → summary_child/teen/adult.txt
     ▼
[STEP 4] TTS 더빙 (F5-TTS)
     │  → tts_segments/seg_XXXX.wav (748개)
     │  → dubbed_audio.wav (전체 트랙)
     │  → 영상명_dubbed.mp4
     ▼
[STEP 5] 텍스트 임베딩 (Qwen3-Embedding-0.6B)
     │  → segments에 "embedding" 벡터 추가
     ▼
[STEP 6] 벡터 DB 저장 (ChromaDB)
     │  → data/vector_db/chroma.sqlite3
     │  → segments.json
     ▼
[STEP 7] RAG 챗봇 (선택)
```

---

## STEP 1 — 오디오 추출

**파일**: `utils/audio.py` → `extract_audio()`

**역할**: 영상에서 음성만 분리

**동작**:
- ffmpeg를 subprocess로 호출
- 비디오 스트림 제거 (`-vn`)
- 16kHz 모노 PCM WAV로 변환 (Whisper 최적 입력 형식)
- `ffprobe`로 영상 전체 길이(초)도 별도 측정

**출력**: `data/output/영상명/extracted_audio.wav`

---

## STEP 2 — 음성인식 (STT)

**파일**: `modules/stt.py` → `transcribe()`

**모델**: `faster-whisper large-v3-turbo` (CUDA float16)

**동작 순서**:
1. `WhisperModel` 로드 (GPU VRAM ~2GB)
2. VAD 필터로 무음 구간 자동 제거 (300ms 이상 무음 제거)
3. 음성 구간만 beam_size=3으로 디코딩
4. 세그먼트 단위로 텍스트 + 타임스탬프 추출
5. 완료 후 VRAM 즉시 해제

**출력 형식**:
```python
[
  {"start": 0.0, "end": 3.5, "text": "안녕하세요.", "words": []},
  {"start": 3.5, "end": 7.2, "text": "이번 시간에는...", "words": []},
  ...
]
```

**출력 파일**: `subtitles_original.srt`

**속도 최적화 포인트**:
- `best_of=1` (후보 1개만 생성, 원래 5개)
- `beam_size=3` (원래 5)
- `word_timestamps=False` (단어 단위 타임스탬프 불필요)

---

## STEP 3 — LLM 번역 + 요약

**파일**: `modules/llm.py` → `LLMProcessor`

**모델**: `Qwen/Qwen3-4B` (bitsandbytes 4-bit NF4 양자화, VRAM ~2.5GB)

### 3-1. 번역 (translate_segments)

원본 언어 ≠ 목표 언어일 때만 실행 (기본 ko→ko이면 스킵)

**배치 처리 방식** (속도 최적화):
- 10개 세그먼트를 하나의 프롬프트로 묶어서 처리
- 748개 세그먼트 → 75번 LLM 호출 (기존 748번)
- 응답 파싱: `"1. 번역문\n2. 번역문..."` 형식을 정규식으로 분리
- 파싱 실패 시 원문 그대로 사용 (안전 장치)

**프롬프트 형식**:
```
아래 10개의 문장을 각각 자연스러운 한국어로 번역하세요.
반드시 '번호. 번역문' 형식으로만 출력하세요.

1. Hello everyone.
2. Today we will learn about...
...
```

### 3-2. 연령별 요약 (summarize_by_age)

전체 자막 텍스트를 이어붙여 3가지 버전으로 요약:

| 대상 | 조건 |
|------|------|
| child | 초등학생 수준, 비유 사용, 짧은 문장 |
| teen | 중고등학생 수준, 논리적 설명 |
| adult | 성인 학습자, 전문 용어 활용 |

**출력 파일**: `summary_child.txt`, `summary_teen.txt`, `summary_adult.txt`

---

## STEP 4 — TTS AI 더빙

**파일**: `modules/tts.py` → `TTSProcessor`

**모델**: `F5-TTS F5TTS_v1_Base` (Flow Matching 기반, VRAM ~0.7GB)

### 4-1. 화자 참조 음성 확보

`--ref-audio`를 지정하지 않으면 **자동 추출**:
- `utils/audio.py` → `extract_ref_clip()` 호출
- STT 세그먼트 중 5~15초 길이, 텍스트 15자 이상인 구간 선택
- 해당 구간을 ffmpeg로 24kHz WAV 추출
- 이 음성으로 화자 목소리 클로닝

### 4-2. 세그먼트별 합성 (synthesize_all)

**GPU 추론 + CPU 후처리 병렬 오버랩** 구조:

```
세그먼트 N:   [GPU 추론] → 결과 큐 → [CPU 후처리: 스트레칭+저장]
세그먼트 N+1:             [GPU 추론] → 결과 큐 → [CPU 후처리]
                          ↑ 동시 실행
```

- GPU: F5-TTS `infer()` 호출 (nfe_step=8, 기본 32보다 4배 빠름)
- CPU: ThreadPoolExecutor(max_workers=2)로 오디오 스트레칭 병렬 처리

### 4-3. 타임스탬프 동기화 (stretch_audio_to_duration)

**파일**: `utils/timestamp_sync.py`

TTS로 생성된 음성 길이 ≠ 원본 세그먼트 길이 → 맞춰야 함

동작:
1. soundfile로 TTS WAV 읽기 (빠른 I/O)
2. 실제 길이 vs 목표 길이 비율 계산
3. 비율이 [0.5 ~ 2.5] 범위면 librosa STFT 위상 보코더로 스트레칭 (n_fft=512)
4. 범위 벗어나면 클램프 후 무음 패딩
5. 출력 파일 저장

### 4-4. 전체 더빙 트랙 합성 (merge_dubbed_audio)

- 748개 세그먼트 WAV를 영상 전체 길이 버퍼에 타임스탬프 위치에 배치
- 세그먼트 간 빈 구간은 자동으로 무음 처리
- 최종 피크 정규화 후 단일 WAV 저장

### 4-5. 영상과 합성 (mix_dubbed_into_video)

ffmpeg amix 필터:
- 더빙 음성 100% 볼륨
- 원본 음성 8% 볼륨 (배경음으로 유지)
- 비디오 스트림은 그대로 복사 (`-c:v copy`)

**출력 파일**: `영상명_dubbed.mp4`

---

## STEP 5 — 텍스트 임베딩

**파일**: `modules/embedder.py` → `EmbeddingProcessor`

**모델**: `Qwen/Qwen3-Embedding-0.6B` (float16, VRAM ~1.1GB)

**동작**:
1. 번역된 텍스트(없으면 원문) 추출
2. batch_size=32로 묶어 토크나이저 처리
3. 모델 forward pass → last_hidden_state
4. Attention mask 기반 Mean Pooling
5. L2 정규화 → 1024차원 단위 벡터

**출력**: 각 세그먼트에 `"embedding": [float × 1024]` 필드 추가

---

## STEP 6 — 벡터 DB 저장 (RAG 인덱싱)

**파일**: `modules/rag.py` → `RAGChatbot.index_segments()`

**DB**: ChromaDB (로컬 SQLite 기반, 서버 불필요)

**저장 내용**:

| 필드 | 내용 |
|------|------|
| id | `seg_00001` 형식 |
| embedding | 1024차원 벡터 |
| document | 번역된 텍스트 |
| metadata.start | 시작 시간(초) |
| metadata.end | 종료 시간(초) |
| metadata.original | 원본 텍스트 |
| metadata.ts | `MM:SS` 타임스탬프 |

- 컬렉션 이름: `edu_영상명` (한글 자동 변환 처리)
- 중복 실행 시 upsert로 덮어쓰기

**출력 파일**: `data/vector_db/chroma.sqlite3`, `segments.json`

---

## STEP 7 — RAG 챗봇 (선택)

**파일**: `modules/rag.py` → `RAGChatbot.chat()` / `query()`

`--chat` 또는 `--chat-only` 옵션 시 실행

**질의응답 흐름**:
```
질문 입력
    │
    ▼
질문 임베딩 (Qwen3-Embedding-0.6B)
    │
    ▼
ChromaDB 코사인 유사도 검색 (상위 5개)
    │
    ▼
거리 임계값(0.60) 필터링 (관련성 낮은 청크 제거)
    │
    ├── 관련 청크 없음 → "정보 없음" 고정 응답 반환
    │
    ▼
LLM 답변 생성 (Qwen3-4B)
    │  프롬프트: 시스템 규칙 + 참고자료(타임스탬프 포함) + 질문
    ▼
답변 출력 (타임스탬프 포함)
```

**Hallucination 방지 장치**:
1. 시스템 프롬프트: "참고 자료 외 추측 금지"
2. 코사인 거리 0.60 초과 청크 제외
3. 관련 청크 없으면 고정 문장만 반환
4. max_new_tokens=512, temperature=0.1 (창의적 생성 억제)

---

## VRAM 관리 전략

8GB GPU에서 모든 모델을 순차 실행:

```
STT 로드 (2.0GB)
    → STT 완료 후 VRAM 해제
LLM 로드 (2.5GB)
    → LLM 완료 후 VRAM 해제
TTS 로드 (0.7GB)
    → TTS 완료 후 VRAM 해제
Embed 로드 (1.1GB)
    → Embed 완료 후 VRAM 해제
```

각 단계 후 `torch.cuda.empty_cache()` 호출로 예약 메모리도 반환

---

## 파일 구조

```
edu_pipeline/
├── main.py                  # 파이프라인 진입점 + CLI
├── config.py                # 전체 설정 (모델명, 경로, 파라미터)
│
├── modules/
│   ├── stt.py               # 음성인식 (faster-whisper)
│   ├── llm.py               # 번역 + 요약 (Qwen3-4B)
│   ├── tts.py               # AI 더빙 (F5-TTS)
│   ├── embedder.py          # 텍스트 임베딩 (Qwen3-Embedding)
│   └── rag.py               # RAG 챗봇 (ChromaDB + LLM)
│
├── utils/
│   ├── audio.py             # ffmpeg 래퍼 (추출, 합성, SRT)
│   ├── timestamp_sync.py    # 오디오 스트레칭 + 트랙 합성
│   └── memory.py            # VRAM 모니터링 + 해제
│
└── data/
    ├── input/               # 입력 영상 넣는 곳
    ├── output/영상명/        # 모든 출력물 저장
    │   ├── extracted_audio.wav
    │   ├── subtitles_original.srt
    │   ├── subtitles_ko.srt
    │   ├── summary_child.txt
    │   ├── summary_teen.txt
    │   ├── summary_adult.txt
    │   ├── ref_clip.wav          # 자동 추출된 화자 참조 음성
    │   ├── tts_segments/         # 세그먼트별 TTS 음성
    │   ├── dubbed_audio.wav
    │   ├── 영상명_dubbed.mp4
    │   └── segments.json
    └── vector_db/
        └── chroma.sqlite3   # 벡터 DB
```

---

## CLI 옵션 정리

```bash
# 기본 실행 (한국어 영상)
python main.py -i data/input/lecture.mp4

# TTS 스킵 (자막 + 요약만)
python main.py -i data/input/lecture.mp4 --skip-tts

# 영어 강의 → 한국어 번역 + 더빙
python main.py -i data/input/lecture.mp4 --lang en --target-lang ko

# 직접 참조 음성 지정 (화자 클로닝)
python main.py -i data/input/lecture.mp4 --ref-audio speaker.wav --ref-text "안녕하세요."

# 파이프라인 완료 후 챗봇 실행
python main.py -i data/input/lecture.mp4 --chat

# 챗봇만 실행 (이미 인덱싱된 경우)
python main.py --chat-only --collection lecture
```

---

## 사용 모델 요약

| 단계 | 모델 | VRAM | 역할 |
|------|------|------|------|
| STT | faster-whisper large-v3-turbo | ~2.0GB | 음성 → 텍스트 |
| LLM | Qwen/Qwen3-4B (4bit) | ~2.5GB | 번역 + 요약 |
| TTS | F5-TTS F5TTS_v1_Base | ~0.7GB | 텍스트 → 음성 |
| Embed | Qwen/Qwen3-Embedding-0.6B | ~1.1GB | 텍스트 → 벡터 |
| RAG DB | ChromaDB (로컬) | 0GB | 벡터 검색 |

---

---

# 코드 레벨 상세 설명

---

## config.py — 전체 설정 관리

```python
_DEFAULT_MODE = "mac" if platform.system() == "Darwin" else "pc"
MODE = os.environ.get("EDU_PIPELINE_MODE", _DEFAULT_MODE).lower()
```
- 실행 환경(Mac/PC)을 자동 감지
- 환경변수 `EDU_PIPELINE_MODE`로 수동 오버라이드 가능
- Mac → MLX 백엔드, PC → CUDA + bitsandbytes 백엔드 자동 선택

```python
@dataclass
class _LLMConfig:
    model_id: str       = "Qwen/Qwen3-4B"
    load_in_4bit: bool  = field(default_factory=lambda: MODE == "pc")
    max_new_tokens: int = 1024
    temperature: float  = 0.1
```
- `@dataclass`를 사용해 설정을 타입 안전하게 관리
- `field(default_factory=lambda: ...)` 패턴으로 MODE에 따라 값이 동적 결정
- `cfg = Config()` 한 줄로 전체 설정 인스턴스 생성, 모든 모듈에서 `from config import cfg`로 공유

---

## utils/audio.py — ffmpeg 래퍼

### `_run()` — subprocess 공통 실행기

```python
def _run(cmd: list[str], label: str = "ffmpeg") -> None:
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(...)
```
- 모든 ffmpeg/ffprobe 호출의 공통 래퍼
- `capture_output=True` → stdout/stderr 캡처 (터미널 출력 억제)
- `encoding="utf-8", errors="replace"` → 한글 경로 처리 (Windows cp949 오류 방지)
- 실패 시 명령어와 stderr 마지막 800자를 포함한 상세 오류 발생

### `extract_audio()` — 오디오 추출

```python
_run([
    "ffmpeg", "-y", "-i", video_path,
    "-vn",                    # 비디오 스트림 제거
    "-acodec", "pcm_s16le",   # 무압축 PCM 16bit
    "-ar", "16000",           # 16kHz (Whisper 최적)
    "-ac", "1",               # 모노 채널
    output_path,
])
```
- `-y`: 기존 파일 덮어쓰기 (재실행 시 오류 방지)
- `pcm_s16le`: Whisper가 요구하는 포맷 (16bit Little Endian PCM)
- 스테레오 → 모노 변환으로 파일 크기 절반

### `get_video_duration()` — 영상 길이 측정

```python
result = subprocess.run([
    "ffprobe", "-v", "quiet",
    "-print_format", "json",
    "-show_format",
    video_path,
], ...)
info = json.loads(result.stdout)
return float(info["format"]["duration"])
```
- ffprobe로 영상 메타데이터를 JSON으로 출력
- `format.duration` 필드에서 초 단위 길이 추출
- TTS 트랙 합성 시 전체 길이 버퍼 크기 결정에 사용

### `extract_ref_clip()` — 화자 참조 음성 자동 추출

```python
candidates = [
    s for s in segments
    if 5.0 <= (s["end"] - s["start"]) <= 15.0 and len(s["text"]) >= 15
]
best = max(candidates, key=lambda s: s["end"] - s["start"])
```
- STT 결과에서 5~15초 길이, 텍스트 15자 이상인 세그먼트 필터링
- 그 중 가장 긴 구간 선택 (F5-TTS 클로닝 품질 최적화)
- ffmpeg `-ss`/`-to`로 정확한 구간 추출, 24kHz로 변환 (F5-TTS 입력 포맷)

### `mix_dubbed_into_video()` — 더빙 영상 합성

```python
"-filter_complex",
(
    f"[0:a]volume={original_volume}[orig];"   # 원본 8%로 낮춤
    "[orig][1:a]amix=inputs=2:duration=first:normalize=0[aout]"  # 두 트랙 믹스
),
"-map", "0:v",        # 비디오는 원본 그대로
"-map", "[aout]",     # 오디오는 믹스 결과
"-c:v", "copy",       # 비디오 재인코딩 없음 (빠름)
"-c:a", "aac", "-b:a", "192k",  # 오디오만 AAC로 인코딩
```
- `amix` 필터로 두 오디오 트랙을 실시간 믹싱
- `duration=first`: 원본 영상 길이에 맞춤
- `normalize=0`: 자동 볼륨 정규화 비활성화 (더빙 음성 볼륨 유지)
- `-c:v copy`: 비디오 스트림을 복사만 해서 화질 손실 없이 빠르게 처리

### `export_srt()` — SRT 자막 파일 생성

```python
def _fmt_srt_time(secs: float) -> str:
    h  = int(secs // 3600)
    m  = int((secs % 3600) // 60)
    s  = int(secs % 60)
    ms = int(round((secs % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```
- SRT 표준 형식 `00:00:00,000 --> 00:00:03,500`으로 변환
- 각 세그먼트를 번호 / 타임코드 / 텍스트 / 빈줄 구조로 저장

---

## modules/stt.py — 음성인식

### `transcribe()` — Whisper 추론

```python
model = WhisperModel(
    cfg.stt.model_size,      # "large-v3-turbo"
    device="cuda",
    compute_type="float16",  # 반정밀도 → VRAM 절감 + 속도 향상
    num_workers=2,           # 데이터 로딩 병렬 워커
)
```
- `faster-whisper`는 OpenAI Whisper를 CTranslate2로 재구현한 최적화 버전
- `float16` 추론으로 FP32 대비 VRAM 절반, 속도 2배
- `large-v3-turbo`: v3 대비 파라미터 40% 감소, 속도 2-3배 향상

```python
segments_gen, info = model.transcribe(
    audio_path,
    beam_size=3,                    # 빔 서치 후보 수 (많을수록 정확하지만 느림)
    best_of=1,                      # 후보 샘플링 수 (1 = 가장 빠름)
    word_timestamps=False,          # 단어 단위 타임스탬프 비활성화
    vad_filter=True,                # Voice Activity Detection 활성화
    vad_parameters={
        "min_silence_duration_ms": 300,  # 300ms 이상 무음 → 세그먼트 분리
        "speech_pad_ms": 100,            # 음성 앞뒤 100ms 여유
    },
    condition_on_previous_text=True,     # 이전 문맥 반영 (더 자연스러운 전사)
    compression_ratio_threshold=2.4,     # 반복 텍스트 감지 임계값
    no_speech_threshold=0.6,             # 음성 없음 판단 임계값
)
```
- `transcribe()`는 **제너레이터**를 반환 → 메모리 효율적 (전체 로드 없이 순차 처리)
- VAD 필터가 무음 구간을 미리 제거 → 실제 음성만 Whisper에 전달 → 처리 시간 단축
- `condition_on_previous_text`: 이전 세그먼트 텍스트를 컨텍스트로 제공 → 문맥 연속성 유지

```python
for seg in segments_gen:
    segments.append({
        "start": round(seg.start, 3),
        "end":   round(seg.end,   3),
        "text":  seg.text.strip(),
        "words": words,
    })
del model
clear_vram()
```
- 제너레이터를 순회하며 딕셔너리 리스트로 변환
- `round(..., 3)`: 소수점 3자리까지만 저장 (ms 단위)
- **완료 즉시 `del model` + `clear_vram()`** → 다음 단계(LLM)를 위한 VRAM 확보

---

## modules/llm.py — 번역 + 요약

### 4-bit 양자화 로드 `_load_pc()`

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,   # 연산은 FP16으로 수행
    bnb_4bit_use_double_quant=True,         # 이중 양자화: 양자화 상수도 4bit로 압축
    bnb_4bit_quant_type="nf4",             # NF4: 정규분포 데이터에 최적화된 4bit
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda",          # GPU에 자동 배치
    torch_dtype=torch.float16,  # 가중치 저장 포맷
)
model.eval()  # 드롭아웃, 배치 정규화 비활성화 → 추론 전용 모드
```
- NF4(Normal Float 4): 신경망 가중치가 정규분포를 따른다는 점을 활용한 최적 4bit 표현
- 이중 양자화: 양자화에 쓰이는 스케일 상수까지 한 번 더 압자화 → 추가 메모리 절감
- 결과: 원본 FP16 대비 메모리 75% 절감, 정확도 손실 최소화

### 채팅 템플릿 적용 `_infer_pc()`

```python
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,   # Qwen3 thinking 모드 비활성화 (번역/요약 불필요)
)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():        # 그래디언트 계산 비활성화 → 메모리 절감
    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,         # 그리디 디코딩 (확정적 출력)
        pad_token_id=tokenizer.eos_token_id,
    )
generated = out[0][inputs["input_ids"].shape[1]:]  # 입력 부분 제거, 생성 부분만 추출
```
- `apply_chat_template`: 모델별 특수 토큰 형식 자동 적용 (`<|im_start|>user...`)
- `enable_thinking=False`: Qwen3의 사고 과정 토큰 생성 비활성화 → 빠른 출력
- `do_sample=False`: 번역/요약은 창의적 출력 불필요 → 그리디로 가장 확률 높은 토큰만 선택

### 배치 번역 `translate_segments()`

```python
for batch_start in range(0, total, batch_size):  # 10개씩 슬라이싱
    batch = segments[batch_start:batch_start + batch_size]
    numbered = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(batch))
    prompt = f"아래 {n}개의 문장을 각각 자연스러운 {lang_name}로 번역하세요.\n..."

    raw = self._infer(prompt, 256 * n)  # 최대 토큰 = 문장 수 × 256

    # 정규식 파싱: "1. 번역문\n2. 번역문" 형식 추출
    for m in re.finditer(r"(\d+)\.\s*(.+?)(?=\n\d+\.|$)", raw, re.DOTALL):
        idx = int(m.group(1)) - 1
        parsed[idx] = m.group(2).strip()
```
- `range(0, total, 10)`: 0, 10, 20, ... 인덱스로 10개씩 배치 처리
- `re.DOTALL`: `.`이 줄바꿈도 매칭 → 다줄 번역문 처리
- `(?=\n\d+\.|$)`: 다음 번호 또는 문자열 끝까지를 번역문으로 캡처 (룩어헤드)
- 파싱 실패한 인덱스는 `parsed.get(i, seg["text"])`로 원문 폴백

---

## modules/tts.py — AI 더빙

### 기본 참조 음성 로드 `_get_default_ref()`

```python
def _get_default_ref():
    from importlib.resources import files
    return (
        str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        "Some call me nature, others call me mother nature.",
    )
```
- `importlib.resources.files()`: 패키지 내부 파일을 경로 하드코딩 없이 접근
- 가상환경 내 f5_tts 패키지에 번들된 영어 기본 음성 사용
- 반환값: (음성 파일 경로, 해당 음성의 텍스트) 튜플

### GPU 추론 `_infer()`

```python
def _infer(self, text: str):
    wav, sr, _ = self._tts.infer(
        ref_file=self._ref_audio,   # 참조 음성 (화자 클로닝 소스)
        ref_text=self._ref_text,    # 참조 음성의 텍스트 (정렬용)
        gen_text=text,              # 생성할 텍스트
        target_rms=0.1,             # 목표 RMS 에너지 (볼륨 정규화)
        nfe_step=self._nfe_step,    # Flow Matching 추론 스텝 (8 = 빠름)
    )
    return wav, sr
```
- F5-TTS는 Flow Matching 기반 → `nfe_step`이 적을수록 빠르지만 음질 저하
- `ref_file` + `ref_text`로 화자 목소리 특성 추출 → 동일 화자로 생성
- `target_rms`: 생성 음성의 에너지를 참조 음성에 맞게 정규화

### GPU + CPU 병렬 오버랩 `synthesize_all()`

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    pending: dict[int, Future] = {}

    for i, seg in enumerate(segments):
        # 2개 앞 세그먼트의 CPU 후처리 완료 확인
        if i - 2 in pending:
            fut = pending.pop(i - 2)
            fut.result()  # 완료 대기 (blocking)

        # GPU 추론 (동기 - GPU는 한 번에 하나만)
        wav, sr = self._infer(text)

        # CPU 후처리 비동기 제출 (다음 GPU 추론과 겹침)
        pending[i] = executor.submit(
            self._postprocess, wav, sr, duration, out_path
        )
```
- **핵심 아이디어**: GPU 추론(동기) 도중 이전 세그먼트의 CPU 후처리(비동기)가 동시 실행
- `pending` 딕셔너리로 미완료 Future 관리
- `i - 2` 시점에 완료 확인 → 2개의 CPU 후처리가 항상 파이프라인에 유지
- GPU는 동기 실행 (CUDA 연산은 GIL 해제 → Python 스레드와 병렬 가능)

### CPU 후처리 `_postprocess()`

```python
@staticmethod
def _postprocess(wav, sr, target_duration, output_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, wav, sr)           # numpy 배열 → 임시 WAV 파일
        stretch_audio_to_duration(...)        # 길이 맞춤
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)               # 임시 파일 반드시 삭제
```
- `@staticmethod`: 인스턴스 상태 불필요 → ThreadPoolExecutor에서 안전하게 호출
- `try/finally`: 예외 발생해도 임시 파일 반드시 정리

---

## utils/timestamp_sync.py — 오디오 시간 동기화

### `stretch_audio_to_duration()` — 길이 맞춤

```python
y, file_sr = sf.read(input_path)   # soundfile: 빠른 WAV I/O (C 백엔드)
y = y.astype(np.float32)

ratio = target_duration / actual_duration  # >1: 늘림, <1: 줄임

if abs(ratio - 1.0) < 0.02:
    sf.write(output_path, y, sr)   # 2% 이내 차이 → 스트레칭 생략
    return
```
- `soundfile.read()`: librosa.load() 대비 5배 빠름 (리샘플링 없음)
- 2% 이내 차이는 스트레칭 없이 그대로 저장 (불필요한 연산 방지)

```python
clamped = max(min_ratio, min(ratio, max_ratio))  # [0.5, 2.5] 범위 제한

stretched = librosa.effects.time_stretch(
    y,
    rate=1.0 / clamped,  # rate > 1이면 빠르게, < 1이면 느리게
    n_fft=512,           # FFT 윈도우 크기 (작을수록 빠름, 품질 저하)
)
```
- `librosa.effects.time_stretch`: STFT 위상 보코더 알고리즘
  - 신호를 주파수 영역으로 변환 → 위상 유지하며 길이 조정 → 시간 영역으로 역변환
  - `n_fft=512`: 기본값 2048 대비 4배 빠름 (짧은 구간 처리에 적합)
- `rate=1.0/clamped`: librosa의 rate 파라미터는 역수 관계 (rate=2.0 → 2배 빠름)

```python
target_samples = int(sr * target_duration)
if len(stretched) < target_samples:
    stretched = np.pad(stretched, (0, target_samples - len(stretched)))
else:
    stretched = stretched[:target_samples]

peak = np.abs(stretched).max()
if peak > 0.99:
    stretched = stretched * (0.95 / peak)  # 클리핑 방지 정규화
```
- 스트레칭 후 샘플 수가 목표와 정확히 일치하지 않을 수 있음 → 패딩/트리밍으로 보정
- 피크 > 0.99 → 0.95로 정규화 (DAC 클리핑 방지)

### `merge_dubbed_audio()` — 전체 트랙 합성

```python
total_samples = int(sr * total_duration)
merged = np.zeros(total_samples, dtype=np.float32)  # 전체 길이 무음 버퍼

for seg in segments:
    start_sample = int(seg["start"] * sr)            # 타임스탬프 → 샘플 위치
    end_sample   = min(start_sample + len(y), total_samples)
    merged[start_sample:end_sample] += y[:copy_len]  # 해당 위치에 덧씌우기

peak = np.abs(merged).max()
if peak > 0.98:
    merged = merged * (0.95 / peak)  # 최종 정규화
```
- 전체 영상 길이만큼의 무음 numpy 배열 생성
- 각 세그먼트를 타임스탬프 위치(`start * sr` 샘플)에 삽입 (`+=`)
- 세그먼트 사이 빈 공간은 자동으로 무음(0.0) 유지
- 마지막에 전체 피크 정규화 → 오디오 클리핑 방지

---

## modules/embedder.py — 텍스트 임베딩

### Mean Pooling `_mean_pool()`

```python
def _mean_pool(self, token_embeddings, attention_mask):
    # attention_mask: 실제 토큰 위치 1, 패딩 위치 0
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)  # 패딩 제외 합산
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)     # 패딩 아닌 토큰 수
    return summed / counts                               # 평균
```
- **Mean Pooling**: 문장의 모든 토큰 임베딩을 평균내어 단일 벡터로 압축
- 패딩 토큰(0)은 mask로 제외 → 실제 토큰만 평균
- `clamp(min=1e-9)`: 0으로 나누기 방지

### 배치 임베딩 `embed()`

```python
for i in range(0, len(texts), batch_size):  # 32개씩 배치
    batch = texts[i : i + batch_size]
    encoded = self._tokenizer(
        batch,
        padding=True,        # 배치 내 최장 문장 길이로 패딩
        truncation=True,     # 512 토큰 초과 시 자름
        max_length=512,
        return_tensors="pt", # PyTorch 텐서로 반환
    ).to(self._device)

    with torch.no_grad():
        output = self._model(**encoded)

    pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
    normalized = F.normalize(pooled, p=2, dim=1)  # L2 정규화 → 단위 벡터
    all_vectors.extend(normalized.cpu().float().tolist())
```
- `padding=True`: 가변 길이 문장을 동일한 길이로 맞춰 배치 처리 가능
- `last_hidden_state`: 마지막 Transformer 레이어의 출력 (각 토큰별 768/1024차원 벡터)
- `F.normalize(p=2)`: L2 정규화 → 벡터 크기를 1로 통일 → 코사인 유사도 = 내적

---

## modules/rag.py — RAG 챗봇

### 컬렉션 이름 정규화 `_sanitize_name()`

```python
def _sanitize_name(name: str) -> str:
    import re, hashlib
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)   # 허용 문자 외 → 언더스코어
    sanitized = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized)  # 앞 비영숫자 제거
    sanitized = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized)  # 뒤 비영숫자 제거
    if len(sanitized) < 3:
        sanitized = "col" + hashlib.md5(name.encode()).hexdigest()[:8]
    return sanitized[:512]
```
- ChromaDB 규칙: 영숫자+`._-`만 허용, 시작/끝은 반드시 영숫자
- 한글 파일명(`샘플강의영상`) → `_____` → 앞뒤 제거 → 짧으면 MD5 해시로 고유 이름 생성
- `hashlib.md5`: 원본 이름의 해시 → 이름이 달라도 충돌 없는 고유 ID 보장

### 세그먼트 인덱싱 `index_segments()`

```python
self._collection.upsert(
    ids=ids,              # ["seg_00001", "seg_00002", ...]
    embeddings=embeddings, # [[0.12, -0.34, ...], ...]  1024차원
    documents=documents,   # ["번역된 텍스트", ...]
    metadatas=metadatas,   # [{"start": 0.0, "ts": "00:00"}, ...]
)
```
- `upsert`: insert + update → 같은 id 존재 시 덮어쓰기 (중복 실행 안전)
- ChromaDB는 SQLite + HNSW 인덱스로 로컬 파일에 저장 → 서버 불필요

### 유사도 검색 `_retrieve()`

```python
q_vec = embedder.embed([question])[0]  # 질문 임베딩

results = self._collection.query(
    query_embeddings=[q_vec],
    n_results=min(cfg.rag.top_k, n_stored),  # 최대 5개 검색
    include=["documents", "metadatas", "distances"],
)

for doc, meta, dist in zip(...):
    if dist > cfg.rag.distance_threshold:  # 코사인 거리 0.60 초과 → 제외
        continue
    chunks.append({"doc": doc, "meta": meta, "dist": dist})
```
- ChromaDB는 코사인 거리(0=동일, 2=반대)로 유사도 측정
- `distance_threshold=0.60`: 거리 0.60 초과 = 관련성 낮음 → 제외
- L2 정규화된 벡터끼리의 코사인 거리 = 1 - 코사인 유사도

### 답변 생성 `query()`

```python
context_lines = [
    f"[{c['meta']['ts']}] {c['doc']}"   # "[03:24] 번역된 텍스트"
    for c in chunks
]
context = "\n".join(context_lines)

prompt = (
    f"{_SYSTEM_PROMPT}\n\n"
    f"[참고 자료]\n{context}\n\n"
    f"[질문]\n{question}\n\n"
    f"[답변]"
)
answer = llm._infer(prompt, 512)
```
- 검색된 청크에 타임스탬프를 붙여 프롬프트에 삽입
- LLM은 참고 자료만 보고 답변 → Hallucination 방지
- 시스템 프롬프트 규칙: "참고 자료에 없으면 정해진 문장만 출력"
