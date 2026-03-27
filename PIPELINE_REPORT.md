# edu_pipeline 기술 보고서

> 교육 영상 자동 처리 파이프라인 — 설계, 구현, 동작 원리

---

## 1. 프로젝트 개요

### 1.1 목적

교육 영상에서 자막을 추출하고, 다국어 번역 및 학습자 연령별(초등/중등/고등/대학생) 수준에 맞춘 요약을 생성하며, AI 더빙까지 수행하는 **엔드투엔드 자동화 파이프라인**입니다. 추가로 영상 내용 기반의 **제한된 RAG 챗봇**을 제공합니다.

### 1.2 지원 환경

| 항목 | PC (Windows) | Mac (Apple Silicon) |
|------|-------------|---------------------|
| GPU | NVIDIA CUDA (8GB+) | Apple Metal (MPS) |
| 프레임워크 | PyTorch + CUDA | MLX / PyTorch + MPS |
| LLM 백엔드 | llama.cpp (GGUF) → bitsandbytes 폴백 | mlx-lm |

### 1.3 지원 언어 및 연령

- **언어**: 한국어, 영어, 베트남어, 라오스어
- **연령**: 초등학생, 중학생, 고등학생, 대학생

---

## 2. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                    edu_pipeline                           │
│                                                          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │
│  │  STEP 1  │──▶│  STEP 2  │──▶│  STEP 3  │──▶│  STEP 4  │ │
│  │  오디오   │   │   STT    │   │   LLM    │   │   TTS    │ │
│  │  추출    │   │  Whisper  │   │  Qwen3   │   │  F5-TTS  │ │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘ │
│       │                            │               │      │
│       │                            ▼               │      │
│       │              ┌──────────────────┐          │      │
│       │              │  연령별 요약 4종   │          │      │
│       │              │  + 용어 해설       │          │      │
│       │              └──────────────────┘          │      │
│       │                                            │      │
│       ▼                                            ▼      │
│  ┌─────────┐   ┌─────────┐   ┌─────────────────────┐    │
│  │  STEP 5  │──▶│  STEP 6  │──▶│     STEP 7 (선택)    │    │
│  │ Embedding│   │ ChromaDB │   │    RAG 챗봇 실행     │    │
│  │ Qwen3-E  │   │  인덱싱   │   │  (영상 내용 기반 QA) │    │
│  └─────────┘   └─────────┘   └─────────────────────┘    │
│                                                          │
│  ※ 각 모델은 순차 로드 → 사용 → VRAM 해제 (8GB GPU 최적화)  │
└──────────────────────────────────────────────────────────┘
```

### 2.1 VRAM 순차 관리 전략

8GB GPU에서 모든 모델을 동시에 로드할 수 없으므로, **한 번에 하나의 모델만 GPU에 올리고 작업 완료 후 즉시 해제**합니다.

```
시간 →
STT 로드 (1.5GB) ████░░░░░░░░░░░░░░░░
    → STT 완료 후 VRAM 해제
LLM 로드 (4.5GB) ░░░░████████░░░░░░░░
    → LLM 완료 후 VRAM 해제
TTS 로드 (0.7GB) ░░░░░░░░░░░░████░░░░
    → TTS 완료 후 VRAM 해제
Embed 로드 (1.1GB) ░░░░░░░░░░░░░░░░████
```

---

## 3. 각 단계별 상세 동작 원리

### STEP 1: 오디오 추출

**파일**: `utils/audio.py` → `extract_audio()`

```python
# ffmpeg를 사용해 영상에서 16kHz 모노 WAV 오디오를 추출
ffmpeg -i input.mp4 -ar 16000 -ac 1 -vn output.wav
```

- **왜 16kHz?**: Whisper 모델의 입력 요구사항이 16kHz 모노 오디오
- **왜 WAV?**: 무손실 포맷으로 STT 정확도 보장
- 영상 길이(duration)도 이 단계에서 측정하여 이후 TTS 타임스탬프 동기화에 활용

---

### STEP 2: STT (Speech-to-Text)

**파일**: `modules/stt.py` → `transcribe()`
**모델**: `faster-whisper large-v3-turbo` (VRAM ~1.5GB)

```python
# faster-whisper는 CTranslate2 기반 최적화 Whisper 구현
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
segments, info = model.transcribe(audio_path, beam_size=3, vad_filter=True)
```

**동작 과정**:
1. **VAD(Voice Activity Detection)** 필터가 무음 구간을 자동 제거 → 처리 시간 단축
2. **언어 자동 감지**: 첫 30초를 분석하여 언어 판별 (수동 지정도 가능)
3. **beam search** (beam_size=3)로 후보군 중 최적 텍스트 선택
4. 각 세그먼트에 `start`, `end` 타임스탬프와 `text` 포함
5. 결과를 **SRT 자막 파일**로 저장

**출력 예시**:
```json
[
  {"start": 0.0, "end": 3.52, "text": "안녕하세요, 오늘 강의를 시작하겠습니다."},
  {"start": 3.52, "end": 7.84, "text": "오늘의 주제는 인공지능의 기초입니다."}
]
```

**VRAM 해제**: 완료 즉시 `model` 객체 삭제 + `torch.cuda.empty_cache()`

---

### STEP 3: LLM (번역 + 연령별 요약)

**파일**: `modules/llm.py` → `LLMProcessor`
**모델**: `Qwen3-8B` (GGUF Q4_K_M ~5GB / BnB NF4 ~4.5GB)

#### 3-A. 번역 (translate_segments)

세그먼트를 **20개씩 배치**로 묶어 한 번에 번역합니다.

```python
# 배치 번역 프롬프트 예시
prompt = """아래 20개의 문장을 각각 자연스러운 한국어로 번역하세요.
반드시 '번호. 번역문' 형식으로만 출력하세요.

1. Hello, let's start today's lecture.
2. Today's topic is the basics of AI.
...
20. Thank you for listening."""
```

**왜 배치 번역?**: 문장 하나씩 번역하면 330개 세그먼트 × 330번 LLM 호출 → 매우 느림. 20개씩 묶으면 **17번**으로 끝남.

**파싱 로직**: 정규표현식으로 `번호. 번역문` 형식을 파싱하고, 실패한 항목은 원문 유지.

```python
_NUMBERED_LINE_RE = re.compile(r"(\d+)\s*[.\)]\s*(.+?)(?=\n\s*\d+\s*[.\)]|\Z)", re.DOTALL)
```

#### 3-B. 연령별 요약 + 용어 해설 (summarize_all_ages)

4개 연령(초등/중등/고등/대학생)별로 **각각 독립 호출**합니다.

```python
# 초등학생용 프롬프트 (요약 부분)
"""다음 교육 영상 내용을 요약해 주세요.
요약 조건: 초등학생(8~13세)도 이해할 수 있도록 쉽고 친근한 단어를 사용하고,
짧은 문장으로 핵심만 설명해 주세요. 어려운 용어는 비유로 풀어주세요.

출력 형식:
1. 먼저 요약문을 작성하세요.
2. 요약문 아래에 '---'을 쓰고, 그 아래에 [용어 해설] 섹션을 추가하세요.
   본문에 사용된 어려운 단어나 개념을 5개 이내로 골라,
   초등학생이 이해할 수 있도록 쉬운 비유와 함께 설명해 주세요.
   각 용어는 '• 용어: 설명' 형식으로 작성하세요."""
```

**출력 예시 (초등학생)**:
```
인공지능은 컴퓨터가 사람처럼 생각하고 배우는 기술이에요.
마치 강아지가 "앉아"를 배우는 것처럼, 컴퓨터도 많은 예시를 보면서 배워요.

---
[용어 해설]
• 인공지능: 컴퓨터가 사람처럼 생각하는 기술. 로봇 친구를 만드는 것과 비슷해요.
• 알고리즘: 문제를 풀기 위한 순서. 요리 레시피처럼 순서대로 따라하는 거예요.
• 데이터: 컴퓨터가 공부하는 재료. 교과서에 있는 글자나 그림 같은 거예요.
```

**왜 4번 개별 호출?**: 한 번에 4개 연령을 통합 생성하면 파싱 실패 시 최대 8번 재시도 → 개별 4번이 더 안정적이고 빠름.

#### 3-C. Qwen3 Thinking 블록 최적화

Qwen3는 기본적으로 `<think>...</think>` 블록을 생성하여 추론합니다. 번역/요약에서는 불필요하므로 **thinking을 미리 닫아** 토큰 낭비를 방지합니다.

```python
def _build_qwen3_prompt(user_msg: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n/no_think\n{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n</think>\n"  # ← 미리 닫기
    )
```

**VRAM 해제**: 모든 파라미터를 CPU로 이동 → 삭제 → `gc.collect()` → `torch.cuda.empty_cache()`

---

### STEP 4: TTS (AI 더빙)

**파일**: `utils/tts_worker.py` (별도 프로세스), `modules/tts.py`
**모델**: `F5-TTS v1 Base` (VRAM ~0.7GB)

#### 왜 별도 프로세스?

STT/LLM이 사용한 CUDA 컨텍스트가 F5-TTS와 충돌하여 크래시가 발생합니다. 별도 프로세스로 실행하면 **깨끗한 CUDA 컨텍스트**를 보장합니다.

```python
# main.py에서 subprocess로 TTS 실행
proc = subprocess.run([sys.executable, "-m", "utils.tts_worker",
                       "--segments-json", seg_json, "--tts-dir", tts_dir])
```

**동작 과정**:
1. **화자 참조 음성 추출**: 영상에서 가장 긴 발화 구간을 자동 선택 (또는 사용자 지정)
2. **Voice Cloning**: F5-TTS가 참조 음성의 톤/스타일을 학습
3. **세그먼트별 TTS 생성**: 번역된 각 문장을 음성으로 변환
4. **타임스탬프 동기화**: 원본 세그먼트의 시작/종료 시간에 맞춰 속도 조절 (0.5x~2.5x)
5. **더빙 트랙 합성**: 모든 TTS 클립을 하나의 WAV로 합침
6. **영상 합성**: ffmpeg로 원본 영상 + 더빙 오디오 → 최종 더빙 영상

```python
# 타임스탬프 동기화 — 원본 구간 길이에 맞춰 TTS 속도 조절
original_duration = segment["end"] - segment["start"]
tts_duration = len(audio) / sample_rate
stretch_ratio = tts_duration / original_duration  # 1.0 = 동일 속도
# 0.5 ~ 2.5 범위 내에서만 스트레칭
```

---

### STEP 5: 임베딩

**파일**: `modules/embedder.py` → `EmbeddingProcessor`
**모델**: `Qwen3-Embedding-0.6B` (VRAM ~1.1GB)

각 세그먼트의 텍스트를 **벡터(고차원 숫자 배열)**로 변환합니다.

```python
# "인공지능은 컴퓨터가 사람처럼 생각하는 기술입니다"
# → [0.023, -0.156, 0.891, ..., 0.045]  (768차원 벡터)
```

**왜 Qwen3-Embedding?**: 한국어/영어/베트남어 등 다국어 텍스트를 하나의 벡터 공간에서 비교 가능. 의미가 비슷한 문장은 벡터도 가까움.

- **배치 처리**: 64개씩 묶어 한 번에 임베딩 → GPU 효율 극대화
- **max_length=512**: 세그먼트 하나가 512 토큰을 넘지 않으므로 충분

---

### STEP 6: ChromaDB 인덱싱

**파일**: `modules/rag.py` → `RAGChatbot.index_segments()`

임베딩된 세그먼트를 **ChromaDB (벡터 데이터베이스)**에 저장합니다.

```python
# 각 세그먼트를 벡터 DB에 저장
collection.add(
    ids=["seg_0", "seg_1", ...],
    embeddings=[[0.023, -0.156, ...], [0.045, 0.312, ...], ...],
    documents=["인공지능은...", "오늘의 주제는...", ...],
    metadatas=[{"start": 0.0, "end": 3.52}, ...]
)
```

**왜 ChromaDB?**: 로컬 파일 기반으로 서버 없이 동작, 코사인 유사도 검색을 네이티브 지원.

---

### STEP 7: RAG 챗봇 (선택)

**파일**: `modules/rag.py` → `RAGChatbot.chat()`

사용자 질문에 대해 **영상 내용에서만** 관련 정보를 검색하여 답변합니다.

```
사용자: "인공지능의 학습 방법에 대해 설명해줘"
     ↓
[1] 질문을 벡터로 변환 (Qwen3-Embedding)
     ↓
[2] ChromaDB에서 코사인 유사도 상위 5개 세그먼트 검색
     ↓
[3] 검색된 세그먼트를 컨텍스트로 LLM에 전달
     ↓
[4] LLM이 컨텍스트 기반으로만 답변 생성
     (영상에 없는 내용은 "해당 내용이 영상에 없습니다" 응답)
```

**환각 방지 전략**: 시스템 프롬프트에 "반드시 제공된 검색 결과만으로 답변하라"는 제약을 부여.

---

## 4. 사용 모델 요약

| 단계 | 모델 | 크기 | VRAM | 역할 |
|------|------|------|------|------|
| STT | faster-whisper large-v3-turbo | 1.5GB | ~1.5GB | 음성→텍스트 |
| LLM | Qwen3-8B (Q4_K_M GGUF) | 5GB | ~4.5GB | 번역, 요약, 챗봇 |
| TTS | F5-TTS v1 Base | 0.7GB | ~0.7GB | 음성 합성 + 화자 클로닝 |
| Embedding | Qwen3-Embedding-0.6B | 1.1GB | ~1.1GB | 텍스트→벡터 |
| Vector DB | ChromaDB | - | CPU | 벡터 검색 |

---

## 5. 출력물 구조

```
data/output/영상이름/
├── extracted_audio.wav          # 추출된 오디오
├── subtitles_original.srt       # 원본 자막 (SRT)
├── subtitles_ko.srt             # 번역 자막 (번역 시)
├── summary_elementary.txt       # 초등학생용 요약 + 용어 해설
├── summary_middle.txt           # 중학생용 요약 + 용어 해설
├── summary_high.txt             # 고등학생용 요약 + 용어 해설
├── summary_university.txt       # 대학생용 요약 + 용어 해설
├── dubbed_audio.wav             # AI 더빙 오디오
├── 영상이름_dubbed.mp4           # 더빙 합성 영상
├── segments.json                # 전체 세그먼트 매니페스트
├── ref_clip.wav                 # TTS 참조 음성 (자동 추출)
└── tts_segments/                # 개별 TTS 클립
    ├── seg_000.wav
    ├── seg_001.wav
    └── ...
```

---

## 6. 실행 방법

### 6-A. 명령줄 (CLI)

```bash
# 기본 실행
python main.py -i data/input/영상.mp4

# 영어 강의 → 한국어 번역 + 더빙 + 챗봇
python main.py -i lecture.mp4 --lang en --target-lang ko --chat

# TTS 건너뛰고 자막/요약만
python main.py -i lecture.mp4 --skip-tts

# 3단계(LLM)부터 재개
python main.py -i lecture.mp4 --start-from 3
```

### 6-B. GUI 런처 (launcher.pyw)

`launcher.pyw` 더블클릭 → 파일 선택 → 옵션 설정 → 실행

### 6-C. 웹 UI (web_ui.py)

```bash
python web_ui.py
# 브라우저에서 http://localhost:5000 자동 오픈
```

브라우저에서 파일 선택, 옵션 설정, 실시간 로그 확인, 단계별 진행 표시 제공.

---

## 7. 핵심 설계 원칙

| 원칙 | 구현 |
|------|------|
| **8GB VRAM 최적화** | 모델 순차 로드/해제, GGUF 4bit 양자화 |
| **안정성 우선** | TTS 별도 프로세스, 개별 요약 호출, 폴백 백엔드 |
| **재개 가능성** | `--start-from`으로 중간 단계부터 재실행 |
| **다국어 지원** | 한국어/영어/베트남어/라오스어 번역 |
| **교육 맞춤** | 4단계 연령별 요약 + 용어 해설 자동 생성 |
| **환각 방지** | RAG 챗봇이 영상 내용 기반으로만 응답 |

---

*최종 업데이트: 2026-03-27*
*LLM 모델: Qwen3-8B (Q4_K_M GGUF)*
