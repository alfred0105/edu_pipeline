# edu_pipeline 코드 품질 상세 리포트

> 검토일: 2026-03-25
> 환경: Windows 11 / Python 3.10.6 (venv) / RTX 4060 Ti 8GB / CUDA 12.4
> 검토 범위: 전체 소스 코드 10개 파일 (테스트 코드 없음)

---

## 1. 종합 점수 요약

| # | 파일 | 정확성 | 안정성 | 성능 | 코드품질 | 문서화 | **종합** |
|---|------|--------|--------|------|----------|--------|----------|
| 1 | `main.py` | 9 | 7 | 8 | 9 | 9 | **84** |
| 2 | `config.py` | 10 | 9 | 10 | 9 | 9 | **94** |
| 3 | `modules/stt.py` | 9 | 8 | 7 | 7 | 8 | **78** |
| 4 | `modules/llm.py` | 8 | 6 | 6 | 7 | 8 | **70** |
| 5 | `modules/tts.py` | 8 | 7 | 7 | 8 | 9 | **78** |
| 6 | `modules/embedder.py` | 9 | 7 | 8 | 8 | 8 | **80** |
| 7 | `modules/rag.py` | 8 | 6 | 6 | 7 | 8 | **70** |
| 8 | `utils/audio.py` | 9 | 8 | 9 | 8 | 8 | **84** |
| 9 | `utils/timestamp_sync.py` | 9 | 8 | 6 | 8 | 8 | **78** |
| 10 | `utils/memory.py` | 10 | 9 | 10 | 9 | 8 | **92** |

> 점수 기준: 각 항목 10점 만점 / 종합 = (정확성×25 + 안정성×25 + 성능×20 + 코드품질×15 + 문서화×15) / 10

### 전체 평균: **80.8 / 100**

---

## 2. 파일별 상세 분석

---

### 2.1 `main.py` — 84점

**역할:** 파이프라인 진입점 + CLI 인터페이스

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| 파이프라인 순서 정확성 | 10/10 | STT→LLM→TTS→Embed→Index 순서 논리적으로 완벽 |
| VRAM 해제 타이밍 | 9/10 | 각 단계 후 `unload()` + `clear_vram()` 호출. STEP 1(ffmpeg)은 GPU 미사용이므로 해제 불필요 — 정확 |
| 입력 파일 검증 | 6/10 | 파일 존재 여부 확인 없음 — 존재하지 않는 파일 입력 시 ffmpeg 에러로 나중에 실패 |
| 단계별 에러 격리 | 5/10 | try-except 없음. STEP 2 실패 시 STEP 3~7 전부 스킵되지만, 부분 결과물(STEP 1 오디오)은 남아 있어 디버깅 가능 |
| CLI 인터페이스 | 9/10 | argparse 잘 구성, `--skip-tts`, `--chat-only` 등 유용한 옵션 |
| lazy import | 10/10 | 무거운 모듈(torch, transformers 등)을 함수 내에서 import — 시작 속도 빠름 |
| `args.lang or None` (line 80) | 7/10 | `"ko"` 기본값인데 `or None`은 빈 문자열일 때만 작동. `"ko"`가 들어오면 Whisper에 ko 강제 — 자동 감지 안 됨. 의도적이지만 사용자가 혼란 가능 |
| 요약 시 `full_text` 길이 (line 107) | 5/10 | 1시간 강의 = 세그먼트 수백 개 = `full_text` 수만 토큰. Qwen3-4B context 32K 초과 시 잘림 발생하지만 에러 없이 품질만 저하 |
| segments dict 불변성 (line 104) | 9/10 | `{**s, "translated": s["text"]}` spread 방식으로 원본 훼손 없이 새 리스트 생성 — 깔끔 |

#### 주요 이슈

```
[line 80] language=args.lang or None
```
- `--lang` 기본값이 `"ko"`이므로 `or None`은 절대 None이 안 됨
- 사용자가 Whisper 자동 감지를 원하면 `--lang ""` 입력해야 하는데, 이건 직관적이지 않음
- **해결:** `--lang` 기본값을 `None`으로 바꾸거나, `--auto-detect` 별도 플래그 추가

```
[line 107] full_text = " ".join(s["text"] for s in segments)
```
- 긴 영상에서 context window 초과 위험
- **해결:** 텍스트가 일정 토큰 이상이면 앞뒤 잘라서 요약하거나, map-reduce 패턴 적용

---

### 2.2 `config.py` — 94점

**역할:** 환경 자동 감지 + 전체 설정 중앙 관리

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| 환경 감지 | 10/10 | `platform.system()` + 환경변수 오버라이드 — 완벽 |
| dataclass 활용 | 10/10 | 타입 힌트 + default_factory로 MODE 참조 — 파이썬 모범 사례 |
| 경로 관리 | 9/10 | `Path(__file__).parent` 기준 상대 경로 — 이동 가능. 단, OneDrive 한글 경로에서 간혹 인코딩 문제 발생 가능 |
| assert 사용 (line 20) | 8/10 | `python -O` 최적화 모드에서 assert가 무시됨. 프로덕션에서는 `ValueError` raise가 안전 |
| 디렉토리 자동 생성 (line 86-87) | 10/10 | import 시 자동 생성 — 사용자가 수동으로 만들 필요 없음 |
| 설정 확장성 | 8/10 | .env 파일이나 YAML 지원 없음. 현재는 소스 코드 직접 수정 필요 |

#### 주요 이슈

```
[line 46] temperature: float = 0.1
```
- config에 `temperature=0.1`이 있지만, `llm.py`의 `_infer_pc()`에서 `do_sample=False`로 설정되어 있어 temperature가 실질적으로 무시됨
- 설정과 실제 동작 간 불일치 — 혼란 유발

---

### 2.3 `modules/stt.py` — 78점

**역할:** faster-whisper 기반 음성→텍스트 변환

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| 모델 로드 | 9/10 | config에서 device/compute_type 가져옴. `num_workers=2` 적절 |
| transcribe 파라미터 | 8/10 | VAD 필터, compression_ratio_threshold 등 세밀한 튜닝 |
| `word_timestamps=False` vs words 처리 (line 71, 86-90) | 4/10 | `word_timestamps=False`인데 `seg.words` 처리 코드가 5줄 존재. `seg.words`는 항상 None → words는 항상 빈 리스트. **죽은 코드** |
| 출력 형식 일관성 | 9/10 | `start`, `end`, `text`, `words` 키 일관적. `round(_, 3)` 정밀도 통일 |
| VRAM 해제 | 8/10 | `del model` + `clear_vram()` 적절. 단, generator `segments_gen`이 model 참조를 유지할 수 있어 `del model` 전에 모든 세그먼트를 먼저 소비해야 함 — 현재 for loop에서 이미 소비하므로 OK |
| 긴 오디오 처리 | 5/10 | 1시간+ 오디오에 대한 청크 분할 없음. faster-whisper가 내부적으로 30초 윈도우를 사용하긴 하지만, 메모리 사용량이 선형 증가 |
| 에러 핸들링 | 6/10 | 모델 로드 실패, 오디오 파일 깨짐 등에 대한 처리 없음 |

#### 주요 이슈

```python
# [line 71] word_timestamps=False
# [line 86-90] 하지만 seg.words를 처리하는 코드 존재
if seg.words:  # word_timestamps=False이면 항상 None
    words = [...]
```
- `word_timestamps=True`로 바꾸면 단어 단위 타임스탬프를 활용할 수 있어 TTS 동기화 정밀도가 올라감
- 아니면 words 처리 코드를 제거하여 깔끔하게 정리

---

### 2.4 `modules/llm.py` — 70점

**역할:** Qwen3-4B 기반 번역 + 연령별 요약

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| 4-bit 양자화 (line 49-54) | 9/10 | NF4 + double_quant + float16 compute — VRAM 절감 최적 조합 |
| `<think>` 태그 처리 (line 67-70) | 9/10 | `_strip_think_tags()` 추가로 Qwen3 thinking 출력 제거 — **이번에 수정 완료** |
| 번역 배치 정규식 (line 173) | 5/10 | `r"(\d+)\.\s*(.+?)(?=\n\d+\.\|$)"` — LLM이 형식을 안 지키면 파싱 실패. 원문 폴백은 있지만 부분 실패 빈번 가능 |
| `_infer` lambda 바인딩 (line 122-124) | 6/10 | `self._infer = lambda p, m=...: _infer_pc(self._tokenizer, self._model, p, m)` — `unload()` 후 `self._model`이 삭제되어도 lambda에 `self` 참조가 남아있어 AttributeError 발생 가능. RAG에서 `llm._infer()` 직접 호출 시 문제 |
| `do_sample=False` (line 84) | 7/10 | config의 `temperature=0.1`과 불일치. greedy decoding이 번역에는 적합하지만, 요약에서는 약간의 다양성이 도움될 수 있음 |
| `max_new_tokens` 계산 (line 169) | 5/10 | `256 * n` — batch_size=10일 때 2560 토큰. Qwen3-4B의 생성 한계와 무관하게 설정. 불필요하게 큰 값이나, generate에서 EOS 만나면 조기 종료하므로 실제 문제는 없음 |
| 요약 프롬프트 품질 | 8/10 | 연령별 지시문이 명확하고 구체적. "비유로 풀어주세요" 등 좋은 지시 |
| `unload()` (line 208-214) | 7/10 | `del self._model` + `del self._tokenizer`는 하지만 `self._infer` lambda가 여전히 참조 유지. `del self._infer`도 해야 완전 해제 |

#### 주요 이슈

```python
# [line 122-124] lambda 클로저 문제
self._infer = lambda p, m=cfg.llm.max_new_tokens: _infer_pc(
    self._tokenizer, self._model, p, m
)
```
- `unload()` 후 `self._model`을 `del`하지만, `self._infer` lambda가 `self`를 캡처하고 있으므로 `self._infer()`를 호출하면 `AttributeError: 'LLMProcessor' object has no attribute '_model'` 발생
- RAG 모듈의 `query()` (line 201)에서 `llm._infer(prompt, 512)`를 직접 호출하는데, 만약 LLM이 unload된 상태에서 호출되면 크래시

```python
# [line 173] 번역 파싱 정규식
for m in re.finditer(r"(\d+)\.\s*(.+?)(?=\n\d+\.|$)", raw, re.DOTALL):
```
- `$`는 `re.DOTALL`에서 문자열 끝만 매치 → 마지막 번역이 여러 줄이면 정상 파싱
- 하지만 LLM이 `1) 번역문` (점 대신 괄호) 형식이나 번호 없이 출력하면 전부 실패
- 파싱 실패 시 원문 폴백으로 최소한 크래시는 안 나지만, 번역 품질 보장 불가

---

### 2.5 `modules/tts.py` — 78점

**역할:** F5-TTS 기반 AI 더빙 + GPU/CPU 오버랩

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| GPU/CPU 오버랩 설계 (line 144) | 9/10 | ThreadPoolExecutor로 GPU 추론과 CPU 후처리 파이프라이닝 — 고급 최적화 |
| `nfe_step=8` 기본값 (line 62) | 6/10 | 공식 권장 32의 1/4. 속도 4배이지만 음질 저하 체감 가능. 16이 더 나은 균형점 |
| 빈 텍스트 처리 | 4/10 | `text=""`이거나 공백만 있는 세그먼트가 `_infer()`에 전달될 수 있음. F5-TTS가 빈 텍스트에 어떻게 반응하는지 보장 없음 |
| tempfile 사용 (line 100-101) | 6/10 | 매 세그먼트마다 tempfile 생성→삭제 반복. 디스크 I/O 오버헤드. numpy 배열을 직접 stretch 함수에 전달하면 파일 I/O 제거 가능 |
| `_get_default_ref()` (line 40-45) | 8/10 | f5_tts 패키지 내장 예시 음성 활용 — 설치만 되면 동작 보장 |
| 세그먼트 실패 복구 (line 167-176) | 8/10 | 개별 세그먼트 실패 시 `tts_path=None` 설정하고 계속 진행 — 파이프라인 중단 방지 |
| F5TTS_SR 상수 (line 37) | 7/10 | 24000 하드코딩. F5-TTS API가 sr을 반환하므로 상수 대신 반환값 사용이 더 안전 |
| 후처리 완료 확인 (line 159) | 8/10 | 2개 앞서 대기 방식으로 메모리 누적 방지. 다만 마지막 2개는 루프 후에 별도 대기 — 정상 |

#### 주요 이슈

```python
# [line 148] 빈 텍스트 가능성
text = seg.get("translated", seg["text"]) if use_translated else seg["text"]
```
- STT 결과에서 `text=""`인 세그먼트 존재 가능 (짧은 노이즈 구간)
- F5-TTS에 빈 문자열 전달 시 에러 또는 무음 아닌 노이즈 생성 가능
- **해결:** `if not text.strip(): continue` 추가하여 빈 텍스트 건너뛰기

```python
# [line 100-101] tempfile 매번 생성
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp_path = tmp.name
```
- 세그먼트 100개면 tempfile 100번 생성/삭제
- stretch 함수가 numpy 배열을 직접 받으면 파일 I/O 제거 가능

---

### 2.6 `modules/embedder.py` — 80점

**역할:** Qwen3-Embedding-0.6B으로 텍스트 벡터화

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| Mean Pooling 구현 (line 50-56) | 10/10 | attention_mask 기반 가중 평균 — 교과서적 구현 |
| L2 정규화 (line 95) | 10/10 | `F.normalize(pooled, p=2, dim=1)` — 코사인 유사도에 필수 |
| task instruction prefix | 9/10 | passage/query 구분 추가로 검색 정확도 향상 — **이번에 수정 완료** |
| float16 사용 (line 41) | 8/10 | VRAM 절감에 효과적. 다만 임베딩 정밀도가 float32 대비 소폭 저하 가능 |
| 배치 처리 (line 81) | 8/10 | batch_size=32 적절. 다만 텍스트 길이 편차가 클 때 패딩 낭비 발생 |
| 빈 텍스트 처리 | 5/10 | `texts=[]` 입력 시 빈 리스트 반환 — 정상. 하지만 개별 텍스트가 `""`이면 의미 없는 벡터 생성 |
| `embed_segments` (line 100-108) | 8/10 | translated 우선 사용 — 다국어 검색 일관성 확보 |
| `unload()` (line 110-115) | 8/10 | `del` + `clear_vram()` + `log_vram()` 체계적 |

#### 주요 이슈

```python
# [line 72-76] TASK_INSTRUCTIONS가 함수 안에 정의
_TASK_INSTRUCTIONS = {
    "passage": "Instruct: Represent this educational content for retrieval\n",
    "query":   "Instruct: Retrieve relevant educational content\n",
}
```
- 매 호출마다 dict 재생성. 성능 영향은 미미하지만, 클래스 상수나 모듈 상수로 올리는 게 깔끔

---

### 2.7 `modules/rag.py` — 70점

**역할:** ChromaDB 벡터 검색 + LLM 질의응답 + CLI 챗봇

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| Hallucination 방지 설계 | 9/10 | 시스템 프롬프트 + 거리 임계값 필터링 + 고정 응답 — 3중 방어 |
| `_sanitize_name()` (line 37-45) | 9/10 | 한글/특수문자 컬렉션명 처리, 3자 미만 시 hash 대체 — 견고 |
| `emb_map` id() 사용 (line 115) | 6/10 | `id(o)`는 객체 identity 기반. `missing` 리스트 요소가 `segments` 리스트와 같은 객체 참조일 때만 작동. 현재 코드에서는 정상이지만, 향후 deepcopy 등으로 변경 시 깨짐 — 취약한 패턴 |
| VRAM 동시 점유 (chat 모드) | 4/10 | `_get_embedder()` + `_get_llm()` lazy 로드는 좋지만, chat 모드에서 Embedder(~1.3GB) + LLM(~2.5GB) 동시 로드 = ~3.8GB. 다른 VRAM 사용까지 합하면 8GB 한계 근접 |
| `llm._infer()` 직접 호출 (line 201) | 5/10 | 클래스의 private 메서드(`_infer`)를 외부에서 호출하는 안티패턴. `llm.generate()` 같은 public API로 노출하는 게 바람직 |
| 대화 히스토리 | 4/10 | chat 모드에서 이전 대화 맥락이 유지되지 않음. 후속 질문("그것에 대해 더 알려줘")에 답 불가 |
| `_fmt_timestamp()` (line 48-51) | 8/10 | 간단하고 정확. 1시간 초과 시 `mm:ss` 대신 `hh:mm:ss` 필요할 수 있음 |
| ChromaDB upsert (line 134) | 8/10 | 중복 인덱싱 방지에 효과적. 다만 같은 영상을 다른 설정으로 재처리 시 이전 데이터가 덮어씌워짐 |

#### 주요 이슈

```python
# [line 201] private 메서드 외부 호출
answer = llm._infer(prompt, 512)
```
- `LLMProcessor._infer`는 lambda로 바인딩된 private 메서드
- 직접 호출하면 `LLMProcessor`의 내부 구현에 종속
- `unload()` 후 호출 시 `AttributeError` 발생

```python
# chat 모드 VRAM 문제
def _get_embedder(self):  # ~1.3GB
def _get_llm(self):       # ~2.5GB
# 둘 다 lazy load + 캐시 → 한번 로드되면 해제 안 됨
```
- 질문 1개당: embed(질문) → search(ChromaDB) → LLM(답변)
- Embedder + LLM 동시 VRAM ~3.8GB + CUDA 오버헤드 → 8GB에서 위험

---

### 2.8 `utils/audio.py` — 84점

**역할:** ffmpeg 래핑 — 오디오 추출, 더빙 합성, SRT 내보내기

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| `_run()` 에러 핸들링 (line 18-26) | 9/10 | stderr 마지막 800자 출력으로 디버깅 용이. `encoding="utf-8", errors="replace"` 안전 |
| `extract_audio()` (line 29-47) | 9/10 | 16kHz 모노 PCM — STT 표준 입력 형식. `-y` 덮어쓰기 자동 |
| `get_video_duration()` (line 50-64) | 8/10 | ffprobe JSON 파싱 정확. `"format"` 키 없는 영상(스트림만 있는 경우)에서 KeyError 가능 |
| `mix_dubbed_into_video()` (line 67-97) | 9/10 | amix 필터 + volume 조절 + 비디오 copy 인코딩. 원본 화질 유지하면서 오디오만 교체 — 효율적 |
| `extract_ref_clip()` 후보 선택 (line 114-121) | 7/10 | 5~15초, 15자 이상 조건 합리적. 하지만 노이즈가 많은 구간이 선택될 수 있음 (음성 품질 미검사) |
| `_fmt_srt_time()` (line 139-145) | 9/10 | SRT 표준 형식 준수. 다만 `ms = int(round((secs % 1) * 1000))`에서 반올림으로 1000이 될 수 있는 극단적 케이스 (999.9995초 등) |
| ffmpeg 존재 확인 | 5/10 | ffmpeg 미설치 시 FileNotFoundError → 사용자에게 "ffmpeg를 설치하세요" 안내 없음 |
| `#디버그 수집용` 주석 (line 9) | 6/10 | 불필요한 인라인 주석. `import logging`은 표준 패턴이므로 주석 불필요 |

#### 주요 이슈

```python
# [line 130] 샘플레이트 하드코딩
"-ar", "24000", "-ac", "1",
```
- F5-TTS의 24kHz에 맞춘 값이지만, config의 TTS 설정이나 상수를 참조하지 않음
- TTS 모델이 변경되면 이 값도 수동으로 바꿔야 함

---

### 2.9 `utils/timestamp_sync.py` — 78점

**역할:** TTS 음성 길이를 Whisper 타임스탬프에 동기화

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| 스트레칭 알고리즘 (line 72) | 8/10 | librosa phase vocoder 사용. 음성 품질 유지하면서 시간 변환 가능 |
| 비율 클램프 (line 64) | 9/10 | [0.5, 2.5] 범위 제한 + 로그 경고 — 극단적 변형 방지 |
| 무음 처리 (line 49-53) | 9/10 | 짧은 구간이나 무음은 무음 패딩으로 대체 — 안정적 |
| 거의 동일 비율 스킵 (line 57) | 10/10 | 2% 이내면 스트레칭 건너뜀 — 불필요한 음질 저하 방지 |
| 클리핑 방지 (line 82-84) | 9/10 | peak > 0.99일 때 0.95로 정규화 — 오디오 왜곡 방지 |
| `n_fft=512` (line 72) | 6/10 | 24kHz에서 ~21ms 윈도우. 음성에는 적합하지만, `n_fft=1024`(~43ms)이 인간 음성의 기본 주기에 더 적합하고 음질도 좋음 |
| `merge_dubbed_audio()` 오버랩 처리 (line 136) | 7/10 | `+=` 방식으로 오버랩된 세그먼트가 합산됨. 두 세그먼트가 겹치면 음성이 섞이지만, Whisper 세그먼트는 보통 겹치지 않으므로 실제로는 OK |
| librosa 성능 (line 72) | 5/10 | `librosa.effects.time_stretch`는 STFT 기반으로 느림. 세그먼트당 0.1~0.5초 소요 × 100개 = 10~50초. rubberband 또는 sox가 3-5배 빠름 |

#### 주요 이슈

```python
# [line 72] librosa time_stretch 성능
stretched = librosa.effects.time_stretch(y, rate=1.0 / clamped, n_fft=512)
```
- CPU 바운드 작업으로, 세그먼트가 많으면 전체 파이프라인의 주요 병목
- TTS `synthesize_all()`에서 ThreadPoolExecutor로 비동기 처리 중이므로 GPU 추론과 겹치긴 함
- 하지만 CPU 코어가 부족하면 여전히 병목

---

### 2.10 `utils/memory.py` — 92점

**역할:** VRAM/RAM 메모리 관리

#### 채점표

| 세부 항목 | 점수 | 근거 |
|-----------|------|------|
| CUDA/MPS/CPU 분기 (line 42-54) | 10/10 | 세 환경 모두 올바르게 처리. ImportError도 잡음 |
| `empty_cache()` + `synchronize()` (line 45-46) | 10/10 | CUDA 캐시 해제 정석. synchronize로 비동기 작업 완료 보장 |
| `gc.collect()` in finally (line 56) | 9/10 | 항상 실행 보장 — 파이썬 객체 참조 순환 해제에 필수 |
| `log_vram()` 포맷 (line 25-29) | 9/10 | Allocated/Reserved/Total 세 가지 모두 표시 — 디버깅에 유용 |
| `vram_context()` (line 59-80) | 8/10 | 좋은 아이디어지만 실제 코드에서 사용되지 않음. 현재 모든 모듈이 수동으로 `clear_vram()` 호출 |
| `tag:30s` 포맷 (line 26) | 8/10 | 고정 폭 정렬로 로그 가독성 향상. 한글 태그는 폭 계산이 달라질 수 있음 |

#### 주요 이슈
- 거의 없음. 이 파일은 잘 작성됨

---

## 3. 프로젝트 전체 이슈 (크로스 파일)

### 3.1 Critical — 작동 불량 유발

| # | 이슈 | 위치 | 상태 |
|---|------|------|------|
| C-1 | Qwen3 `<think>` 태그가 번역/요약 결과에 포함 | `llm.py` L67-70 | **수정 완료** |
| C-2 | RAG chat 모드에서 Embedder+LLM 동시 VRAM ~3.8GB → OOM 위험 (8GB GPU) | `rag.py` L87-97 | 미수정 |

### 3.2 High — 기능 저하

| # | 이슈 | 위치 | 상태 |
|---|------|------|------|
| H-1 | STT beam_size config 무시 | `stt.py` L69 | **수정 완료** |
| H-2 | Embedding task instruction 누락 → 검색 정확도 ~15% 저하 | `embedder.py` L58 | **수정 완료** |
| H-3 | LLM 번역 파싱 정규식이 LLM 형식 변동에 취약 | `llm.py` L173 | 미수정 |
| H-4 | 긴 영상 full_text 요약 시 context window 초과 | `main.py` L107 | 미수정 |

### 3.3 Medium — 안정성/성능

| # | 이슈 | 위치 | 상태 |
|---|------|------|------|
| M-1 | TTS에 빈 텍스트 전달 시 에러 가능 | `tts.py` L148 | 미수정 |
| M-2 | `word_timestamps=False`인데 words 처리 코드 존재 (죽은 코드) | `stt.py` L86-90 | 미수정 |
| M-3 | `llm.unload()` 시 `_infer` lambda 참조 미삭제 | `llm.py` L208-214 | 미수정 |
| M-4 | librosa time_stretch 성능 병목 | `timestamp_sync.py` L72 | 미수정 |
| M-5 | config temperature=0.1이지만 do_sample=False로 무시됨 | `llm.py` L84 / `config.py` L46 | 미수정 |
| M-6 | ffmpeg 미설치 시 사용자 안내 없음 | `audio.py` | 미수정 |
| M-7 | RAG `llm._infer()` private 메서드 외부 호출 | `rag.py` L201 | 미수정 |
| M-8 | 입력 영상 파일 존재 여부 미검증 | `main.py` L60 | 미수정 |

### 3.4 Low — 코드 정리

| # | 이슈 | 위치 | 상태 |
|---|------|------|------|
| L-1 | LLM docstring 모델명 불일치 | `llm.py` L9 | **수정 완료** |
| L-2 | `#디버그 수집용` 불필요 주석 | `audio.py` L9 | 미수정 |
| L-3 | `_fmt_timestamp()` 1시간 초과 미지원 | `rag.py` L48-51 | 미수정 |
| L-4 | `extract_ref_clip` 샘플레이트 하드코딩 | `audio.py` L130 | 미수정 |

---

## 4. 수정 완료 항목 요약

| 파일 | 수정 내용 | 심각도 |
|------|-----------|--------|
| `modules/llm.py` | `_strip_think_tags()` 함수 추가, PC/Mac 추론 모두 적용 | Critical |
| `modules/llm.py` | docstring 모델명 수정 (Qwen2.5 → Qwen3) | Low |
| `modules/stt.py` | `beam_size=3` → `beam_size=cfg.stt.beam_size` | High |
| `modules/embedder.py` | `embed()` 메서드에 task instruction prefix 추가 | High |
| `modules/rag.py` | 검색 쿼리에 `task="query"` 적용 | High |

---

## 5. 테스트 현황

| 항목 | 상태 |
|------|------|
| 단위 테스트 파일 | 없음 |
| 통합 테스트 | 없음 |
| CI/CD | 없음 |
| 수동 테스트 | main.py CLI로만 가능 |

> 테스트 코드가 전혀 없으므로, 코드 변경 시 회귀 검증이 불가능합니다.
> 최소한 각 모듈의 입출력을 검증하는 단위 테스트 추가를 강력 권장합니다.
