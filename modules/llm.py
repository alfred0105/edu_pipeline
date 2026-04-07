"""
modules/llm.py
────────────────────────────────────────────────────────────────────
LLM 모듈 — 번역 및 연령별 수준 맞춤 요약.

PC  (CUDA):
    1순위: llama.cpp (GGUF Q4_K_M) — GPU 가속, transformers 대비 2~3배 빠름
    2순위: transformers + bitsandbytes NF4 (llama.cpp 미설치 시 폴백)
Mac (Apple Silicon): mlx-lm (Apple MLX 프레임워크)

모델: google/gemma-4-26b-it (MoE: 26B 전체 / 4B 활성 파라미터)
  - GGUF Q4_K_M: 가중치 ~14GB 저장, 추론 시 ~4B 활성 (GPU 부분 오프로드 가능)
  - n_gpu_layers=-1 이면 전체 GPU 시도; VRAM 부족 시 정수로 조정하세요.
  - 뛰어난 한국어·영어·다국어 지원
"""

import logging
import os
import re
from config import cfg, MODE
from utils.memory import clear_vram, log_vram

logger = logging.getLogger(__name__)


# ── 연령별 수준 프롬프트 ──────────────────────────────────────────────────────
AGE_LEVELS = {
    "elementary": (
        "초등학생(8~13세)도 이해할 수 있도록 쉽고 친근한 단어를 사용하고, "
        "짧은 문장으로 핵심만 설명해 주세요. 어려운 용어는 비유로 풀어주세요."
    ),
    "middle": (
        "중학생(14~16세) 수준의 언어를 사용하고, 핵심 개념과 그 이유를 "
        "논리적으로 설명해 주세요. 교과서 수준의 용어를 활용해도 됩니다."
    ),
    "high": (
        "고등학생(17~19세) 수준의 언어를 사용하고, 개념 간 관계와 원리를 "
        "심층적으로 설명해 주세요. 전문 용어를 적절히 사용하되 맥락 안에서 이해 가능하게 작성하세요."
    ),
    "university": (
        "대학생 및 성인 학습자를 위해 전문 용어를 적극 활용하며 체계적으로 설명해 주세요. "
        "학술적이면서도 이해하기 쉽게 작성하고, 관련 배경 지식도 간략히 언급해 주세요."
    ),
}

# 연령별 용어 해설 지시 (요약 하단에 어려운 용어 설명 추가)
AGE_GLOSSARY = {
    "elementary": "본문에 사용된 어려운 단어나 개념을 5개 이내로 골라, 초등학생이 이해할 수 있도록 쉬운 비유와 함께 설명해 주세요.",
    "middle":     "본문에 사용된 전문 용어나 핵심 개념을 5개 이내로 골라, 중학생 수준에서 이해할 수 있도록 간결하게 설명해 주세요.",
    "high":       "본문에 사용된 전문 용어나 학술 개념을 5개 이내로 골라, 고등학생이 참고할 수 있도록 정의와 맥락을 설명해 주세요.",
    "university": "본문에 사용된 고급 전문 용어를 5개 이내로 골라, 학술적 정의와 함께 간략히 설명해 주세요.",
}

LANG_NAMES = {
    "ko": "한국어", "en": "영어", "vi": "베트남어", "lo": "라오스어",
}


# ── Gemma 채팅 템플릿 (llama.cpp용) ─────────────────────────────────────────
def _build_gemma_prompt(user_msg: str) -> str:
    """Gemma 채팅 템플릿을 수동으로 구성합니다."""
    return (
        f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


# ── llama.cpp 백엔드 사용 가능 여부 체크 ──────────────────────────────────────
def _setup_dll_dirs():
    """CUDA / PyTorch DLL 경로를 등록합니다 (llama.dll 의존성 해결)."""
    import os
    try:
        import torch
        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib'))
    except Exception:
        pass
    # 1순위: CUDA_PATH 환경변수 (사용자 설치 경로를 신뢰)
    cuda_path_env = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path_env:
        cuda_bin = os.path.join(cuda_path_env, "bin")
        if os.path.isdir(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
                logger.debug(f"[LLM] CUDA DLL 경로 등록 (env): {cuda_bin}")
            except Exception as e:
                logger.debug(f"[LLM] CUDA DLL 경로 등록 실패: {e}")
        return
    # 2순위: 표준 설치 경로에서 버전 디렉토리 자동 스캔 (최신 버전 우선)
    toolkit_base = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if os.path.isdir(toolkit_base):
        try:
            versions = sorted(
                (d for d in os.listdir(toolkit_base)
                 if os.path.isdir(os.path.join(toolkit_base, d))),
                reverse=True,
            )
            for ver in versions:
                cuda_bin = os.path.join(toolkit_base, ver, "bin")
                if os.path.isdir(cuda_bin):
                    try:
                        os.add_dll_directory(cuda_bin)
                        logger.debug(f"[LLM] CUDA DLL 경로 등록 (scan {ver}): {cuda_bin}")
                    except Exception as e:
                        logger.debug(f"[LLM] CUDA DLL 등록 실패 ({cuda_bin}): {e}")
                    break
        except Exception as e:
            logger.debug(f"[LLM] CUDA 디렉토리 스캔 실패: {e}")


def _check_llamacpp() -> bool:
    """llama-cpp-python 설치 여부와 GGUF 파일 존재 여부를 빠르게 확인합니다.
    실제 모델 로드 없이 import + GGUF 파일 존재 여부만 체크합니다."""
    _setup_dll_dirs()
    try:
        from llama_cpp import Llama  # noqa: F401
    except (ImportError, RuntimeError):
        logger.debug("[LLM] llama-cpp-python 미설치 또는 DLL 로드 실패")
        return False

    try:
        from huggingface_hub import hf_hub_download
        # GGUF 파일이 캐시에 있는지만 확인 (네트워크 호출 없음)
        hf_hub_download(
            repo_id=cfg.llm.gguf_repo,
            filename=cfg.llm.gguf_file,
            local_files_only=True,
        )
        # 파일이 존재하면 사용 가능으로 판단 (실제 호환성은 _load_pc_llamacpp에서 확인)
        return True
    except Exception as e:
        logger.debug(f"[LLM] llama.cpp GGUF 미발견: {e}")
        return False


# ── PC 백엔드 1: llama.cpp (GGUF) ────────────────────────────────────────────
def _load_pc_llamacpp():
    """llama.cpp GGUF 모델 로드 (GPU 가속, transformers 대비 2~3배 빠름)"""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(
        repo_id=cfg.llm.gguf_repo,
        filename=cfg.llm.gguf_file,
    )
    logger.info(f"[LLM] llama.cpp 모델 로드: {cfg.llm.gguf_file}")

    model = Llama(
        model_path=model_path,
        n_gpu_layers=cfg.llm.n_gpu_layers,
        n_ctx=cfg.llm.n_ctx,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    logger.info(f"[LLM] llama.cpp 로드 완료  n_gpu_layers={cfg.llm.n_gpu_layers}")
    return "llamacpp", model, None


def _infer_llamacpp(model, prompt: str, max_tokens: int) -> str:
    """llama.cpp 추론 — Gemma 채팅 포맷"""
    full_prompt = _build_gemma_prompt(prompt)
    output = model(
        full_prompt,
        max_tokens=max_tokens,
        temperature=cfg.llm.temperature,
        stop=["<end_of_turn>", "<eos>"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


# ── PC 백엔드 2: transformers + bitsandbytes NF4 (폴백) ──────────────────────
def _load_pc_bnb():
    """bitsandbytes NF4 모델 로드 (llama.cpp 미설치 시 사용)"""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    )
    logger.info(f"[LLM] transformers + bitsandbytes NF4 로드: {cfg.llm.model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    # transformers의 _patch_mistral_regex()가 is_base_mistral()로 HF API를 호출함.
    # 로컬 경로를 직접 넘기면 _is_local=True → is_base_mistral() 호출 자체가 스킵됨.
    from huggingface_hub import snapshot_download
    try:
        _model_path = snapshot_download(cfg.llm.model_id, local_files_only=True)
    except Exception:
        _model_path = cfg.llm.model_id
    tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm.model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    # stop 토큰 ID 캐싱 (매 호출 변환 방지)
    eos_ids = [tokenizer.eos_token_id]
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id != tokenizer.unk_token_id:
        eos_ids.append(end_of_turn_id)

    return "bnb", tokenizer, model, eos_ids


def _infer_bnb(tokenizer, model, eos_ids: list[int], prompt: str, max_tokens: int) -> str:
    """transformers 추론 — Gemma 채팅 포맷"""
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=cfg.llm.temperature,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Ollama 백엔드 ────────────────────────────────────────────────────────────
def _check_ollama() -> bool:
    """Ollama가 실행 중이고 모델이 로드돼 있는지 확인합니다."""
    try:
        import urllib.request, json
        url = f"{cfg.llm.ollama_url}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        # gemma4:27b, gemma4:27b-instruct-q4_K_M 등 prefix 매칭
        base = cfg.llm.ollama_model.split(":")[0]
        found = any(base in m for m in models)
        if not found:
            logger.warning(f"[LLM] Ollama 실행 중이나 {cfg.llm.ollama_model} 없음. 목록: {models}")
        return found
    except Exception as e:
        logger.debug(f"[LLM] Ollama 미실행: {e}")
        return False


def _ollama_payload(
    prompt: str,
    max_tokens: int,
    stream: bool,
    system: str | None = None,
    history: list | None = None,
) -> bytes:
    import json
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)   # [{"role": "user"|"assistant", "content": str}, ...]
    messages.append({"role": "user", "content": prompt})
    return json.dumps({
        "model": cfg.llm.ollama_model,
        "messages": messages,
        "stream": stream,
        "think": False,
        "options": {
            "temperature": cfg.llm.temperature,
            "num_predict": max_tokens,
            "num_ctx": 4096,
            "num_batch": 512,
        },
    }, ensure_ascii=False).encode("utf-8")


def _infer_ollama(prompt: str, max_tokens: int,
                  system: str | None = None,
                  history: list | None = None) -> str:
    """Ollama /api/chat으로 추론합니다."""
    import urllib.request, json
    req = urllib.request.Request(
        f"{cfg.llm.ollama_url}/api/chat",
        data=_ollama_payload(prompt, max_tokens, stream=False,
                             system=system, history=history),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"[LLM] Ollama 추론 실패: {e}", exc_info=True)
        raise


def _infer_ollama_stream(prompt: str, max_tokens: int,
                         system: str | None = None,
                         history: list | None = None):
    """Ollama /api/chat 스트리밍 — 토큰 단위 yield"""
    import urllib.request, json
    req = urllib.request.Request(
        f"{cfg.llm.ollama_url}/api/chat",
        data=_ollama_payload(prompt, max_tokens, stream=True,
                             system=system, history=history),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    logger.info(f"[LLM] Ollama 스트리밍 시작  model={cfg.llm.ollama_model}  max_tokens={max_tokens}")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if not line.strip():
                    continue
                data = json.loads(line.decode("utf-8"))
                if "error" in data:
                    logger.error(f"[LLM] Ollama 오류: {data['error']}")
                    raise RuntimeError(data["error"])
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except Exception as e:
        logger.error(f"[LLM] Ollama 스트리밍 실패: {e}", exc_info=True)
        raise


# ── PC 로드 (자동 선택) ──────────────────────────────────────────────────────
def _load_pc():
    """PC 모델 로드: Ollama → llama.cpp → bitsandbytes 순으로 시도"""
    # 1순위: Ollama (외부 프로세스, 로드 없음)
    if _check_ollama():
        logger.info(f"[LLM] Ollama 백엔드 사용: {cfg.llm.ollama_model}")
        return "ollama", None, None, None

    # 2순위: llama.cpp
    _setup_dll_dirs()
    try:
        from llama_cpp import Llama  # noqa: F401
        try:
            backend, model, _ = _load_pc_llamacpp()
            return backend, None, model, None
        except Exception as e:
            logger.warning(f"[LLM] llama.cpp 로드 실패 → bitsandbytes 폴백: {e}")
    except (ImportError, RuntimeError):
        logger.info("[LLM] llama.cpp 미설치 → bitsandbytes 사용")

    # 3순위: bitsandbytes
    return _load_pc_bnb()


# ── Mac 백엔드 (mlx-lm) ──────────────────────────────────────────────────────
def _load_mac(model_id: str):
    from mlx_lm import load
    model, tokenizer = load(model_id)
    return "mlx", tokenizer, model


def _infer_mac(tokenizer, model, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model, tokenizer, prompt=text, max_tokens=max_tokens,
        temp=cfg.llm.temperature, verbose=False,
    ).strip()


# ── 번역 파싱 ────────────────────────────────────────────────────────────────
# 지원 형식: "1. text", "1) text", "1: text", "**1.** text"
_NUMBERED_LINE_RE = re.compile(
    r"\*{0,2}(\d+)\*{0,2}\s*[.):\-]\s*\*{0,2}\s*(.+?)(?=\n\s*\*{0,2}\d+\*{0,2}\s*[.):\-]|\Z)",
    re.DOTALL,
)


def _parse_numbered_fallback(raw: str, n: int) -> dict[int, str]:
    """
    정규식 파싱이 부분 실패했을 때 줄 수 기반으로 번역문을 추출합니다.
    모델이 번호 없이 줄바꿈만으로 구분할 때도 동작합니다.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        m = re.match(r"^\*{0,2}\d+\*{0,2}\s*[.):\-]\s*\*{0,2}\s*(.*)", ln)
        cleaned.append(m.group(1).strip() if m else ln)
    if len(cleaned) == n:
        return {i: t for i, t in enumerate(cleaned)}
    return {}

# ── 요약 파싱 (3개 연령별 블록 분리) ──────────────────────────────────────────
_SECTION_RE = re.compile(
    r"\[(\w+)\]\s*\n(.*?)(?=\n\[\w+\]|\Z)",
    re.DOTALL,
)


# ── LLMProcessor ────────────────────────────────────────────────────────────
class LLMProcessor:
    """
    번역 + 요약을 담당하는 LLM 래퍼.
    파이프라인 실행 시 한 번만 로드하고, 완료 후 .unload()로 VRAM을 해제합니다.
    """

    def __init__(self):
        logger.info(f"[LLM] 모델 로드 시작: {cfg.llm.model_id}  mode={MODE}")
        log_vram("LLM 로드 전")

        self._eos_ids = None
        if MODE == "pc":
            self._backend, self._tokenizer, self._model, self._eos_ids = _load_pc()
        else:
            self._backend, self._tokenizer, self._model = _load_mac(cfg.llm.model_id)
            self._eos_ids = None

        self._loaded = True
        log_vram("LLM 로드 후")
        logger.info(f"[LLM] 모델 준비 완료  backend={self._backend}")

    # ── 공용 추론 API ────────────────────────────────────────────────────
    def generate(self, prompt: str, max_tokens: int | None = None,
                system: str | None = None,
                history: list | None = None) -> str:
        """프롬프트를 받아 텍스트를 생성합니다."""
        if not self._loaded:
            raise RuntimeError("[LLM] 모델이 이미 해제되었습니다. 새 인스턴스를 생성하세요.")
        mt = max_tokens or cfg.llm.max_new_tokens

        if self._backend == "ollama":
            return _infer_ollama(prompt, mt, system=system, history=history)
        elif self._backend == "llamacpp":
            return _infer_llamacpp(self._model, prompt, mt)
        elif self._backend == "bnb":
            return _infer_bnb(self._tokenizer, self._model, self._eos_ids, prompt, mt)
        else:  # mlx
            return _infer_mac(self._tokenizer, self._model, prompt, mt)

    # ── 번역 ────────────────────────────────────────────────────────────────
    def translate(self, text: str, target_lang: str = "ko") -> str:
        """텍스트를 target_lang으로 번역합니다."""
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        prompt = (
            f"다음 텍스트를 자연스러운 {lang_name}로 번역해 주세요.\n"
            "번역문만 출력하세요. 설명이나 주석을 추가하지 마세요.\n\n"
            f"원문:\n{text}"
        )
        return self.generate(prompt, 512)

    def translate_segments(
        self,
        segments: list[dict],
        target_lang: str = "ko",
        batch_size: int = 20,
    ) -> list[dict]:
        """세그먼트를 batch_size개씩 묶어 한 번에 번역합니다."""
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        result = []
        total = len(segments)

        for batch_start in range(0, total, batch_size):
            batch = segments[batch_start:batch_start + batch_size]
            n = len(batch)
            numbered = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(batch))
            prompt = (
                f"아래 {n}개의 문장을 각각 자연스러운 {lang_name}로 번역하세요.\n"
                "반드시 '번호. 번역문' 형식으로만 출력하세요. 다른 내용은 출력하지 마세요.\n\n"
                f"{numbered}"
            )
            logger.info(f"[LLM] 번역 배치 {batch_start // batch_size + 1}/"
                        f"{(total + batch_size - 1) // batch_size}  ({n}개)")

            try:
                raw = self.generate(prompt, 256 * n)
            except Exception as e:
                logger.error(
                    f"[LLM] 번역 배치 {batch_start // batch_size + 1} 실패: {e}"
                    f" — 해당 배치 {n}개를 원문으로 유지합니다."
                )
                for seg in batch:
                    result.append({**seg, "translated": seg["text"]})
                continue

            parsed = {}
            for m in _NUMBERED_LINE_RE.finditer(raw):
                idx = int(m.group(1)) - 1
                if 0 <= idx < n:
                    parsed[idx] = m.group(2).strip()

            parsed_count = len(parsed)
            if parsed_count < n:
                logger.warning(
                    f"[LLM] 번역 파싱 {parsed_count}/{n}개만 성공 — "
                    f"줄 수 기반 폴백 파서 시도"
                )
                fallback = _parse_numbered_fallback(raw, n)
                if len(fallback) > parsed_count:
                    parsed = fallback
                    logger.info(f"[LLM] 폴백 파서로 {len(parsed)}/{n}개 복원")
                else:
                    logger.warning(f"[LLM] 폴백도 실패 — {n - len(parsed)}개 원문 유지")

            for i, seg in enumerate(batch):
                translated = parsed.get(i, seg["text"])
                result.append({**seg, "translated": translated})

        logger.info(f"[LLM] 번역 완료: {total}개 세그먼트")
        return result

    # ── 연령별 요약 (단건) ────────────────────────────────────────────────
    def generate_title(self, text: str, max_tokens: int = 60) -> str:
        """콘텐츠 앞부분을 보고 짧은 제목 생성 (10자 내외)."""
        prompt = (
            "다음 교육 콘텐츠의 핵심 주제를 20자 이내 한국어 제목으로만 출력하세요.\n"
            "예: '양자역학 기초 개념과 파동함수'\n"
            "주의: 제목만 출력하고 다른 설명은 쓰지 마세요.\n\n"
            f"내용:\n{text[:600]}"
        )
        return self.generate(prompt, max_tokens).strip().strip('"\'').split('\n')[0]

    def generate_topic(self, text: str, max_tokens: int = 30) -> str:
        """콘텐츠의 주제 카테고리를 한국어 단어/짧은 구로 생성 (라이브러리 분류용).

        예: '물리학', '한국 역사', '파이썬 프로그래밍', '유기화학', '미적분'
        """
        prompt = (
            "다음 콘텐츠의 학문 분야나 주제를 나타내는 카테고리를 10자 이내 한국어로만 출력하세요.\n"
            "예시: 물리학 / 한국 역사 / 파이썬 프로그래밍 / 유기화학 / 미적분 / 영어 문법\n"
            "카테고리 단어만 출력하고 다른 설명은 쓰지 마세요.\n\n"
            f"내용:\n{text[:500]}"
        )
        result = self.generate(prompt, max_tokens).strip().strip('"\'').split('\n')[0]
        # 최대 20자로 제한
        return result[:20] if result else "기타"

    def generate_topics(self, text: str, max_tokens: int = 60) -> list[str]:
        """다중 분야 태그 생성. 같은 콘텐츠가 여러 분야에 속할 수 있음.

        예: '전자현미경 렌즈' → ['광학', '전자현미경', '물리학']
        """
        prompt = (
            "다음 콘텐츠에 해당하는 학문 분야 또는 세부 주제 태그를 1~4개 생성하세요.\n"
            "쉼표(,)로만 구분하고 다른 설명은 쓰지 마세요. 각 태그는 10자 이내 한국어 명사구.\n"
            "예: 광학, 전자현미경, 물리학\n\n"
            f"내용:\n{text[:600]}"
        )
        raw = self.generate(prompt, max_tokens).strip().strip('"\'').split('\n')[0]
        tags = [t.strip()[:20] for t in raw.split(',') if t.strip()]
        # 중복 제거 (순서 유지) + 최대 4개
        seen, out = set(), []
        for t in tags:
            if t and t not in seen:
                seen.add(t); out.append(t)
            if len(out) >= 4:
                break
        return out or ["기타"]

    def generate_summary(self, text: str, max_tokens: int = 200) -> str:
        """콘텐츠가 어떤 내용인지 2~3문장으로 중립 설명 생성 (라이브러리 카드용)."""
        prompt = (
            "다음 콘텐츠가 어떤 주제와 내용을 다루는지 2~3문장으로 간결하게 설명하세요.\n"
            "연령이나 난이도 언급 없이, 핵심 주제와 다루는 내용만 사실적으로 서술하세요.\n"
            "제목·부연 설명·글머리 기호 없이 문장만 출력하세요.\n\n"
            f"내용:\n{text}"
        )
        return self.generate(prompt, max_tokens).strip()

    def summarize_by_age(
        self,
        text: str,
        age_group: str = "university",
        max_tokens: int = 800,
    ) -> str:
        instruction = AGE_LEVELS.get(age_group, AGE_LEVELS["university"])
        glossary = AGE_GLOSSARY.get(age_group, AGE_GLOSSARY["university"])
        prompt = (
            f"다음 교육 영상 내용을 요약해 주세요.\n"
            f"요약 조건: {instruction}\n\n"
            "출력 형식:\n"
            "1. 먼저 요약문을 작성하세요.\n"
            f"2. 요약문 아래에 '---'을 쓰고, 그 아래에 [용어 해설] 섹션을 추가하세요.\n"
            f"   {glossary}\n"
            "   각 용어는 '• 용어: 설명' 형식으로 작성하세요.\n\n"
            "제목이나 부연 설명은 생략하세요.\n\n"
            f"내용:\n{text}"
        )
        return self.generate(prompt, max_tokens)

    # ── 연령별 요약 4개 (개별 호출 — 안정성 + 속도 우선) ─────────────────
    def summarize_all_ages(self, text: str) -> dict[str, str]:
        """
        4개 연령별 요약을 각각 생성합니다.

        Returns:
            {"elementary": "...", "middle": "...", "high": "...", "university": "..."}
        """
        results = {}
        for age in AGE_LEVELS:
            logger.info(f"[LLM] 요약 생성 중: {age}")
            results[age] = self.summarize_by_age(text, age, max_tokens=800)
        return results

    # ── 스트리밍 추론 ────────────────────────────────────────────────────────
    def generate_stream(self, prompt: str, max_tokens: int | None = None,
                        system: str | None = None,
                        history: list | None = None):
        """
        토큰을 스트리밍으로 생성합니다.
        llama.cpp: 실시간 토큰 단위 yield
        bnb/mlx:  전체 생성 후 청크로 나눠 yield (폴백)
        """
        if not self._loaded:
            raise RuntimeError("[LLM] 모델이 이미 해제되었습니다.")
        mt = max_tokens or cfg.llm.max_new_tokens

        if self._backend == "ollama":
            yield from _infer_ollama_stream(prompt, mt, system=system, history=history)
        elif self._backend == "llamacpp":
            full_prompt = _build_gemma_prompt(prompt)
            for chunk in self._model(
                full_prompt, max_tokens=mt,
                temperature=cfg.llm.temperature,
                stop=["<end_of_turn>", "<eos>"],
                echo=False, stream=True,
            ):
                token = chunk["choices"][0]["text"]
                if token:
                    yield token
        else:
            # bnb/mlx 폴백: 전체 생성 후 20자 단위 청크
            full = self.generate(prompt, mt)
            size = 20
            for i in range(0, len(full), size):
                yield full[i:i + size]

    # ── VRAM 해제 ───────────────────────────────────────────────────────────
    def unload(self) -> None:
        """모델을 메모리에서 완전히 해제합니다."""
        if not self._loaded:
            return
        self._loaded = False
        if self._backend == "ollama":
            logger.info("[LLM] Ollama 백엔드 — 해제 불필요 (외부 프로세스)")
            return
        import torch, gc
        # 모델의 모든 파라미터를 CPU로 이동 후 삭제 (CUDA 메모리 즉시 반환)
        if self._model is not None and hasattr(self._model, 'cpu'):
            try:
                self._model.cpu()
            except Exception:
                pass
        del self._model
        self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        clear_vram()
        log_vram("LLM 해제 후")
        logger.info("[LLM] 모델 해제 완료")
