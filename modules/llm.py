"""
modules/llm.py
────────────────────────────────────────────────────────────────────
LLM 모듈 — 번역 및 연령별 수준 맞춤 요약.

PC  (CUDA 8GB):
    1순위: llama.cpp (GGUF Q4_K_M) — GPU 가속, transformers 대비 2~3배 빠름
    2순위: transformers + bitsandbytes NF4 (llama.cpp 미설치 시 폴백)
Mac (Apple Silicon): mlx-lm (Apple MLX 프레임워크)

모델: Qwen/Qwen3-4B
  - GGUF Q4_K_M VRAM ~2.3 GB / BnB NF4 VRAM ~2.5 GB
  - 한국어·영어·일본어·중국어 다국어 지원 우수
"""

import logging
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


# ── Qwen3 후처리 ──────────────────────────────────────────────────────────────
def _strip_think_tags(text: str) -> str:
    """Qwen3의 <think>...</think> 태그를 제거합니다."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Qwen3 채팅 템플릿 (llama.cpp용) ──────────────────────────────────────────
def _build_qwen3_prompt(user_msg: str) -> str:
    """Qwen3 채팅 템플릿을 수동으로 구성합니다.
    thinking 블록을 미리 닫아서 <think> 토큰 낭비를 방지합니다."""
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n/no_think\n{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n</think>\n"
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
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)


def _check_llamacpp() -> bool:
    """llama-cpp-python이 Qwen3를 지원하는지 빠르게 확인합니다.
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
    import os
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
    """llama.cpp 추론 — thinking 이미 닫힌 상태에서 시작"""
    full_prompt = _build_qwen3_prompt(prompt)
    output = model(
        full_prompt,
        max_tokens=max_tokens,
        temperature=cfg.llm.temperature,
        top_k=1,
        stop=["<|im_end|>", "<|endoftext|>", "<think>"],
        echo=False,
    )
    text = output["choices"][0]["text"].strip()
    return _strip_think_tags(text)


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
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model_id, trust_remote_code=True)
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
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)

    return "bnb", tokenizer, model, eos_ids


def _infer_bnb(tokenizer, model, eos_ids: list[int], prompt: str, max_tokens: int) -> str:
    """transformers 추론 — thinking 블록을 미리 닫아 토큰 낭비 방지"""
    import torch
    messages = [{"role": "user", "content": f"/no_think\n{prompt}"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    text += "<think>\n</think>\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=cfg.llm.temperature,
            top_k=1,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _strip_think_tags(decoded)


# ── PC 로드 (자동 선택) ──────────────────────────────────────────────────────
def _load_pc():
    """PC 모델 로드: llama.cpp 우선, 실패 시 bitsandbytes 폴백"""
    if _check_llamacpp():
        try:
            backend, model, _ = _load_pc_llamacpp()
            return backend, None, model, None  # llamacpp는 tokenizer/eos_ids 불필요
        except Exception as e:
            logger.warning(f"[LLM] llama.cpp 로드 실패 → bitsandbytes 폴백: {e}")
    else:
        logger.info("[LLM] llama.cpp 미설치 또는 미지원 → bitsandbytes 사용")
    return _load_pc_bnb()  # (backend, tokenizer, model, eos_ids)


# ── Mac 백엔드 (mlx-lm) ──────────────────────────────────────────────────────
def _load_mac(model_id: str):
    from mlx_lm import load
    model, tokenizer = load(model_id)
    return "mlx", tokenizer, model


def _infer_mac(tokenizer, model, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate
    messages = [{"role": "user", "content": f"/no_think\n{prompt}"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    text += "<think>\n</think>\n"
    result = generate(
        model, tokenizer, prompt=text, max_tokens=max_tokens,
        temp=cfg.llm.temperature, verbose=False,
    )
    return _strip_think_tags(result)


# ── 번역 파싱 ────────────────────────────────────────────────────────────────
_NUMBERED_LINE_RE = re.compile(
    r"(\d+)\s*[.\)]\s*(.+?)(?=\n\s*\d+\s*[.\)]|\Z)",
    re.DOTALL,
)

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
    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """프롬프트를 받아 텍스트를 생성합니다."""
        if not self._loaded:
            raise RuntimeError("[LLM] 모델이 이미 해제되었습니다. 새 인스턴스를 생성하세요.")
        mt = max_tokens or cfg.llm.max_new_tokens

        if self._backend == "llamacpp":
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

            raw = self.generate(prompt, 256 * n)

            parsed = {}
            for m in _NUMBERED_LINE_RE.finditer(raw):
                idx = int(m.group(1)) - 1
                if 0 <= idx < n:
                    parsed[idx] = m.group(2).strip()

            parsed_count = len(parsed)
            if parsed_count < n:
                logger.warning(f"[LLM] 번역 파싱 {parsed_count}/{n}개만 성공 — 실패분은 원문 유지")

            for i, seg in enumerate(batch):
                translated = parsed.get(i, seg["text"])
                result.append({**seg, "translated": translated})

        logger.info(f"[LLM] 번역 완료: {total}개 세그먼트")
        return result

    # ── 연령별 요약 (단건) ────────────────────────────────────────────────
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

    # ── VRAM 해제 ───────────────────────────────────────────────────────────
    def unload(self) -> None:
        """모델을 메모리에서 완전히 해제합니다."""
        if not self._loaded:
            return
        self._loaded = False
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
