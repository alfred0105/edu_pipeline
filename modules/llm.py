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
    "child": (
        "초등학생도 이해할 수 있도록 쉽고 친근한 단어를 사용하고, "
        "짧은 문장으로 핵심만 설명해 주세요. 어려운 용어는 비유로 풀어주세요."
    ),
    "teen": (
        "중·고등학생 수준의 언어를 사용하고, 핵심 개념과 그 이유를 "
        "논리적으로 설명해 주세요."
    ),
    "adult": (
        "성인 학습자를 위해 전문 용어를 적절히 활용하며 체계적으로 설명해 주세요. "
        "학술적이면서도 이해하기 쉽게 작성해 주세요."
    ),
}

LANG_NAMES = {
    "ko": "한국어", "en": "영어", "ja": "일본어",
    "zh": "중국어", "es": "스페인어", "fr": "프랑스어",
}


# ── Qwen3 후처리 ──────────────────────────────────────────────────────────────
def _strip_think_tags(text: str) -> str:
    """Qwen3의 <think>...</think> 태그를 제거합니다."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Qwen3 채팅 템플릿 (llama.cpp용) ──────────────────────────────────────────
def _build_qwen3_prompt(user_msg: str) -> str:
    """Qwen3 채팅 템플릿을 수동으로 구성합니다."""
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ── llama.cpp 백엔드 사용 가능 여부 체크 ──────────────────────────────────────
def _check_llamacpp() -> bool:
    """llama-cpp-python이 Qwen3를 지원하는지 확인합니다."""
    try:
        import os
        # CUDA DLL 경로 등록
        import torch
        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib'))
        cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin'
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)

        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        # GGUF 모델 파일이 캐시에 있는지 확인
        model_path = hf_hub_download(
            repo_id=cfg.llm.gguf_repo,
            filename=cfg.llm.gguf_file,
            local_files_only=True,
        )
        # 테스트 로드 (n_ctx=32로 최소 메모리)
        test = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=32, verbose=False)
        del test
        return True
    except Exception as e:
        logger.debug(f"[LLM] llama.cpp 사용 불가: {e}")
        return False


# ── PC 백엔드 1: llama.cpp (GGUF) ────────────────────────────────────────────
def _load_pc_llamacpp():
    """llama.cpp GGUF 모델 로드 (GPU 가속, transformers 대비 2~3배 빠름)"""
    import os
    import torch
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib'))
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)

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
    """llama.cpp 추론"""
    full_prompt = _build_qwen3_prompt(prompt)
    output = model(
        full_prompt,
        max_tokens=max_tokens,
        temperature=cfg.llm.temperature,
        top_k=1,
        stop=["<|im_end|>", "<|endoftext|>"],
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
    return "bnb", tokenizer, model


def _infer_bnb(tokenizer, model, prompt: str, max_tokens: int) -> str:
    """transformers 추론"""
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=cfg.llm.temperature,
            top_k=1,
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
            return _load_pc_llamacpp()
        except Exception as e:
            logger.warning(f"[LLM] llama.cpp 로드 실패 → bitsandbytes 폴백: {e}")
    else:
        logger.info("[LLM] llama.cpp 미설치 또는 미지원 → bitsandbytes 사용")
    backend, tokenizer, model = _load_pc_bnb()
    return backend, tokenizer, model


# ── Mac 백엔드 (mlx-lm) ──────────────────────────────────────────────────────
def _load_mac(model_id: str):
    from mlx_lm import load
    model, tokenizer = load(model_id)
    return "mlx", tokenizer, model


def _infer_mac(tokenizer, model, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
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

        if MODE == "pc":
            self._backend, self._tokenizer, self._model = _load_pc()
        else:
            self._backend, self._tokenizer, self._model = _load_mac(cfg.llm.model_id)

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
            return _infer_bnb(self._tokenizer, self._model, prompt, mt)
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
        age_group: str = "adult",
        max_tokens: int = 600,
    ) -> str:
        instruction = AGE_LEVELS.get(age_group, AGE_LEVELS["adult"])
        prompt = (
            f"다음 교육 영상 내용을 요약해 주세요.\n"
            f"요약 조건: {instruction}\n"
            "요약문만 출력하세요. 제목이나 부연 설명은 생략하세요.\n\n"
            f"내용:\n{text}"
        )
        return self.generate(prompt, max_tokens)

    # ── 연령별 요약 3개 통합 (1회 호출로 child/teen/adult 동시 생성) ───────
    def summarize_all_ages(self, text: str, max_tokens: int = 1500) -> dict[str, str]:
        """
        3개 연령별 요약을 1회 LLM 호출로 생성합니다.

        Returns:
            {"child": "...", "teen": "...", "adult": "..."}
        """
        prompt = (
            "다음 교육 영상 내용을 3가지 수준으로 요약해 주세요.\n"
            "각 요약은 반드시 [child], [teen], [adult] 태그로 시작하세요.\n"
            "요약문만 출력하세요. 다른 내용은 출력하지 마세요.\n\n"
            f"[child] 조건: {AGE_LEVELS['child']}\n"
            f"[teen] 조건: {AGE_LEVELS['teen']}\n"
            f"[adult] 조건: {AGE_LEVELS['adult']}\n\n"
            f"내용:\n{text}"
        )

        raw = self.generate(prompt, max_tokens)

        # 파싱: [child] ... [teen] ... [adult] ...
        parsed = {}
        for m in _SECTION_RE.finditer(raw):
            key = m.group(1).strip().lower()
            if key in AGE_LEVELS:
                parsed[key] = m.group(2).strip()

        # 파싱 실패 시 개별 호출로 폴백
        if len(parsed) < 3:
            logger.warning(f"[LLM] 통합 요약 파싱 {len(parsed)}/3 — 누락분은 개별 호출로 보완")
            for age in AGE_LEVELS:
                if age not in parsed:
                    parsed[age] = self.summarize_by_age(text, age)

        return parsed

    # ── VRAM 해제 ───────────────────────────────────────────────────────────
    def unload(self) -> None:
        """모델을 메모리에서 완전히 해제합니다."""
        if not self._loaded:
            return
        self._loaded = False
        del self._model
        self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        clear_vram()
        log_vram("LLM 해제 후")
        logger.info("[LLM] 모델 해제 완료")
