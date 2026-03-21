"""
modules/llm.py
────────────────────────────────────────────────────────────────────
LLM 모듈 — 번역 및 연령별 수준 맞춤 요약.

PC  (CUDA 8GB): transformers + bitsandbytes 4-bit 양자화 (NF4)
Mac (Apple Silicon): mlx-lm (Apple MLX 프레임워크)

모델: Qwen/Qwen2.5-4B-Instruct
  - 4-bit 양자화 시 VRAM ~2.5 GB (RTX 4060 Ti에서 안정적)
  - 한국어·영어·일본어·중국어 다국어 지원 우수
"""

import logging
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


# ── PC 백엔드 (bitsandbytes 4-bit) ──────────────────────────────────────────
def _load_pc(model_id: str):
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,   # 이중 양자화로 추가 메모리 절감
        bnb_4bit_quant_type="nf4",        # NF4 = 4-bit Normal Float (가장 정확)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _infer_pc(tokenizer, model, prompt: str, max_tokens: int) -> str:
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Mac 백엔드 (mlx-lm) ──────────────────────────────────────────────────────
def _load_mac(model_id: str):
    from mlx_lm import load
    model, tokenizer = load(model_id)
    return tokenizer, model


def _infer_mac(tokenizer, model, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    result = generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
    return result.strip()


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
            self._tokenizer, self._model = _load_pc(cfg.llm.model_id)
            self._infer = lambda p, m=cfg.llm.max_new_tokens: _infer_pc(
                self._tokenizer, self._model, p, m
            )
        else:
            self._tokenizer, self._model = _load_mac(cfg.llm.model_id)
            self._infer = lambda p, m=cfg.llm.max_new_tokens: _infer_mac(
                self._tokenizer, self._model, p, m
            )

        log_vram("LLM 로드 후")
        logger.info("[LLM] 모델 준비 완료")

    # ── 번역 ────────────────────────────────────────────────────────────────
    def translate(self, text: str, target_lang: str = "ko") -> str:
        """텍스트를 target_lang으로 번역합니다."""
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        prompt = (
            f"다음 텍스트를 자연스러운 {lang_name}로 번역해 주세요.\n"
            "번역문만 출력하세요. 설명이나 주석을 추가하지 마세요.\n\n"
            f"원문:\n{text}"
        )
        return self._infer(prompt, 512)

    def translate_segments(
        self,
        segments: list[dict],
        target_lang: str = "ko",
        batch_size: int = 10,
    ) -> list[dict]:
        """세그먼트를 batch_size개씩 묶어 한 번에 번역합니다."""
        import re
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

            raw = self._infer(prompt, 256 * n)

            # 번호 파싱
            parsed = {}
            for m in re.finditer(r"(\d+)\.\s*(.+?)(?=\n\d+\.|$)", raw, re.DOTALL):
                idx = int(m.group(1)) - 1
                if 0 <= idx < n:
                    parsed[idx] = m.group(2).strip()

            for i, seg in enumerate(batch):
                translated = parsed.get(i, seg["text"])  # 파싱 실패 시 원문 사용
                result.append({**seg, "translated": translated})

        logger.info(f"[LLM] 번역 완료: {total}개 세그먼트")
        return result

    # ── 연령별 요약 ─────────────────────────────────────────────────────────
    def summarize_by_age(
        self,
        text: str,
        age_group: str = "adult",
        max_tokens: int = 600,
    ) -> str:
        """
        교육 콘텐츠를 연령별 수준에 맞게 요약합니다.

        Args:
            age_group: "child" | "teen" | "adult"
        """
        instruction = AGE_LEVELS.get(age_group, AGE_LEVELS["adult"])
        prompt = (
            f"다음 교육 영상 내용을 요약해 주세요.\n"
            f"요약 조건: {instruction}\n"
            "요약문만 출력하세요. 제목이나 부연 설명은 생략하세요.\n\n"
            f"내용:\n{text}"
        )
        return self._infer(prompt, max_tokens)

    # ── VRAM 해제 ───────────────────────────────────────────────────────────
    def unload(self) -> None:
        """모델을 메모리에서 해제합니다. 다음 단계(TTS) 실행 전 호출하세요."""
        del self._model
        del self._tokenizer
        clear_vram()
        log_vram("LLM 해제 후")
        logger.info("[LLM] 모델 해제 완료")
