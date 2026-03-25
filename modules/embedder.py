"""
modules/embedder.py
────────────────────────────────────────────────────────────────────
임베딩 모듈 — Qwen3-Embedding-0.6B으로 세그먼트 텍스트를 벡터화합니다.

Qwen3-Embedding-0.6B 특징:
  - 파라미터 0.6B의 초경량 임베딩 전용 모델
  - 512 토큰까지 처리, 1024차원 출력
  - 한국어·영어 등 다국어 지원
  - float16으로 CUDA VRAM ~1.3 GB

Mean Pooling + L2 정규화 방식으로 벡터를 추출합니다.
"""

import logging
from config import cfg, MODE
from utils.memory import clear_vram, log_vram

logger = logging.getLogger(__name__)

# Qwen3-Embedding task instruction prefix (인덱싱 vs 검색 구분)
_TASK_INSTRUCTIONS = {
    "passage": "Instruct: Represent this educational content for retrieval\n",
    "query":   "Instruct: Retrieve relevant educational content\n",
}


class EmbeddingProcessor:
    """
    텍스트 배열 → 임베딩 벡터 배열 변환기.
    STT 완료 후 사용하며, 완료 후 .unload()로 VRAM을 반납합니다.
    """

    def __init__(self):
        logger.info(f"[Embed] 모델 로드: {cfg.embed.model_id}  device={cfg.embed.device}")
        log_vram("Embed 로드 전")

        import torch
        from transformers import AutoTokenizer, AutoModel

        self._device = cfg.embed.device
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.embed.model_id, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            cfg.embed.model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()
        # 추론 전용 — gradient 관련 메모리 할당 차단
        for p in self._model.parameters():
            p.requires_grad_(False)

        log_vram("Embed 로드 후")
        logger.info("[Embed] 모델 준비 완료")

    def _mean_pool(self, token_embeddings, attention_mask):
        """Attention mask 기반 평균 풀링"""
        import torch
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: list[str], task: str = "passage") -> list[list[float]]:
        """
        텍스트 리스트를 임베딩 벡터 리스트로 변환합니다.

        Args:
            task: "passage" (인덱싱) 또는 "query" (검색 질문)

        Returns:
            각 텍스트에 대한 float 벡터 리스트 (L2 정규화됨)
        """
        import torch
        import torch.nn.functional as F

        prefix = _TASK_INSTRUCTIONS.get(task, "")

        all_vectors: list[list[float]] = []
        batch_size = cfg.embed.batch_size

        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i : i + batch_size]]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.embed.max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                output = self._model(**encoded)

            pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            all_vectors.extend(normalized.cpu().float().tolist())

        return all_vectors

    def embed_segments(self, segments: list[dict]) -> list[dict]:
        """
        세그먼트 리스트에 'embedding' 필드를 추가합니다.
        번역된 텍스트가 있으면 우선 사용합니다 (검색 언어 통일).
        """
        texts = [seg.get("translated", seg["text"]) for seg in segments]
        vectors = self.embed(texts)
        logger.info(f"[Embed] {len(segments)}개 세그먼트 벡터화 완료  dim={len(vectors[0]) if vectors else 0}")
        return [{**seg, "embedding": vec} for seg, vec in zip(segments, vectors)]

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        clear_vram()
        log_vram("Embed 해제 후")
        logger.info("[Embed] 모델 해제 완료")
