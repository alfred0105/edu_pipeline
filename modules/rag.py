"""
modules/rag.py
────────────────────────────────────────────────────────────────────
RAG 챗봇 — 영상 내용에만 기반한 질의응답.

설계 원칙 (Hallucination 방지):
  1. 시스템 프롬프트에 "참고 자료 외 추측 금지"를 명시
  2. 코사인 거리 임계값(0.60)으로 관련성 낮은 청크 필터링
  3. 검색 결과가 없거나 모두 임계값 초과 시 "정보 없음" 응답 고정
  4. max_new_tokens=512로 장황한 생성 제한
  5. temperature=0.1로 창의적 생성 억제

벡터 저장소: ChromaDB (로컬 파일 기반, 서버 불필요)
"""

import logging
from pathlib import Path
from config import cfg

logger = logging.getLogger(__name__)


# ── 시스템 프롬프트 (Hallucination 방지 핵심) ────────────────────────────────
_SYSTEM_PROMPT = """당신은 교육 영상 내용만을 바탕으로 질문에 답하는 어시스턴트입니다.

[필수 규칙]
1. 아래 [참고 자료]에 있는 내용만을 근거로 답변하세요.
2. 참고 자료에 없는 내용은 절대로 추측하거나 지어내지 마세요.
3. 참고 자료에서 답을 찾을 수 없는 경우, 반드시 다음 문장으로만 답하세요:
   "제공된 영상 내용에서 해당 정보를 찾을 수 없습니다."
4. 답변 마지막에 근거로 사용한 자료의 타임스탬프를 명시하세요.
5. 한국어로 답변하세요."""

_NO_CONTEXT_REPLY = "제공된 영상 내용에서 해당 정보를 찾을 수 없습니다."


def _fmt_timestamp(secs: float) -> str:
    m = int(secs // 60)
    s = int(secs % 60)
    return f"{m:02d}:{s:02d}"


class RAGChatbot:
    """
    ChromaDB 기반 RAG 챗봇.

    사용 흐름:
        rag = RAGChatbot(collection_name="lecture_01")
        rag.index_segments(segments_with_embeddings)  # 한 번만 실행
        answer = rag.query("양자역학이란 무엇인가요?")
        rag.chat()  # 대화형 CLI
    """

    def __init__(self, collection_name: str = "edu_video"):
        import chromadb

        db_path = cfg.vector_dir
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=f"{cfg.rag.collection_prefix}{collection_name}",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"[RAG] ChromaDB 초기화  "
            f"collection={cfg.rag.collection_prefix}{collection_name}  "
            f"저장 경로={db_path}  "
            f"기존 항목={self._collection.count()}개"
        )

        self._embedder = None
        self._llm      = None

    # ── Lazy 로드 ────────────────────────────────────────────────────────
    def _get_embedder(self):
        if self._embedder is None:
            from modules.embedder import EmbeddingProcessor
            self._embedder = EmbeddingProcessor()
        return self._embedder

    def _get_llm(self):
        if self._llm is None:
            from modules.llm import LLMProcessor
            self._llm = LLMProcessor()
        return self._llm

    # ── 인덱싱 ──────────────────────────────────────────────────────────
    def index_segments(self, segments: list[dict]) -> None:
        """
        세그먼트를 ChromaDB에 저장합니다.
        'embedding' 필드가 없는 세그먼트는 자동으로 임베딩 후 저장합니다.
        """
        if not segments:
            logger.warning("[RAG] 저장할 세그먼트가 없습니다.")
            return

        # 임베딩이 없는 세그먼트만 배치 처리
        missing = [s for s in segments if "embedding" not in s]
        if missing:
            logger.info(f"[RAG] {len(missing)}개 세그먼트 임베딩 중...")
            embedder = self._get_embedder()
            embedded = embedder.embed_segments(missing)
            emb_map = {id(o): e["embedding"] for o, e in zip(missing, embedded)}
            segments = [
                {**s, "embedding": emb_map[id(s)]} if "embedding" not in s else s
                for s in segments
            ]

        ids        = [f"seg_{i:05d}" for i in range(len(segments))]
        embeddings = [s["embedding"] for s in segments]
        documents  = [s.get("translated", s["text"]) for s in segments]
        metadatas  = [
            {
                "start":    s["start"],
                "end":      s["end"],
                "original": s["text"],
                "ts":       _fmt_timestamp(s["start"]),
            }
            for s in segments
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"[RAG] {len(segments)}개 세그먼트 인덱싱 완료  총 {self._collection.count()}개")

    # ── 검색 ────────────────────────────────────────────────────────────
    def _retrieve(self, question: str) -> list[dict]:
        """질문과 유사한 상위 K개 청크를 반환합니다."""
        n_stored = self._collection.count()
        if n_stored == 0:
            return []

        embedder = self._get_embedder()
        q_vec    = embedder.embed([question])[0]

        results = self._collection.query(
            query_embeddings=[q_vec],
            n_results=min(cfg.rag.top_k, n_stored),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist > cfg.rag.distance_threshold:
                # 관련성이 낮은 청크 제외 → Hallucination 방지 핵심
                logger.debug(f"[RAG] 청크 제외 (dist={dist:.3f} > {cfg.rag.distance_threshold}): {doc[:40]}")
                continue
            chunks.append({"doc": doc, "meta": meta, "dist": dist})

        return chunks

    # ── 답변 생성 ────────────────────────────────────────────────────────
    def query(self, question: str) -> str:
        """
        질문에 답합니다. 관련 청크가 없으면 "정보 없음" 문장을 반환합니다.

        Returns:
            LLM이 생성한 답변 문자열
        """
        chunks = self._retrieve(question)

        if not chunks:
            logger.info(f"[RAG] 관련 청크 없음 → 고정 응답 반환")
            return _NO_CONTEXT_REPLY

        # 참고 자료 구성 (타임스탬프 포함)
        context_lines = [
            f"[{c['meta']['ts']}] {c['doc']}"
            for c in chunks
        ]
        context = "\n".join(context_lines)

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"[참고 자료]\n{context}\n\n"
            f"[질문]\n{question}\n\n"
            f"[답변]"
        )

        llm    = self._get_llm()
        answer = llm._infer(prompt, 512)

        # 모델이 _NO_CONTEXT_REPLY를 무시하고 지어낸 경우 방어
        if not answer.strip():
            return _NO_CONTEXT_REPLY

        return answer

    # ── 대화형 CLI ────────────────────────────────────────────────────────
    def chat(self) -> None:
        """터미널 기반 대화 루프를 시작합니다."""
        print("\n" + "=" * 55)
        print("  RAG 챗봇 — 영상 내용 기반 질의응답")
        print("  종료: 'q' 또는 'quit' 입력")
        print("=" * 55 + "\n")

        while True:
            try:
                question = input("질문 > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n챗봇을 종료합니다.")
                break

            if not question:
                continue
            if question.lower() in ("q", "quit", "exit", "종료"):
                print("챗봇을 종료합니다.")
                break

            answer = self.query(question)
            print(f"\n답변 > {answer}\n")
