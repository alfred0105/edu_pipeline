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
import threading as _threading
from pathlib import Path
from config import cfg

# BM25 하이브리드 검색 (rank_bm25 미설치 시 자동 fallback)
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

logger = logging.getLogger(__name__)

# ── ChromaDB 공유 클라이언트 싱글톤 ────────────────────────────────────────
_chroma_client = None
_chroma_lock   = _threading.Lock()

def _get_chroma_client():
    global _chroma_client
    with _chroma_lock:
        if _chroma_client is None:
            import chromadb
            db_path = cfg.vector_dir
            Path(db_path).mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=db_path)
        return _chroma_client


def _reset_chroma_client():
    """클라이언트 싱글톤을 초기화 (인덱스 손상 복구용)."""
    global _chroma_client
    with _chroma_lock:
        _chroma_client = None


# ── 공유 임베더 싱글톤 (쿼리 임베딩 전용 — 매 요청마다 재로드 방지) ────────────
_shared_embedder      = None
_shared_embedder_lock = _threading.Lock()

def _get_shared_embedder():
    global _shared_embedder
    with _shared_embedder_lock:
        if _shared_embedder is None:
            from modules.embedder import EmbeddingProcessor
            logger.info("[RAG] 공유 임베더 로드")
            _shared_embedder = EmbeddingProcessor()
        return _shared_embedder

def _release_shared_embedder():
    global _shared_embedder
    with _shared_embedder_lock:
        if _shared_embedder is not None:
            _shared_embedder.unload()
            _shared_embedder = None


# ── 시스템 프롬프트 (Hallucination 방지 핵심) ────────────────────────────────
_SYSTEM_PROMPT = """당신은 교육 영상 내용을 바탕으로 학습자의 질문에 친절하게 답하는 어시스턴트입니다.

[답변 방침]
1. [참고 자료]에 있는 내용을 최대한 활용하여 답변하세요.
2. 참고 자료에 직접적인 내용이 없더라도 관련된 내용을 바탕으로 도움이 되는 답변을 하세요.
3. 답변할 때 근거로 사용한 자료의 타임스탬프를 함께 언급하면 좋습니다.
4. 한국어로 답변하세요."""

_ANALYSIS_PROMPT = """당신은 실험 데이터 분석 전문가입니다.
아래의 실험 데이터와 참고 자료를 바탕으로 정밀하게 분석합니다.

[필수 규칙]
1. 데이터에 있는 패턴, 추세, 이상치를 구체적 수치와 함께 설명하세요.
2. 예측 시 반드시 데이터 기반 근거를 명시하세요.
3. 참고 자료(영상/문서)와 데이터를 연계해 맥락을 설명하세요.
4. 불확실한 예측은 "(예측)" 또는 "(추정)" 표시를 붙이세요.
5. 한국어로 답변하세요."""

_NO_CONTEXT_REPLY = (
    "업로드된 자료에서 해당 내용을 찾지 못했습니다.\n\n"
    "다음을 확인해 보세요:\n"
    "• 질문이 자료의 주제와 관련 있는지 확인하세요.\n"
    "• 사이드바에서 검색할 소스를 선택했는지 확인하세요.\n"
    "• 더 구체적으로 질문해 보세요."
)

# 검색 결과는 있으나 유사도 임계값 미달 시 사용 (NO_CONTEXT_REPLY와 구분)
_THRESHOLD_REPLY = (
    "관련 내용이 자료에서 발견되었으나 유사도가 기준치에 미치지 못합니다.\n\n"
    "다음을 시도해 보세요:\n"
    "• 더 구체적인 키워드나 표현으로 질문해 보세요.\n"
    "• 질문을 짧게 나눠서 시도해 보세요."
)


def _sanitize_name(name: str) -> str:
    """ChromaDB 컬렉션 이름에 사용 불가한 문자를 제거합니다.
    ASCII 이외 문자(한글 등)가 포함된 경우 MD5 해시로 고유성을 보장합니다."""
    import re, hashlib
    # ASCII 범위 밖 문자가 있으면 해시 suffix 추가
    has_non_ascii = bool(re.search(r"[^\x00-\x7F]", name))
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    sanitized = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)  # 연속 언더스코어 정리
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    if len(sanitized) < 3 or has_non_ascii:
        # 비ASCII 포함: 해시로 고유 이름 생성 (충돌 방지)
        prefix = sanitized[:20] if sanitized else "col"
        sanitized = f"{prefix}_{hash_suffix}"
    return sanitized[:512]


def _fmt_timestamp(secs: float) -> str:
    """초 → 사람이 읽을 수 있는 타임스탬프 (1시간 이상 지원)"""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _resolve_collection(client, col_name: str, collection_name: str) -> tuple:
    """
    prefixed 컬렉션 우선, 비어있으면 레거시(non-prefixed) 폴백으로 컬렉션을 반환합니다.

    Args:
        client:          ChromaDB PersistentClient
        col_name:        sanitized prefixed 이름 (e.g. "edu_lecture01")
        collection_name: 원본 이름 (e.g. "lecture01")

    Returns:
        (collection, count) 튜플
    """
    col: object = None
    count: int  = 0

    # 1) prefixed 이름 시도
    try:
        col   = client.get_or_create_collection(
            name=col_name, metadata={"hnsw:space": "cosine"})
        count = col.count()
        if count > 0:
            return col, count
    except Exception:
        col   = None
        count = 0

    # 2) non-prefixed 레거시 이름 시도 (이전 인덱싱 결과 유지)
    legacy_name = _sanitize_name(collection_name)
    if legacy_name != col_name:
        try:
            leg       = client.get_collection(legacy_name)
            leg_count = leg.count()
            if leg_count > 0:
                logger.info(
                    f"[RAG] 레거시 컬렉션 '{legacy_name}' 사용 "
                    f"(데이터: {leg_count}개)"
                )
                return leg, leg_count
        except Exception:
            pass

    # 3) 빈 prefixed 컬렉션 반환 (새 인덱싱용)
    if col is None:
        col   = client.get_or_create_collection(
            name=col_name, metadata={"hnsw:space": "cosine"})
        count = 0
    return col, count


class RAGChatbot:
    """
    ChromaDB 기반 RAG 챗봇.

    사용 흐름:
        rag = RAGChatbot(collection_name="lecture_01")
        rag.index_segments(segments_with_embeddings)  # 한 번만 실행
        answer, chunks = rag.query("양자역학이란 무엇인가요?")
        rag.chat()  # 대화형 CLI
    """

    def __init__(self, collection_name: str = "edu_video", create_if_missing: bool = True):
        self._client = _get_chroma_client()
        prefix   = cfg.rag.collection_prefix
        col_name = _sanitize_name(f"{prefix}{collection_name}")

        # 읽기 전용 모드: 기존 컬렉션 없으면 빈 컬렉션 생성하지 않고 예외
        if not create_if_missing:
            existing = {c if isinstance(c, str) else c.name
                        for c in self._client.list_collections()}
            legacy_name = _sanitize_name(collection_name)
            if col_name not in existing and legacy_name not in existing:
                raise ValueError(f"컬렉션 '{collection_name}' 없음 (생성 차단됨)")

        try:
            self._collection, count = _resolve_collection(
                self._client, col_name, collection_name)
        except Exception as e:
            logger.warning(f"[RAG] 컬렉션 초기화 오류, 클라이언트 재시작: {e}")
            _reset_chroma_client()
            self._client = _get_chroma_client()
            try:
                self._collection, count = _resolve_collection(
                    self._client, col_name, collection_name)
            except Exception as e2:
                logger.error(f"[RAG] 컬렉션 '{col_name}' 복구 불가, 재생성: {e2}")
                try:
                    self._client.delete_collection(col_name)
                except Exception:
                    pass
                self._collection = self._client.create_collection(
                    name=col_name, metadata={"hnsw:space": "cosine"})
                count = 0

        logger.info(
            f"[RAG] ChromaDB 초기화  "
            f"collection={col_name}  기존 항목={count}개"
        )

        self._embedder  = None
        self._llm       = None
        self._bm25      = None   # (BM25Okapi, ids, docs, metas) 캐시
        self._bm25_dirty = True  # index_segments 후 무효화

    # ── Lazy 로드 + VRAM 관리 ──────────────────────────────────────────────
    def _get_embedder(self):
        if self._embedder is None:
            from modules.embedder import EmbeddingProcessor
            self._embedder = EmbeddingProcessor()
        return self._embedder

    def _release_embedder(self):
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None

    def _get_llm(self):
        if self._llm is None:
            from modules.llm import LLMProcessor
            self._llm = LLMProcessor()
        return self._llm

    def _release_llm(self):
        if self._llm is not None:
            self._llm.unload()
            self._llm = None

    # ── BM25 헬퍼 ────────────────────────────────────────────────────────
    def _build_bm25(self):
        """ChromaDB 전체 문서로 BM25 인덱스를 빌드합니다 (캐싱)."""
        if not _HAS_BM25:
            return
        if not self._bm25_dirty and self._bm25 is not None:
            return
        data = self._collection.get(include=["documents", "metadatas", "ids"])
        ids   = data.get("ids", [])
        docs  = data.get("documents", [])
        metas = data.get("metadatas", [])
        if not docs:
            self._bm25 = None
            return
        tokenized  = [d.lower().split() for d in docs]
        self._bm25 = (_BM25Okapi(tokenized), ids, docs, metas)
        self._bm25_dirty = False

    def _bm25_query(self, question: str, n: int) -> list[tuple]:
        """BM25 상위 n개 (id, doc, meta) 반환. rank_bm25 없으면 빈 리스트."""
        if not _HAS_BM25:
            return []
        self._build_bm25()
        if self._bm25 is None:
            return []
        bm25_idx, ids, docs, metas = self._bm25
        scores = bm25_idx.get_scores(question.lower().split())
        top_idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [(ids[i], docs[i], metas[i]) for i in top_idxs if scores[i] > 0]

    # ── 인덱싱 ──────────────────────────────────────────────────────────
    def index_segments(self, segments: list[dict],
                       window_size: int = 3, window_stride: int = 2) -> None:
        """
        세그먼트를 ChromaDB에 저장합니다.
        슬라이딩 윈도우(window_size, window_stride)로 연속 세그먼트를 묶어
        경계 손실을 줄이고 검색 품질을 높입니다.

        'embedding' 필드가 없는 세그먼트는 자동으로 임베딩 후 저장합니다.
        기존보다 세그먼트 수가 적으면 낡은 항목을 삭제합니다.
        """
        if not segments:
            logger.warning("[RAG] 저장할 세그먼트가 없습니다.")
            return

        # ── 슬라이딩 윈도우 청킹 ───────────────────────────────────────
        if window_size > 1 and len(segments) > window_size:
            windowed: list[dict] = []
            for i in range(0, len(segments), window_stride):
                ws = segments[i:i + window_size]
                if not ws:
                    break
                windowed.append({
                    "text":       " ".join(s["text"] for s in ws),
                    "translated": " ".join(s.get("translated", s["text"]) for s in ws),
                    "start":      ws[0]["start"],
                    "end":        ws[-1]["end"],
                    "data_type":  ws[0].get("data_type", "video"),
                })
            segments = windowed
            logger.info(f"[RAG] 슬라이딩 윈도우 청킹 완료: {len(segments)}개 청크 (window={window_size}, stride={window_stride})")

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

        new_count  = len(segments)
        old_count  = self._collection.count()

        ids        = [f"seg_{i:05d}" for i in range(new_count)]
        embeddings = [s["embedding"] for s in segments]
        documents  = [s.get("translated", s["text"]) for s in segments]
        metadatas  = [
            {
                "start":     s["start"],
                "end":       s["end"],
                "original":  s["text"],
                "ts":        _fmt_timestamp(s["start"]),
                "data_type": s.get("data_type", "video"),
            }
            for s in segments
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # 재인덱싱 시 낡은 세그먼트 삭제
        if old_count > new_count:
            stale_ids = [f"seg_{i:05d}" for i in range(new_count, old_count)]
            self._collection.delete(ids=stale_ids)
            logger.info(f"[RAG] 낡은 세그먼트 {len(stale_ids)}개 삭제")

        self._bm25_dirty = True   # BM25 캐시 무효화
        logger.info(f"[RAG] {new_count}개 세그먼트 인덱싱 완료  총 {self._collection.count()}개")

    # ── 검색 ────────────────────────────────────────────────────────────
    def _retrieve(self, question: str, threshold: float | None = None,
                  embedder=None) -> tuple[list[dict], int]:
        """
        하이브리드 검색 (벡터 + BM25 RRF fusion).

        rank_bm25 미설치 시 벡터 검색만 사용합니다.

        Returns:
            (filtered_chunks, raw_count) — raw_count는 임계값 적용 전 총 결과 수.
        """
        n_stored = self._collection.count()
        if n_stored == 0:
            return [], 0

        thr      = threshold if threshold is not None else cfg.rag.distance_threshold
        embedder = embedder or _get_shared_embedder()
        q_vec    = embedder.embed([question], task="query")[0]

        # 벡터 검색 (상위 K*3으로 넓게 검색)
        n_query  = min(cfg.rag.top_k * 3, n_stored)
        results  = self._collection.query(
            query_embeddings=[q_vec],
            n_results=n_query,
            include=["documents", "metadatas", "distances"],  # ids는 항상 포함됨
        )
        raw_count = len(results["documents"][0])

        # ids는 ChromaDB가 항상 반환 (include에 명시 불필요)
        vec_ids  = results.get("ids", [[]])[0]
        vec_docs = results["documents"][0]
        vec_metas = results["metadatas"][0]
        vec_dists = results["distances"][0]
        vec_hits: list[tuple] = list(zip(vec_ids, vec_docs, vec_metas, vec_dists))

        # BM25 검색 (실패해도 벡터 검색으로 폴백)
        try:
            bm25_hits = self._bm25_query(question, n=cfg.rag.top_k * 3)
        except Exception as e:
            logger.warning(f"[RAG] BM25 검색 실패, 벡터 검색 단독 사용: {e}")
            bm25_hits = []

        if bm25_hits and vec_ids:
            # ── RRF (Reciprocal Rank Fusion) ──────────────────────────
            RRF_K    = 60
            rrf: dict[str, float] = {}
            meta_map: dict[str, tuple] = {}
            dist_map: dict[str, float] = {}

            for rank, (id_, doc, meta, dist) in enumerate(vec_hits):
                rrf[id_]      = rrf.get(id_, 0) + 1.0 / (RRF_K + rank + 1)
                meta_map[id_] = (doc, meta)
                dist_map[id_] = dist

            for rank, (id_, doc, meta) in enumerate(bm25_hits):
                rrf[id_]      = rrf.get(id_, 0) + 1.0 / (RRF_K + rank + 1)
                if id_ not in meta_map:
                    meta_map[id_] = (doc, meta)
                if id_ not in dist_map:
                    dist_map[id_] = thr   # BM25 전용 히트 → 거리 불명확, 임계값으로 설정

            sorted_ids = sorted(rrf, key=lambda x: -rrf[x])
            chunks: list[dict] = []
            for id_ in sorted_ids:
                dist = dist_map[id_]
                if dist > thr:
                    logger.debug(f"[RAG] RRF 청크 제외 (dist={dist:.3f}): {id_}")
                    continue
                doc, meta = meta_map[id_]
                chunks.append({"doc": doc, "meta": meta, "dist": dist})
                if len(chunks) >= cfg.rag.top_k:
                    break
            logger.debug(f"[RAG] 하이브리드 검색: vec={len(vec_hits)} bm25={len(bm25_hits)} → {len(chunks)}개")
            return chunks, raw_count

        # BM25 없음 → 벡터 검색 단독
        chunks = []
        for id_, doc, meta, dist in vec_hits:
            if dist > thr:
                logger.debug(f"[RAG] 청크 제외 (dist={dist:.3f} > {thr}): {doc[:40]}")
                continue
            chunks.append({"doc": doc, "meta": meta, "dist": dist})
        return chunks, raw_count

    # ── 답변 생성 ────────────────────────────────────────────────────────
    def query(self, question: str, keep_llm: bool = False,
              threshold: float | None = None) -> tuple[str, list[dict]]:
        """
        질문에 답합니다. 관련 청크가 없으면 "정보 없음" 문장을 반환합니다.

        8GB VRAM 보호: 임베딩 완료 → embedder 해제 → LLM 로드 → 답변 생성

        Args:
            keep_llm:  True면 LLM을 해제하지 않음 (연속 질문 시 재로드 방지)
            threshold: 유사도 임계값 (None이면 config 기본값 사용)

        Returns:
            (answer: str, chunks_meta: list[dict])
        """
        chunks, raw_count = self._retrieve(question, threshold=threshold)

        # 검색 완료 → embedder VRAM 해제 (LLM이 메모리를 쓸 수 있도록)
        self._release_embedder()
        _release_shared_embedder()  # 공유 싱글톤도 해제

        if not chunks:
            if raw_count > 0:
                # 결과는 있으나 전부 임계값 초과 → 별도 안내
                logger.info(f"[RAG] {raw_count}개 결과 모두 임계값 초과 → threshold 응답 반환")
                return _THRESHOLD_REPLY, []
            logger.info("[RAG] 관련 청크 없음 → no-context 응답 반환")
            return _NO_CONTEXT_REPLY, []

        # 참고 자료 구성 (타임스탬프 포함)
        context_lines = [
            f"[{c['meta']['ts']}] {c['doc']}"
            for c in chunks
        ]
        context = "\n".join(context_lines)

        user_prompt = f"[참고 자료]\n{context}\n\n[질문]\n{question}"

        try:
            llm = self._get_llm()
            answer = llm.generate(user_prompt, 800, system=_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"[RAG] LLM 로드/생성 실패: {e}", exc_info=True)
            answer = ""
        finally:
            if not keep_llm:
                self._release_llm()

        if not answer.strip():
            return _NO_CONTEXT_REPLY, []

        chunks_meta = [
            {"ts": c["meta"]["ts"], "text": c["doc"][:300], "dist": round(c["dist"], 3)}
            for c in chunks
        ]
        return answer, chunks_meta

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

            # keep_llm=True → 연속 질문 시 LLM 재로드 없이 즉시 응답
            answer, _ = self.query(question, keep_llm=True)
            print(f"\n답변 > {answer}\n")

        # 챗봇 종료 시 모델 해제
        self._release_llm()
        self._release_embedder()


# ── 멀티 컬렉션 RAG ─────────────────────────────────────────────────────────
class MultiRAGChatbot:
    """
    여러 컬렉션을 동시에 검색하는 RAG 챗봇.
    선택된 컬렉션들에서 청크를 합산 후 상위 K개로 답변 생성.
    """

    def __init__(self, collection_names: list[str]):
        self._bots = {}
        for name in collection_names:
            try:
                self._bots[name] = RAGChatbot(collection_name=name, create_if_missing=False)
            except Exception as e:
                logger.warning(f"[MultiRAG] 컬렉션 '{name}' 건너뜀: {e}")

    def retrieve(self, question: str, threshold: float | None = None) -> list[dict]:
        """모든 컬렉션에서 청크만 검색 반환 (공유 임베더 한 번만 로드)"""
        if not self._bots:
            return []
        # 공유 임베더를 한 번만 로드해 모든 봇에 전달
        embedder = _get_shared_embedder()
        all_chunks: list[dict] = []
        for name, bot in self._bots.items():
            try:
                chunks, _ = bot._retrieve(question, threshold=threshold, embedder=embedder)
            except Exception as e:
                logger.error(f"[RAG] 컬렉션 '{name}' 검색 실패: {e}", exc_info=True)
                chunks = []
            for c in chunks:
                c["source"] = name
            all_chunks.extend(chunks)
        all_chunks.sort(key=lambda x: x["dist"])
        return all_chunks[:cfg.rag.top_k]

    def query(self, question: str, threshold: float | None = None,
              history: list | None = None) -> tuple[str, list[dict]]:
        """여러 컬렉션에서 검색 후 통합 답변 생성. (answer, chunks_meta) 반환"""
        if not self._bots:
            return _NO_CONTEXT_REPLY, []

        top_chunks = self.retrieve(question, threshold)

        if not top_chunks:
            return _NO_CONTEXT_REPLY, []

        context = "\n".join(
            f"[{c['source']} / {c['meta']['ts']}] {c['doc']}"
            for c in top_chunks
        )
        first_bot = next(iter(self._bots.values()))
        llm = first_bot._get_llm()
        user_prompt = f"[참고 자료]\n{context}\n\n[질문]\n{question}"
        try:
            answer = llm.generate(
                user_prompt, 512, system=_SYSTEM_PROMPT, history=history
            ).strip() or _NO_CONTEXT_REPLY
        except Exception as e:
            logger.error(f"[RAG] MultiRAG LLM 생성 실패: {e}", exc_info=True)
            answer = _NO_CONTEXT_REPLY
        # LLM 유지 — 연속 질문 시 재로드 방지 (unload()는 MultiRAGChatbot.unload()에서만 호출)
        chunks_meta = [
            {"source": c["source"], "ts": c["meta"]["ts"],
             "text": c["doc"][:300], "dist": round(c["dist"], 3)}
            for c in top_chunks
        ]
        return answer, chunks_meta

    def query_stream(self, question: str, threshold: float | None = None,
                     history: list | None = None):
        """
        스트리밍으로 답변을 생성합니다.
        yields: ('status', str) 로딩 상태
                ('chunks', list[dict]) 검색 결과
                ('token', str) 답변 토큰
        """
        if not self._bots:
            yield ("token", _NO_CONTEXT_REPLY)
            return

        yield ("status", "searching")
        top_chunks = self.retrieve(question, threshold)
        # 검색 완료 → 공유 임베더 해제 (LLM VRAM 확보)
        _release_shared_embedder()

        # 청크 메타 먼저 전송
        chunks_meta = [
            {"source": c["source"], "ts": c["meta"]["ts"],
             "text": c["doc"][:300], "dist": round(c["dist"], 3)}
            for c in top_chunks
        ]
        yield ("chunks", chunks_meta)

        if not top_chunks:
            yield ("token", _NO_CONTEXT_REPLY)
            return

        context = "\n".join(
            f"[{c['source']} / {c['meta']['ts']}] {c['doc']}"
            for c in top_chunks
        )
        first_bot = next(iter(self._bots.values()))
        llm_loaded = first_bot._llm is not None
        if not llm_loaded:
            yield ("status", "loading_llm")
        try:
            llm = first_bot._get_llm()
        except Exception as e:
            logger.error(f"[RAG] LLM 로드 실패: {e}", exc_info=True)
            yield ("token", _NO_CONTEXT_REPLY)
            return
        yield ("status", "generating")

        # 실험 데이터 청크가 포함된 경우 분석 전용 프롬프트 사용
        has_exp = any(c.get("meta", {}).get("data_type") == "experimental"
                      for c in top_chunks)
        sys_prompt = _ANALYSIS_PROMPT if has_exp else _SYSTEM_PROMPT
        # system/user 분리: Ollama가 system role을 제대로 처리하도록
        user_prompt = f"[참고 자료]\n{context}\n\n[질문]\n{question}"
        has_content = False
        try:
            for token in llm.generate_stream(user_prompt, 800, system=sys_prompt, history=history):
                has_content = True
                yield ("token", token)
        except Exception as e:
            logger.error(f"[RAG] 스트리밍 생성 실패: {e}", exc_info=True)
            if not has_content:
                yield ("token", _NO_CONTEXT_REPLY)
        # LLM 유지 — 연속 질문 시 재로드 방지 (unload()는 MultiRAGChatbot.unload()에서만 호출)
        if not has_content:
            yield ("token", _NO_CONTEXT_REPLY)

    def unload(self):
        _release_shared_embedder()
        for bot in self._bots.values():
            bot._release_llm()


def list_collections() -> list[str]:
    """ChromaDB에 저장된 컬렉션 목록 반환 (prefix 제거, 중복 제거).

    동일 stem에 prefixed/non-prefixed 두 컬렉션이 모두 있으면
    데이터가 있는 것을 우선합니다.
    """
    client = _get_chroma_client()
    prefix = cfg.rag.collection_prefix
    raw = client.list_collections()

    # stem → (actual_col_name, count) 매핑 (데이터가 많은 것 우선)
    stem_map: dict[str, tuple[str, int]] = {}
    for c in raw:
        name = c if isinstance(c, str) else c.name
        stem = name[len(prefix):] if name.startswith(prefix) else name
        try:
            cnt = client.get_collection(name).count()
        except Exception:
            cnt = 0
        # 같은 stem 이면 데이터가 더 많은 것 유지
        prev = stem_map.get(stem)
        if prev is None or cnt > prev[1]:
            stem_map[stem] = (name, cnt)

    # 데이터가 0개인 컬렉션은 숨김 (아직 인덱싱 전이거나 비어있는 잔여물)
    return [stem for stem, (_, cnt) in stem_map.items() if cnt > 0]


def index_summaries(output_dir: "str | Path", collection_name: str) -> int:
    """
    파이프라인이 생성한 요약 파일을 기존 RAG 컬렉션에 추가 인덱싱합니다.
    summary_adult.txt / summary_teen.txt / summary_child.txt → data_type='summary'

    Returns:
        인덱싱된 요약 청크 수 (0이면 요약 파일 없음)
    """
    output_dir = Path(output_dir)
    SUMMARY_FILES = {
        "summary_adult.txt":  "대학/성인",
        "summary_teen.txt":   "고등학생",
        "summary_child.txt":  "초등학생",
    }

    segments: list[dict] = []
    for fname, level in SUMMARY_FILES.items():
        fpath = output_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # 문단(빈 줄 구분) 단위로 분리
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paras:
            segments.append({
                "text":      para,
                "translated": para,
                "start":     0.0,
                "end":       0.0,
                "data_type": "summary",
                "level":     level,
            })

    if not segments:
        logger.info(f"[RAG] 요약 파일 없음: {output_dir}")
        return 0

    bot = RAGChatbot(collection_name=collection_name)

    # 임베딩
    embedder = _get_shared_embedder()
    texts     = [s["text"] for s in segments]
    vecs      = embedder.embed(texts, task="passage")
    for s, vec in zip(segments, vecs):
        s["embedding"] = vec

    # 요약 청크는 sum_ 접두사 ID 사용 (seg_ 와 충돌하지 않음)
    n = len(segments)
    ids       = [f"sum_{i:05d}" for i in range(n)]
    embeddings = [s["embedding"] for s in segments]
    documents  = [s["text"] for s in segments]
    metadatas  = [
        {
            "start":     0.0,
            "end":       0.0,
            "original":  s["text"],
            "ts":        "요약",
            "data_type": "summary",
            "level":     s.get("level", ""),
        }
        for s in segments
    ]

    bot._collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    bot._bm25_dirty = True
    logger.info(f"[RAG] 요약 {n}개 청크 인덱싱 완료  collection={collection_name}")
    return n
