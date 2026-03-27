"""
utils/memory.py
────────────────────────────────────────────────────────────────────
VRAM / RAM 메모리 관리 유틸리티.

PC(CUDA 8GB) 환경에서 각 파이프라인 단계 종료 후
명시적으로 VRAM 캐시를 해제해 다음 단계가 메모리를 확보할 수 있도록 합니다.
Mac(MPS) 환경에서는 가비지 컬렉션만 수행합니다.
"""

import gc
import logging

logger = logging.getLogger(__name__)


def log_vram(tag: str = "") -> None:
    """현재 VRAM 사용량을 로그에 출력합니다. CUDA가 없으면 조용히 무시."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc   = torch.cuda.memory_allocated()  / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            total   = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(
                f"[VRAM] {tag:30s} | "
                f"Allocated: {alloc:.2f} GB | "
                f"Reserved: {reserved:.2f} GB | "
                f"Total: {total:.1f} GB"
            )
    except (ImportError, RuntimeError):
        pass


def clear_vram(verbose: bool = True) -> None:
    """
    GPU VRAM 캐시를 해제합니다.
    - CUDA (PC/RTX 4060 Ti): gc → empty_cache → ipc_collect → synchronize
    - MPS  (Mac Apple Silicon): gc.collect만 수행 (MPS는 별도 캐시 API 없음)
    - CPU  : gc.collect만 수행
    """
    # gc를 먼저 실행해야 Python 객체가 해제되고 CUDA 텐서도 반환됨
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            # Reserved가 여전히 높으면 한 번 더 시도
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            if reserved > 0.5:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if verbose:
                log_vram("after clear_vram")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if verbose:
                logger.debug("[VRAM] MPS device — skipping CUDA cache clear")
    except ImportError:
        pass


def vram_context(tag: str = ""):
    """
    with 문으로 사용하는 컨텍스트 매니저.
    블록 진입·종료 시 VRAM 로그를 찍고, 종료 시 캐시를 해제합니다.

    사용 예:
        with vram_context("STT"):
            model = WhisperModel(...)
            segments = model.transcribe(...)
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        log_vram(f"{tag} start")
        try:
            yield
        finally:
            clear_vram(verbose=False)
            log_vram(f"{tag} end")

    return _ctx()
