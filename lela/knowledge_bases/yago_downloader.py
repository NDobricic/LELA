"""Utility to download and cache YAGO 4.5 entities as the default knowledge base."""

import logging
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

from lela.registry import knowledge_bases

logger = logging.getLogger(__name__)

YAGO_URL = "https://yago-knowledge.org/data/yago4.5/yago-entities.jsonl.zip"
YAGO_DIR = Path("data/yago")
YAGO_PATH = YAGO_DIR / "yago-entities.jsonl"


def _format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _make_progress_hook():
    """Return a urlretrieve reporthook that prints a one-line progress bar."""
    state = {"last_pct": -1}

    def hook(block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        if pct != state["last_pct"]:
            state["last_pct"] = pct
            sys.stderr.write(
                f"\r  [{pct:3d}%] {_format_bytes(downloaded)} / {_format_bytes(total_size)}"
            )
            sys.stderr.flush()
            if pct >= 100:
                sys.stderr.write("\n")

    return hook


def ensure_yago_kb() -> str:
    """Return the path to the YAGO entities JSONL, downloading it if needed."""
    if YAGO_PATH.exists():
        logger.info("YAGO KB already on disk at %s", YAGO_PATH)
        return str(YAGO_PATH)

    YAGO_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = YAGO_DIR / "yago-entities.jsonl.zip"

    # Print to stderr so the message is visible even when the root logger is
    # at WARNING (the Python default). This download is a one-time event the
    # user should always see.
    print(
        "\nNo knowledge base was specified; LELA will download the YAGO 4.5\n"
        f"entity dump from {YAGO_URL}\n"
        "This is a one-time download (a few hundred MB); subsequent runs\n"
        f"reuse the cached copy at {YAGO_PATH}.\n",
        file=sys.stderr,
        flush=True,
    )

    urllib.request.urlretrieve(YAGO_URL, zip_path, reporthook=_make_progress_hook())
    print("Extracting archive...", file=sys.stderr, flush=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("yago-entities.jsonl", YAGO_DIR)

    zip_path.unlink()
    print(f"YAGO KB ready at {YAGO_PATH}\n", file=sys.stderr, flush=True)
    return str(YAGO_PATH)


@knowledge_bases.register("yago")
def yago_kb(
    cache_dir: Optional[str] = None,
    cancel_event: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
):
    """KB factory that auto-downloads YAGO 4.5 if missing, then loads it as
    a JSONLKnowledgeBase. Lets users write ``{"name": "yago"}`` in their
    config without worrying about the file path.
    """
    from lela.knowledge_bases.jsonl import JSONLKnowledgeBase

    path = ensure_yago_kb()
    return JSONLKnowledgeBase(
        path=path,
        cache_dir=cache_dir,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
