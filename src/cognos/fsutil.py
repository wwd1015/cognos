"""Atomic file writes that survive Windows.

Checkpoints and run state are written to a temp file and renamed over the target, so a concurrent
reader (the UI polling, another process) never sees a torn file. On Windows the rename can fail
transiently with ``PermissionError`` while a reader or an antivirus scanner briefly holds the target
open, so the rename is retried with a short backoff before giving up.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def atomic_write(path: str | Path, text: str, *, attempts: int = 20) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.02 * (i + 1))
