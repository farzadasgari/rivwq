from __future__ import annotations

import os
from pathlib import Path

from rivwq.gee import status


def acquire(path: Path, pid: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = os.getpid() if pid is None else pid
    for _ in range(2):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if status.pid_alive(path):
                raise RuntimeError(f"another process holds {path}") from None
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(str(owner))
        return
    raise RuntimeError(f"could not take {path}")


def release(path: Path) -> None:
    path.unlink(missing_ok=True)
