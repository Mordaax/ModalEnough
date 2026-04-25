"""Atomic artifacts and a cross-platform, process-lifetime directory lock."""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def atomic_text(path: Path, text: str) -> None:
    """Flush a replacement beside its destination before atomically installing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    """Write strict JSON with the same durability guarantees as text artifacts."""
    atomic_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


@contextmanager
def directory_lock(directory: Path) -> Iterator[None]:
    """Prevent concurrent writers; the OS releases the lock after a crash."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".modalenough.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"Another process is already writing to {directory}."
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
